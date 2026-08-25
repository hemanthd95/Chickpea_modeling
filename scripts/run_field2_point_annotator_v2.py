#!/usr/bin/env python
"""Run the main-only Field 2 vegetation chickpea-review annotator."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from chickpea_ssl.field2_chickpea_review import MANUAL_LABEL_PROVENANCE, validate_policy
from chickpea_ssl.field2_point_spectra import PredictionFreeSpectrumStore
from chickpea_ssl.field2_blind_review import atomic_write_bytes, atomic_write_json, validate_annotation_display_contract
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_point_annotator import ANNOTATION_COLUMNS, HTML, PointAnnotationStore


MANUAL_LABELS = tuple(MANUAL_LABEL_PROVENANCE)
OPTIONAL_WEED_SUBTYPES = ("ordinary_weed", "tall_grass_weed", "other_weed")
MANUAL_COLUMNS = ANNOTATION_COLUMNS + ["manual_decision", "optional_weed_subtype", "reference_provenance"]


def manual_csv_bytes(payload: dict) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=MANUAL_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for sample_id in payload["sample_order"]:
        record = dict(payload["annotations"][sample_id])
        record["source_preview_checksums"] = json.dumps(record["source_preview_checksums"], sort_keys=True)
        writer.writerow(record)
    return stream.getvalue().encode()


class ChickpeaReviewStore(PointAnnotationStore):
    """Preserve all 800 records while accepting responses only for the frozen queue."""

    def __init__(self, *args, references: pd.DataFrame, policy_contract_hash: str, **kwargs):
        super().__init__(*args, allowed_labels=MANUAL_LABELS, boundary_correction_exclusive=False, **kwargs)
        manual = references[references.manual_review_required.astype(str).str.lower().eq("true")]
        self.queue_ids = manual.sample_id.astype(str).tolist()
        if len(self.queue_ids) != 477 or not set(self.queue_ids).issubset(set(self.sample_order)):
            raise ValueError("Frozen manual chickpea-review queue must contain exactly 477 v2 main records")
        self.queue_set = set(self.queue_ids)
        self.policy_contract_hash = policy_contract_hash

    @staticmethod
    def _manual_fields(record: dict) -> None:
        record.setdefault("manual_decision", record.get("selected_label", ""))
        record.setdefault("optional_weed_subtype", "")
        record.setdefault("reference_provenance", MANUAL_LABEL_PROVENANCE.get(record.get("selected_label", ""), ""))

    def load(self) -> dict:
        payload = json.loads(self.json_path.read_text()) if self.json_path.exists() else self.initial_payload()
        payload, _ = self.migrate_display_version(payload)
        payload = self.migrate_area_membership(payload)
        payload["chickpea_review_policy_contract_sha256"] = self.policy_contract_hash
        for record in payload["annotations"].values():
            self._manual_fields(record)
        self.validate(payload)
        return payload

    def validate(self, payload: dict) -> dict[str, list[str]]:
        issues = super().validate(payload)
        if payload.get("chickpea_review_policy_contract_sha256") != self.policy_contract_hash:
            raise ValueError("Point annotations use the wrong chickpea-review policy contract")
        for sample_id in self.sample_order:
            record = payload["annotations"][sample_id]
            self._manual_fields(record)
            item = issues[sample_id]
            label = record.get("selected_label", "")
            if record.get("manual_decision") != label:
                item.append("manual_decision_must_equal_selected_label")
            if record.get("reference_provenance") != MANUAL_LABEL_PROVENANCE.get(label, ""):
                item.append("manual_reference_provenance_mismatch")
            subtype = record.get("optional_weed_subtype", "")
            if subtype and subtype not in OPTIONAL_WEED_SUBTYPES:
                item.append("unknown_optional_weed_subtype")
            if subtype and label != "weed_unspecified":
                item.append("weed_subtype_requires_not_chickpea")
            if sample_id not in self.queue_set and (
                label or record.get("reviewed") or record.get("confidence") or subtype
                or record.get("boundary_needs_correction")
            ):
                item.append("rule_or_area_constrained_record_not_manually_editable")
        return issues

    def save(self, payload: dict) -> tuple[int, int, int]:
        with self.lock:
            current = self.load()
            if int(payload.get("revision", -1)) != int(current["revision"]):
                raise ValueError("Annotation revision conflict; reload before saving")
            issues = self.validate(payload)
            invalid = {sample_id: values for sample_id, values in issues.items() if values}
            if invalid:
                raise ValueError("Logical chickpea-review validation failed: " + json.dumps(invalid, sort_keys=True))
            saved = json.loads(json.dumps(payload))
            saved["revision"] = current["revision"] + 1
            saved["updated_utc"] = datetime.now(timezone.utc).isoformat()
            saved["automatic_metadata"] = {
                "biological_labels_assigned_automatically": False,
                "reserve_frame_exposed": False, "predictions_or_probabilities_used": False,
                "rule_based_references_stored_separately": True, "manual_queue_count": 477,
            }
            atomic_write_json(self.json_path, saved)
            atomic_write_bytes(self.csv_path, manual_csv_bytes(saved))
            atomic_write_bytes(self.audit_path, self._audit_bytes(saved, issues))
            self._write_overview(saved)
            reviewed = sum(saved["annotations"][sample_id]["reviewed"] for sample_id in self.queue_ids)
            return len(saved["annotations"]), reviewed, saved["revision"]


def build_store(
    project: Path, v2: dict, policy: dict, annotation_root: Path | None = None,
) -> tuple[ChickpeaReviewStore, PredictionFreeSpectrumStore, dict]:
    result = validate_policy(project, policy)
    base = yaml.safe_load((project / v2["base_config"]).read_text())
    area = yaml.safe_load((project / v2["area_config"]).read_text())
    main = result["main"].copy()
    main["zone_type"] = main["domain_name"]
    main["domain"] = main["domain_reporting_stratum"]
    display_path = project / base["annotation_display"]["display_contract"]
    display = validate_annotation_display_contract(project, base, display_path)
    policy_path = project / policy["outputs"]["contract"]
    output_root = annotation_root.resolve() if annotation_root is not None else project / v2["point_annotation"]["output_root"]
    store = ChickpeaReviewStore(
        project, main, result["contract"]["inputs"]["v2_main_frame_sha256"],
        project / base["review"]["package_manifest"], display["manifest"],
        base["annotation_display"]["version"], sha256(display_path),
        output_root,
        sha256(project / area["freeze"]["contract"]),
        references=result["references"], policy_contract_hash=sha256(policy_path),
    )
    if annotation_root is not None and not store.json_path.is_file():
        if store.save(store.load()) != (800, 0, 1):
            raise RuntimeError("Temporary neutral chickpea-review store failed to initialize")
    if not store.json_path.is_file():
        raise RuntimeError("The preserved neutral v2 annotation store is required")
    spectra = PredictionFreeSpectrumStore(
        project, project / base["inputs"]["readiness_inventory"],
        project / base["inputs"]["valid_support_manifest"], main,
    )
    return store, spectra, result


def handler_factory(store: ChickpeaReviewStore, spectra: PredictionFreeSpectrumStore, policy: dict):
    label_display = policy["manual_review"]["decisions"]

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(content)

        def do_GET(self):
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}: return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/field2-point-viewer.js": return self.send_bytes(Path(__file__).with_name("field2_point_viewer.js").read_bytes(), "text/javascript; charset=utf-8")
            if clean == "/api/samples":
                queue = store.frame[store.frame.sample_id.astype(str).isin(store.queue_set)].copy()
                queue["manual_review_required"] = True
                return self.send_bytes(queue.to_json(orient="records").encode(), "application/json")
            if clean == "/api/manifest": return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if clean == "/api/annotations": return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if clean.startswith("/api/spectrum/"):
                sample_id = clean.rsplit("/", 1)[-1]
                if sample_id not in store.queue_set: return self.send_bytes(b"Unknown or non-queue sample", "text/plain", 404)
                try: return self.send_bytes(json.dumps(spectra.sample_spectrum(sample_id)).encode(), "application/json")
                except (KeyError, ValueError) as error: return self.send_bytes(str(error).encode(), "text/plain", 404)
            if clean == "/api/schema":
                return self.send_bytes(json.dumps({
                    "workflow": "chickpea_review_v1", "labels": MANUAL_LABELS,
                    "label_display": label_display, "confidence": ["high", "medium", "low"],
                    "optional_weed_subtypes": OPTIONAL_WEED_SUBTYPES,
                    "label_shortcuts": {label: str(index + 1) for index, label in enumerate(MANUAL_LABELS)},
                    "confidence_shortcuts": {"high": "h", "medium": "m", "low": "l"},
                    "frame": "main", "total_main_count": 800, "manual_review_count": 477,
                    "reserve_exposed": False, "default_layer": "natural_rgb",
                    "display_version": store.display_version, "area_geometry_frozen": True,
                    "raw_spectrum_available": True, "model_outputs_available": False,
                    "center_instruction": policy["manual_review"]["center_instruction"],
                }).encode(), "application/json")
            if clean.startswith("/layers/"):
                parts = clean.strip("/").split("/")
                if len(parts) == 3 and parts[2].endswith(".png"):
                    try: return self.send_bytes(store.layer_path(parts[1], parts[2][:-4]).read_bytes(), "image/png")
                    except (KeyError, FileNotFoundError): pass
            return self.send_bytes(b"Not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/annotations": return self.send_bytes(b"Not found", "text/plain", 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 50_000_000: raise ValueError("Invalid request size")
                records, reviewed, revision = store.save(json.loads(self.rfile.read(length)))
                return self.send_bytes(json.dumps({"records": records, "reviewed": reviewed, "revision": revision, "file": str(store.json_path)}).encode(), "application/json")
            except Exception as error: return self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)

        def log_message(self, format, *args): return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_area_sampling_v2.yaml"), type=Path)
    parser.add_argument("--policy-config", default=Path("configs/field2_chickpea_review_policy.yaml"), type=Path)
    parser.add_argument("--annotation-root", type=Path, help="Isolated annotation root for automated testing only")
    parser.add_argument("--port", type=int); parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(); project = Path(yaml.safe_load(args.paths.read_text())["project_root"]).resolve()
    v2, policy = yaml.safe_load(args.config.read_text()), yaml.safe_load(args.policy_config.read_text())
    store, spectra, _ = build_store(project, v2, policy, args.annotation_root); current = store.load()
    port = args.port or int(v2["point_annotation"]["port"])
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(store, spectra, policy)); url = f"http://127.0.0.1:{port}"
    reviewed = sum(current["annotations"][sample_id]["reviewed"] for sample_id in store.queue_ids)
    print(f"Field 2 prediction-free chickpea-review annotator: {url}", flush=True)
    print(f"Manual queue: 477 vegetation points; reviewed: {reviewed}; total main: 800", flush=True)
    print("Reserve served: 0; default layer: natural RGB; no class preselected", flush=True)
    print(f"Resume file: {store.json_path}", flush=True)
    if not args.no_browser: threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nChickpea-review annotator stopped. Saved responses remain on disk.", flush=True)


if __name__ == "__main__": main()
