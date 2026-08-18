#!/usr/bin/env python
"""Run the prediction-free main-frame Field 2 point annotator."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from chickpea_ssl.field2_blind_review import (
    CONFIDENCE_VALUES, POINT_LABELS, atomic_write_bytes,
    atomic_write_json, require_main_sampling_frame, validate_annotation_display_contract,
    verify_preview_hashes,
)
from chickpea_ssl.field2_area_review import ALLEY_POINT_LABELS, require_frozen_area_contract
from chickpea_ssl.field2_readiness import sha256


ANNOTATION_COLUMNS = [
    "sample_id", "cube_id", "frozen_cube_role", "selected_label", "confidence",
    "investigator_note", "reviewed", "review_timestamp", "reviewer_identifier",
    "role_contradiction", "source_preview_checksums", "sampling_frame_sha256",
    "annotation_display_version", "annotation_display_contract_sha256",
    "natural_rgb_preview_sha256", "requires_visual_rereview",
    "area_zone_contract_sha256", "zone_type", "domain", "boundary_needs_correction",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_annotation(
    row,
    frame_hash: str,
    display_version: str,
    display_contract_hash: str,
    natural_rgb_hash: str,
) -> dict:
    return {
        "sample_id": str(row.sample_id), "cube_id": str(row.cube_id),
        "frozen_cube_role": str(row.cube_evaluation_role), "selected_label": "",
        "confidence": "", "investigator_note": "", "reviewed": False,
        "review_timestamp": "", "reviewer_identifier": "", "role_contradiction": False,
        "source_preview_checksums": json.loads(str(row.source_preview_checksums)),
        "sampling_frame_sha256": frame_hash,
        "annotation_display_version": display_version,
        "annotation_display_contract_sha256": display_contract_hash,
        "natural_rgb_preview_sha256": natural_rgb_hash,
        "requires_visual_rereview": False,
        "area_zone_contract_sha256": "", "zone_type": "", "domain": "",
        "boundary_needs_correction": False,
    }


def annotation_csv_bytes(payload: dict) -> bytes:
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=ANNOTATION_COLUMNS); writer.writeheader()
    for sample_id in payload["sample_order"]:
        record = dict(payload["annotations"][sample_id])
        record["source_preview_checksums"] = json.dumps(record["source_preview_checksums"], sort_keys=True)
        writer.writerow(record)
    return buffer.getvalue().encode()


class PointAnnotationStore:
    """Atomic, resumable, main-frame-only annotation state."""

    def __init__(
        self,
        project: Path,
        main_frame: pd.DataFrame,
        frame_hash: str,
        manifest_path: Path,
        natural_rgb_manifest: pd.DataFrame,
        display_version: str,
        display_contract_hash: str,
        output_root: Path,
        area_contract_hash: str = "",
    ):
        self.project = project.resolve(); self.frame = main_frame.copy(); self.frame_hash = frame_hash
        self.display_version = display_version; self.display_contract_hash = display_contract_hash
        self.area_contract_hash = area_contract_hash
        if set(self.frame.sampling_frame) != {"main"} or len(self.frame) != 800:
            raise ValueError("Point annotator accepts only the exact frozen 800-point main frame")
        self.sample_order = self.frame.sample_id.astype(str).tolist()
        self.by_sample = {str(row.sample_id): row for row in self.frame.itertuples(index=False)}
        self.manifest_frame = pd.read_csv(manifest_path).fillna("").sort_values("cube_id")
        preview_issues = verify_preview_hashes(self.manifest_frame, self.project)
        if preview_issues:
            raise ValueError(f"Prediction-free preview checksum validation failed: {preview_issues}")
        self.manifest = {str(row.cube_id): row for row in self.manifest_frame.itertuples(index=False)}
        if natural_rgb_manifest.cube_id.astype(str).tolist() != self.manifest_frame.cube_id.astype(str).tolist():
            raise ValueError("Natural RGB manifest differs from the 40-cube review manifest")
        self.natural_rgb = {
            str(row.cube_id): row for row in natural_rgb_manifest.itertuples(index=False)
        }
        self.output_root = output_root.resolve(); self.output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_root / "field2_blind_main_point_annotations.json"
        self.csv_path = self.output_root / "field2_blind_main_point_annotations.csv"
        self.audit_path = self.output_root / "field2_blind_main_point_annotation_audit.csv"
        self.overview_path = self.output_root / "field2_blind_main_point_annotation_overview.png"
        self.lock = threading.Lock()

    def initial_payload(self) -> dict:
        return {
            "version": "field2_blind_main_point_annotations_v1", "revision": 0,
            "sampling_frame_sha256": self.frame_hash, "sample_order": self.sample_order,
            "annotation_display_version": self.display_version,
            "annotation_display_contract_sha256": self.display_contract_hash,
            "area_zone_contract_sha256": self.area_contract_hash,
            "automatic_metadata": {
                "biological_labels_assigned_automatically": False,
                "reserve_frame_exposed": False, "predictions_or_probabilities_used": False,
            },
            "annotations": {
                sample_id: default_annotation(
                    self.by_sample[sample_id], self.frame_hash, self.display_version,
                    self.display_contract_hash,
                    str(self.natural_rgb[str(self.by_sample[sample_id].cube_id)].natural_rgb_sha256),
                )
                for sample_id in self.sample_order
            },
        }

    def migrate_display_version(self, payload: dict) -> tuple[dict, int]:
        """Preserve annotations while marking old reviewed records for visual re-review."""
        migrated = json.loads(json.dumps(payload))
        prior_version = migrated.get("annotation_display_version", "")
        reviewed_requiring_rereview = 0
        migrated["annotation_display_version"] = self.display_version
        migrated["annotation_display_contract_sha256"] = self.display_contract_hash
        for sample_id in self.sample_order:
            record = migrated["annotations"][sample_id]
            old_record_version = record.get("annotation_display_version", prior_version)
            requires = bool(record.get("reviewed")) and old_record_version != self.display_version
            record["annotation_display_version"] = self.display_version
            record["annotation_display_contract_sha256"] = self.display_contract_hash
            record["natural_rgb_preview_sha256"] = str(
                self.natural_rgb[str(record["cube_id"])].natural_rgb_sha256
            )
            record["requires_visual_rereview"] = requires or bool(
                record.get("requires_visual_rereview", False)
            )
            reviewed_requiring_rereview += int(record["requires_visual_rereview"])
        migrated["automatic_metadata"] = {
            "biological_labels_assigned_automatically": False,
            "reserve_frame_exposed": False,
            "predictions_or_probabilities_used": False,
            "display_version_migration_changed_biological_labels": False,
        }
        return migrated, reviewed_requiring_rereview

    def migrate_area_membership(self, payload: dict) -> dict:
        """Attach immutable frozen zone membership without changing biological labels."""
        migrated = json.loads(json.dumps(payload))
        migrated["area_zone_contract_sha256"] = self.area_contract_hash
        for sample_id in self.sample_order:
            record = migrated["annotations"][sample_id]
            row = self.by_sample[sample_id]
            record["area_zone_contract_sha256"] = self.area_contract_hash
            record["zone_type"] = str(getattr(row, "zone_type", ""))
            record["domain"] = str(getattr(row, "domain", ""))
            record["boundary_needs_correction"] = bool(record.get("boundary_needs_correction", False))
        return migrated

    def load(self) -> dict:
        payload = json.loads(self.json_path.read_text()) if self.json_path.exists() else self.initial_payload()
        payload, _ = self.migrate_display_version(payload)
        payload = self.migrate_area_membership(payload)
        self.validate(payload)
        return payload

    def validate(self, payload: dict) -> dict[str, list[str]]:
        if payload.get("version") != "field2_blind_main_point_annotations_v1":
            raise ValueError("Unknown point-annotation schema version")
        if payload.get("sampling_frame_sha256") != self.frame_hash or payload.get("sample_order") != self.sample_order:
            raise ValueError("Point annotations differ from the frozen main frame")
        if payload.get("annotation_display_version") != self.display_version:
            raise ValueError("Point annotations use the wrong annotation-display version")
        if payload.get("annotation_display_contract_sha256") != self.display_contract_hash:
            raise ValueError("Point annotations use the wrong annotation-display contract")
        if payload.get("area_zone_contract_sha256", "") != self.area_contract_hash:
            raise ValueError("Point annotations use the wrong frozen area-zone contract")
        annotations = payload.get("annotations")
        if not isinstance(annotations, dict) or list(annotations) != self.sample_order:
            raise ValueError("Point annotation inventory is missing, duplicated, reordered, or contains reserve samples")
        issues: dict[str, list[str]] = {}
        for sample_id in self.sample_order:
            row, record = self.by_sample[sample_id], annotations[sample_id]
            item_issues = []
            if record.get("sample_id") != sample_id or record.get("cube_id") != row.cube_id:
                item_issues.append("sample_or_cube_mismatch")
            if record.get("frozen_cube_role") != row.cube_evaluation_role:
                item_issues.append("frozen_cube_role_mismatch")
            label, confidence, reviewed = record.get("selected_label", ""), record.get("confidence", ""), record.get("reviewed")
            if label and label not in POINT_LABELS: item_issues.append("unknown_label")
            if confidence and confidence not in CONFIDENCE_VALUES: item_issues.append("unknown_confidence")
            if not isinstance(reviewed, bool): item_issues.append("reviewed_must_be_boolean")
            if reviewed and not label: item_issues.append("reviewed_requires_label")
            if reviewed and confidence not in CONFIDENCE_VALUES: item_issues.append("reviewed_requires_confidence")
            if reviewed and not str(record.get("review_timestamp", "")).strip(): item_issues.append("reviewed_requires_timestamp")
            expected_contradiction = label == "chickpea" and row.cube_evaluation_role == "chickpea_absent_negative_control"
            if record.get("role_contradiction") is not expected_contradiction: item_issues.append("role_contradiction_mismatch")
            if record.get("sampling_frame_sha256") != self.frame_hash: item_issues.append("sampling_frame_hash_mismatch")
            if record.get("annotation_display_version") != self.display_version: item_issues.append("annotation_display_version_mismatch")
            if record.get("annotation_display_contract_sha256") != self.display_contract_hash: item_issues.append("annotation_display_contract_mismatch")
            expected_rgb_hash = str(self.natural_rgb[str(row.cube_id)].natural_rgb_sha256)
            if record.get("natural_rgb_preview_sha256") != expected_rgb_hash: item_issues.append("natural_rgb_preview_hash_mismatch")
            expected_zone = str(getattr(row, "zone_type", "")); expected_domain = str(getattr(row, "domain", ""))
            if record.get("area_zone_contract_sha256", "") != self.area_contract_hash: item_issues.append("area_zone_contract_hash_mismatch")
            if record.get("zone_type", "") != expected_zone or record.get("domain", "") != expected_domain: item_issues.append("frozen_zone_membership_mismatch")
            if not isinstance(record.get("boundary_needs_correction"), bool): item_issues.append("boundary_needs_correction_must_be_boolean")
            if record.get("boundary_needs_correction") and label: item_issues.append("boundary_correction_must_not_force_label")
            if record.get("boundary_needs_correction") and reviewed: item_issues.append("boundary_correction_cannot_be_reviewed")
            if expected_zone == "alley" and label and label not in ALLEY_POINT_LABELS: item_issues.append("label_not_allowed_in_alley")
            if not isinstance(record.get("requires_visual_rereview"), bool): item_issues.append("requires_visual_rereview_must_be_boolean")
            expected_previews = json.loads(str(row.source_preview_checksums))
            if record.get("source_preview_checksums") != expected_previews: item_issues.append("source_preview_checksums_mismatch")
            for key in ("investigator_note", "review_timestamp", "reviewer_identifier"):
                if not isinstance(record.get(key, ""), str): item_issues.append(f"{key}_must_be_string")
            issues[sample_id] = item_issues
        return issues

    def _audit_bytes(self, payload: dict, issues: dict[str, list[str]]) -> bytes:
        fields = ["sample_id", "cube_id", "reviewed", "selected_label", "logical_validation", "role_contradiction", "issues"]
        buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=fields); writer.writeheader()
        for sample_id in self.sample_order:
            record = payload["annotations"][sample_id]
            writer.writerow({"sample_id": sample_id, "cube_id": record["cube_id"], "reviewed": record["reviewed"], "selected_label": record["selected_label"], "logical_validation": "pass" if not issues[sample_id] else "fail", "role_contradiction": record["role_contradiction"], "issues": "|".join(issues[sample_id])})
        return buffer.getvalue().encode()

    def _write_overview(self, payload: dict) -> None:
        fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
        for axis, (cube_id, group) in zip(axes.flat, self.frame.groupby("cube_id", sort=True)):
            image = plt.imread(self.layer_path(cube_id, "false_colour")); axis.imshow(image)
            reviewed = sum(payload["annotations"][sample_id]["reviewed"] for sample_id in group.sample_id)
            total = len(group); color = "#2ca25f" if reviewed == total else "#f0ad4e" if reviewed else "#7f8c8d"
            for spine in axis.spines.values(): spine.set_visible(True); spine.set_linewidth(5); spine.set_color(color)
            axis.set_title(f"{cube_id}\n{reviewed}/{total} reviewed", fontsize=8); axis.set_xticks([]); axis.set_yticks([])
        fig.suptitle("Field 2 blind MAIN-frame annotation progress (no reserve points)")
        descriptor, name = tempfile.mkstemp(prefix=f".{self.overview_path.name}.", suffix=".png", dir=self.output_root)
        os.close(descriptor); temporary = Path(name)
        try:
            fig.savefig(temporary, dpi=160); plt.close(fig); temporary.replace(self.overview_path)
        except Exception:
            plt.close(fig); temporary.unlink(missing_ok=True); raise

    def save(self, payload: dict) -> tuple[int, int, int]:
        with self.lock:
            current = self.load()
            if int(payload.get("revision", -1)) != int(current["revision"]):
                raise ValueError("Annotation revision conflict; reload before saving")
            issues = self.validate(payload)
            invalid = {sample_id: values for sample_id, values in issues.items() if values}
            if invalid: raise ValueError("Logical annotation validation failed: " + json.dumps(invalid, sort_keys=True))
            preview_issues = verify_preview_hashes(self.manifest_frame, self.project)
            if preview_issues: raise ValueError(f"Preview checksums changed: {preview_issues}")
            saved = json.loads(json.dumps(payload)); saved["revision"] = current["revision"] + 1; saved["updated_utc"] = utc_now()
            saved["automatic_metadata"] = {"biological_labels_assigned_automatically": False, "reserve_frame_exposed": False, "predictions_or_probabilities_used": False, "display_version_migration_changed_biological_labels": False}
            atomic_write_json(self.json_path, saved); atomic_write_bytes(self.csv_path, annotation_csv_bytes(saved)); atomic_write_bytes(self.audit_path, self._audit_bytes(saved, issues)); self._write_overview(saved)
            reviewed = sum(record["reviewed"] for record in saved["annotations"].values())
            return len(saved["annotations"]), reviewed, saved["revision"]

    def public_manifest(self) -> dict:
        result = {}
        for cube_id, row in self.manifest.items():
            rgb = self.natural_rgb[cube_id]
            selected = json.loads(str(rgb.selected_bands_json))
            result[cube_id] = {
                "preview_step": int(row.preview_step), "natural_rgb_preview_step": 1,
                "natural_rgb_width": int(rgb.width), "natural_rgb_height": int(rgb.height),
                "natural_rgb_wavelengths_nm": {
                    channel: selected[channel]["selected_wavelength_nm"]
                    for channel in ("red", "green", "blue")
                },
            }
        return result

    def layer_path(self, cube_id: str, layer: str) -> Path:
        if cube_id not in self.manifest or layer not in {"natural_rgb", "false_colour", "pca", "stored_index", "support_outline"}:
            raise KeyError((cube_id, layer))
        if layer == "natural_rgb":
            return self.project / str(self.natural_rgb[cube_id].natural_rgb_path)
        return self.project / str(getattr(self.manifest[cube_id], f"{layer}_path"))


LEGACY_NONINTERACTIVE_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Field 2 blind MAIN point annotator</title>
<style>:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#101418;color:#edf2f7}header{padding:9px 12px;background:#182028;position:sticky;top:0;z-index:5;border-bottom:1px solid #3f4b56}.bar{display:flex;flex-wrap:wrap;gap:7px;align-items:center}button,select,input,textarea{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px}.primary{background:#18794e}.danger{background:#7f1d1d}.layout{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:calc(100vh - 126px)}.stage{display:grid;grid-template-columns:minmax(260px,1fr) minmax(320px,1.15fr);gap:10px;padding:10px;overflow:auto;background:#050708}.panel{display:flex;min-width:0;flex-direction:column;align-items:center;justify-content:center;gap:5px}.panel-title{font-size:.8rem;color:#b9c4ce}canvas{display:block;max-width:100%;max-height:calc(100vh - 190px);border:1px solid #52606d;background:#000}.overview{image-rendering:auto}.detail{image-rendering:pixelated}.form{padding:12px;overflow:auto;background:#151b21}.field{display:block;margin:8px 0}.field span{display:block;color:#b9c4ce;font-size:.84rem}.form textarea,.form input,.form select{width:100%}#status{padding-top:7px}.dirty,.warning{color:#ffd166}.rereview{color:#ff9f1c}.help{font-size:.78rem;color:#b9c4ce}.metadata{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:.8rem}@media(max-width:1100px){.stage{grid-template-columns:1fr}.layout{grid-template-columns:minmax(0,1fr) 350px}canvas{max-height:42vh}}@media(max-width:760px){.layout{grid-template-columns:1fr;height:auto}.form{max-height:none}}</style></head><body>
<header><div class="bar"><button id="prev">← Previous</button><span id="counter"></span><button id="next">Next →</button><label>Cube <select id="cubeFilter"></select></label><label>Role <select id="roleFilter"></select></label><label>Review <select id="reviewFilter"><option value="">All records</option><option value="rereview">Requires visual re-review</option></select></label><label>View <select id="layer"><option value="natural_rgb">Natural RGB (default)</option><option value="false_colour">False colour reflectance</option><option value="pca">PCA</option><option value="stored_index">Stored scalar index</option><option value="support_outline">Support boundary</option></select></label><label>Zoom <select id="zoom"><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option><option value="16">16×</option></select></label><label><input id="grid" type="checkbox"> Pixel grid ≥8×</label><button id="save" class="primary">Save all</button></div><div id="status">Loading frozen MAIN frame…</div><div class="help">Shortcuts: ←/→ navigate · N/F/P/I/B views · 2/4/8/X zoom · G grid · Ctrl/Cmd+S saves. The outlined square is the exact sampled pixel; surrounding pixels are context only.</div></header>
<div class="layout"><div class="stage"><section class="panel"><div class="panel-title">Complete cube overview</div><canvas id="overview" class="overview"></canvas></section><section class="panel"><div class="panel-title">Magnified sampled-pixel neighborhood</div><canvas id="detail" class="detail" width="640" height="640"></canvas></section></div><aside class="form"><h3>Investigator annotation</h3><label class="field"><span>Selected label (no default)</span><select id="label"><option value="">Unlabeled</option></select></label><label class="field"><span>Confidence</span><select id="confidence"><option value="">Not set</option></select></label><label class="field"><span>Reviewer identifier (optional)</span><input id="reviewer"></label><label class="field"><span>Investigator note (optional)</span><textarea id="note"></textarea></label><button id="review" class="primary">Mark reviewed</button> <button id="clear" class="danger">Clear current</button><p id="rereview" class="rereview"></p><p id="contradiction" class="warning"></p><h3>Frozen sample metadata</h3><div id="metadata" class="metadata"></div><p class="help">Mixed labels apply only when the sampled pixel is spatially mixed. Uncertainty is preferable to a forced class. Chickpea remains available on negative-control cubes; selecting it records a role contradiction without changing the frozen cube role. Reserve samples are locked and unavailable here.</p></aside></div>
<script>
const $=id=>document.getElementById(id),overview=$('overview'),overviewCtx=overview.getContext('2d'),detail=$('detail'),detailCtx=detail.getContext('2d');let samples=[],visible=[],manifest={},schema={},state={},index=0,image=new Image(),dirty=false;function sample(){return visible[index]}function record(){return state.annotations[sample().sample_id]}function status(x,bad=dirty){$('status').textContent=x;$('status').className=bad?'dirty':''}function setDirty(){dirty=true;status('UNSAVED changes — use Save all.',true)}function imageCoordinates(s,m){const step=$('layer').value==='natural_rgb'?+m.natural_rgb_preview_step:+m.preview_step;return{step,cx:(+s.column+.5)/step,cy:(+s.row+.5)/step}}
function filter(){const cube=$('cubeFilter').value,role=$('roleFilter').value,review=$('reviewFilter').value,current=sample()?.sample_id;visible=samples.filter(s=>(!cube||s.cube_id===cube)&&(!role||s.cube_evaluation_role===role)&&(!review||state.annotations[s.sample_id].requires_visual_rereview));let found=visible.findIndex(s=>s.sample_id===current);index=found>=0?found:0;if(!visible.length){status('No MAIN samples match this filter.',true);return}load()}
function drawOverview(s,m,cx,cy,step){const maxW=760,maxH=720,scale=Math.min(maxW/image.naturalWidth,maxH/image.naturalHeight,1);overview.width=Math.max(1,Math.round(image.naturalWidth*scale));overview.height=Math.max(1,Math.round(image.naturalHeight*scale));overviewCtx.imageSmoothingEnabled=true;overviewCtx.drawImage(image,0,0,overview.width,overview.height);const x=(cx-.5/step)*scale,y=(cy-.5/step)*scale,size=Math.max(2,scale/step);overviewCtx.strokeStyle='#ff2d55';overviewCtx.lineWidth=Math.max(1,Math.min(2,scale));overviewCtx.strokeRect(x,y,size,size)}
function drawDetail(cx,cy){const zoom=+$('zoom').value,w=detail.width,h=detail.height,sourceW=w/zoom,sourceH=h/zoom;detailCtx.imageSmoothingEnabled=false;detailCtx.clearRect(0,0,w,h);detailCtx.drawImage(image,cx-sourceW/2,cy-sourceH/2,sourceW,sourceH,0,0,w,h);if($('grid').checked&&zoom>=8){detailCtx.strokeStyle='rgba(255,255,255,.28)';detailCtx.lineWidth=1;detailCtx.beginPath();const ox=w/2-(Math.floor(sourceW/2)*zoom),oy=h/2-(Math.floor(sourceH/2)*zoom);for(let x=ox;x<=w;x+=zoom){detailCtx.moveTo(Math.round(x)+.5,0);detailCtx.lineTo(Math.round(x)+.5,h)}for(let y=oy;y<=h;y+=zoom){detailCtx.moveTo(0,Math.round(y)+.5);detailCtx.lineTo(w,Math.round(y)+.5)}detailCtx.stroke()}detailCtx.strokeStyle='#ff2d55';detailCtx.lineWidth=2;detailCtx.strokeRect(w/2-zoom/2,h/2-zoom/2,zoom,zoom)}
function render(){const s=sample(),r=record(),m=manifest[s.cube_id],coords=imageCoordinates(s,m);drawOverview(s,m,coords.cx,coords.cy,coords.step);drawDetail(coords.cx,coords.cy);const reviewed=Object.values(state.annotations).filter(x=>x.reviewed).length,rereview=Object.values(state.annotations).filter(x=>x.requires_visual_rereview).length;$('counter').textContent=`${index+1} / ${visible.length} filtered · ${reviewed}/800 reviewed · ${rereview} re-review`;$('label').value=r.selected_label;$('confidence').value=r.confidence;$('reviewer').value=r.reviewer_identifier;$('note').value=r.investigator_note;$('review').textContent=r.reviewed&&!r.requires_visual_rereview?'Reviewed ✓':'Mark reviewed';$('rereview').textContent=r.requires_visual_rereview?'VISUAL RE-REVIEW REQUIRED: this preserved label predates the frozen natural RGB display.':'';$('contradiction').textContent=r.role_contradiction?'ROLE CONTRADICTION RECORDED: investigator selected chickpea on a frozen absent-negative-control cube.':'';const wl=m.natural_rgb_wavelengths_nm;$('metadata').textContent=`sample ID: ${s.sample_id}\ncube ID: ${s.cube_id}\nfrozen cube role: ${s.cube_evaluation_role}\ncube notes: ${s.cube_investigator_notes||'(none)'}\nRGB wavelengths: ${wl.red} / ${wl.green} / ${wl.blue} nm\ndisplay version: ${r.annotation_display_version}\nrank stratum: ${s.scalar_index_rank_stratum}\nspatial block: ${s.spatial_group_id}\nframe: MAIN only`;status(`${s.sample_id} · ${r.reviewed?'REVIEWED':'not reviewed'}${r.requires_visual_rereview?' · RE-REVIEW REQUIRED':''} · ${r.selected_label||'unlabeled'}${dirty?' · UNSAVED':''}`,dirty)}
function load(){const s=sample();if(!s)return;image.onload=render;image.onerror=()=>status('Prediction-free display layer failed to load.',true);image.src='/layers/'+s.cube_id+'/'+$('layer').value+'.png?v='+Date.now()}function move(d){index=Math.max(0,Math.min(visible.length-1,index+d));load()}function update(){const s=sample(),r=record();r.selected_label=$('label').value;r.confidence=$('confidence').value;r.reviewer_identifier=$('reviewer').value.trim();r.investigator_note=$('note').value;r.role_contradiction=r.selected_label==='chickpea'&&s.cube_evaluation_role==='chickpea_absent_negative_control';r.reviewed=false;r.review_timestamp='';setDirty();render()}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('cubeFilter').onchange=filter;$('roleFilter').onchange=filter;$('reviewFilter').onchange=filter;$('layer').onchange=load;$('zoom').onchange=render;$('grid').onchange=render;for(const id of ['label','confidence','reviewer','note'])$(id).addEventListener('change',update);$('review').onclick=()=>{const r=record();update();if(!r.selected_label||!r.confidence){status('Select a label and confidence before marking reviewed.',true);return}r.reviewed=true;r.review_timestamp=new Date().toISOString();r.requires_visual_rereview=false;setDirty();render()};$('clear').onclick=()=>{if(!confirm('Clear this MAIN annotation?'))return;const s=sample(),old=record();state.annotations[s.sample_id]={sample_id:s.sample_id,cube_id:s.cube_id,frozen_cube_role:s.cube_evaluation_role,selected_label:'',confidence:'',investigator_note:'',reviewed:false,review_timestamp:'',reviewer_identifier:'',role_contradiction:false,source_preview_checksums:JSON.parse(s.source_preview_checksums),sampling_frame_sha256:state.sampling_frame_sha256,annotation_display_version:state.annotation_display_version,annotation_display_contract_sha256:state.annotation_display_contract_sha256,natural_rgb_preview_sha256:old.natural_rgb_preview_sha256,requires_visual_rereview:false};setDirty();render()};
async function save(){const response=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!response.ok){status(await response.text(),true);return}const result=await response.json();state.revision=result.revision;dirty=false;status(`SAVED: ${result.records} MAIN records · ${result.reviewed} reviewed. File: ${result.file}`)}$('save').onclick=save;function setLayer(name){$('layer').value=name;load()}function setZoom(value){$('zoom').value=String(value);render()}document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)&&!(e.ctrlKey||e.metaKey))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key.toLowerCase()==='n')setLayer('natural_rgb');else if(e.key.toLowerCase()==='f')setLayer('false_colour');else if(e.key.toLowerCase()==='p')setLayer('pca');else if(e.key.toLowerCase()==='i')setLayer('stored_index');else if(e.key.toLowerCase()==='b')setLayer('support_outline');else if(['2','4','8'].includes(e.key))setZoom(e.key);else if(e.key.toLowerCase()==='x')setZoom(16);else if(e.key.toLowerCase()==='g'){$('grid').checked=!$('grid').checked;render()}else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save()}};window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
async function init(){samples=await(await fetch('/api/samples')).json();manifest=await(await fetch('/api/manifest')).json();schema=await(await fetch('/api/schema')).json();state=await(await fetch('/api/annotations')).json();if(samples.length!==800||new Set(samples.map(s=>s.sampling_frame)).size!==1||samples[0].sampling_frame!=='main'||schema.reserve_exposed)throw new Error('Expected exact frozen 800-point MAIN frame with reserve locked');if(schema.default_layer!=='natural_rgb'||schema.display_version!==state.annotation_display_version)throw new Error('Natural RGB display provenance mismatch');for(const name of schema.labels){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');$('label').appendChild(o)}for(const name of schema.confidence){const o=document.createElement('option');o.value=name;o.textContent=name;$('confidence').appendChild(o)}for(const value of ['',...new Set(samples.map(s=>s.cube_id))]){const o=document.createElement('option');o.value=value;o.textContent=value||'All cubes';$('cubeFilter').appendChild(o)}for(const value of ['',...new Set(samples.map(s=>s.cube_evaluation_role))]){const o=document.createElement('option');o.value=value;o.textContent=value?value.replaceAll('_',' '):'All roles';$('roleFilter').appendChild(o)}visible=samples;load()}init().catch(e=>status(String(e),true));
</script></body></html>'''


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field 2 blind MAIN point annotator</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}
body{margin:0;background:#101418;color:#edf2f7;overflow:hidden}header{padding:7px 10px;background:#182028;border-bottom:1px solid #3f4b56}.bar,.button-group{display:flex;flex-wrap:wrap;gap:6px;align-items:center}button,select,input,textarea{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px}
button{cursor:pointer}.primary{background:#18794e}.danger{background:#7f1d1d}.active{outline:2px solid #67e8f9;outline-offset:1px}
.layout{display:grid;grid-template-columns:minmax(420px,1.05fr) minmax(350px,.85fr) 390px;height:calc(100vh - 80px);min-height:0}.column{min-width:0;min-height:0;padding:8px;background:#050708;overflow:auto}.center{display:grid;grid-template-rows:auto 1fr 1fr;gap:7px;border-left:1px solid #26313b}
.panel{display:flex;min-width:0;min-height:0;flex-direction:column;align-items:center;justify-content:center;gap:5px}.panel-title{font-size:.82rem;color:#b9c4ce}canvas{display:block;max-width:100%;max-height:100%;border:1px solid #52606d;background:#000;touch-action:none;position:relative;z-index:1}.viewer{width:100%;height:auto;cursor:crosshair;image-rendering:pixelated}.viewer.dragging{cursor:grabbing}.magnifier{width:min(100%,500px);height:auto;image-rendering:pixelated}
.legend{display:flex;gap:16px;font-size:.8rem}.yellow{color:#ffd166}.cyan{color:#67e8f9}.form{height:calc(100vh - 80px);padding:9px;overflow:auto;background:#151b21;position:sticky;top:0;z-index:20;border-left:1px solid #3f4b56}.field{display:block;margin:6px 0}.field span{display:block;color:#b9c4ce;font-size:.8rem}.form textarea,.form input,.form select{width:100%}
.choice{min-width:110px;min-height:42px;flex:1 1 45%;text-transform:capitalize;font-weight:650}.choice[aria-pressed=true]{background:#075985;border-color:#67e8f9}.choice:disabled{opacity:.35;cursor:not-allowed}.confidence-choice{min-height:36px}.confidence-choice[aria-pressed=true]{background:#166534}.message{margin:5px 0;color:#ffd166;font-weight:600}#status{padding-top:4px}.dirty,.warning{color:#ffd166}.rereview{color:#ff9f1c}.help{font-size:.75rem;color:#b9c4ce}.metadata{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:.75rem}.zone{padding:7px;background:#1e293b;border-left:4px solid #67e8f9}.form-actions{position:sticky;bottom:0;background:#151b21;padding:7px 0;border-top:1px solid #3f4b56}.controls-shield{pointer-events:none;position:absolute;inset:0;z-index:-1}.hidden{display:none!important}
@media(max-width:1200px){.layout{grid-template-columns:minmax(360px,1fr) minmax(310px,.8fr) 350px}.choice{min-width:100px}}
</style></head><body>
<header><div class="bar">
<div id="status">Loading frozen MAIN frame…</div><div id="safetyMessage" class="message">Sample location is frozen; clicking only changes the inspection view.</div><div class="help">Point annotation remains gated on the frozen area contract. Image clicks never choose a biological label.</div>
</div></header><div class="layout">
<section class="column"><div class="panel-title">Cube overview · frozen yellow sample · cyan inspection</div><canvas id="viewer" class="viewer" width="900" height="760" aria-label="Cube overview"></canvas><div class="legend"><span class="yellow">□ Frozen sample</span><span class="cyan">＋ Inspection location</span></div></section>
<section class="column center"><div class="bar"><label>Layer <select id="layer"><option value="natural_rgb">Natural RGB (default)</option><option value="false_colour">False color</option><option value="pca">PCA</option><option value="stored_index">Stored index</option><option value="support_outline">Support</option></select></label><button class="zoom-choice" data-zoom="2">2×</button><button class="zoom-choice" data-zoom="4">4×</button><button class="zoom-choice" data-zoom="8">8×</button><button class="zoom-choice" data-zoom="16">16×</button><button id="resetView">Reset</button><label><input id="grid" type="checkbox" checked> Grid</label><label><input id="showInspection" type="checkbox" checked> Inspection</label></div><section class="panel"><div class="panel-title">Large frozen-sample magnifier</div><canvas id="frozenMagnifier" class="magnifier" width="500" height="360"></canvas></section><section class="panel" id="inspectionPanel"><div class="panel-title">Inspection magnifier</div><canvas id="inspectionMagnifier" class="magnifier" width="500" height="300"></canvas></section></section>
<aside class="form"><div class="controls-shield" aria-hidden="true"></div><div class="bar"><button id="prev">← Previous</button><button id="next">Next →</button><span id="counter"></span></div><label class="field"><span>Cube</span><select id="cubeFilter"></select></label><div class="bar"><label>Role <select id="roleFilter"></select></label><label>Review <select id="reviewFilter"><option value="">All</option><option value="rereview">Re-review</option></select></label></div><h3>Biological annotation</h3>
<div id="zoneMembership" class="zone">Frozen zone membership loading…</div>
<div class="field"><span>Selected label (no default)</span><div id="labelButtons" class="button-group" role="group" aria-label="Selected label"></div></div>
<div class="field"><span>Confidence</span><div id="confidenceButtons" class="button-group" role="group" aria-label="Confidence"></div></div>
<label class="field"><span>Reviewer identifier (optional)</span><input id="reviewer"></label><label class="field"><span>Investigator note (optional)</span><textarea id="note"></textarea></label>
<button id="boundaryCorrection">Boundary needs correction</button><p id="rereview" class="rereview"></p><p id="contradiction" class="warning"></p><div id="metadata" class="metadata"></div><div class="form-actions"><button id="review" class="primary">Mark reviewed</button> <button id="save" class="primary">Save all</button> <button id="clear" class="danger">Clear</button></div>
</aside></div><script src="/field2-point-viewer.js"></script></body></html>'''


def handler_factory(store: PointAnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(content)
        def do_GET(self):
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}: return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/field2-point-viewer.js":
                viewer_js = Path(__file__).with_name("field2_point_viewer.js")
                return self.send_bytes(viewer_js.read_bytes(), "text/javascript; charset=utf-8")
            if clean == "/api/samples": return self.send_bytes(store.frame.to_json(orient="records").encode(), "application/json")
            if clean == "/api/manifest": return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if clean == "/api/annotations": return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if clean == "/api/schema": return self.send_bytes(json.dumps({"labels": POINT_LABELS, "confidence": CONFIDENCE_VALUES, "alley_labels": ALLEY_POINT_LABELS, "frame": "main", "reserve_exposed": False, "default_layer": "natural_rgb", "display_version": store.display_version, "area_geometry_frozen": bool(store.area_contract_hash)}).encode(), "application/json")
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
                print(f"SAVED: {records} MAIN point annotations; {reviewed} reviewed", flush=True); print(f"File: {store.json_path}", flush=True)
                return self.send_bytes(json.dumps({"records": records, "reviewed": reviewed, "revision": revision, "file": str(store.json_path)}).encode(), "application/json")
            except Exception as error: return self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)
        def log_message(self, format, *args): return
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", required=True, type=Path); parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path); parser.add_argument("--area-config", default=Path("configs/field2_area_annotation.yaml"), type=Path); parser.add_argument("--port", type=int); parser.add_argument("--no-browser", action="store_true"); args = parser.parse_args()
    paths, config = yaml.safe_load(args.paths.read_text()), yaml.safe_load(args.config.read_text()); project = Path(paths["project_root"]).resolve()
    released = require_main_sampling_frame(project, config)
    area_config = yaml.safe_load(args.area_config.read_text())
    area_contract, area_membership = require_frozen_area_contract(project, area_config)
    main_frame = released["main"].merge(
        area_membership[["sample_id", "zone_type", "domain", "primary_external_validation_eligible"]],
        on="sample_id", how="left", validate="one_to_one",
    )
    if main_frame[["zone_type", "domain"]].eq("").any().any() or main_frame[["zone_type", "domain"]].isna().any().any():
        raise ValueError("Frozen area membership is incomplete for the 800-point main frame")
    display_contract_path = project / config["annotation_display"]["display_contract"]
    display_result = validate_annotation_display_contract(project, config, display_contract_path)
    store = PointAnnotationStore(
        project, main_frame, released["main_frame_sha256"],
        project / config["review"]["package_manifest"], display_result["manifest"],
        config["annotation_display"]["version"], sha256(display_contract_path),
        project / config["point_annotation"]["output_root"], sha256(project / area_config["freeze"]["contract"]),
    )
    current, rereview_count = store.migrate_display_version(store.load())
    port = args.port or int(config["point_annotation"]["port"]); server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(store)); url = f"http://127.0.0.1:{port}"
    reviewed_count = sum(record["reviewed"] for record in current["annotations"].values())
    print(f"Field 2 prediction-free MAIN point annotator: {url}", flush=True); print("Default layer: full-resolution natural RGB", flush=True); print("Reserve frame: locked and not served", flush=True); print(f"Reviewed records: {reviewed_count}; requiring RGB visual re-review: {rereview_count}", flush=True); print(f"Resume file: {store.json_path}", flush=True)
    if not args.no_browser: threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nPoint annotator stopped. Saved MAIN annotations remain on disk.", flush=True)


if __name__ == "__main__": main()
