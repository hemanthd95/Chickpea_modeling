#!/usr/bin/env python
"""Run the main-only Field 2 area-stratified v2 point annotator."""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from chickpea_ssl.field2_area_sampling_v2 import require_v2_main_frame
from chickpea_ssl.field2_point_spectra import PredictionFreeSpectrumStore
from chickpea_ssl.field2_blind_review import validate_annotation_display_contract
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_point_annotator import PointAnnotationStore, handler_factory


def build_store(project: Path, v2: dict) -> tuple[PointAnnotationStore, PredictionFreeSpectrumStore]:
    base = yaml.safe_load((project / v2["base_config"]).read_text())
    area = yaml.safe_load((project / v2["area_config"]).read_text())
    released = require_v2_main_frame(project, v2)
    main = released["main"].copy()
    main["zone_type"] = main["domain_name"]
    main["domain"] = main["domain_reporting_stratum"]
    display_path = project / base["annotation_display"]["display_contract"]
    display = validate_annotation_display_contract(project, base, display_path)
    store = PointAnnotationStore(
        project, main, released["main_frame_sha256"],
        project / base["review"]["package_manifest"], display["manifest"],
        base["annotation_display"]["version"], sha256(display_path),
        project / v2["point_annotation"]["output_root"],
        sha256(project / area["freeze"]["contract"]),
    )
    spectra = PredictionFreeSpectrumStore(
        project, project / base["inputs"]["readiness_inventory"],
        project / base["inputs"]["valid_support_manifest"], main,
    )
    return store, spectra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_area_sampling_v2.yaml"), type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    project = Path(yaml.safe_load(args.paths.read_text())["project_root"]).resolve()
    v2 = yaml.safe_load(args.config.read_text())
    store, spectra = build_store(project, v2)
    if not store.json_path.exists():
        records, reviewed, revision = store.save(store.load())
        if (records, reviewed, revision) != (800, 0, 1):
            raise RuntimeError("Neutral v2 annotation namespace did not initialize exactly")
    current = store.load()
    port = args.port or int(v2["point_annotation"]["port"])
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(store, spectra))
    url = f"http://127.0.0.1:{port}"
    reviewed = sum(record["reviewed"] for record in current["annotations"].values())
    print(f"Field 2 prediction-free area-stratified v2 MAIN annotator: {url}", flush=True)
    print("Main records served: 800; reserve records served: 0", flush=True)
    print("Default layer: full-resolution natural RGB; no biological label preselected", flush=True)
    print(f"Reviewed records: {reviewed}; resume file: {store.json_path}", flush=True)
    if not args.no_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nV2 point annotator stopped. Saved annotations remain on disk.", flush=True)


if __name__ == "__main__":
    main()
