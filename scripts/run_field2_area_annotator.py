#!/usr/bin/env python
"""Run the prediction-free Field 2 experimental-area zone annotator."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_area_review import (
    CONFIDENCE_VALUES, COVERAGE_MODES, ZONE_DEFINITIONS, ZONE_TYPES, Field2AreaAnnotationStore,
)
from chickpea_ssl.field2_area_operational import PRECEDENCE_RULE, ZONE_PRECEDENCE
from chickpea_ssl.field2_readiness import sha256


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field 2 prediction-free area annotator</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#0b1015;color:#edf2f7;overflow:hidden}
.app{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:100vh}.workspace{min-width:0;display:grid;grid-template-rows:auto minmax(0,1fr);overflow:hidden}
.viewer-toolbar{position:relative;z-index:5;display:flex;flex-wrap:wrap;gap:7px;align-items:center;padding:8px 10px;background:#182028;border-bottom:1px solid #3f4b56}
button,select,input,textarea{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px}button{cursor:pointer}button:hover{background:#33414d}button.active,[aria-pressed=true]{outline:2px solid #67e8f9;outline-offset:1px}.primary{background:#18794e}.danger{background:#7f1d1d}
.stage-wrap{min-height:0;overflow:auto;padding:10px;display:flex;align-items:center;justify-content:center;background:#020406}.stage{position:relative;max-width:100%;max-height:100%}canvas{display:block;width:min(100%,1100px);height:auto;max-height:calc(100vh - 76px);background:#000;border:1px solid #52606d;touch-action:none;cursor:crosshair;image-rendering:pixelated}
.side{position:sticky;top:0;height:100vh;min-height:0;background:#151b21;border-left:1px solid #3f4b56;display:grid;grid-template-rows:auto minmax(0,1fr) auto;z-index:20}.side-top,.side-bottom{position:relative;z-index:3;background:#182028;padding:9px;border-bottom:1px solid #3f4b56}.side-bottom{border-top:1px solid #3f4b56;border-bottom:0}.side-scroll{overflow:auto;padding:10px}.row,.button-grid{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.button-grid button{flex:1 1 44%;min-width:135px}.field{display:block;margin:8px 0}.field>span{display:block;color:#b9c4ce;font-size:.8rem;margin-bottom:3px}.field textarea,.field input,.field select{width:100%}
.zone-research_crop_area{border-color:#22c55e!important}.zone-alley{border-color:#f97316!important}.zone-outside_research_field{border-color:#3b82f6!important}.zone-uncertain_boundary{border-color:#a855f7!important}.help{font-size:.76rem;line-height:1.35;color:#b9c4ce}.warning{color:#ffd166}.ok{color:#86efac}.policy{margin-top:6px;padding:6px;border-left:4px solid #67e8f9;background:#0f2a35;font-weight:700}#status{font-size:.82rem;margin-top:6px}.tool-status{font-weight:700;color:#67e8f9}
@media(max-width:900px){.app{grid-template-columns:minmax(0,1fr) 340px}.button-grid button{min-width:115px}}
</style></head><body><div class="app">
<main class="workspace"><div class="viewer-toolbar">
<label>Layer <select id="layer"><option value="natural_rgb">Natural RGB (default)</option><option value="false_colour">False color</option><option value="pca">PCA</option><option value="stored_index">Stored scalar index</option><option value="valid_support">Valid support</option><option value="support_outline">Support outline</option></select></label>
<button class="zoom" data-zoom="2">2×</button><button class="zoom" data-zoom="4">4×</button><button class="zoom" data-zoom="8">8×</button><button class="zoom" data-zoom="16">16×</button><button id="resetView">Reset view</button>
<span class="tool-status" id="toolStatus">Tool: pan</span><span class="help">Wheel zoom · drag pan · click polygons to select</span>
</div><div class="stage-wrap"><div class="stage"><canvas id="canvas" width="1100" height="780" aria-label="Field 2 area drawing canvas"></canvas></div></div></main>
<aside class="side"><div class="side-top">
<div class="row"><button id="previous">← Previous</button><button id="next">Next →</button><label>Cube <select id="cube"></select></label></div>
<div id="progress"></div><div id="status">Loading frozen prediction-free layers…</div><div class="policy">Small overlaps are resolved automatically: outside &gt; alley &gt; uncertain &gt; research crop.</div>
</div><div class="side-scroll">
<label class="field"><span>Coverage mode</span><select id="coverage"><option value="">Choose coverage mode…</option></select></label>
<div id="effectiveMode" class="help policy"></div>
<div class="field"><span>Zone type</span><div id="zoneButtons" class="button-grid"></div></div>
<div class="field"><span>Drawing and editing</span><div class="button-grid">
<button id="draw">Draw polygon</button><button id="finish">Finish polygon</button><button id="cancelDrawing" class="danger">Cancel current polygon</button><button id="edit">Edit vertices</button><button id="insert">Insert vertex</button><button id="deleteVertex">Delete vertex</button><button id="movePolygon">Move polygon</button><button id="pan">Pan</button><button id="deletePolygon" class="danger">Delete polygon</button><button id="undo">Undo</button><button id="redo">Redo</button><button id="clearSelected" class="danger">Clear selected</button>
</div></div>
<label class="field"><span>Polygon opacity</span><input id="opacity" type="range" min="0" max="0.7" step="0.05" value="0.25"></label>
<div class="field"><span>Visible zone types</span><div id="zoneToggles" class="button-grid"></div></div>
<button id="unassignedOutside">Treat unassigned valid support as outside research field</button><p id="unassignedStatus" class="help"></p>
<label class="field"><span>Confidence</span><select id="confidence"><option value="">Choose confidence…</option></select></label>
<label class="field"><span>Reviewer identifier (optional)</span><input id="reviewer"></label>
<label class="field"><span>Investigator notes</span><textarea id="notes" rows="3"></textarea></label>
<p id="audit" class="help"></p><pre id="auditDetails" class="help warning"></pre>
<details class="help"><summary>Scientific zone definitions</summary><p><b>Research crop area:</b> investigator-identified planted-row/research-plot domain; not a biological chickpea label.</p><p><b>Alley:</b> investigator-confirmed plot or tractor alley; only soil and weeds occur by investigator rule.</p><p><b>Outside research field:</b> valid imagery outside the intended experiment; it may contain soil or varied vegetation.</p><p><b>Uncertain boundary:</b> geometry that cannot be placed confidently.</p><p>Zones are contextual domains and never automatic biological labels.</p></details>
</div><div class="side-bottom"><div class="row"><button id="review" class="primary">Mark cube reviewed</button><button id="save" class="primary">Save all</button></div><div class="help">Save is atomic and resumable. Geometry is not frozen here.</div></div></aside>
</div><script src="/field2-area-annotator.js"></script></body></html>'''


def validate_config_inputs(project: Path, config: dict) -> None:
    for name, item in config["inputs"].items():
        path = project / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen area-annotator input mismatch: {name}")


def handler_factory(store: Field2AreaAnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(content)

        def do_GET(self):
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}: return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/field2-area-annotator.js":
                return self.send_bytes(Path(__file__).with_name("field2_area_annotator.js").read_bytes(), "text/javascript; charset=utf-8")
            if clean == "/api/manifest": return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if clean == "/api/annotations": return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if clean == "/api/schema":
                return self.send_bytes(json.dumps({
                    "coverage_modes": COVERAGE_MODES, "zone_types": ZONE_TYPES,
                    "zone_definitions": ZONE_DEFINITIONS,
                    "pixel_domain_precedence_low_to_high": ZONE_PRECEDENCE,
                    "precedence_rule": PRECEDENCE_RULE,
                    "blank_mode_with_polygons": "mixed_manual_boundaries",
                    "confidence": CONFIDENCE_VALUES, "reserve_exposed": False,
                    "biological_labels_available": False, "default_layer": "natural_rgb",
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
                reviewed, polygons, vertices, revision = store.save(json.loads(self.rfile.read(length)))
                print(f"SAVED: {reviewed}/40 reviewed; {polygons} polygons; {vertices} vertices", flush=True)
                return self.send_bytes(json.dumps({
                    "reviewed": reviewed, "polygons": polygons, "vertices": vertices,
                    "revision": revision, "file": str(store.json_path),
                }).encode(), "application/json")
            except Exception as error:
                return self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)

        def log_message(self, format, *args): return
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_area_annotation.yaml"), type=Path)
    parser.add_argument("--port", type=int); parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths, config = yaml.safe_load(args.paths.read_text()), yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve(); validate_config_inputs(project, config)
    store = Field2AreaAnnotationStore(
        project, project / config["inputs"]["review_manifest"]["path"],
        project / config["inputs"]["natural_rgb_manifest"]["path"],
        project / config["inputs"]["valid_support_manifest"]["path"],
        project / config["annotation"]["output_root"],
    )
    port = args.port or int(config["port"]); server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(store))
    url = f"http://127.0.0.1:{port}"
    payload = store.load(); reviewed = sum(item["reviewed"] for item in payload["annotations"].values())
    print(f"Field 2 prediction-free area annotator: {url}", flush=True)
    print("Default layer: full-resolution natural RGB", flush=True)
    print("Biological labels/model outputs: unavailable", flush=True)
    print(f"Area review progress: {reviewed}/40 cubes", flush=True)
    print(f"Resume file: {store.json_path}", flush=True)
    if not args.no_browser: threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nArea annotator stopped. Saved annotations remain on disk.", flush=True)
    finally: server.server_close()


if __name__ == "__main__": main()
