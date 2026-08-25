#!/usr/bin/env python
"""Run a localhost-only browser annotator for observed planter tyre tracks."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Field 1 planter-track annotator</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #101418; color: #edf2f7; }
header { position: sticky; top: 0; z-index: 10; padding: 10px 14px; background: #182028; border-bottom: 1px solid #39434d; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
button, select, input { font: inherit; color: inherit; background: #26313b; border: 1px solid #52606d; border-radius: 5px; padding: 6px 9px; }
button:hover { background: #33414d; }
button.primary { background: #18794e; border-color: #2fb171; }
button.danger { background: #7f1d1d; border-color: #b64242; }
label { display: flex; gap: 5px; align-items: center; }
#stage { position: relative; margin: 12px auto; width: min(95vw, 1100px); }
canvas { width: 100%; height: auto; display: block; background: #063b3d; border: 1px solid #52606d; cursor: crosshair; }
#status { padding-top: 8px; color: #cbd5e1; }
#help { padding: 0 14px 12px; color: #aeb9c4; font-size: 0.92rem; }
.dirty { color: #ffcc66 !important; }
</style>
</head>
<body>
<header>
  <div class="toolbar">
    <button id="previous">← Previous</button>
    <label>Cube <select id="cube"></select></label>
    <button id="next">Next →</button>
    <label>Layer <select id="layer">
      <option value="first_difference">First difference</option>
      <option value="second_difference">Second difference</option>
      <option value="pca">PCA RGB</option>
    </select></label>
    <label>Feature <select id="kind">
      <option value="tyre_track">Tyre track</option>
      <option value="alley_boundary">Alley boundary</option>
      <option value="uncertain_structure">Uncertain structure</option>
    </select></label>
    <label>Note <input id="note" size="18" placeholder="optional"></label>
    <button id="finish">Finish line</button>
    <button id="undoPoint">Undo point</button>
    <button id="undoLine">Undo last line</button>
    <button id="clear" class="danger">Clear cube</button>
    <button id="save" class="primary">Save annotations</button>
    <button id="backup">Download backup JSON</button>
  </div>
  <div id="status">Loading…</div>
</header>
<div id="stage"><canvas id="canvas"></canvas></div>
<div id="help">Click along the centre of a visible tyre mark. Use several points for a curve. Press Enter or “Finish line” when complete. Mark each physical tyre track separately; switch between first and second difference as needed.</div>
<script>
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const cubeSelect = document.getElementById('cube');
const layerSelect = document.getElementById('layer');
const kindSelect = document.getElementById('kind');
const noteInput = document.getElementById('note');
const statusNode = document.getElementById('status');
let manifest = [];
let state = {version: 1, cubes: {}};
let image = new Image();
let drawing = [];
let pointer = null;
let dirty = false;

const colors = {tyre_track: '#ffd84d', alley_boundary: '#42d392', uncertain_structure: '#ff66c4'};
function currentCube() { return cubeSelect.value; }
function lines() { const id = currentCube(); if (!state.cubes[id]) state.cubes[id] = []; return state.cubes[id]; }
function setStatus(message, isDirty = dirty) { statusNode.textContent = message; statusNode.className = isDirty ? 'dirty' : ''; }
function markDirty() { dirty = true; setStatus('Unsaved changes — use Save annotations.'); }
function imageUrl() { return '/layers/' + encodeURIComponent(currentCube()) + '/' + layerSelect.value + '.png'; }
function loadImage() {
  drawing = []; pointer = null;
  image = new Image();
  image.onload = () => { canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; draw(); setStatus(currentCube() + ' • ' + lines().length + ' saved lines' + (dirty ? ' • unsaved changes' : '')); };
  image.onerror = () => setStatus('Layer could not be loaded.', true);
  image.src = imageUrl() + '?v=' + Date.now();
}
function pointFromEvent(event) {
  const box = canvas.getBoundingClientRect();
  return {x: (event.clientX - box.left) * canvas.width / box.width, y: (event.clientY - box.top) * canvas.height / box.height};
}
function stroke(points, kind, temporary = false) {
  if (!points.length) return;
  context.save();
  context.strokeStyle = colors[kind] || colors.uncertain_structure;
  context.fillStyle = context.strokeStyle;
  context.lineWidth = temporary ? 2 : 3;
  context.beginPath(); context.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) context.lineTo(points[i].x, points[i].y);
  context.stroke();
  for (const point of points) { context.beginPath(); context.arc(point.x, point.y, 3, 0, Math.PI * 2); context.fill(); }
  context.restore();
}
function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (image.complete && image.naturalWidth) context.drawImage(image, 0, 0);
  for (const line of lines()) stroke(line.points, line.kind);
  if (drawing.length) {
    const preview = pointer ? drawing.concat([pointer]) : drawing;
    stroke(preview, kindSelect.value, true);
  }
}
function finishLine() {
  if (drawing.length < 2) { setStatus('A line needs at least two points.', true); return; }
  lines().push({
    id: currentCube() + '-' + Date.now(), kind: kindSelect.value,
    layer: layerSelect.value, note: noteInput.value.trim(),
    points: drawing.map(point => ({x: Number(point.x.toFixed(3)), y: Number(point.y.toFixed(3))}))
  });
  drawing = []; pointer = null; noteInput.value = ''; markDirty(); draw();
}
function finishBeforeAction() {
  if (!drawing.length) return true;
  if (drawing.length >= 2) { finishLine(); return true; }
  setStatus('The current line has only one point. Add another point or press Escape to discard it.', true);
  return false;
}
canvas.addEventListener('click', event => { drawing.push(pointFromEvent(event)); markDirty(); draw(); });
canvas.addEventListener('mousemove', event => { pointer = pointFromEvent(event); draw(); });
canvas.addEventListener('mouseleave', () => { pointer = null; draw(); });
document.getElementById('finish').onclick = finishLine;
document.getElementById('undoPoint').onclick = () => { if (drawing.length) { drawing.pop(); markDirty(); draw(); } };
document.getElementById('undoLine').onclick = () => { if (lines().length) { lines().pop(); markDirty(); draw(); } };
document.getElementById('clear').onclick = () => { if (confirm('Delete all annotations for ' + currentCube() + '?')) { state.cubes[currentCube()] = []; drawing = []; markDirty(); draw(); } };
document.getElementById('previous').onclick = () => { if (!finishBeforeAction()) return; cubeSelect.selectedIndex = Math.max(0, cubeSelect.selectedIndex - 1); loadImage(); };
document.getElementById('next').onclick = () => { if (!finishBeforeAction()) return; cubeSelect.selectedIndex = Math.min(cubeSelect.options.length - 1, cubeSelect.selectedIndex + 1); loadImage(); };
cubeSelect.onchange = () => { if (finishBeforeAction()) loadImage(); };
layerSelect.onchange = () => { if (finishBeforeAction()) loadImage(); };
kindSelect.onchange = draw;
document.addEventListener('keydown', event => {
  if (event.key === 'Enter') { event.preventDefault(); finishLine(); }
  if (event.key === 'Escape') { drawing = []; pointer = null; draw(); }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); }
});
async function save() {
  if (!finishBeforeAction()) return;
  const button = document.getElementById('save');
  button.disabled = true; button.textContent = 'Saving…'; setStatus('Saving annotations to disk…', true);
  try {
    const response = await fetch('/api/annotations', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(state)});
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json(); dirty = false;
    button.textContent = 'Saved ✓';
    setStatus('Saved ' + result.lines + ' lines and ' + result.vertices + ' vertices to disk.');
    window.setTimeout(() => { button.textContent = 'Save annotations'; }, 1800);
  } catch (error) {
    button.textContent = 'Save failed';
    setStatus('Save failed: ' + String(error) + '. Use Download backup JSON before refreshing.', true);
  } finally {
    button.disabled = false;
  }
}
document.getElementById('save').onclick = save;
document.getElementById('backup').onclick = () => {
  if (!finishBeforeAction()) return;
  const blob = new Blob([JSON.stringify(state, null, 2)], {type: 'application/json'});
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = 'field1_planter_track_annotations_backup.json';
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  setStatus('Backup JSON downloaded. This does not replace Save annotations.');
};
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
async function initialize() {
  manifest = await (await fetch('/api/manifest')).json();
  state = await (await fetch('/api/annotations')).json();
  for (const item of manifest) { const option = document.createElement('option'); option.value = item.cube_id; option.textContent = item.cube_id + (item.analysis_role === 'sensitivity_only' ? ' (sensitivity)' : ''); cubeSelect.appendChild(option); }
  if (!manifest.length) { setStatus('No annotation layers found. Run the standardized turn-band audit first.', true); return; }
  loadImage();
}
initialize().catch(error => setStatus(String(error), true));
</script>
</body>
</html>'''


class AnnotationStore:
    def __init__(self, manifest_path: Path, output_root: Path):
        self.manifest_path = manifest_path
        self.manifest = pd.read_csv(manifest_path).fillna("")
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = output_root / "field1_planter_track_annotations.json"
        self.csv_path = output_root / "field1_planter_track_vertices.csv"
        self.geojson_path = output_root / "field1_planter_track_annotations.geojson"
        self.lock = threading.Lock()
        self.by_cube = {str(row.cube_id): row for row in self.manifest.itertuples(index=False)}

    def public_manifest(self) -> list[dict]:
        rows = []
        for row in self.manifest.to_dict("records"):
            rows.append({
                "cube_id": str(row["cube_id"]),
                "analysis_role": str(row.get("analysis_role", "primary_candidate")),
                "preview_width": int(row["preview_width"]),
                "preview_height": int(row["preview_height"]),
                "preview_step": int(row["preview_step"]),
            })
        return rows

    def load(self) -> dict:
        if self.json_path.exists():
            return json.loads(self.json_path.read_text())
        return {"version": 1, "cubes": {}}

    def layer_path(self, cube_id: str, layer: str) -> Path:
        if cube_id not in self.by_cube:
            raise KeyError(cube_id)
        if layer not in {"pca", "first_difference", "second_difference"}:
            raise KeyError(layer)
        return Path(getattr(self.by_cube[cube_id], f"{layer}_path"))

    def validate(self, payload: dict) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("cubes"), dict):
            raise ValueError("Expected an object with a cubes mapping")
        allowed_kinds = {"tyre_track", "alley_boundary", "uncertain_structure"}
        for cube_id, annotations in payload["cubes"].items():
            if cube_id not in self.by_cube or not isinstance(annotations, list):
                raise ValueError(f"Unknown or invalid cube: {cube_id}")
            row = self.by_cube[cube_id]
            for annotation in annotations:
                if annotation.get("kind") not in allowed_kinds:
                    raise ValueError("Unknown annotation type")
                points = annotation.get("points")
                if not isinstance(points, list) or len(points) < 2:
                    raise ValueError("Every annotation must contain at least two points")
                for point in points:
                    x, y = float(point["x"]), float(point["y"])
                    if not (0 <= x < int(row.preview_width) and 0 <= y < int(row.preview_height)):
                        raise ValueError(f"Point outside preview bounds for {cube_id}")

    def save(self, payload: dict) -> tuple[int, int]:
        self.validate(payload)
        payload["version"] = 1
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        vertices = []
        features = []
        line_count = 0
        for cube_id, annotations in payload["cubes"].items():
            row = self.by_cube[cube_id]
            for annotation in annotations:
                line_count += 1
                coordinates = []
                for index, point in enumerate(annotation["points"]):
                    preview_x, preview_y = float(point["x"]), float(point["y"])
                    column = (preview_x + 0.5) * float(row.preview_step)
                    raster_row = (preview_y + 0.5) * float(row.preview_step)
                    map_x = float(row.transform_a) * column + float(row.transform_b) * raster_row + float(row.transform_c)
                    map_y = float(row.transform_d) * column + float(row.transform_e) * raster_row + float(row.transform_f)
                    coordinates.append([map_x, map_y])
                    vertices.append({
                        "cube_id": cube_id, "annotation_id": annotation["id"],
                        "kind": annotation["kind"], "layer": annotation.get("layer", ""),
                        "note": annotation.get("note", ""), "vertex_index": index,
                        "preview_x": preview_x, "preview_y": preview_y,
                        "original_column": column, "original_row": raster_row,
                        "map_x": map_x, "map_y": map_y, "crs": str(row.crs),
                    })
                features.append({
                    "type": "Feature",
                    "properties": {
                        "cube_id": cube_id, "annotation_id": annotation["id"],
                        "kind": annotation["kind"], "layer": annotation.get("layer", ""),
                        "note": annotation.get("note", ""), "crs": str(row.crs),
                    },
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                })
        with self.lock:
            temporary = self.json_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(self.json_path)
            columns = [
                "cube_id", "annotation_id", "kind", "layer", "note", "vertex_index",
                "preview_x", "preview_y", "original_column", "original_row", "map_x", "map_y", "crs",
            ]
            with self.csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader(); writer.writerows(vertices)
            self.geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
        return line_count, len(vertices)


def handler_factory(store: AnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(content)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if self.path == "/api/manifest":
                return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if self.path == "/api/annotations":
                return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if self.path.startswith("/layers/"):
                path = self.path.split("?", 1)[0]
                parts = path.strip("/").split("/")
                if len(parts) != 3 or not parts[2].endswith(".png"):
                    return self.send_bytes(b"Not found", "text/plain", 404)
                try:
                    layer_path = store.layer_path(parts[1], parts[2][:-4])
                    return self.send_bytes(layer_path.read_bytes(), "image/png")
                except (KeyError, FileNotFoundError):
                    return self.send_bytes(b"Not found", "text/plain", 404)
            return self.send_bytes(b"Not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/api/annotations":
                return self.send_bytes(b"Not found", "text/plain", 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                lines, vertices = store.save(payload)
                print(
                    f"Saved planter annotations: {lines} lines, {vertices} vertices -> {store.json_path}",
                    flush=True,
                )
                return self.send_bytes(json.dumps({"lines": lines, "vertices": vertices}).encode(), "application/json")
            except Exception as error:
                return self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)

        def log_message(self, format, *args):
            return
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    report_root = project / "metadata" / "local" / "reports" / "standardized_planter_turn_audit"
    manifest_path = report_root / "annotation_layer_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            "Annotation layers are missing. Run scripts/audit_standardized_planter_turn_bands.py first."
        )
    output_root = project / "metadata" / "local" / "annotations" / "planter_tracks"
    store = AnnotationStore(manifest_path, output_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory(store))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Planter-track annotator: {url}")
    print("Click along each tyre-track centreline and save. Press Ctrl+C here when finished.")
    print(f"Annotations: {store.json_path}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotator stopped. Saved annotations remain on disk.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
