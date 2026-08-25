#!/usr/bin/env python
"""Collect investigator-confirmed chickpea and weed reference points locally."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
from pathlib import Path
import threading
import webbrowser

import pandas as pd
import yaml


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field 1 vegetation-reference annotator</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#101418;color:#edf2f7}
header{position:sticky;top:0;z-index:10;padding:10px 14px;background:#182028;border-bottom:1px solid #39434d}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}button,select,input{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px 9px}button:hover{background:#33414d}.primary{background:#18794e;border-color:#2fb171}.danger{background:#7f1d1d;border-color:#b64242}label{display:flex;gap:5px;align-items:center}
#stage{position:relative;margin:12px;overflow:auto;max-height:78vh;border:1px solid #52606d;background:#063b3d}canvas{height:auto;display:block;cursor:crosshair}#status{padding-top:8px;color:#cbd5e1}.dirty{color:#ffcc66!important}#help{padding:0 16px 14px;color:#b9c4ce;line-height:1.45}.role{font-weight:700;color:#ffd166}
</style></head><body>
<header><div class="toolbar">
<button id="previous">← Previous</button><label>Cube <select id="cube"></select></label><button id="next">Next →</button>
<label>Layer <select id="layer"><option value="false_colour">NIR–red–green</option><option value="pca">PCA RGB</option><option value="first_difference">PCA first difference</option><option value="second_difference">PCA second difference</option><option value="ndvi">NDVI</option><option value="current_chickpea">Historical chickpea overlay</option></select></label>
<label>Reference <select id="kind"><option value="confirmed_chickpea">Confirmed chickpea</option><option value="confirmed_weed">Confirmed weed</option><option value="uncertain_do_not_use">Uncertain—do not use</option></select></label>
<label>Radius <select id="radius"><option value="0">single point</option><option value="0.03" selected>3 cm</option><option value="0.05">5 cm</option></select></label>
<label>Zoom <select id="zoom"><option value="1">100%</option><option value="1.5">150%</option><option value="2" selected>200%</option><option value="3">300%</option></select></label>
<label><input id="references" type="checkbox" checked>Show locked row/alley references</label><label>Note <input id="note" size="13" placeholder="optional"></label>
<button id="undo">Undo point</button><button id="clear" class="danger">Clear cube</button><button id="save" class="primary">Save annotations</button><button id="backup">Download backup</button>
</div><div id="status">Loading…</div></header>
<div id="stage"><canvas id="canvas"></canvas></div>
<div id="help"><b>Goal:</b> on each <b>primary_candidate</b> cube, mark about 10 unmistakable chickpea and 10 unmistakable weed locations. Use 200–300% zoom and switch layers when useful. Click plant interiors only—avoid soil, shadows, boundaries, mixed plants, alleys, and ambiguous pixels. The green row polygons and orange/magenta planter footprints are locked context. Cubes 12/14/15 are projection-only and 24/28 are sensitivity-only, so they do not need reference points now. Ctrl+S saves.</div>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),cube=document.getElementById('cube'),layer=document.getElementById('layer'),kind=document.getElementById('kind'),radius=document.getElementById('radius'),zoom=document.getElementById('zoom'),note=document.getElementById('note'),statusNode=document.getElementById('status'),references=document.getElementById('references');
let manifest=[],state={version:1,cubes:{}},referenceState={cubes:{}},image=new Image(),dirty=false,activeCube='',activeLayer=layer.value;
const colors={confirmed_chickpea:'#39ff88',confirmed_weed:'#ff4d6d',uncertain_do_not_use:'#ffd166'},referenceColors={chickpea_growing_block:'#35f28b',exclude_non_chickpea:'#ff5d5d',uncertain_boundary:'#ffd84d',alley_boundary:'#ff9f1c',tyre_track:'#e754c6',uncertain_structure:'#9aa6b2'};
function points(){if(!state.cubes[activeCube])state.cubes[activeCube]=[];return state.cubes[activeCube]}
function refs(){return(referenceState.cubes&&referenceState.cubes[activeCube])||[]}
function item(){return manifest.find(x=>x.cube_id===activeCube)}
function setStatus(s,bad=dirty){statusNode.textContent=s;statusNode.className=bad?'dirty':''}function markDirty(){dirty=true}
function imageUrl(){return'/layers/'+encodeURIComponent(activeCube)+'/'+activeLayer+'.png'}
function updateCanvasSize(){if(!image.naturalWidth)return;const z=+zoom.value;canvas.style.width=(image.naturalWidth*z)+'px';canvas.style.height=(image.naturalHeight*z)+'px'}
function counts(){const p=points();return{c:p.filter(x=>x.kind==='confirmed_chickpea').length,w:p.filter(x=>x.kind==='confirmed_weed').length,u:p.filter(x=>x.kind==='uncertain_do_not_use').length}}
function status(){const n=counts(),m=item();setStatus(activeCube+' • '+m.analysis_role+' • chickpea '+n.c+' • weed '+n.w+' • uncertain '+n.u+(dirty?' • UNSAVED':''),dirty)}
function loadImage(){activeCube=cube.value;activeLayer=layer.value;image=new Image();image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;updateCanvasSize();draw();status()};image.onerror=()=>setStatus('Layer could not be loaded.',true);image.src=imageUrl()+'?v='+Date.now()}
function pt(e){const b=canvas.getBoundingClientRect();return{x:(e.clientX-b.left)*canvas.width/b.width,y:(e.clientY-b.top)*canvas.height/b.height}}
function path(points,color,closed,width=2){if(!points||!points.length)return;ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)ctx.lineTo(points[i].x,points[i].y);if(closed)ctx.closePath();ctx.stroke();ctx.restore()}
function drawPoint(p){const m=item(),displayRadius=Math.max(4,(+p.radius_m||0)/m.preview_pixel_size_m);ctx.save();ctx.strokeStyle=colors[p.kind]||'#fff';ctx.fillStyle=(colors[p.kind]||'#fff')+'55';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,displayRadius,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(p.x-4,p.y);ctx.lineTo(p.x+4,p.y);ctx.moveTo(p.x,p.y-4);ctx.lineTo(p.x,p.y+4);ctx.stroke();ctx.restore()}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,0);if(references.checked)for(const r of refs())path(r.points,referenceColors[r.kind]||'#aaa',true,2);for(const p of points())drawPoint(p)}
canvas.onclick=e=>{const p=pt(e);points().push({id:activeCube+'-'+Date.now(),kind:kind.value,layer:activeLayer,radius_m:+radius.value,note:note.value.trim(),x:+p.x.toFixed(3),y:+p.y.toFixed(3)});note.value='';markDirty();draw();status()};
document.getElementById('undo').onclick=()=>{if(points().length){points().pop();markDirty();draw();status()}};document.getElementById('clear').onclick=()=>{if(confirm('Delete all vegetation reference points for '+activeCube+'?')){state.cubes[activeCube]=[];markDirty();draw();status()}};
document.getElementById('previous').onclick=()=>{cube.selectedIndex=Math.max(0,cube.selectedIndex-1);loadImage()};document.getElementById('next').onclick=()=>{cube.selectedIndex=Math.min(cube.options.length-1,cube.selectedIndex+1);loadImage()};cube.onchange=loadImage;layer.onchange=loadImage;zoom.onchange=()=>{updateCanvasSize()};kind.onchange=draw;references.onchange=draw;
document.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save()}};
async function save(){const b=document.getElementById('save');b.disabled=true;b.textContent='Saving…';try{const response=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!response.ok)throw new Error(await response.text());const result=await response.json();dirty=false;b.textContent='Saved ✓';setStatus('SAVED: '+result.points+' reference points across '+result.cubes+' cubes.');setTimeout(()=>{b.textContent='Save annotations';status()},2200)}catch(error){b.textContent='Save failed';setStatus('Save failed: '+error+'. Download a backup before refreshing.',true)}finally{b.disabled=false}}
document.getElementById('save').onclick=save;document.getElementById('backup').onclick=()=>{const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='field1_vegetation_reference_annotations_backup.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
async function initialize(){manifest=await(await fetch('/api/manifest')).json();state=await(await fetch('/api/annotations')).json();referenceState=await(await fetch('/api/references')).json();for(const m of manifest){const o=document.createElement('option');o.value=m.cube_id;o.textContent=m.cube_id+' ['+m.analysis_role+']';cube.appendChild(o)}if(!manifest.length){setStatus('No layers found. Run the chickpea-region layer preparation first.',true);return}activeCube=cube.value;loadImage()}initialize().catch(e=>setStatus(String(e),true));
</script></body></html>'''


class Store:
    def __init__(
        self,
        manifest_path: Path,
        output_root: Path,
        region_reference_path: Path,
        planter_reference_path: Path,
    ):
        self.manifest = pd.read_csv(manifest_path).fillna("")
        self.by_cube = {
            str(row.cube_id): row
            for row in self.manifest.itertuples(index=False)
        }
        self.output_root = output_root
        output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = output_root / "field1_vegetation_reference_annotations.json"
        self.csv_path = output_root / "field1_vegetation_reference_points.csv"
        self.geojson_path = output_root / "field1_vegetation_reference_annotations.geojson"
        self.region_reference_path = region_reference_path
        self.planter_reference_path = planter_reference_path
        self.lock = threading.Lock()

    def public_manifest(self) -> list[dict]:
        result = []
        for row in self.manifest.itertuples(index=False):
            determinant = (
                float(row.transform_a) * float(row.transform_e)
                - float(row.transform_b) * float(row.transform_d)
            )
            result.append({
                "cube_id": str(row.cube_id),
                "analysis_role": str(row.analysis_role),
                "preview_width": int(row.preview_width),
                "preview_height": int(row.preview_height),
                "preview_pixel_size_m": (
                    math.sqrt(abs(determinant)) * float(row.preview_step)
                ),
            })
        return result

    def load(self) -> dict:
        if self.json_path.exists():
            return json.loads(self.json_path.read_text())
        return {"version": 1, "cubes": {}}

    def references(self) -> dict:
        combined: dict[str, list[dict]] = {}
        for source, path in (
            ("investigator_row_polygon", self.region_reference_path),
            ("investigator_planter_footprint", self.planter_reference_path),
        ):
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for cube_id, items in payload.get("cubes", {}).items():
                for item in items:
                    reference = dict(item)
                    reference["source"] = source
                    combined.setdefault(cube_id, []).append(reference)
        return {"version": 1, "cubes": combined}

    def layer_path(self, cube_id: str, layer: str) -> Path:
        allowed = {
            "false_colour", "current_chickpea", "pca",
            "first_difference", "second_difference", "ndvi",
        }
        if cube_id not in self.by_cube or layer not in allowed:
            raise KeyError((cube_id, layer))
        return Path(getattr(self.by_cube[cube_id], f"{layer}_path"))

    def validate(self, payload: dict) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("cubes"), dict):
            raise ValueError("Expected an object with a cubes mapping")
        allowed = {"confirmed_chickpea", "confirmed_weed", "uncertain_do_not_use"}
        for cube_id, points in payload["cubes"].items():
            if cube_id not in self.by_cube or not isinstance(points, list):
                raise ValueError(f"Unknown or invalid cube: {cube_id}")
            row = self.by_cube[cube_id]
            for point in points:
                if point.get("kind") not in allowed:
                    raise ValueError("Unknown reference type")
                x, y = float(point["x"]), float(point["y"])
                if not (
                    0 <= x < int(row.preview_width)
                    and 0 <= y < int(row.preview_height)
                ):
                    raise ValueError(f"Point outside preview bounds for {cube_id}")
                radius_m = float(point.get("radius_m", 0))
                if radius_m not in {0.0, 0.03, 0.05}:
                    raise ValueError("Reference radius must be 0, 0.03, or 0.05 m")

    def save(self, payload: dict) -> tuple[int, int]:
        self.validate(payload)
        payload["version"] = 1
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        rows: list[dict] = []
        features: list[dict] = []
        cubes_with_points = 0
        for cube_id, points in payload["cubes"].items():
            if points:
                cubes_with_points += 1
            row = self.by_cube[cube_id]
            for point in points:
                px, py = float(point["x"]), float(point["y"])
                column = (px + 0.5) * float(row.preview_step)
                raster_row = (py + 0.5) * float(row.preview_step)
                map_x = (
                    float(row.transform_a) * column
                    + float(row.transform_b) * raster_row
                    + float(row.transform_c)
                )
                map_y = (
                    float(row.transform_d) * column
                    + float(row.transform_e) * raster_row
                    + float(row.transform_f)
                )
                properties = {
                    "cube_id": cube_id,
                    "annotation_id": str(point["id"]),
                    "kind": str(point["kind"]),
                    "layer": str(point.get("layer", "")),
                    "radius_m": float(point.get("radius_m", 0)),
                    "note": str(point.get("note", "")),
                    "preview_x": px,
                    "preview_y": py,
                    "original_column": column,
                    "original_row": raster_row,
                    "map_x": map_x,
                    "map_y": map_y,
                    "crs": str(row.crs),
                    "analysis_role": str(row.analysis_role),
                }
                rows.append(properties)
                features.append({
                    "type": "Feature",
                    "properties": {
                        key: value for key, value in properties.items()
                        if key not in {"map_x", "map_y"}
                    },
                    "geometry": {"type": "Point", "coordinates": [map_x, map_y]},
                })
        columns = [
            "cube_id", "annotation_id", "kind", "layer", "radius_m", "note",
            "preview_x", "preview_y", "original_column", "original_row",
            "map_x", "map_y", "crs", "analysis_role",
        ]
        with self.lock:
            temporary = self.json_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(self.json_path)
            with self.csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            self.geojson_path.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": features,
            }, indent=2))
        return len(rows), cubes_with_points


def handler_factory(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(
            self, content: bytes, content_type: str, status: int = 200
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}:
                return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/api/manifest":
                return self.send_bytes(
                    json.dumps(store.public_manifest()).encode(), "application/json"
                )
            if clean == "/api/annotations":
                return self.send_bytes(
                    json.dumps(store.load()).encode(), "application/json"
                )
            if clean == "/api/references":
                return self.send_bytes(
                    json.dumps(store.references()).encode(), "application/json"
                )
            if clean.startswith("/layers/"):
                parts = clean.strip("/").split("/")
                if len(parts) == 3 and parts[2].endswith(".png"):
                    try:
                        return self.send_bytes(
                            store.layer_path(parts[1], parts[2][:-4]).read_bytes(),
                            "image/png",
                        )
                    except (KeyError, FileNotFoundError):
                        pass
            return self.send_bytes(b"Not found", "text/plain", 404)

        def do_POST(self) -> None:
            if self.path != "/api/annotations":
                return self.send_bytes(b"Not found", "text/plain", 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20_000_000:
                    raise ValueError("Invalid request size")
                points, cubes = store.save(json.loads(self.rfile.read(length)))
                print(
                    f"SAVED: {points} reference points across {cubes} cubes",
                    flush=True,
                )
                print(f"File: {store.json_path}", flush=True)
                return self.send_bytes(
                    json.dumps({"points": points, "cubes": cubes}).encode(),
                    "application/json",
                )
            except Exception as error:
                return self.send_bytes(
                    str(error).encode(), "text/plain; charset=utf-8", 400
                )

        def log_message(self, format, *args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    report_root = local / "reports" / "chickpea_region_annotation"
    manifest_path = report_root / "chickpea_region_annotation_layer_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            "Annotation layers are missing. Run "
            "prepare_chickpea_region_annotation_layers.py first."
        )
    output_root = local / "annotations" / "vegetation_references"
    store = Store(
        manifest_path,
        output_root,
        local / "annotations" / "chickpea_regions"
        / "field1_chickpea_region_annotations.json",
        local / "annotations" / "planter_tracks"
        / "field1_planter_track_annotations.json",
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_factory(store)
    )
    url = f"http://127.0.0.1:{args.port}"
    print(f"Vegetation-reference annotator: {url}")
    print(
        "Mark clean chickpea/weed reference points and save. "
        "Press Ctrl+C here when finished."
    )
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
