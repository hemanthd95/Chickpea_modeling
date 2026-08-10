#!/usr/bin/env python
"""Run a localhost-only annotator for chickpea-growing region polygons."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import webbrowser

import pandas as pd
import yaml


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field 1 chickpea-region annotator</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#101418;color:#edf2f7}
header{position:sticky;top:0;z-index:10;padding:10px 14px;background:#182028;border-bottom:1px solid #39434d}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}button,select,input{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px 9px}button:hover{background:#33414d}.primary{background:#18794e;border-color:#2fb171}.danger{background:#7f1d1d;border-color:#b64242}label{display:flex;gap:5px;align-items:center}
#stage{position:relative;margin:12px auto;width:min(96vw,1200px)}canvas{width:100%;height:auto;display:block;background:#063b3d;border:1px solid #52606d;cursor:crosshair}#status{padding-top:8px;color:#cbd5e1}.dirty{color:#ffcc66!important}#help{padding:0 16px 14px;color:#b9c4ce;line-height:1.4}.role{font-weight:700;color:#ffd166}
</style></head><body>
<header><div class="toolbar">
<button id="previous">← Previous</button><label>Cube <select id="cube"></select></label><button id="next">Next →</button>
<label>Layer <select id="layer"><option value="false_colour">NIR–red–green</option><option value="current_chickpea">Current chickpea overlay</option><option value="pca">PCA RGB</option><option value="first_difference">PCA first difference</option><option value="second_difference">PCA second difference</option><option value="ndvi">NDVI</option></select></label>
<label>Polygon <select id="kind"><option value="chickpea_growing_block">Chickpea-growing block</option><option value="exclude_non_chickpea">Exclude from chickpea</option><option value="uncertain_boundary">Uncertain boundary</option></select></label>
<label><input id="references" type="checkbox" checked>Show alley/tyre references</label><label>Note <input id="note" size="15" placeholder="optional"></label>
<button id="finish">Close polygon</button><button id="undoPoint">Undo point</button><button id="undoPolygon">Undo polygon</button><button id="clear" class="danger">Clear cube</button><button id="save" class="primary">Save annotations</button><button id="backup">Download backup</button>
</div><div id="status">Loading…</div></header>
<div id="stage"><canvas id="canvas"></canvas></div>
<div id="help"><b>Recommended:</b> draw one polygon around each distinct six-row planting block/pass. Do not trace individual plants. Use “Exclude from chickpea” for known non-crop vegetation inside a block. Existing alleys are orange and tyre footprints are magenta reference outlines; they are locked. Press Enter to close a polygon and Ctrl+S to save.</div>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),cube=document.getElementById('cube'),layer=document.getElementById('layer'),kind=document.getElementById('kind'),note=document.getElementById('note'),statusNode=document.getElementById('status'),references=document.getElementById('references');
let manifest=[],state={version:1,cubes:{}},referenceState={cubes:{}},image=new Image(),drawing=[],pointer=null,dirty=false,activeCube='';
const colors={chickpea_growing_block:'#35f28b',exclude_non_chickpea:'#ff5d5d',uncertain_boundary:'#ffd84d'},referenceColors={alley_boundary:'#ff9f1c',tyre_track:'#e754c6',uncertain_structure:'#9aa6b2'};
function polygons(){if(!state.cubes[activeCube])state.cubes[activeCube]=[];return state.cubes[activeCube]}
function refs(){return(referenceState.cubes&&referenceState.cubes[activeCube])||[]}
function setStatus(s,bad=dirty){statusNode.textContent=s;statusNode.className=bad?'dirty':''}function markDirty(){dirty=true;setStatus('Unsaved changes — use Save annotations.')}
function imageUrl(){return'/layers/'+encodeURIComponent(activeCube)+'/'+layer.value+'.png'}
function loadImage(){drawing=[];pointer=null;activeCube=cube.value;image=new Image();image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw();const item=manifest.find(x=>x.cube_id===activeCube);setStatus(activeCube+' • '+item.analysis_role+' • '+polygons().length+' polygons'+(dirty?' • unsaved':''))};image.onerror=()=>setStatus('Layer could not be loaded.',true);image.src=imageUrl()+'?v='+Date.now()}
function pt(e){const b=canvas.getBoundingClientRect();return{x:(e.clientX-b.left)*canvas.width/b.width,y:(e.clientY-b.top)*canvas.height/b.height}}
function path(points,color,closed,width=3,fill=false){if(!points||!points.length)return;ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color+'28';ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)ctx.lineTo(points[i].x,points[i].y);if(closed)ctx.closePath();if(fill)ctx.fill();ctx.stroke();for(const p of points){ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fillStyle=color;ctx.fill()}ctx.restore()}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,0);if(references.checked)for(const r of refs())path(r.points,referenceColors[r.kind]||'#aaa',true,2,false);for(const p of polygons())path(p.points,colors[p.kind]||'#ffd84d',true,3,true);if(drawing.length)path(pointer?drawing.concat([pointer]):drawing,colors[kind.value],false,2,false)}
function finish(){if(drawing.length<3){setStatus('A polygon needs at least three points.',true);return false}polygons().push({id:activeCube+'-'+Date.now(),kind:kind.value,layer:layer.value,note:note.value.trim(),points:drawing.map(p=>({x:+p.x.toFixed(3),y:+p.y.toFixed(3)}))});drawing=[];pointer=null;note.value='';markDirty();draw();return true}
function canNavigate(){if(!drawing.length)return true;setStatus('Close the current polygon or press Escape before changing cube/layer.',true);return false}
canvas.onclick=e=>{drawing.push(pt(e));markDirty();draw()};canvas.onmousemove=e=>{pointer=pt(e);draw()};canvas.onmouseleave=()=>{pointer=null;draw()};
document.getElementById('finish').onclick=finish;document.getElementById('undoPoint').onclick=()=>{if(drawing.length){drawing.pop();markDirty();draw()}};document.getElementById('undoPolygon').onclick=()=>{if(polygons().length){polygons().pop();markDirty();draw()}};document.getElementById('clear').onclick=()=>{if(confirm('Delete all chickpea-region polygons for '+activeCube+'?')){state.cubes[activeCube]=[];drawing=[];markDirty();draw()}};
document.getElementById('previous').onclick=()=>{if(!canNavigate())return;cube.selectedIndex=Math.max(0,cube.selectedIndex-1);loadImage()};document.getElementById('next').onclick=()=>{if(!canNavigate())return;cube.selectedIndex=Math.min(cube.options.length-1,cube.selectedIndex+1);loadImage()};
cube.onchange=()=>{if(!canNavigate()){cube.value=activeCube;return}loadImage()};layer.onchange=()=>{if(!canNavigate())return;loadImage()};kind.onchange=draw;references.onchange=draw;
document.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();finish()}if(e.key==='Escape'){drawing=[];pointer=null;draw()}if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save()}};
async function save(){if(drawing.length){setStatus('Close or discard the current polygon before saving.',true);return}const b=document.getElementById('save');b.disabled=true;b.textContent='Saving…';try{const response=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!response.ok)throw new Error(await response.text());const result=await response.json();dirty=false;b.textContent='Saved ✓';setStatus('Saved '+result.polygons+' polygons and '+result.vertices+' vertices to disk.');setTimeout(()=>b.textContent='Save annotations',1800)}catch(error){b.textContent='Save failed';setStatus('Save failed: '+error+'. Download a backup before refreshing.',true)}finally{b.disabled=false}}
document.getElementById('save').onclick=save;document.getElementById('backup').onclick=()=>{const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='field1_chickpea_region_annotations_backup.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
async function initialize(){manifest=await(await fetch('/api/manifest')).json();state=await(await fetch('/api/annotations')).json();referenceState=await(await fetch('/api/references')).json();for(const item of manifest){const o=document.createElement('option');o.value=item.cube_id;o.textContent=item.cube_id+' ['+item.analysis_role+']';cube.appendChild(o)}if(!manifest.length){setStatus('No layers found. Run the preparation script first.',true);return}activeCube=cube.value;loadImage()}initialize().catch(e=>setStatus(String(e),true));
</script></body></html>'''


class Store:
    def __init__(self, manifest_path: Path, output_root: Path, reference_path: Path):
        self.manifest = pd.read_csv(manifest_path).fillna("")
        self.by_cube = {str(row.cube_id): row for row in self.manifest.itertuples(index=False)}
        self.output_root = output_root
        output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = output_root / "field1_chickpea_region_annotations.json"
        self.csv_path = output_root / "field1_chickpea_region_vertices.csv"
        self.geojson_path = output_root / "field1_chickpea_region_annotations.geojson"
        self.reference_path = reference_path
        self.lock = threading.Lock()

    def public_manifest(self) -> list[dict]:
        return [{
            "cube_id": str(r.cube_id), "analysis_role": str(r.analysis_role),
            "preview_width": int(r.preview_width), "preview_height": int(r.preview_height),
        } for r in self.manifest.itertuples(index=False)]

    def load(self) -> dict:
        return json.loads(self.json_path.read_text()) if self.json_path.exists() else {"version": 1, "cubes": {}}

    def references(self) -> dict:
        return json.loads(self.reference_path.read_text()) if self.reference_path.exists() else {"version": 1, "cubes": {}}

    def layer_path(self, cube_id: str, layer: str) -> Path:
        allowed = {"false_colour", "current_chickpea", "pca", "first_difference", "second_difference", "ndvi"}
        if cube_id not in self.by_cube or layer not in allowed:
            raise KeyError((cube_id, layer))
        return Path(getattr(self.by_cube[cube_id], f"{layer}_path"))

    def validate(self, payload: dict) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("cubes"), dict):
            raise ValueError("Expected an object with a cubes mapping")
        allowed = {"chickpea_growing_block", "exclude_non_chickpea", "uncertain_boundary"}
        for cube_id, polygons in payload["cubes"].items():
            if cube_id not in self.by_cube or not isinstance(polygons, list):
                raise ValueError(f"Unknown or invalid cube: {cube_id}")
            row = self.by_cube[cube_id]
            for polygon in polygons:
                if polygon.get("kind") not in allowed:
                    raise ValueError("Unknown polygon type")
                points = polygon.get("points")
                if not isinstance(points, list) or len(points) < 3:
                    raise ValueError("Every polygon must contain at least three points")
                for point in points:
                    x, y = float(point["x"]), float(point["y"])
                    if not (0 <= x < int(row.preview_width) and 0 <= y < int(row.preview_height)):
                        raise ValueError(f"Point outside preview bounds for {cube_id}")

    def save(self, payload: dict) -> tuple[int, int]:
        self.validate(payload)
        payload["version"] = 1
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        vertices, features, polygon_count = [], [], 0
        for cube_id, polygons in payload["cubes"].items():
            row = self.by_cube[cube_id]
            for polygon in polygons:
                polygon_count += 1
                coordinates = []
                for index, point in enumerate(polygon["points"]):
                    px, py = float(point["x"]), float(point["y"])
                    column = (px + 0.5) * float(row.preview_step)
                    raster_row = (py + 0.5) * float(row.preview_step)
                    map_x = float(row.transform_a) * column + float(row.transform_b) * raster_row + float(row.transform_c)
                    map_y = float(row.transform_d) * column + float(row.transform_e) * raster_row + float(row.transform_f)
                    coordinates.append([map_x, map_y])
                    vertices.append({"cube_id": cube_id, "annotation_id": polygon["id"], "kind": polygon["kind"], "layer": polygon.get("layer", ""), "note": polygon.get("note", ""), "vertex_index": index, "preview_x": px, "preview_y": py, "original_column": column, "original_row": raster_row, "map_x": map_x, "map_y": map_y, "crs": str(row.crs), "analysis_role": str(row.analysis_role)})
                ring = coordinates + [coordinates[0]]
                features.append({"type": "Feature", "properties": {"cube_id": cube_id, "annotation_id": polygon["id"], "kind": polygon["kind"], "layer": polygon.get("layer", ""), "note": polygon.get("note", ""), "crs": str(row.crs), "analysis_role": str(row.analysis_role)}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
        with self.lock:
            temporary = self.json_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2)); temporary.replace(self.json_path)
            columns = ["cube_id", "annotation_id", "kind", "layer", "note", "vertex_index", "preview_x", "preview_y", "original_column", "original_row", "map_x", "map_y", "crs", "analysis_role"]
            with self.csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(vertices)
            self.geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
        return polygon_count, len(vertices)


def handler_factory(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(content)
        def do_GET(self):
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}: return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/api/manifest": return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if clean == "/api/annotations": return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if clean == "/api/references": return self.send_bytes(json.dumps(store.references()).encode(), "application/json")
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
                if length <= 0 or length > 20_000_000: raise ValueError("Invalid request size")
                polygons, vertices = store.save(json.loads(self.rfile.read(length)))
                print(f"SAVED: {polygons} polygons, {vertices} vertices", flush=True)
                print(f"File: {store.json_path}", flush=True)
                return self.send_bytes(json.dumps({"polygons": polygons, "vertices": vertices}).encode(), "application/json")
            except Exception as error: return self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)
        def log_message(self, format, *args): return
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    report_root = project / "metadata" / "local" / "reports" / "chickpea_region_annotation"
    manifest_path = report_root / "chickpea_region_annotation_layer_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("Annotation layers are missing. Run prepare_chickpea_region_annotation_layers.py first.")
    output_root = project / "metadata" / "local" / "annotations" / "chickpea_regions"
    reference_path = project / "metadata" / "local" / "annotations" / "planter_tracks" / "field1_planter_track_annotations.json"
    store = Store(manifest_path, output_root, reference_path)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory(store))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Chickpea-region annotator: {url}")
    print("Draw planting-block polygons and save. Press Ctrl+C here when finished.")
    print(f"Annotations: {store.json_path}")
    if not args.no_browser: threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nAnnotator stopped. Saved annotations remain on disk.")
    finally: server.server_close()


if __name__ == "__main__":
    main()
