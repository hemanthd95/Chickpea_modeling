#!/usr/bin/env python
"""Run the localhost-only prediction-free Field 2 cube-role reviewer."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import webbrowser

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_blind_review import (
    CONFIDENCE_VALUES,
    CubeRoleReviewStore,
    PREVIEW_LAYERS,
    PRIMARY_ROLES,
    REVIEW_FLAGS,
)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field 2 prediction-free cube-role reviewer</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#101418;color:#edf2f7}
header{position:sticky;top:0;z-index:20;background:#182028;border-bottom:1px solid #3f4b56;padding:9px 12px}.toolbar{display:flex;flex-wrap:wrap;gap:7px;align-items:center}
button,select,input,textarea{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px 8px}button:hover{background:#344451}.primary{background:#18794e;border-color:#2fb171}.danger{background:#7f1d1d;border-color:#b64242}.dirty{color:#ffd166!important}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;height:calc(100vh - 104px)}#stage{position:relative;overflow:hidden;background:#050708;cursor:grab}#stage.dragging{cursor:grabbing}#image{position:absolute;left:0;top:0;transform-origin:0 0;max-width:none;user-select:none;pointer-events:none}
#form{overflow:auto;padding:14px;background:#151b21;border-left:1px solid #3f4b56}.section{padding:10px 0;border-bottom:1px solid #38434d}.section h3{margin:0 0 8px;font-size:1rem}.field{display:block;margin:8px 0}.field>span{display:block;color:#b9c4ce;font-size:.86rem;margin-bottom:3px}.flags{display:grid;grid-template-columns:1fr;gap:5px}.flags label{display:flex;gap:7px;align-items:flex-start}.flags input{margin-top:3px}textarea{width:100%;min-height:68px}#metadata{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:.78rem;color:#b9c4ce;line-height:1.5}#status{padding-top:7px;color:#cbd5e1;font-size:.9rem}@media(max-width:900px){main{grid-template-columns:1fr;height:auto}#stage{height:65vh}#form{border-left:0;border-top:1px solid #3f4b56}}
</style></head><body>
<header><div class="toolbar"><button id="previous">← Previous</button><label>Cube <select id="cube"></select></label><button id="next">Next →</button>
<label>View <select id="layer"></select></label><button id="zoomOut">−</button><button id="zoomIn">+</button><button id="reset">Reset view</button>
<button id="clear" class="danger">Clear current review</button><button id="save" class="primary">Save all reviews</button></div><div id="status">Loading…</div></header>
<main><div id="stage"><img id="image" alt="Prediction-free review layer"></div><aside id="form">
<div class="section"><h3>Investigator decision</h3><label class="field"><span>Primary evaluation role</span><select id="role"></select></label><label class="field"><span>Confidence</span><select id="confidence"><option value="">Not set</option></select></label><label class="field"><span>Reviewer identifier (optional)</span><input id="reviewer" autocomplete="off"></label><label><input type="checkbox" id="reviewed"> Review complete</label></div>
<div class="section"><h3>Independent investigator flags</h3><div id="flags" class="flags"></div></div>
<div class="section"><label class="field"><span>Investigator notes</span><textarea id="notes"></textarea></label><label class="field"><span>Exclusion reason (required only for exclusion)</span><textarea id="exclusion"></textarea></label></div>
<div class="section"><h3>Automatic non-biological metadata</h3><div id="metadata"></div></div>
</aside></main>
<script>
const cube=document.getElementById('cube'),layer=document.getElementById('layer'),image=document.getElementById('image'),stage=document.getElementById('stage'),statusNode=document.getElementById('status');
const role=document.getElementById('role'),confidence=document.getElementById('confidence'),reviewer=document.getElementById('reviewer'),reviewed=document.getElementById('reviewed'),notes=document.getElementById('notes'),exclusion=document.getElementById('exclusion'),flagsNode=document.getElementById('flags'),metadata=document.getElementById('metadata');
let manifest=[],schema={},state={},dirty=false,scale=1,panX=0,panY=0,drag=null;
function item(){return manifest.find(x=>x.cube_id===cube.value)}function record(){return state.reviews[cube.value]}function setStatus(text,bad=dirty){statusNode.textContent=text;statusNode.className=bad?'dirty':''}
function markDirty(){dirty=true;setStatus('UNSAVED changes — use Save all reviews.',true)}function layerUrl(){return'/layers/'+encodeURIComponent(cube.value)+'/'+layer.value+'.png?v='+Date.now()}
function transform(){image.style.transform=`translate(${panX}px,${panY}px) scale(${scale})`}function resetView(){if(!image.naturalWidth)return;scale=Math.min(stage.clientWidth/image.naturalWidth,stage.clientHeight/image.naturalHeight);panX=(stage.clientWidth-image.naturalWidth*scale)/2;panY=(stage.clientHeight-image.naturalHeight*scale)/2;transform()}
function loadImage(){image.onload=resetView;image.onerror=()=>setStatus('Review layer failed to load.',true);image.src=layerUrl()}
function updateRecord(){const r=record();r.primary_role=role.value;r.confidence=confidence.value;r.reviewer_identifier=reviewer.value.trim();r.investigator_notes=notes.value;r.exclusion_reason=exclusion.value;r.reviewed=reviewed.checked;if(r.reviewed&&!r.review_timestamp)r.review_timestamp=new Date().toISOString();if(!r.reviewed)r.review_timestamp='';for(const name of schema.flags)r.flags[name]=document.getElementById('flag-'+name).checked;markDirty();updateStatus()}
function updateStatus(){const done=Object.values(state.reviews).filter(x=>x.reviewed).length,r=record();setStatus(cube.value+' • '+r.primary_role+' • '+(r.reviewed?'REVIEWED':'unreviewed')+' • '+done+'/40 complete'+(dirty?' • UNSAVED':''),dirty)}
function renderForm(){const r=record();role.value=r.primary_role;confidence.value=r.confidence;reviewer.value=r.reviewer_identifier;reviewed.checked=r.reviewed;notes.value=r.investigator_notes;exclusion.value=r.exclusion_reason;for(const name of schema.flags)document.getElementById('flag-'+name).checked=!!r.flags[name];const m=item();metadata.textContent=`cube ID: ${m.cube_id}\ndimensions: ${m.width} × ${m.height}\nGSD: ${m.gsd_m} m\nvalid fraction: ${(+m.valid_fraction).toFixed(6)}\nbounds: ${m.bounds}\nfootprint overlap warning: ${m.footprint_overlap_warning}\nspectral QC: ${m.spectral_qc_status}\nalignment: ${m.alignment_status}\nmetadata source: automatic contract fields\nrole/flags source: investigator only`;updateStatus()}
function changeCube(index){cube.selectedIndex=Math.max(0,Math.min(cube.options.length-1,index));renderForm();loadImage()}
for(const node of [role,confidence,reviewer,reviewed,notes,exclusion])node.addEventListener('change',updateRecord);
document.getElementById('previous').onclick=()=>changeCube(cube.selectedIndex-1);document.getElementById('next').onclick=()=>changeCube(cube.selectedIndex+1);cube.onchange=()=>{renderForm();loadImage()};layer.onchange=loadImage;
document.getElementById('zoomIn').onclick=()=>{scale=Math.min(12,scale*1.25);transform()};document.getElementById('zoomOut').onclick=()=>{scale=Math.max(.05,scale/1.25);transform()};document.getElementById('reset').onclick=resetView;
stage.onwheel=e=>{e.preventDefault();const box=stage.getBoundingClientRect(),mx=e.clientX-box.left,my=e.clientY-box.top,old=scale;scale=Math.max(.05,Math.min(12,scale*(e.deltaY<0?1.15:1/1.15)));panX=mx-(mx-panX)*scale/old;panY=my-(my-panY)*scale/old;transform()};stage.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,px:panX,py:panY};stage.setPointerCapture(e.pointerId);stage.classList.add('dragging')};stage.onpointermove=e=>{if(drag){panX=drag.px+e.clientX-drag.x;panY=drag.py+e.clientY-drag.y;transform()}};stage.onpointerup=()=>{drag=null;stage.classList.remove('dragging')};
document.getElementById('clear').onclick=()=>{if(!confirm('Clear the current review for '+cube.value+'?'))return;const m=item(),hashes=typeof m.preview_sha256_json==='string'?JSON.parse(m.preview_sha256_json):m.preview_sha256_json;state.reviews[cube.value]={cube_id:cube.value,primary_role:'unreviewed',flags:Object.fromEntries(schema.flags.map(x=>[x,false])),investigator_notes:'',exclusion_reason:'',confidence:'',reviewed:false,review_timestamp:'',reviewer_identifier:'',source_preview_checksums:hashes};markDirty();renderForm()};
async function save(){const b=document.getElementById('save');b.disabled=true;b.textContent='Saving…';try{const response=await fetch('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!response.ok)throw new Error(await response.text());const result=await response.json();state.revision=result.revision;dirty=false;b.textContent='Saved ✓';setStatus('SAVED: '+result.records+' cube-role reviews • '+result.reviewed+' reviewed. File: '+result.file);setTimeout(()=>{b.textContent='Save all reviews';updateStatus()},2500)}catch(error){b.textContent='Save failed';setStatus('Save failed: '+String(error)+'. Resolve the reported logical rule or reload.',true)}finally{b.disabled=false}}
document.getElementById('save').onclick=save;document.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save()}};window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
async function initialize(){manifest=await(await fetch('/api/manifest')).json();schema=await(await fetch('/api/schema')).json();state=await(await fetch('/api/reviews')).json();for(const m of manifest){const o=document.createElement('option');o.value=m.cube_id;o.textContent=m.cube_id;cube.appendChild(o)}for(const name of schema.layers){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');layer.appendChild(o)}for(const name of schema.primary_roles){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');role.appendChild(o)}for(const name of schema.confidence){const o=document.createElement('option');o.value=name;o.textContent=name;confidence.appendChild(o)}for(const name of schema.flags){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.id='flag-'+name;input.onchange=updateRecord;label.append(input,document.createTextNode(name.replaceAll('_',' ')));flagsNode.appendChild(label)}if(manifest.length!==40)throw new Error('Expected exactly 40 review packages');renderForm();loadImage()}initialize().catch(e=>setStatus(String(e),true));
</script></body></html>'''


def handler_factory(store: CubeRoleReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(content)

        def do_GET(self) -> None:
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}:
                return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/api/manifest":
                return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if clean == "/api/reviews":
                return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if clean == "/api/schema":
                schema = {"primary_roles": PRIMARY_ROLES, "flags": REVIEW_FLAGS,
                          "confidence": CONFIDENCE_VALUES, "layers": PREVIEW_LAYERS}
                return self.send_bytes(json.dumps(schema).encode(), "application/json")
            if clean.startswith("/layers/"):
                parts = clean.strip("/").split("/")
                if len(parts) == 3 and parts[2].endswith(".png"):
                    try:
                        return self.send_bytes(store.layer_path(parts[1], parts[2][:-4]).read_bytes(), "image/png")
                    except (KeyError, FileNotFoundError):
                        pass
            return self.send_bytes(b"Not found", "text/plain", 404)

        def do_POST(self) -> None:
            if self.path != "/api/reviews":
                return self.send_bytes(b"Not found", "text/plain", 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10_000_000:
                    raise ValueError("Invalid request size")
                records, reviewed_count = store.save(json.loads(self.rfile.read(length)))
                saved = store.load()
                print(f"SAVED: {records} cube-role reviews", flush=True)
                print(f"File: {store.json_path}", flush=True)
                response = {"records": records, "reviewed": reviewed_count,
                            "revision": saved["revision"], "file": str(store.json_path)}
                return self.send_bytes(json.dumps(response).encode(), "application/json")
            except Exception as error:
                return self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)

        def log_message(self, format, *args):
            return
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()
    manifest_path = project / config["review"]["package_manifest"]
    if not manifest_path.is_file():
        raise SystemExit("Review package missing. Run prepare_field2_cube_role_review.py first.")
    store = CubeRoleReviewStore(project, manifest_path, project / config["review"]["annotations_root"])
    if store.expected_cube_ids != list(config["expected_cube_ids"]):
        raise SystemExit("Review package does not contain the exact frozen 40-cube inventory")
    port = args.port or int(config["review"]["port"])
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(store))
    url = f"http://127.0.0.1:{port}"
    print(f"Field 2 prediction-free cube-role reviewer: {url}", flush=True)
    print(f"Resume file: {store.json_path}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReviewer stopped. Saved cube-role reviews remain on disk.", flush=True)


if __name__ == "__main__":
    main()
