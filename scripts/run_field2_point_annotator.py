#!/usr/bin/env python
"""Future prediction-free point-annotation UI; requires a frozen sampling frame."""

from __future__ import annotations

import argparse
import csv
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

from chickpea_ssl.field2_blind_review import POINT_LABELS, atomic_write_bytes, atomic_write_json
from chickpea_ssl.field2_readiness import sha256


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Field 2 blind point annotator</title><style>:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#101418;color:#edf2f7}header{padding:10px;background:#182028;position:sticky;top:0;z-index:5}.bar{display:flex;flex-wrap:wrap;gap:7px;align-items:center}button,select,input{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px}.primary{background:#18794e}.stage{display:grid;place-items:center;height:78vh;overflow:hidden}canvas{max-width:96%;max-height:96%;border:1px solid #52606d}#status{padding-top:7px}.dirty{color:#ffd166}</style></head><body>
<header><div class="bar"><button id="prev">← Previous</button><span id="counter"></span><button id="next">Next →</button><label>View <select id="layer"><option value="false_colour">Reflectance</option><option value="pca">PCA</option><option value="stored_index">Stored index</option><option value="support_outline">Support boundary</option></select></label><label>Context <select id="context"><option value="close">Close</option><option value="medium">Medium</option></select></label><label>Label <select id="label"></select></label><label>Note <input id="note"></label><button id="save" class="primary">Save annotations</button></div><div id="status">Loading frozen sampling frame…</div></header><div class="stage"><canvas id="canvas"></canvas></div>
<script>const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),layer=document.getElementById('layer'),context=document.getElementById('context'),label=document.getElementById('label'),note=document.getElementById('note'),statusNode=document.getElementById('status');let samples=[],manifest={},state={revision:0,annotations:{}},index=0,image=new Image(),dirty=false;function sample(){return samples[index]}function setStatus(x,bad=dirty){statusNode.textContent=x;statusNode.className=bad?'dirty':''}function render(){const s=sample(),m=manifest[s.cube_id],step=+m.preview_step,cx=(+s.column+.5)/step,cy=(+s.row+.5)/step,half=context.value==='close'?45:110;canvas.width=half*2;canvas.height=half*2;ctx.drawImage(image,cx-half,cy-half,half*2,half*2,0,0,half*2,half*2);ctx.strokeStyle='#ff3355';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(half-12,half);ctx.lineTo(half+12,half);ctx.moveTo(half,half-12);ctx.lineTo(half,half+12);ctx.stroke();document.getElementById('counter').textContent=(index+1)+' / '+samples.length+' • '+s.sample_id+' • '+s.cube_id;const a=state.annotations[s.sample_id]||{};label.value=a.label||'';note.value=a.note||'';setStatus(s.sample_id+' • prediction-free • '+(a.label||'unlabeled')+(dirty?' • UNSAVED':''),dirty)}function load(){const s=sample();image.onload=render;image.src='/layers/'+s.cube_id+'/'+layer.value+'.png?v='+Date.now()}function move(d){index=Math.max(0,Math.min(samples.length-1,index+d));load()}document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);layer.onchange=load;context.onchange=render;label.onchange=()=>{const s=sample();state.annotations[s.sample_id]={sample_id:s.sample_id,label:label.value,note:note.value};dirty=true;render()};note.onchange=()=>{const s=sample();if(state.annotations[s.sample_id])state.annotations[s.sample_id].note=note.value;dirty=true;render()};async function save(){const response=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!response.ok){setStatus(await response.text(),true);return}const result=await response.json();state.revision=result.revision;dirty=false;setStatus('SAVED: '+result.count+' blind point annotations')}document.getElementById('save').onclick=save;window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};async function init(){samples=await(await fetch('/api/samples')).json();manifest=await(await fetch('/api/manifest')).json();state=await(await fetch('/api/annotations')).json();const schema=await(await fetch('/api/schema')).json();const empty=document.createElement('option');empty.value='';empty.textContent='Unlabeled';label.appendChild(empty);for(const name of schema.labels){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');label.appendChild(o)}if(!samples.length)throw new Error('Frozen sampling frame is empty');load()}init().catch(e=>setStatus(String(e),true));</script></body></html>'''


class PointAnnotationStore:
    def __init__(self, project: Path, frame_path: Path, manifest_path: Path, output_root: Path):
        self.project = project
        self.frame = pd.read_csv(frame_path).fillna("")
        self.samples = set(self.frame.sample_id.astype(str))
        manifest = pd.read_csv(manifest_path).fillna("")
        self.manifest = {str(row.cube_id): row for row in manifest.itertuples(index=False)}
        self.output_root = output_root; output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = output_root / "field2_blind_point_annotations.json"
        self.csv_path = output_root / "field2_blind_point_annotations.csv"
        self.lock = threading.Lock()

    def load(self) -> dict:
        return json.loads(self.json_path.read_text()) if self.json_path.exists() else {"version": 1, "revision": 0, "annotations": {}}

    def validate(self, payload: dict) -> None:
        if not isinstance(payload.get("annotations"), dict):
            raise ValueError("Expected annotations mapping")
        for sample_id, record in payload["annotations"].items():
            if sample_id not in self.samples or record.get("label") not in POINT_LABELS:
                raise ValueError(f"Unknown sample or label: {sample_id}")

    def save(self, payload: dict) -> tuple[int, int]:
        with self.lock:
            current = self.load()
            if int(payload.get("revision", -1)) != int(current["revision"]):
                raise ValueError("Annotation revision conflict; reload")
            self.validate(payload)
            saved = json.loads(json.dumps(payload)); saved["version"] = 1; saved["revision"] = current["revision"] + 1
            atomic_write_json(self.json_path, saved)
            buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=["sample_id", "label", "note"]); writer.writeheader()
            writer.writerows(saved["annotations"].values()); atomic_write_bytes(self.csv_path, buffer.getvalue().encode())
            return len(saved["annotations"]), saved["revision"]

    def public_manifest(self) -> dict:
        return {cube_id: {"preview_step": int(row.preview_step)} for cube_id, row in self.manifest.items()}

    def layer_path(self, cube_id: str, layer: str) -> Path:
        if cube_id not in self.manifest or layer not in {"false_colour", "pca", "stored_index", "support_outline"}:
            raise KeyError((cube_id, layer))
        return self.project / str(getattr(self.manifest[cube_id], f"{layer}_path"))


def handler_factory(store: PointAnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(content)
        def do_GET(self):
            clean=self.path.split("?",1)[0]
            if clean in {"/","/index.html"}: return self.send_bytes(HTML.encode(),"text/html; charset=utf-8")
            if clean=="/api/samples": return self.send_bytes(self.frame_json,"application/json")
            if clean=="/api/manifest": return self.send_bytes(json.dumps(store.public_manifest()).encode(),"application/json")
            if clean=="/api/annotations": return self.send_bytes(json.dumps(store.load()).encode(),"application/json")
            if clean=="/api/schema": return self.send_bytes(json.dumps({"labels":POINT_LABELS}).encode(),"application/json")
            if clean.startswith("/layers/"):
                parts=clean.strip("/").split("/")
                if len(parts)==3 and parts[2].endswith(".png"):
                    try:return self.send_bytes(store.layer_path(parts[1],parts[2][:-4]).read_bytes(),"image/png")
                    except (KeyError,FileNotFoundError):pass
            return self.send_bytes(b"Not found","text/plain",404)
        @property
        def frame_json(self): return store.frame.to_json(orient="records").encode()
        def do_POST(self):
            if self.path!="/api/annotations":return self.send_bytes(b"Not found","text/plain",404)
            try:
                length=int(self.headers.get("Content-Length","0")); count,revision=store.save(json.loads(self.rfile.read(length))); return self.send_bytes(json.dumps({"count":count,"revision":revision}).encode(),"application/json")
            except Exception as error:return self.send_bytes(str(error).encode(),"text/plain; charset=utf-8",400)
        def log_message(self,format,*args):return
    return Handler


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--paths",required=True,type=Path);parser.add_argument("--config",default=Path("configs/field2_blind_evaluation.yaml"),type=Path);parser.add_argument("--port",type=int);parser.add_argument("--no-browser",action="store_true");args=parser.parse_args()
    paths=yaml.safe_load(args.paths.read_text());config=yaml.safe_load(args.config.read_text());project=Path(paths["project_root"]).resolve();sampling=config["sampling"]
    contract_path=project/sampling["sampling_contract"]
    if not contract_path.is_file():raise SystemExit("REFUSED: frozen blind sampling-frame contract is required before point annotation")
    contract=yaml.safe_load(contract_path.read_text());frame_path=project/sampling["sampling_frame"]
    if contract.get("status")!="field2_blind_sampling_frame_frozen" or contract.get("sampling_frame_sha256")!=sha256(frame_path):raise SystemExit("REFUSED: invalid or changed blind sampling frame")
    store=PointAnnotationStore(project,frame_path,project/config["review"]["package_manifest"],project/config["point_annotation"]["output_root"]);port=args.port or int(config["point_annotation"]["port"]);server=ThreadingHTTPServer(("127.0.0.1",port),handler_factory(store));url=f"http://127.0.0.1:{port}";print(f"Field 2 blind point annotator: {url}",flush=True)
    if not args.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:print("\nPoint annotator stopped.",flush=True)


if __name__=="__main__":main()
