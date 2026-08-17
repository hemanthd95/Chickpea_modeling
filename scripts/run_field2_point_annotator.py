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
    atomic_write_json, require_main_sampling_frame, verify_preview_hashes,
)


ANNOTATION_COLUMNS = [
    "sample_id", "cube_id", "frozen_cube_role", "selected_label", "confidence",
    "investigator_note", "reviewed", "review_timestamp", "reviewer_identifier",
    "role_contradiction", "source_preview_checksums", "sampling_frame_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_annotation(row, frame_hash: str) -> dict:
    return {
        "sample_id": str(row.sample_id), "cube_id": str(row.cube_id),
        "frozen_cube_role": str(row.cube_evaluation_role), "selected_label": "",
        "confidence": "", "investigator_note": "", "reviewed": False,
        "review_timestamp": "", "reviewer_identifier": "", "role_contradiction": False,
        "source_preview_checksums": json.loads(str(row.source_preview_checksums)),
        "sampling_frame_sha256": frame_hash,
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

    def __init__(self, project: Path, main_frame: pd.DataFrame, frame_hash: str, manifest_path: Path, output_root: Path):
        self.project = project.resolve(); self.frame = main_frame.copy(); self.frame_hash = frame_hash
        if set(self.frame.sampling_frame) != {"main"} or len(self.frame) != 800:
            raise ValueError("Point annotator accepts only the exact frozen 800-point main frame")
        self.sample_order = self.frame.sample_id.astype(str).tolist()
        self.by_sample = {str(row.sample_id): row for row in self.frame.itertuples(index=False)}
        self.manifest_frame = pd.read_csv(manifest_path).fillna("").sort_values("cube_id")
        preview_issues = verify_preview_hashes(self.manifest_frame, self.project)
        if preview_issues:
            raise ValueError(f"Prediction-free preview checksum validation failed: {preview_issues}")
        self.manifest = {str(row.cube_id): row for row in self.manifest_frame.itertuples(index=False)}
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
            "automatic_metadata": {
                "biological_labels_assigned_automatically": False,
                "reserve_frame_exposed": False, "predictions_or_probabilities_used": False,
            },
            "annotations": {
                sample_id: default_annotation(self.by_sample[sample_id], self.frame_hash)
                for sample_id in self.sample_order
            },
        }

    def load(self) -> dict:
        payload = json.loads(self.json_path.read_text()) if self.json_path.exists() else self.initial_payload()
        self.validate(payload)
        return payload

    def validate(self, payload: dict) -> dict[str, list[str]]:
        if payload.get("version") != "field2_blind_main_point_annotations_v1":
            raise ValueError("Unknown point-annotation schema version")
        if payload.get("sampling_frame_sha256") != self.frame_hash or payload.get("sample_order") != self.sample_order:
            raise ValueError("Point annotations differ from the frozen main frame")
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
            saved["automatic_metadata"] = {"biological_labels_assigned_automatically": False, "reserve_frame_exposed": False, "predictions_or_probabilities_used": False}
            atomic_write_json(self.json_path, saved); atomic_write_bytes(self.csv_path, annotation_csv_bytes(saved)); atomic_write_bytes(self.audit_path, self._audit_bytes(saved, issues)); self._write_overview(saved)
            reviewed = sum(record["reviewed"] for record in saved["annotations"].values())
            return len(saved["annotations"]), reviewed, saved["revision"]

    def public_manifest(self) -> dict:
        return {cube_id: {"preview_step": int(row.preview_step)} for cube_id, row in self.manifest.items()}

    def layer_path(self, cube_id: str, layer: str) -> Path:
        if cube_id not in self.manifest or layer not in {"false_colour", "pca", "stored_index", "support_outline"}:
            raise KeyError((cube_id, layer))
        return self.project / str(getattr(self.manifest[cube_id], f"{layer}_path"))


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Field 2 blind MAIN point annotator</title>
<style>:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}body{margin:0;background:#101418;color:#edf2f7}header{padding:9px 12px;background:#182028;position:sticky;top:0;z-index:5;border-bottom:1px solid #3f4b56}.bar{display:flex;flex-wrap:wrap;gap:7px;align-items:center}button,select,input,textarea{font:inherit;color:inherit;background:#26313b;border:1px solid #52606d;border-radius:5px;padding:6px}.primary{background:#18794e}.danger{background:#7f1d1d}.layout{display:grid;grid-template-columns:minmax(0,1fr) 390px;height:calc(100vh - 120px)}.stage{display:grid;place-items:center;overflow:hidden;background:#050708}canvas{max-width:96%;max-height:96%;border:1px solid #52606d;image-rendering:auto}.form{padding:12px;overflow:auto;background:#151b21}.field{display:block;margin:8px 0}.field span{display:block;color:#b9c4ce;font-size:.84rem}.form textarea,.form input,.form select{width:100%}#status{padding-top:7px}.dirty,.warning{color:#ffd166}.help{font-size:.78rem;color:#b9c4ce}.metadata{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:.8rem}@media(max-width:900px){.layout{grid-template-columns:1fr;height:auto}.stage{height:60vh}}</style></head><body>
<header><div class="bar"><button id="prev">← Previous</button><span id="counter"></span><button id="next">Next →</button><label>Cube <select id="cubeFilter"></select></label><label>Role <select id="roleFilter"></select></label><label>View <select id="layer"><option value="false_colour">Reflectance</option><option value="pca">PCA</option><option value="stored_index">Stored scalar index</option><option value="support_outline">Support boundary</option></select></label><label>Context <select id="context"><option value="close">Close</option><option value="medium">Medium</option></select></label><button id="save" class="primary">Save all</button></div><div id="status">Loading frozen MAIN frame…</div><div class="help">Shortcuts: ←/→ navigate · F/P/I/B views · Z toggles context · Ctrl/Cmd+S saves. Label the center pixel at the crosshair; context is supporting evidence only.</div></header>
<div class="layout"><div class="stage"><canvas id="canvas"></canvas></div><aside class="form"><h3>Investigator annotation</h3><label class="field"><span>Selected label (no default)</span><select id="label"><option value="">Unlabeled</option></select></label><label class="field"><span>Confidence</span><select id="confidence"><option value="">Not set</option></select></label><label class="field"><span>Reviewer identifier (optional)</span><input id="reviewer"></label><label class="field"><span>Investigator note (optional)</span><textarea id="note"></textarea></label><button id="review" class="primary">Mark reviewed</button> <button id="clear" class="danger">Clear current</button><p id="contradiction" class="warning"></p><h3>Frozen sample metadata</h3><div id="metadata" class="metadata"></div><p class="help">Mixed labels apply only when the central pixel is spatially mixed. Uncertainty is preferable to a forced class. Chickpea remains available on negative-control cubes; selecting it records a role contradiction without changing the frozen cube role. Reserve samples are locked and unavailable here.</p></aside></div>
<script>
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d');let samples=[],visible=[],manifest={},schema={},state={},index=0,image=new Image(),dirty=false;function sample(){return visible[index]}function record(){return state.annotations[sample().sample_id]}function status(x,bad=dirty){$('status').textContent=x;$('status').className=bad?'dirty':''}function setDirty(){dirty=true;status('UNSAVED changes — use Save all.',true)}
function filter(){const cube=$('cubeFilter').value,role=$('roleFilter').value,current=sample()?.sample_id;visible=samples.filter(s=>(!cube||s.cube_id===cube)&&(!role||s.cube_evaluation_role===role));index=Math.max(0,visible.findIndex(s=>s.sample_id===current));if(index<0)index=0;if(!visible.length){status('No MAIN samples match this filter.',true);return}load()}
function render(){const s=sample(),r=record(),m=manifest[s.cube_id],step=+m.preview_step,cx=(+s.column+.5)/step,cy=(+s.row+.5)/step,half=$('context').value==='close'?45:110;canvas.width=half*2;canvas.height=half*2;ctx.drawImage(image,cx-half,cy-half,half*2,half*2,0,0,half*2,half*2);ctx.strokeStyle='#ff2d55';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(half-14,half);ctx.lineTo(half+14,half);ctx.moveTo(half,half-14);ctx.lineTo(half,half+14);ctx.stroke();$('counter').textContent=(index+1)+' / '+visible.length+' filtered · '+Object.values(state.annotations).filter(x=>x.reviewed).length+'/800 reviewed';$('label').value=r.selected_label;$('confidence').value=r.confidence;$('reviewer').value=r.reviewer_identifier;$('note').value=r.investigator_note;$('review').textContent=r.reviewed?'Reviewed ✓':'Mark reviewed';$('contradiction').textContent=r.role_contradiction?'ROLE CONTRADICTION RECORDED: investigator selected chickpea on a frozen absent-negative-control cube.':'';$('metadata').textContent=`sample ID: ${s.sample_id}\ncube ID: ${s.cube_id}\nfrozen cube role: ${s.cube_evaluation_role}\ncube notes: ${s.cube_investigator_notes||'(none)'}\nrank stratum: ${s.scalar_index_rank_stratum}\nspatial block: ${s.spatial_group_id}\nframe: MAIN only`;status(`${s.sample_id} · ${r.reviewed?'REVIEWED':'not reviewed'} · ${r.selected_label||'unlabeled'}${dirty?' · UNSAVED':''}`,dirty)}
function load(){const s=sample();if(!s)return;image.onload=render;image.onerror=()=>status('Prediction-free preview failed to load.',true);image.src='/layers/'+s.cube_id+'/'+$('layer').value+'.png?v='+Date.now()}function move(d){index=Math.max(0,Math.min(visible.length-1,index+d));load()}function update(){const s=sample(),r=record();r.selected_label=$('label').value;r.confidence=$('confidence').value;r.reviewer_identifier=$('reviewer').value.trim();r.investigator_note=$('note').value;r.role_contradiction=r.selected_label==='chickpea'&&s.cube_evaluation_role==='chickpea_absent_negative_control';r.reviewed=false;r.review_timestamp='';setDirty();render()}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('cubeFilter').onchange=filter;$('roleFilter').onchange=filter;$('layer').onchange=load;$('context').onchange=render;for(const id of ['label','confidence','reviewer','note'])$(id).addEventListener('change',update);$('review').onclick=()=>{const r=record();update();if(!r.selected_label||!r.confidence){status('Select a label and confidence before marking reviewed.',true);return}r.reviewed=true;r.review_timestamp=new Date().toISOString();setDirty();render()};$('clear').onclick=()=>{if(!confirm('Clear this MAIN annotation?'))return;const s=sample();state.annotations[s.sample_id]={sample_id:s.sample_id,cube_id:s.cube_id,frozen_cube_role:s.cube_evaluation_role,selected_label:'',confidence:'',investigator_note:'',reviewed:false,review_timestamp:'',reviewer_identifier:'',role_contradiction:false,source_preview_checksums:JSON.parse(s.source_preview_checksums),sampling_frame_sha256:state.sampling_frame_sha256};setDirty();render()};
async function save(){const response=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!response.ok){status(await response.text(),true);return}const result=await response.json();state.revision=result.revision;dirty=false;status(`SAVED: ${result.records} MAIN records · ${result.reviewed} reviewed. File: ${result.file}`)}$('save').onclick=save;document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)&&!(e.ctrlKey||e.metaKey))return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(e.key.toLowerCase()==='f'){$('layer').value='false_colour';load()}else if(e.key.toLowerCase()==='p'){$('layer').value='pca';load()}else if(e.key.toLowerCase()==='i'){$('layer').value='stored_index';load()}else if(e.key.toLowerCase()==='b'){$('layer').value='support_outline';load()}else if(e.key.toLowerCase()==='z'){$('context').value=$('context').value==='close'?'medium':'close';render()}else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save()}};window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue=''}};
async function init(){samples=await(await fetch('/api/samples')).json();manifest=await(await fetch('/api/manifest')).json();schema=await(await fetch('/api/schema')).json();state=await(await fetch('/api/annotations')).json();if(samples.length!==800||new Set(samples.map(s=>s.sampling_frame)).size!==1||samples[0].sampling_frame!=='main')throw new Error('Expected exact frozen 800-point MAIN frame');for(const name of schema.labels){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');$('label').appendChild(o)}for(const name of schema.confidence){const o=document.createElement('option');o.value=name;o.textContent=name;$('confidence').appendChild(o)}for(const value of ['',...new Set(samples.map(s=>s.cube_id))]){const o=document.createElement('option');o.value=value;o.textContent=value||'All cubes';$('cubeFilter').appendChild(o)}for(const value of ['',...new Set(samples.map(s=>s.cube_evaluation_role))]){const o=document.createElement('option');o.value=value;o.textContent=value?value.replaceAll('_',' '):'All roles';$('roleFilter').appendChild(o)}visible=samples;load()}init().catch(e=>status(String(e),true));
</script></body></html>'''


def handler_factory(store: PointAnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(content))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(content)
        def do_GET(self):
            clean = self.path.split("?", 1)[0]
            if clean in {"/", "/index.html"}: return self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            if clean == "/api/samples": return self.send_bytes(store.frame.to_json(orient="records").encode(), "application/json")
            if clean == "/api/manifest": return self.send_bytes(json.dumps(store.public_manifest()).encode(), "application/json")
            if clean == "/api/annotations": return self.send_bytes(json.dumps(store.load()).encode(), "application/json")
            if clean == "/api/schema": return self.send_bytes(json.dumps({"labels": POINT_LABELS, "confidence": CONFIDENCE_VALUES, "frame": "main", "reserve_exposed": False}).encode(), "application/json")
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
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", required=True, type=Path); parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path); parser.add_argument("--port", type=int); parser.add_argument("--no-browser", action="store_true"); args = parser.parse_args()
    paths, config = yaml.safe_load(args.paths.read_text()), yaml.safe_load(args.config.read_text()); project = Path(paths["project_root"]).resolve()
    released = require_main_sampling_frame(project, config)
    store = PointAnnotationStore(project, released["main"], released["main_frame_sha256"], project / config["review"]["package_manifest"], project / config["point_annotation"]["output_root"])
    port = args.port or int(config["point_annotation"]["port"]); server = ThreadingHTTPServer(("127.0.0.1", port), handler_factory(store)); url = f"http://127.0.0.1:{port}"
    print(f"Field 2 prediction-free MAIN point annotator: {url}", flush=True); print("Reserve frame: locked and not served", flush=True); print(f"Resume file: {store.json_path}", flush=True)
    if not args.no_browser: threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nPoint annotator stopped. Saved MAIN annotations remain on disk.", flush=True)


if __name__ == "__main__": main()
