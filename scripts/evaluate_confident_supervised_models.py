#!/usr/bin/env python
"""Paired evaluation of historical and confident-label center-context models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi
import torch
import yaml

from chickpea_ssl.data import load_band_indices, load_records
from chickpea_ssl.model import build_supervised_model
from chickpea_ssl.spatial import map_block_indices, neighbour_boundary_safe_mask, spatial_group_id
from scripts.prepare_confident_supervised_data import frame_for_labels, valid_spectra
from scripts.run_supervised_outer_evaluation import gather_patches, metrics_from_confusion


MODELS = ["legacy_historical_labels", "confident_with_alley", "confident_without_alley"]
SEEDS = [42, 43, 44]
CLASS_NAMES = ["soil", "chickpea", "weed"]


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk:=stream.read(8*1024*1024): digest.update(chunk)
    return digest.hexdigest()


def load_model_set(project, outer, bands, device, training_reports, old_contract):
    result={}
    old_root=project/"results/supervised_nested_training"/f"outer_fold{outer}"
    old=[]
    for seed in SEEDS:
        path=old_root/f"center_context_fusion_seed{seed}_best.pt"
        expected=old_contract["checkpoint_hashes"][f"outer{outer}:center_context_fusion:seed{seed}"]
        if sha256(path)!=expected: raise ValueError(f"Legacy checkpoint hash mismatch: {path}")
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        model=build_supervised_model("center_context_fusion",bands,3).to(device)
        model.load_state_dict(checkpoint["model_state"]);model.eval();old.append(model)
    result[MODELS[0]]=old
    report=pd.concat([pd.read_csv(path) for path in training_reports])
    for variant,name in (("with_alley",MODELS[1]),("without_alley",MODELS[2])):
        models=[]
        for seed in SEEDS:
            row=report[(report.outer_fold==outer)&(report.variant==variant)&(report.seed==seed)]
            if len(row)!=1: raise ValueError(f"Missing unique trained checkpoint {outer} {variant} {seed}")
            path=Path(row.iloc[0].checkpoint)
            if sha256(path)!=row.iloc[0].checkpoint_sha256: raise ValueError(f"New checkpoint hash mismatch: {path}")
            checkpoint=torch.load(path,map_location="cpu",weights_only=False)
            model=build_supervised_model("center_context_fusion",bands,3).to(device)
            model.load_state_dict(checkpoint["model_state"]);model.eval();models.append(model)
        result[name]=models
    return result


def add_counts(storage, model, seed, role, cube, groups, provenance, truth, predicted):
    for level, units in (("fold",np.full(len(truth),"all",object)),("cube",np.full(len(truth),cube,object)),
                         ("spatial_group",groups),("provenance",provenance.astype(str))):
        order=np.argsort(units); sorted_units=units[order]
        starts=np.r_[0,np.flatnonzero(sorted_units[1:]!=sorted_units[:-1])+1];stops=np.r_[starts[1:],len(order)]
        for start,stop in zip(starts,stops):
            matrix=np.bincount(truth[order[start:stop]]*3+predicted[order[start:stop]],minlength=9).reshape(3,3)
            storage[(model,seed,role,level,str(sorted_units[start]))]+=matrix


def prediction_frame_for_sensitivity(project,config,records,cube_ids,outer,bands,fold_lookup,fold_contract):
    parts=[]
    for cube_id in cube_ids:
        label_dir=project/config["output_root"]/"canonical_outer_evaluation"/cube_id
        with rasterio.open(label_dir/"confident_labels.tif") as ds: labels=ds.read(1);transform=ds.transform;crs=ds.crs.to_string()
        with rasterio.open(label_dir/"provenance.tif") as ds: source=ds.read(1)
        cube=envi.open(str(records[cube_id].header),str(records[cube_id].data)).open_memmap()
        frame=frame_for_labels(cube_id,labels,source,valid_spectra(cube,bands),transform,crs,fold_lookup,
            fold_contract["block_size_m"],config["sampling"]["patch_size_pixels"],.0001,
            config["sampling"]["ground_cell_resolution_m"],config["sampling"]["seed"])
        selected=frame[frame.fold==outer].copy()
        safe=neighbour_boundary_safe_mask(
            selected.x_m.to_numpy(),selected.y_m.to_numpy(),selected.block_x.to_numpy(),selected.block_y.to_numpy(),
            fold_lookup,set(range(1,6))-{outer},fold_contract["block_size_m"],0,0,fold_contract["boundary_exclusion_m"],
        )
        parts.append(selected[safe])
    if not parts:return pd.DataFrame()
    frame=pd.concat(parts,ignore_index=True).sort_values(["_stable","cube_id","row","column"])
    conflicts=frame.groupby(["ground_x","ground_y"])["class_id"].transform("nunique")
    return frame[conflicts==1].drop_duplicates(["ground_x","ground_y"],keep="first")


def worker(args):
    outer=int(args.outer); paths=yaml.safe_load(args.paths.read_text());config=yaml.safe_load(args.config.read_text())
    project=Path(paths["project_root"]);local=project/"metadata/local";root=project/config["contract_root"]
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("Evaluation worker requires exactly one visible CUDA GPU")
    device=torch.device("cuda:0");torch.cuda.set_device(0)
    print(f"outer={outer} physical_gpu={args.physical_gpu} device={device} name={torch.cuda.get_device_name(0)}",flush=True)
    data_contract=yaml.safe_load((root/"field1_confident_supervised_data_contract.yaml").read_text())
    old_contract=yaml.safe_load((local/"contracts/field1_nested_supervised_checkpoints_contract.yaml").read_text())
    fold_contract=yaml.safe_load((local/"contracts/field1_spatial_fold_contract.yaml").read_text())
    folds=pd.read_csv(local/"contracts/field1_spatial_folds.csv")
    fold_lookup={(int(r.block_x),int(r.block_y)):int(r.fold) for r in folds.itertuples()}
    bands=load_band_indices(args.bands,config["normalization"]["band_section"])
    training_reports=sorted((project/config["report_root"]/"training").glob("outer*_gpu*.csv"))
    models=load_model_set(project,outer,len(bands),device,training_reports,old_contract)
    new_stats=pd.read_csv(root/"field1_confident_nested_normalization.csv")
    new_stats=new_stats[new_stats.outer_fold==outer].set_index("band_index").loc[bands]
    old_stats=pd.read_csv(local/"contracts/field1_nested_normalization.csv")
    old_stats=old_stats[old_stats.outer_fold==outer].set_index("band_index").loc[bands]
    stats={MODELS[0]:(old_stats["mean"].to_numpy(np.float32),old_stats["standard_deviation"].to_numpy(np.float32)),
           MODELS[1]:(new_stats["mean"].to_numpy(np.float32),new_stats["standard_deviation"].to_numpy(np.float32)),
           MODELS[2]:(new_stats["mean"].to_numpy(np.float32),new_stats["standard_deviation"].to_numpy(np.float32))}
    records={r.cube_id:r for r in load_records(args.paths,local/"authoritative_manifest.csv")}
    primary=pd.read_csv(root/f"outer{outer}_exhaustive_evaluation.csv.gz")
    populations=[("primary_dense",primary)]
    for role,cubes in (("expansion_sensitivity",config["cube_roles"]["expansion_only"]),
                       ("cubes24_28_sensitivity",config["cube_roles"]["sensitivity_only"]),
                       ("cube32_sensitivity",config["cube_roles"]["transfer_warning_dense_sensitivity_only"])):
        populations.append((role,prediction_frame_for_sensitivity(project,config,records,cubes,outer,bands,fold_lookup,fold_contract)))
    # Investigator points use the model corresponding to their frozen spatial fold.
    points=pd.read_csv(local/"contracts/field1_investigator_vegetation_reference_qc.csv")
    points=points[points.usable_reference.astype(str).str.lower().eq("true")].copy()
    bx,by=map_block_indices(points.map_x,points.map_y,fold_contract["block_size_m"])
    points["fold"]=[fold_lookup[(int(a),int(b))] for a,b in zip(bx,by)]
    points["_block_x"],points["_block_y"]=bx,by
    points=points[points.fold==outer].copy();points["row"]=points.centre_row;points["column"]=points.centre_column
    points["class_id"]=points.kind.map({"confirmed_chickpea":1,"confirmed_weed":2})
    if points.class_id.isna().any(): raise ValueError("Unexpected investigator reference kind")
    points["class_id"]=points.class_id.astype(np.int8)
    points["class_name"]=points.class_id.map(dict(enumerate(CLASS_NAMES)));points["provenance_code"]=points.class_id.map({1:8,2:9})
    points["spatial_group_id"]=[spatial_group_id("EPSG:32617",int(a),int(b)) for a,b in zip(points._block_x,points._block_y)]
    populations.append(("primary_reference_points",points))

    storage=defaultdict(lambda:np.zeros((3,3),np.int64));timing=[];batch_size=1024;radius=config["sampling"]["patch_size_pixels"]//2
    for role,frame in populations:
        if frame.empty: continue
        for cube_id,one in frame.groupby("cube_id"):
            started=time.monotonic();record=records[cube_id];cube=envi.open(str(record.header),str(record.data)).open_memmap()
            raw=np.asarray(cube[...,bands],np.float32);rows=one.row.to_numpy(np.int64,copy=True);columns=one.column.to_numpy(np.int64,copy=True)
            truth=one.class_id.to_numpy(np.int64);groups=one.spatial_group_id.astype(str).to_numpy();provenance=one.provenance_code.to_numpy(np.int64)
            predictions={(name,seed):np.empty(len(one),np.int8) for name in MODELS for seed in [*map(str,SEEDS),"ensemble"]}
            for name in MODELS:
                mean,std=stats[name];normalized=(raw-mean[None,None,:])/np.maximum(std[None,None,:],1e-6)
                tensor=torch.from_numpy(np.moveaxis(normalized,-1,0).copy()).to(device)
                row_tensor=torch.from_numpy(rows).to(device);column_tensor=torch.from_numpy(columns).to(device)
                with torch.inference_mode():
                    for start in range(0,len(one),batch_size):
                        stop=min(start+batch_size,len(one));patches=gather_patches(tensor,row_tensor[start:stop],column_tensor[start:stop],radius)
                        with torch.amp.autocast("cuda",enabled=True): probabilities=[model(patches).softmax(1) for model in models[name]]
                        for seed,probability in zip(SEEDS,probabilities): predictions[(name,str(seed))][start:stop]=probability.argmax(1).cpu().numpy()
                        predictions[(name,"ensemble")][start:stop]=torch.stack(probabilities).mean(0).argmax(1).cpu().numpy()
                del tensor,row_tensor,column_tensor,normalized;torch.cuda.empty_cache()
            for (name,seed),predicted in predictions.items(): add_counts(storage,name,seed,role,cube_id,groups,provenance,truth,predicted)
            elapsed=time.monotonic()-started;timing.append({"outer_fold":outer,"physical_gpu":args.physical_gpu,"dataset_role":role,
                "cube_id":cube_id,"evaluation_units":len(one),"elapsed_seconds":elapsed,"units_per_second":len(one)/elapsed,
                "peak_vram_gib":torch.cuda.max_memory_allocated()/2**30})
            print(f"[outer {outer} {role}] {cube_id}: {len(one):,} paired units in {elapsed:.1f}s",flush=True)
    rows=[]
    for (model,seed,role,level,unit),matrix in storage.items():
        for true in range(3):
            for predicted in range(3):rows.append({"outer_fold":outer,"model":model,"seed":seed,"dataset_role":role,
                "level":level,"unit_id":unit,"true_class_id":true,"predicted_class_id":predicted,"count":int(matrix[true,predicted])})
    report=project/config["report_root"]/"evaluation"/"folds"/f"outer{outer}";report.mkdir(parents=True,exist_ok=True)
    counts_path=report/"confusion_counts.csv.gz";pd.DataFrame(rows).to_csv(counts_path,index=False,compression={"method":"gzip","compresslevel":6,"mtime":0})
    pd.DataFrame(timing).to_csv(report/"inference_timing.csv",index=False)
    (report/"fold_contract.yaml").write_text(yaml.safe_dump({"status":"paired_confident_evaluation_fold_complete","outer_fold":outer,
        "field2_accessed":False,"counts_sha256":sha256(counts_path)},sort_keys=False))


def matrix(frame):
    result=np.zeros((3,3),float)
    for row in frame.itertuples():result[int(row.true_class_id),int(row.predicted_class_id)]+=row.count
    return result


def equal_group_matrix(frame):
    matrices=[]
    for _,group in frame.groupby(["outer_fold","unit_id"]):
        value=matrix(group);matrices.append(value/max(value.sum(),1))
    return np.sum(matrices,axis=0)


def aggregate(args):
    config=yaml.safe_load(args.config.read_text());paths=yaml.safe_load(args.paths.read_text());project=Path(paths["project_root"])
    root=project/config["report_root"]/"evaluation";frames=[];timings=[]
    for outer in config["training"]["outer_folds"]:
        folder=root/"folds"/f"outer{outer}";contract=yaml.safe_load((folder/"fold_contract.yaml").read_text());path=folder/"confusion_counts.csv.gz"
        if contract["status"]!="paired_confident_evaluation_fold_complete" or sha256(path)!=contract["counts_sha256"]:raise ValueError(f"Incomplete outer {outer}")
        frames.append(pd.read_csv(path));timings.append(pd.read_csv(folder/"inference_timing.csv"))
    counts=pd.concat(frames,ignore_index=True);root.mkdir(parents=True,exist_ok=True)
    counts_path=root/"paired_confusion_counts.csv.gz";counts.to_csv(counts_path,index=False,compression={"method":"gzip","compresslevel":6,"mtime":0})
    timing=pd.concat(timings,ignore_index=True);timing.to_csv(root/"inference_timing.csv",index=False)
    metric_rows=[]
    for keys,group in counts[counts.level.isin(["fold","cube","spatial_group","provenance"])].groupby(["model","seed","dataset_role","outer_fold","level","unit_id"]):
        model,seed,role,outer,level,unit=keys;metric_rows.append({"model":model,"seed":seed,"dataset_role":role,"outer_fold":outer,"level":level,"unit_id":unit,**metrics_from_confusion(matrix(group))})
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(root/"metrics_by_fold_cube_group_provenance_seed.csv",index=False)
    primary=[]
    selected=counts[(counts.dataset_role=="primary_dense")&(counts.level=="spatial_group")]
    for (model,seed),group in selected.groupby(["model","seed"]):primary.append({"model":model,"seed":seed,**metrics_from_confusion(equal_group_matrix(group))})
    primary=pd.DataFrame(primary);primary.to_csv(root/"primary_equal_spatial_group_metrics.csv",index=False)
    fold_metrics=metrics[(metrics.dataset_role=="primary_dense")&(metrics.level=="fold")].copy()
    fold_metrics.to_csv(root/"primary_fold_metrics.csv",index=False)
    cube_rows=[]
    cube_counts=counts[(counts.dataset_role=="primary_dense")&(counts.level=="cube")]
    for (model,seed,cube),group in cube_counts.groupby(["model","seed","unit_id"]):
        cube_rows.append({"model":model,"seed":seed,"cube_id":cube,**metrics_from_confusion(matrix(group))})
    cube_metrics=pd.DataFrame(cube_rows);cube_metrics.to_csv(root/"primary_cube_metrics.csv",index=False)
    role_rows=[]
    for (model,seed,role),group in counts[counts.level=="fold"].groupby(["model","seed","dataset_role"]):
        role_rows.append({"model":model,"seed":seed,"dataset_role":role,**metrics_from_confusion(matrix(group))})
    role_metrics=pd.DataFrame(role_rows);role_metrics.to_csv(root/"dataset_role_metrics.csv",index=False)
    seed_variation=(primary[primary.seed.astype(str)!="ensemble"].groupby("model")
                    .agg({column:["mean","std","min","max"] for column in
                          ["balanced_accuracy","macro_f1","soil_f1","chickpea_f1","weed_f1"]}))
    seed_variation.columns=["_".join(column) for column in seed_variation.columns]
    seed_variation.reset_index().to_csv(root/"primary_seed_variation.csv",index=False)
    worst_cube=(cube_metrics[cube_metrics.seed.astype(str)=="ensemble"].groupby("model").macro_f1.min()
                .rename("worst_cube_macro_f1").reset_index())
    worst_cube.to_csv(root/"primary_worst_cube_macro_f1.csv",index=False)
    # Row-normalized confusion for ensembles.
    confusion_rows=[]
    for model,group in selected[selected.seed.astype(str)=="ensemble"].groupby("model"):
        value=equal_group_matrix(group);normalized=value/np.maximum(value.sum(1,keepdims=True),1e-12)
        for true in range(3):
            for predicted in range(3):confusion_rows.append({"model":model,"true_class":CLASS_NAMES[true],"predicted_class":CLASS_NAMES[predicted],"row_normalized":normalized[true,predicted]})
    pd.DataFrame(confusion_rows).to_csv(root/"row_normalized_confusion.csv",index=False)
    # Paired group bootstrap on identical primary units.
    ensemble=selected[selected.seed.astype(str)=="ensemble"];groups=sorted(ensemble.unit_id.unique());rng=np.random.default_rng(42);boot=[]
    lookup={(model,unit):matrix(group) for (model,unit),group in ensemble.groupby(["model","unit_id"])}
    for comparison,first,second in (("cleaned_minus_legacy",MODELS[1],MODELS[0]),("with_minus_without_alley",MODELS[1],MODELS[2])):
        shared=[unit for unit in groups if (first,unit) in lookup and (second,unit) in lookup]
        for replicate in range(1000):
            sampled=rng.choice(shared,len(shared),replace=True)
            values={}
            for name in (first,second):
                normalized=[lookup[(name,unit)]/max(lookup[(name,unit)].sum(),1) for unit in sampled]
                values[name]=metrics_from_confusion(np.sum(normalized,axis=0))
            boot.append({"comparison":comparison,"replicate":replicate,**{f"delta_{key}":values[first][key]-values[second][key]
                for key in ["balanced_accuracy","macro_f1","soil_precision","soil_recall","soil_f1","chickpea_precision","chickpea_recall","chickpea_f1","weed_precision","weed_recall","weed_f1"]}})
    boot=pd.DataFrame(boot);boot.to_csv(root/"paired_spatial_group_bootstrap.csv",index=False)
    intervals=[]
    for comparison,group in boot.groupby("comparison"):
        for column in [value for value in group if value.startswith("delta_")]:intervals.append({"comparison":comparison,"metric":column.removeprefix("delta_"),
            "mean_difference":group[column].mean(),"ci95_lower":group[column].quantile(.025),"ci95_upper":group[column].quantile(.975)})
    pd.DataFrame(intervals).to_csv(root/"paired_differences_95ci.csv",index=False)
    legacy=pd.DataFrame([{"architecture":"center_spectrum","balanced_accuracy":.652364,"macro_f1":.646696,"soil_f1":.918022,"chickpea_f1":.219674,"weed_f1":.802392},
        {"architecture":"spatial_average_cnn","balanced_accuracy":.674820,"macro_f1":.635110,"soil_f1":.852076,"chickpea_f1":.278975,"weed_f1":.774281},
        {"architecture":"center_context_fusion","balanced_accuracy":.699235,"macro_f1":.671206,"soil_f1":.893603,"chickpea_f1":.300510,"weed_f1":.819504}])
    legacy["comparison_semantics"]="contextual_only_evaluation_label_contract_changed";legacy.to_csv(root/"legacy_contextual_metrics.csv",index=False)
    # Freeze all 30 task checkpoints (15 primary plus 15 controlled ablation).
    training_reports=sorted((project/config["report_root"]/"training").glob("outer*_gpu*.csv"))
    training=pd.concat([pd.read_csv(path) for path in training_reports],ignore_index=True)
    if len(training)!=30 or training.groupby(["variant","outer_fold","seed"]).size().ne(1).any():
        raise ValueError("Expected exactly 30 unique task checkpoints")
    training.to_csv(root/"training_run_summary.csv",index=False)
    checkpoint_contract={"status":"confident_supervised_checkpoints_frozen","field":"Field 1","field2_accessed":False,
        "primary_architecture":"center_context_fusion","primary_checkpoint_count":15,"ablation_checkpoint_count":15,
        "outer_folds":[1,2,3,4,5],"seeds":SEEDS,"variants":["with_alley","without_alley"],
        "estimated_two_gpu_wall_seconds":float(training.groupby("physical_gpu").total_seconds.sum().max()),
        "total_checkpoint_runtime_seconds":float(training.total_seconds.sum()),
        "maximum_peak_vram_gib":float(training.peak_vram_gib.max()),
        "checkpoint_hashes":{f"outer{int(row.outer_fold)}:{row.variant}:seed{int(row.seed)}":row.checkpoint_sha256 for row in training.itertuples()},
        "training_summary_sha256":sha256(root/"training_run_summary.csv")}
    (project/config["contract_root"]/"field1_confident_supervised_checkpoints_contract.yaml").write_text(yaml.safe_dump(checkpoint_contract,sort_keys=False))
    outputs=[counts_path,root/"inference_timing.csv",root/"metrics_by_fold_cube_group_provenance_seed.csv",root/"primary_equal_spatial_group_metrics.csv",
             root/"primary_fold_metrics.csv",root/"primary_cube_metrics.csv",root/"dataset_role_metrics.csv",
             root/"primary_seed_variation.csv",root/"primary_worst_cube_macro_f1.csv",root/"training_run_summary.csv",
             root/"row_normalized_confusion.csv",root/"paired_spatial_group_bootstrap.csv",root/"paired_differences_95ci.csv",root/"legacy_contextual_metrics.csv"]
    contract={"status":"paired_confident_supervised_evaluation_complete","field":"Field 1","field2_accessed":False,
              "same_evaluation_units_for_all_models":True,"ground_cell_deduplicated":True,"equal_spatial_group_primary":True,
              "bootstrap_replicates":1000,"outputs":{str(path.relative_to(project)):sha256(path) for path in outputs}}
    (project/config["contract_root"]/"field1_confident_paired_evaluation_contract.yaml").write_text(yaml.safe_dump(contract,sort_keys=False))
    print(primary[primary.seed.astype(str)=="ensemble"].to_string(index=False));print(pd.DataFrame(intervals).to_string(index=False))


def scheduler(args):
    config=yaml.safe_load(args.config.read_text());pending=list(config["training"]["outer_folds"]);running={};gpus=config["training"]["gpu_indices"]
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:continue
            outer=pending.pop(0);env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu)
            command=[sys.executable,str(Path(__file__).resolve()),"--paths",str(args.paths),"--config",str(args.config),"--bands",str(args.bands),"--mode","worker","--outer",str(outer),"--physical-gpu",str(gpu)]
            running[gpu]=(subprocess.Popen(command,cwd=PROJECT_ROOT,env=env),outer);print(f"scheduled evaluation outer {outer} on cuda:{gpu}",flush=True)
        time.sleep(1)
        for gpu,(process,outer) in list(running.items()):
            if process.poll() is not None:
                code=process.returncode;del running[gpu];print(f"evaluation outer {outer} exited {code}",flush=True)
                if code:raise RuntimeError(f"Evaluation outer {outer} failed")
    aggregate(args)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--paths",required=True,type=Path);parser.add_argument("--config",default=Path("configs/confident_labels_v1.yaml"),type=Path)
    parser.add_argument("--bands",default=Path("configs/spectral_bands.yaml"),type=Path);parser.add_argument("--mode",choices=["full","worker","aggregate"],required=True)
    parser.add_argument("--outer",type=int,default=1);parser.add_argument("--physical-gpu",type=int,default=0);args=parser.parse_args()
    if args.mode=="worker":worker(args)
    elif args.mode=="aggregate":aggregate(args)
    else:scheduler(args)


if __name__=="__main__":main()
