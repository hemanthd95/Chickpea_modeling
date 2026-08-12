#!/usr/bin/env python
"""Train and freeze all-eligible Field 1 deployment ensembles on independent GPUs."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import random
import subprocess
import sys
import time

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from chickpea_ssl.data import IndexedPatchDataset,load_band_indices,load_records
from chickpea_ssl.interpretability import sha256
from chickpea_ssl.model import build_supervised_model


BENCHMARK_SHA="da95fd59b7dc5d8ba4add96ed43b398bb7c93749"


def require_gpu():
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; CPU fallback is prohibited")
    if torch.cuda.device_count()!=1:raise RuntimeError(f"Each worker must see exactly one GPU; found {torch.cuda.device_count()}")
    torch.cuda.set_device(0);return torch.device("cuda:0")


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False


def loader(dataset,batch_size,shuffle,seed):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


def load_data(args,variant):
    paths=yaml.safe_load(args.paths.read_text());confident=yaml.safe_load(args.config.read_text());finalization=yaml.safe_load(args.finalization.read_text());project=Path(paths["project_root"])
    root=project/"metadata/local/contracts/supervised_finalization_v1";contract_path=root/"field1_deployment_data_contract.yaml";contract=yaml.safe_load(contract_path.read_text())
    if contract["status"]!="field1_deployment_data_frozen" or contract["benchmark_commit"]!=BENCHMARK_SHA or contract["field2_accessed"] is not False:raise ValueError("Deployment data contract invalid")
    sample_path=root/f"field1_deployment_{variant}_samples.csv"
    if sha256(sample_path)!=contract["sample_sha256"][sample_path.name]:raise ValueError("Deployment sample hash mismatch")
    normalization_path=root/"field1_deployment_normalization.csv"
    if sha256(normalization_path)!=contract["normalization_sha256"]:raise ValueError("Deployment normalization hash mismatch")
    frame=pd.read_csv(sample_path);bands=load_band_indices(args.bands,confident["normalization"]["band_section"])
    stats=pd.read_csv(normalization_path);stats=stats[stats.variant==variant].set_index("band_index").loc[bands]
    records=load_records(args.paths,project/"metadata/local/authoritative_manifest.csv")
    dataset=IndexedPatchDataset(records,frame,bands,stats["mean"].to_numpy(),stats["standard_deviation"].to_numpy())
    return project,confident,finalization,contract,frame,dataset,len(bands)


def gpu_check(args):
    device=require_gpu();project,_,finalization,_,frame,dataset,bands=load_data(args,"with_alley")
    print(f"GPU check physical cuda:{args.physical_gpu}; name={torch.cuda.get_device_name(0)}; PyTorch={torch.__version__}; runtime={torch.version.cuda}",flush=True)
    rows=[]
    for batch_size in finalization["deployment"]["batch_size_candidates"]:
        set_seed(991+int(args.physical_gpu));data_loader=loader(dataset,int(batch_size),False,0);model=build_supervised_model("center_context_fusion",bands,3).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler("cuda",enabled=True);criterion=nn.CrossEntropyLoss()
        torch.cuda.reset_peak_memory_stats();started=time.monotonic();seen=0
        for number,(patches,labels) in enumerate(data_loader,1):
            patches=patches.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=True):loss=criterion(model(patches),labels)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();torch.cuda.synchronize();seen+=len(labels)
            if number>=4:break
        engineering=project/"results/field1_deployment_v1/engineering"/f"gpu{args.physical_gpu}";engineering.mkdir(parents=True,exist_ok=True)
        checkpoint=engineering/f"batch{batch_size}_reload.pt";torch.save({"model_state":model.state_dict(),"batch_size":batch_size},checkpoint)
        loaded=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(loaded["model_state"])
        with torch.inference_mode(),torch.amp.autocast("cuda",enabled=True):model(patches[:min(len(patches),64)])
        torch.cuda.synchronize();elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()/2**30
        rows.append({"physical_gpu":int(args.physical_gpu),"device_name":torch.cuda.get_device_name(0),"pytorch_version":torch.__version__,"cuda_runtime":torch.version.cuda,
                     "batch_size":int(batch_size),"real_patch_samples":seen,"elapsed_seconds":elapsed,"samples_per_second":seen/elapsed,"peak_vram_gib":peak,
                     "forward_backward_passed":True,"checkpoint_reload_passed":True,"vram_safety_passed":peak<32*(1-float(finalization["deployment"]["vram_safety_fraction"]))})
        print(rows[-1],flush=True);del data_loader,model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
    report=project/"metadata/local/reports/supervised_finalization_v1/deployment";report.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(report/f"gpu{args.physical_gpu}_preflight.csv",index=False)


def worker(args):
    device=require_gpu();variant=args.variant;seed=int(args.seed);set_seed(seed)
    project,confident,finalization,data_contract,frame,dataset,bands=load_data(args,variant)
    preflight=yaml.safe_load((project/"metadata/local/contracts/supervised_finalization_v1/field1_deployment_gpu_preflight_contract.yaml").read_text())
    if preflight["status"]!="two_gpu_real_patch_preflight_passed" or preflight["field2_accessed"] is not False:raise ValueError("Two-GPU preflight has not passed")
    batch_size=int(preflight["selected_batch_size"]);epochs=int(data_contract["fixed_epochs"][variant]);data_loader=loader(dataset,batch_size,True,seed)
    model=build_supervised_model("center_context_fusion",bands,3).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=confident["training"]["learning_rate"],weight_decay=confident["training"]["weight_decay"])
    criterion=nn.CrossEntropyLoss();scaler=torch.amp.GradScaler("cuda",enabled=True);history=[];total_started=time.monotonic();total_seen=0;peak=0
    print(f"deployment {variant} seed={seed} physical_gpu={args.physical_gpu} name={torch.cuda.get_device_name(0)} batch={batch_size} samples={len(frame):,} epochs={epochs}",flush=True)
    for epoch in range(1,epochs+1):
        model.train();started=time.monotonic();seen=correct=0;loss_sum=0.;torch.cuda.reset_peak_memory_stats()
        for number,(patches,labels) in enumerate(data_loader,1):
            patches=patches.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=True):logits=model(patches);loss=criterion(logits,labels)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();seen+=len(labels);total_seen+=len(labels);loss_sum+=float(loss.detach())*len(labels);correct+=int((logits.argmax(1)==labels).sum())
            if number==1 or number%25==0 or number==len(data_loader):print(f"{variant} seed {seed} epoch {epoch}/{epochs} batch {number}/{len(data_loader)} loss={loss_sum/seen:.4f} throughput={seen/(time.monotonic()-started):,.0f}/s",flush=True)
        torch.cuda.synchronize();seconds=time.monotonic()-started;peak=max(peak,torch.cuda.max_memory_allocated()/2**30)
        history.append({"variant":variant,"seed":seed,"epoch":epoch,"optimization_loss":loss_sum/seen,"optimization_accuracy":correct/seen,"epoch_seconds":seconds,"samples_per_second":seen/seconds,"peak_vram_gib":peak})
        remaining=(epochs-epoch)*seconds;print(f"epoch complete; optimization_accuracy={correct/seen:.6f}; peak={peak:.3f} GiB; ETA={remaining/60:.1f} min",flush=True)
    output=project/"results/field1_deployment_v1"/variant;output.mkdir(parents=True,exist_ok=True);checkpoint=output/f"center_context_fusion_seed{seed}.pt"
    torch.save({"model_state":model.state_dict(),"architecture":"center_context_fusion","variant":variant,"seed":seed,"epochs":epochs,"benchmark_commit":BENCHMARK_SHA,"normalization_contract":"field1_deployment_data_contract.yaml","calibration":"none"},checkpoint)
    loaded=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(loaded["model_state"]);model.eval()
    inference_loader=loader(dataset,batch_size,False,0);patches,_=next(iter(inference_loader));patches=patches.to(device,non_blocking=True);start=time.monotonic()
    with torch.inference_mode(),torch.amp.autocast("cuda",enabled=True):probability=model(patches).softmax(1)
    torch.cuda.synchronize();inference_seconds=time.monotonic()-start
    if not torch.isfinite(probability).all() or not torch.allclose(probability.sum(1),torch.ones(len(probability),device=device),atol=1e-4):raise ValueError("Reloaded checkpoint inference invalid")
    report=project/"metadata/local/reports/supervised_finalization_v1/deployment";history_path=report/f"{variant}_seed{seed}_history.csv";pd.DataFrame(history).to_csv(history_path,index=False)
    result={"variant":variant,"seed":seed,"physical_gpu":int(args.physical_gpu),"device_name":torch.cuda.get_device_name(0),"epochs":epochs,"fitting_samples":len(frame),"class_0_samples":int((frame.class_id==0).sum()),"class_1_samples":int((frame.class_id==1).sum()),"class_2_samples":int((frame.class_id==2).sum()),"total_seconds":time.monotonic()-total_started,"training_samples_per_second":total_seen/(time.monotonic()-total_started),"peak_vram_gib":peak,"checkpoint_path":str(checkpoint.relative_to(project)),"checkpoint_sha256":sha256(checkpoint),"checkpoint_reload_passed":True,"real_patch_inference_samples_per_second":len(patches)/inference_seconds,"history_sha256":sha256(history_path)}
    pd.DataFrame([result]).to_csv(report/f"{variant}_seed{seed}_result.csv",index=False);print(result,flush=True)


def run_jobs(args,jobs,mode):
    pending=list(jobs);running={}
    while pending or running:
        for gpu in (0,1):
            if gpu in running or not pending:continue
            job=pending.pop(0);env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu)
            command=[sys.executable,str(Path(__file__).resolve()),"--paths",str(args.paths),"--config",str(args.config),"--finalization",str(args.finalization),"--bands",str(args.bands),"--mode",mode,"--physical-gpu",str(gpu)]
            if mode=="worker":command.extend(["--variant",job[0],"--seed",str(job[1])])
            running[gpu]=(subprocess.Popen(command,cwd=PROJECT_ROOT,env=env),job);print(f"scheduled {mode} {job} on cuda:{gpu}",flush=True)
        time.sleep(1)
        for gpu,(process,job) in list(running.items()):
            if process.poll() is not None:
                code=process.returncode;del running[gpu];print(f"{mode} {job} on cuda:{gpu} exited {code}",flush=True)
                if code:raise RuntimeError(f"GPU job failed: {job}")


def preflight(args):
    run_jobs(args,[(0,),(1,)],"gpu-check")
    project=Path(yaml.safe_load(args.paths.read_text())["project_root"]);report=project/"metadata/local/reports/supervised_finalization_v1/deployment"
    frames=[pd.read_csv(report/f"gpu{gpu}_preflight.csv") for gpu in (0,1)];combined=pd.concat(frames,ignore_index=True);combined.to_csv(report/"two_gpu_preflight.csv",index=False)
    passing=combined[combined.forward_backward_passed&combined.checkpoint_reload_passed&combined.vram_safety_passed]
    common=sorted(set(passing[passing.physical_gpu==0].batch_size)&set(passing[passing.physical_gpu==1].batch_size))
    if not common:raise RuntimeError("No batch size passed both GPU preflights")
    selected=int(max(common));driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()[0]
    contract={"status":"two_gpu_real_patch_preflight_passed","benchmark_commit":BENCHMARK_SHA,"field":"Field 1","field2_accessed":False,"gpu_count":2,"device_names":[str(value) for value in sorted(combined.device_name.unique().tolist())],"selected_batch_size":selected,"selection_rule":"largest_candidate_passing_both_gpus_with_20_percent_vram_margin","pytorch_version":str(torch.__version__),"cuda_runtime":str(torch.version.cuda),"driver_version":str(driver),"maximum_peak_vram_gib":float(combined.peak_vram_gib.max()),"report_sha256":sha256(report/"two_gpu_preflight.csv")}
    root=project/"metadata/local/contracts/supervised_finalization_v1";(root/"field1_deployment_gpu_preflight_contract.yaml").write_text(yaml.safe_dump(contract,sort_keys=False));print(contract)


def aggregate(args):
    project=Path(yaml.safe_load(args.paths.read_text())["project_root"]);root=project/"metadata/local/contracts/supervised_finalization_v1";report=project/"metadata/local/reports/supervised_finalization_v1/deployment"
    rows=[]
    for variant in ("with_alley","without_alley"):
        for seed in (42,43,44):rows.append(pd.read_csv(report/f"{variant}_seed{seed}_result.csv").iloc[0].to_dict())
    frame=pd.DataFrame(rows);manifest_path=root/"field1_deployment_checkpoint_manifest.csv";frame.to_csv(manifest_path,index=False)
    if len(frame)!=6 or frame.groupby(["variant","seed"]).size().ne(1).any() or not frame.checkpoint_reload_passed.all():raise ValueError("Deployment checkpoint set incomplete")
    for row in frame.itertuples():
        if sha256(project/row.checkpoint_path)!=row.checkpoint_sha256:raise ValueError(f"Deployment checkpoint hash mismatch: {row.checkpoint_path}")
    data_path=root/"field1_deployment_data_contract.yaml";preflight_path=root/"field1_deployment_gpu_preflight_contract.yaml"
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for variant in ("with_alley","without_alley"):
        for seed in (42,43,44):
            history=pd.read_csv(report/f"{variant}_seed{seed}_history.csv")
            axes[0].plot(history.epoch,history.optimization_loss,label=f"{variant}, seed {seed}")
            axes[1].plot(history.epoch,history.optimization_accuracy,label=f"{variant}, seed {seed}")
    axes[0].set(xlabel="Fixed deployment epoch",ylabel="Optimization loss",title="All-Field-1 fitting loss")
    axes[1].set(xlabel="Fixed deployment epoch",ylabel="Optimization accuracy",title="All-Field-1 fitting accuracy")
    axes[1].legend(fontsize=7,ncol=2);curve_path=report/"deployment_training_curves.png";fig.savefig(curve_path,dpi=220);plt.close(fig)
    contract={"status":"field1_deployment_ensembles_frozen_before_field2_compatibility","benchmark_commit":BENCHMARK_SHA,"field":"Field 1","field2_accessed":False,"field2_predictions_generated":False,"architecture":"center_context_fusion","variants":["with_alley","without_alley"],"seeds":[42,43,44],"deterministic_algorithms":True,"cudnn_benchmark":False,"checkpoint_count":6,"ensemble_rule":"arithmetic_mean_of_three_seed_softmax_probabilities","calibration":"none_frozen_before_external_validation","legacy_deployment_comparator":"not_created_no_scientifically_equivalent_frozen_all_field1_reconstruction","cubes24_28_used_as_blind_data":False,"checkpoint_manifest":str(manifest_path.relative_to(project)),"checkpoint_manifest_sha256":sha256(manifest_path),"training_curves":str(curve_path.relative_to(project)),"training_curves_sha256":sha256(curve_path),"deployment_data_contract_sha256":sha256(data_path),"gpu_preflight_contract_sha256":sha256(preflight_path),"maximum_peak_vram_gib":float(frame.peak_vram_gib.max()),"total_training_seconds":float(frame.total_seconds.sum()),"mean_inference_samples_per_second":float(frame.real_patch_inference_samples_per_second.mean())}
    (root/"field1_deployment_ensemble_contract.yaml").write_text(yaml.safe_dump(contract,sort_keys=False));print(frame.to_string(index=False));print(contract)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--paths",required=True,type=Path);parser.add_argument("--config",default=Path("configs/confident_labels_v1.yaml"),type=Path);parser.add_argument("--finalization",default=Path("configs/supervised_finalization_v1.yaml"),type=Path);parser.add_argument("--bands",default=Path("configs/spectral_bands.yaml"),type=Path);parser.add_argument("--mode",choices=["preflight","gpu-check","full","worker","aggregate"],required=True);parser.add_argument("--physical-gpu",type=int,default=0);parser.add_argument("--variant",default="with_alley");parser.add_argument("--seed",type=int,default=42);args=parser.parse_args()
    if args.mode=="gpu-check":gpu_check(args)
    elif args.mode=="preflight":preflight(args)
    elif args.mode=="worker":worker(args)
    elif args.mode=="aggregate":aggregate(args)
    else:run_jobs(args,[(variant,seed) for variant in ("with_alley","without_alley") for seed in (42,43,44)],"worker");aggregate(args)


if __name__=="__main__":main()
