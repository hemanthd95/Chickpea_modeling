#!/usr/bin/env python
"""Grouped spectral occlusion and center/context reliance for frozen Field 1 models."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from spectral.io import envi
import torch
from torch.utils.data import DataLoader
import yaml

from chickpea_ssl.data import IndexedPatchDataset, load_band_indices, load_records
from chickpea_ssl.interpretability import (
    CLASS_NAMES,
    benjamini_hochberg,
    deterministic_audit_support,
    prediction_metrics,
    sha256,
    wavelength_groups,
)
from chickpea_ssl.model import build_supervised_model


BENCHMARK_SHA = "da95fd59b7dc5d8ba4add96ed43b398bb7c93749"
METRICS = [
    "balanced_accuracy", "macro_f1", "soil_f1", "chickpea_f1", "weed_f1",
    "chickpea_recall", "weed_recall", "negative_log_likelihood", "brier_score",
    "expected_calibration_error",
]
LOWER_IS_BETTER = {"negative_log_likelihood", "brier_score", "expected_calibration_error"}


def routed_forward(model, center_input, context_input):
    row, column = center_input.shape[-2] // 2, center_input.shape[-1] // 2
    center = model.center(center_input[:, :, row, column])
    context = model.context(context_input)
    return model.classifier(torch.cat((center, context), dim=1))


def interpolation_replacement(patches, channels, wavelengths, mean, std):
    replacement = patches.clone()
    first, last = int(channels[0]), int(channels[-1])
    left = first - 1 if first > 0 else None
    right = last + 1 if last + 1 < patches.shape[1] else None
    for channel in channels:
        if left is None:
            raw = patches[:, right] * std[right] + mean[right]
        elif right is None:
            raw = patches[:, left] * std[left] + mean[left]
        else:
            fraction = float((wavelengths[channel] - wavelengths[left]) / (wavelengths[right] - wavelengths[left]))
            raw_left = patches[:, left] * std[left] + mean[left]
            raw_right = patches[:, right] * std[right] + mean[right]
            raw = raw_left + fraction * (raw_right - raw_left)
        replacement[:, channel] = (raw - mean[channel]) / max(float(std[channel]), 1e-6)
    return replacement


def condition_probabilities(models, patches, condition, wavelengths, mean, std, permutation):
    center_input = patches
    context_input = patches
    if condition["audit_type"] == "occlusion":
        channels = np.arange(condition["first_input_channel"], condition["last_input_channel"] + 1)
        if condition["occlusion_method"] == "fold_training_mean":
            modified = patches.clone()
            modified[:, channels] = 0
        else:
            modified = interpolation_replacement(patches, channels, wavelengths, mean, std)
        if condition["route"] in ("both_branches", "center_branch_only"):
            center_input = modified
        if condition["route"] in ("both_branches", "context_branch_only"):
            context_input = modified
    elif condition["audit_type"] == "branch":
        method = condition["branch_ablation"]
        if method in ("center_neutralized", "both_neutralized"):
            center_input = patches.clone()
            center_input[:, :, center_input.shape[-2] // 2, center_input.shape[-1] // 2] = 0
        if method in ("context_neutralized", "both_neutralized"):
            context_input = torch.zeros_like(patches)
        elif method == "context_spatial_shuffle":
            flat = patches.flatten(2)
            context_input = flat[:, :, permutation].reshape_as(patches)
        elif method == "context_spatial_mean":
            context_input = patches.mean(dim=(-2, -1), keepdim=True).expand_as(patches)
    probabilities = []
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=patches.is_cuda):
        for model in models:
            probabilities.append(routed_forward(model, center_input, context_input).softmax(1).float().cpu().numpy())
    return probabilities


def load_worker_inputs(args, outer, device):
    paths = yaml.safe_load(args.paths.read_text())
    confident = yaml.safe_load(args.config.read_text())
    finalization = yaml.safe_load(args.finalization.read_text())
    if finalization["benchmark_commit"] != BENCHMARK_SHA or not finalization["field2_locked"]:
        raise ValueError("Interpretability requires frozen da95fd5 and Field 2 lock")
    project = Path(paths["project_root"])
    source_root = project / confident["contract_root"]
    freeze_root = project / "metadata/local/contracts/supervised_finalization_v1"
    benchmark = yaml.safe_load((freeze_root / "field1_confident_supervised_benchmark_contract.yaml").read_text())
    if benchmark["benchmark_commit"] != BENCHMARK_SHA or benchmark["field2_accessed"] is not False:
        raise ValueError("Benchmark freeze contract invalid")
    bands = load_band_indices(args.bands, confident["normalization"]["band_section"])
    records = load_records(args.paths, project / "metadata/local/authoritative_manifest.csv")
    record_map = {record.cube_id: record for record in records}
    sample_source = pd.read_csv(source_root / f"outer{outer}_exhaustive_evaluation.csv.gz")
    audit = finalization["interpretability"]
    support = deterministic_audit_support(
        sample_source, int(audit["maximum_samples_per_class_spatial_group"]),
        int(audit["maximum_samples_per_outer_fold"]), int(audit["sample_seed"]) + outer,
    )
    stats = pd.read_csv(source_root / "field1_confident_nested_normalization.csv")
    stats = stats[stats.outer_fold == outer].set_index("band_index").loc[bands]
    mean = stats["mean"].to_numpy(np.float32)
    std = stats["standard_deviation"].to_numpy(np.float32)
    manifest = pd.read_csv(freeze_root / "field1_confident_supervised_checkpoint_manifest.csv")
    manifest = manifest[manifest.outer_fold == outer].sort_values("seed")
    models = []
    for row in manifest.itertuples():
        checkpoint_path = project / row.checkpoint_path
        if sha256(checkpoint_path) != row.checkpoint_sha256:
            raise ValueError(f"Frozen checkpoint mismatch: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_supervised_model("center_context_fusion", len(bands), 3).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    first_record = record_map[str(support.iloc[0].cube_id)]
    wavelengths = np.asarray(envi.open(str(first_record.header), str(first_record.data)).metadata["wavelength"], float)[bands]
    groups = wavelength_groups(wavelengths, float(audit["wavelength_group_target_width_nm"]))
    dataset = IndexedPatchDataset(records, support, bands, mean, std)
    loader = DataLoader(dataset, batch_size=int(audit["batch_size"]), shuffle=False, num_workers=8,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)
    patches, truth = [], []
    for patch, label in loader:
        patches.append(patch)
        truth.append(label.numpy())
    return project, confident, finalization, support, torch.cat(patches), np.concatenate(truth), mean, std, wavelengths, groups, models


def group_metric_rows(outer, seeds, support, truth, baseline, probabilities, condition):
    result = []
    for seed_index, seed in enumerate(seeds):
        for group, indices in support.groupby("spatial_group_id").indices.items():
            base = prediction_metrics(truth[indices], baseline[seed_index][indices], np.full(len(indices), group))
            changed = prediction_metrics(truth[indices], probabilities[seed_index][indices], np.full(len(indices), group))
            for metric in METRICS:
                decrease = changed[metric] - base[metric] if metric in LOWER_IS_BETTER else base[metric] - changed[metric]
                result.append({
                    "outer_fold": outer, "seed": seed, "spatial_group_id": group,
                    **condition, "metric": metric, "baseline_value": base[metric],
                    "ablated_value": changed[metric], "performance_decrease": decrease,
                })
    return result


def worker(args):
    outer = int(args.outer)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Importance worker requires exactly one visible CUDA GPU; CPU fallback is prohibited")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    print(f"outer={outer} physical_gpu={args.physical_gpu} name={torch.cuda.get_device_name(0)}", flush=True)
    project, confident, finalization, support, cpu_patches, truth, mean_np, std_np, wavelengths, groups, models = load_worker_inputs(args, outer, device)
    audit = finalization["interpretability"]
    seeds = [42, 43, 44]
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    permutation = torch.randperm(cpu_patches.shape[-2] * cpu_patches.shape[-1], generator=torch.Generator().manual_seed(42000 + outer)).to(device)
    batch_size = int(audit["batch_size"])
    conditions = []
    for group in groups.to_dict("records"):
        for method in audit["occlusion_methods"]:
            for route in audit["routes"]:
                conditions.append({"audit_type": "occlusion", "occlusion_method": method, "route": route,
                                   "branch_ablation": "none", **group})
    for method in audit["branch_ablations"]:
        conditions.append({"audit_type": "branch", "occlusion_method": "none", "route": "frozen_model_reliance",
                           "branch_ablation": method, "wavelength_group_id": "all_bands",
                           "first_input_channel": 0, "last_input_channel": len(wavelengths) - 1,
                           "band_count": len(wavelengths), "wavelength_min_nm": wavelengths[0],
                           "wavelength_max_nm": wavelengths[-1], "wavelength_center_nm": wavelengths.mean()})

    baseline = [np.empty((len(truth), 3), np.float32) for _ in seeds]
    started = time.monotonic()
    for start in range(0, len(truth), batch_size):
        stop = min(start + batch_size, len(truth))
        batch = cpu_patches[start:stop].to(device, non_blocking=True)
        values = condition_probabilities(models, batch, {"audit_type": "baseline"}, wavelengths, mean, std, permutation)
        for index, value in enumerate(values): baseline[index][start:stop] = value
    metric_rows, paired_rows = [], []
    for number, condition in enumerate(conditions, 1):
        probabilities = [np.empty((len(truth), 3), np.float32) for _ in seeds]
        for start in range(0, len(truth), batch_size):
            stop = min(start + batch_size, len(truth))
            batch = cpu_patches[start:stop].to(device, non_blocking=True)
            values = condition_probabilities(models, batch, condition, wavelengths, mean, std, permutation)
            for index, value in enumerate(values): probabilities[index][start:stop] = value
        for index, seed in enumerate(seeds):
            base_metrics = prediction_metrics(truth, baseline[index], support.spatial_group_id, int(audit["calibration_bins"]))
            changed_metrics = prediction_metrics(truth, probabilities[index], support.spatial_group_id, int(audit["calibration_bins"]))
            row = {"outer_fold": outer, "seed": seed, **condition}
            for metric in METRICS:
                row[f"baseline_{metric}"] = base_metrics[metric]
                row[f"ablated_{metric}"] = changed_metrics[metric]
                row[f"decrease_{metric}"] = (changed_metrics[metric] - base_metrics[metric]
                                                  if metric in LOWER_IS_BETTER else base_metrics[metric] - changed_metrics[metric])
            metric_rows.append(row)
        paired_rows.extend(group_metric_rows(outer, seeds, support, truth, baseline, probabilities, condition))
        elapsed = time.monotonic() - started
        print(f"outer {outer}: condition {number}/{len(conditions)} {condition['audit_type']} "
              f"{condition['wavelength_group_id']} {condition['route']} {condition['occlusion_method']} "
              f"throughput={(number * len(truth) * len(seeds))/elapsed:,.0f} model-samples/s "
              f"peak_vram={torch.cuda.max_memory_allocated()/2**30:.3f} GiB", flush=True)

    ensemble = np.mean(np.stack(baseline), axis=0)
    centers = cpu_patches[:, :, cpu_patches.shape[-2] // 2, cpu_patches.shape[-1] // 2].numpy()
    raw_centers = centers * std_np[None, :] + mean_np[None, :]
    nearest = {target: int(np.argmin(abs(wavelengths - target))) for target in (550., 670., 720., 800.)}
    r550, r670, r720, r800 = (raw_centers[:, nearest[value]] for value in (550., 670., 720., 800.))
    epsilon = 1e-12
    index_values = {
        "NDVI": (r800-r670)/(r800+r670+epsilon),
        "GNDVI": (r800-r550)/(r800+r550+epsilon),
        "NDRE": (r800-r720)/(r800+r720+epsilon),
        "NIR_RED_RATIO": r800/(r670+epsilon),
    }
    observations = support[["cube_id", "row", "column", "spatial_group_id", "class_id"]].copy()
    observations["outer_fold"] = outer
    observations["prediction"] = ensemble.argmax(1)
    observations["prediction_confidence"] = ensemble.max(1)
    observations["prediction_correct"] = observations.prediction.to_numpy() == truth
    for name, values in index_values.items(): observations[name] = values
    report = project / "metadata/local/reports/supervised_finalization_v1/interpretability/folds" / f"outer{outer}"
    report.mkdir(parents=True, exist_ok=True)
    support_path = report / "audit_support.csv.gz"
    metrics_path = report / "condition_metrics.csv.gz"
    paired_path = report / "spatial_group_paired_differences.csv.gz"
    index_path = report / "index_observations.csv.gz"
    support.to_csv(support_path, index=False, compression={"method":"gzip","compresslevel":6,"mtime":0})
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, compression={"method":"gzip","compresslevel":6,"mtime":0})
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False, compression={"method":"gzip","compresslevel":6,"mtime":0})
    observations.to_csv(index_path, index=False, compression={"method":"gzip","compresslevel":6,"mtime":0})
    contract = {"status":"field1_importance_fold_complete", "benchmark_commit":BENCHMARK_SHA,
                "field2_accessed":False, "outer_fold":outer, "physical_gpu":int(args.physical_gpu),
                "device_name":torch.cuda.get_device_name(0), "support_count":len(support),
                "spatial_group_count":int(support.spatial_group_id.nunique()),
                "elapsed_seconds":time.monotonic()-started,
                "peak_vram_gib":torch.cuda.max_memory_allocated()/2**30,
                "output_sha256":{path.name:sha256(path) for path in (support_path,metrics_path,paired_path,index_path)}}
    (report / "fold_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
    del cpu_patches, models
    gc.collect(); torch.cuda.empty_cache()


def bootstrap_interval(group, replicates, seed):
    averaged = group.groupby(["outer_fold", "spatial_group_id"], as_index=False).performance_decrease.mean()
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    folds = sorted(averaged.outer_fold.unique())
    by_fold = {fold: averaged[averaged.outer_fold == fold].performance_decrease.to_numpy() for fold in folds}
    for replicate in range(replicates):
        values[replicate] = np.mean([rng.choice(v, len(v), replace=True).mean() for v in by_fold.values()])
    lower, upper = np.quantile(values, [.025, .975])
    p = min(1., 2 * min((values <= 0).mean(), (values >= 0).mean()))
    return float(values.mean()), float(lower), float(upper), float(p)


def index_summary(observations, wavelengths):
    formulas = yaml.safe_load((PROJECT_ROOT / "configs/supervised_finalization_v1.yaml").read_text())["spectral_indices"]["definitions"]
    nearest = {target: float(wavelengths[np.argmin(abs(wavelengths-target))]) for target in (550.,670.,720.,800.)}
    rows = []
    for name, formula in formulas.items():
        if name == "SAVI_L0_5" or name not in observations:
            for class_id, class_name in enumerate(CLASS_NAMES):
                rows.append({"index":name,"formula":formula,"class_id":class_id,"class_name":class_name,
                             "support":0,"audit_status":"not_computed_missing_unit_reflectance_scale_provenance",
                             "chosen_550_nm":nearest[550.],"chosen_670_nm":nearest[670.],"chosen_720_nm":nearest[720.],"chosen_800_nm":nearest[800.],
                             "neural_network_input_channel":False,"threshold_optimized":False})
            continue
        values = observations[name].replace([np.inf,-np.inf],np.nan)
        finite = values.notna()
        mutual = mutual_info_classif(values[finite].to_numpy()[:,None], observations.loc[finite,"class_id"], random_state=42)[0]
        confidence_correlation = values[finite].corr(observations.loc[finite,"prediction_confidence"], method="spearman")
        for class_id, class_name in enumerate(CLASS_NAMES):
            selected = finite & (observations.class_id == class_id)
            target = (observations.loc[finite,"class_id"] == class_id).astype(int)
            auc = roc_auc_score(target, values[finite])
            auc = max(auc, 1-auc)
            correct = finite & observations.prediction_correct.astype(bool)
            error = finite & ~observations.prediction_correct.astype(bool)
            rows.append({"index":name,"formula":formula,"class_id":class_id,"class_name":class_name,
                         "support":int(selected.sum()),"audit_status":"computed_dimensionless_ratio",
                         "mean":values[selected].mean(),"standard_deviation":values[selected].std(),
                         "q05":values[selected].quantile(.05),"median":values[selected].median(),"q95":values[selected].quantile(.95),
                         "one_vs_rest_auc_direction_free":auc,"mutual_information_all_classes":mutual,
                         "spearman_with_prediction_confidence":confidence_correlation,
                         "mean_correct_predictions":values[correct].mean(),"mean_error_predictions":values[error].mean(),
                         "chosen_550_nm":nearest[550.],"chosen_670_nm":nearest[670.],"chosen_720_nm":nearest[720.],"chosen_800_nm":nearest[800.],
                         "neural_network_input_channel":False,"threshold_optimized":False})
    return pd.DataFrame(rows)


def aggregate(args):
    paths = yaml.safe_load(args.paths.read_text()); project = Path(paths["project_root"])
    finalization = yaml.safe_load(args.finalization.read_text()); audit = finalization["interpretability"]
    root = project / "metadata/local/reports/supervised_finalization_v1/interpretability"
    metrics, paired, observations = [], [], []
    for outer in range(1,6):
        folder = root / "folds" / f"outer{outer}"
        contract = yaml.safe_load((folder/"fold_contract.yaml").read_text())
        if contract["status"] != "field1_importance_fold_complete" or contract["field2_accessed"] is not False:
            raise ValueError(f"Incomplete importance fold {outer}")
        for name, collection in (("condition_metrics.csv.gz",metrics),("spatial_group_paired_differences.csv.gz",paired),("index_observations.csv.gz",observations)):
            path=folder/name
            if sha256(path)!=contract["output_sha256"][name]: raise ValueError(f"Importance output hash mismatch: {path}")
            collection.append(pd.read_csv(path))
    metrics=pd.concat(metrics,ignore_index=True); paired=pd.concat(paired,ignore_index=True); observations=pd.concat(observations,ignore_index=True)
    occlusion=metrics[metrics.audit_type=="occlusion"].copy();branch=metrics[metrics.audit_type=="branch"].copy()
    occlusion.to_csv(root/"per_fold_per_seed_occlusion_metrics.csv",index=False)
    paired.to_csv(root/"spatial_group_paired_differences.csv",index=False)
    summary=[];replicates=int(audit["bootstrap_replicates"])
    condition_columns=["occlusion_method","route","wavelength_group_id","first_input_channel","last_input_channel","band_count","wavelength_min_nm","wavelength_max_nm","wavelength_center_nm"]
    for keys, one in paired[paired.audit_type=="occlusion"].groupby(condition_columns+["metric"],dropna=False):
        fields=dict(zip(condition_columns+["metric"],keys))
        mean,low,high,p=bootstrap_interval(one,replicates,int(audit["bootstrap_seed"])+len(summary))
        fold_values=one.groupby(["outer_fold","seed"]).performance_decrease.mean().groupby("outer_fold").mean()
        summary.append({**fields,"mean_performance_decrease":mean,"ci95_lower":low,"ci95_upper":high,
                        "bootstrap_p_value":p,"positive_outer_folds":int((fold_values>0).sum())})
    summary=pd.DataFrame(summary)
    summary["bh_adjusted_p_value"]=np.nan
    for _,indices in summary.groupby(["occlusion_method","route","metric"]).groups.items():
        summary.loc[indices,"bh_adjusted_p_value"]=benjamini_hochberg(summary.loc[indices,"bootstrap_p_value"].to_numpy())
    summary["corrected_significant_positive"]=(summary.ci95_lower>0)&(summary.bh_adjusted_p_value<=float(audit["false_discovery_rate"]))
    summary["stable_region"]=(summary.mean_performance_decrease>0)&summary.corrected_significant_positive&(summary.positive_outer_folds>=int(audit["stable_minimum_positive_folds"]))
    summary.to_csv(root/"aggregate_wavelength_importance.csv",index=False)
    branch_rows=[]
    for (method,metric),one in paired[paired.audit_type=="branch"].groupby(["branch_ablation","metric"]):
        mean,low,high,p=bootstrap_interval(one,replicates,int(audit["bootstrap_seed"])+10000+len(branch_rows))
        branch_rows.append({"summary_level":"aggregate","branch_ablation":method,"metric":metric,
                            "mean_performance_decrease":mean,"ci95_lower":low,"ci95_upper":high,"bootstrap_p_value":p})
    branch_long=pd.DataFrame(branch_rows)
    branch.to_csv(root/"branch_ablation_per_fold_seed.csv",index=False)
    branch_long.to_csv(root/"branch_ablation.csv",index=False)
    first_record=load_records(args.paths,project/"metadata/local/authoritative_manifest.csv")[0]
    bands=load_band_indices(args.bands,"primary")
    wavelengths=np.asarray(envi.open(str(first_record.header),str(first_record.data)).metadata["wavelength"],float)[bands]
    groups=wavelength_groups(wavelengths,float(audit["wavelength_group_target_width_nm"]));groups.to_csv(root/"wavelength_group_definitions.csv",index=False)
    indices=index_summary(observations,wavelengths);indices.to_csv(root/"index_separability.csv",index=False)

    primary=summary[(summary.occlusion_method=="fold_training_mean")&(summary.route=="both_branches")]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for metric,label in (("soil_f1","Soil F1"),("chickpea_f1","Chickpea F1"),("weed_f1","Weed F1")):
        one=primary[primary.metric==metric].sort_values("wavelength_center_nm")
        axes[0,0].plot(one.wavelength_center_nm,one.mean_performance_decrease,marker="o",label=label)
    axes[0,0].axhline(0,color="black",lw=.8);axes[0,0].set(title="Class-specific grouped occlusion",xlabel="Wavelength (nm)",ylabel="F1 decrease");axes[0,0].legend()
    macro=summary[(summary.occlusion_method=="fold_training_mean")&(summary.metric=="macro_f1")]
    for route,label in (("center_branch_only","Center route"),("context_branch_only","Context route"),("both_branches","Both routes")):
        one=macro[macro.route==route].sort_values("wavelength_center_nm");axes[0,1].plot(one.wavelength_center_nm,one.mean_performance_decrease,marker=".",label=label)
    axes[0,1].axhline(0,color="black",lw=.8);axes[0,1].set(title="Center versus context reliance",xlabel="Wavelength (nm)",ylabel="Macro-F1 decrease");axes[0,1].legend()
    stable=primary[primary.metric=="macro_f1"].sort_values("wavelength_center_nm")
    axes[1,0].bar(stable.wavelength_center_nm,stable.positive_outer_folds,width=10,color=np.where(stable.stable_region,"#1976d2","#bdbdbd"));axes[1,0].axhline(4,color="black",ls="--");axes[1,0].set(title="Outer-fold directional stability",xlabel="Wavelength (nm)",ylabel="Positive folds (of 5)")
    branch_macro=branch_long[branch_long.metric=="macro_f1"].sort_values("mean_performance_decrease")
    axes[1,1].barh(branch_macro.branch_ablation,branch_macro.mean_performance_decrease,xerr=[branch_macro.mean_performance_decrease-branch_macro.ci95_lower,branch_macro.ci95_upper-branch_macro.mean_performance_decrease]);axes[1,1].axvline(0,color="black",lw=.8);axes[1,1].set(title="Frozen-model branch reliance",xlabel="Macro-F1 decrease")
    fig.suptitle("Field 1 frozen center-context model interpretability audit")
    overview=root/"publication_importance_overview.png";fig.savefig(overview,dpi=240);plt.close(fig)
    for source,name in ((primary,"class_specific_importance.png"),(macro,"center_vs_context_importance.png"),(stable,"fold_stability.png")):
        fig,ax=plt.subplots(figsize=(10,5));
        for key,one in source.groupby("metric" if name=="class_specific_importance.png" else ("route" if name=="center_vs_context_importance.png" else "stable_region")):
            ax.plot(one.sort_values("wavelength_center_nm").wavelength_center_nm,one.sort_values("wavelength_center_nm").mean_performance_decrease,marker=".",label=str(key))
        ax.axhline(0,color="black",lw=.8);ax.set(xlabel="Wavelength (nm)",ylabel="Performance decrease");ax.legend();fig.tight_layout();fig.savefig(root/name,dpi=220);plt.close(fig)
    outputs=[root/"wavelength_group_definitions.csv",root/"per_fold_per_seed_occlusion_metrics.csv",root/"spatial_group_paired_differences.csv",root/"aggregate_wavelength_importance.csv",root/"branch_ablation_per_fold_seed.csv",root/"branch_ablation.csv",root/"index_separability.csv",overview,root/"class_specific_importance.png",root/"center_vs_context_importance.png",root/"fold_stability.png"]
    contract={"status":"field1_grouped_importance_audit_complete","benchmark_commit":BENCHMARK_SHA,"field":"Field 1","field2_accessed":False,"models_retrained":False,"bands_selected_from_audit":False,"support_count":len(observations),"spatial_groups":int(observations.spatial_group_id.nunique()),"outer_folds":[1,2,3,4,5],"seeds":[42,43,44],"occlusion_methods":audit["occlusion_methods"],"routes":audit["routes"],"bootstrap_replicates":replicates,"multiple_comparison":"benjamini_hochberg","correlated_band_warning":"Adjacent hyperspectral bands are correlated; occlusion or attribution magnitude does not uniquely identify causal wavelengths.","output_sha256":{str(path.relative_to(project)):sha256(path) for path in outputs}}
    contract_path=project/"metadata/local/contracts/supervised_finalization_v1/field1_grouped_importance_contract.yaml";contract_path.write_text(yaml.safe_dump(contract,sort_keys=False))
    print(summary[(summary.metric=="macro_f1")&(summary.occlusion_method=="fold_training_mean")].sort_values("mean_performance_decrease",ascending=False).head(15).to_string(index=False))
    print(branch_long[branch_long.metric=="macro_f1"].to_string(index=False))


def scheduler(args):
    pending=[1,2,3,4,5];running={}
    while pending or running:
        for gpu in (0,1):
            if gpu in running or not pending:continue
            outer=pending.pop(0);env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu)
            command=[sys.executable,str(Path(__file__).resolve()),"--paths",str(args.paths),"--config",str(args.config),"--finalization",str(args.finalization),"--bands",str(args.bands),"--mode","worker","--outer",str(outer),"--physical-gpu",str(gpu)]
            running[gpu]=(subprocess.Popen(command,cwd=PROJECT_ROOT,env=env),outer);print(f"scheduled importance outer {outer} on cuda:{gpu}",flush=True)
        time.sleep(1)
        for gpu,(process,outer) in list(running.items()):
            if process.poll() is not None:
                code=process.returncode;del running[gpu];print(f"importance outer {outer} exited {code}",flush=True)
                if code:raise RuntimeError(f"Importance worker failed: outer {outer}")
    aggregate(args)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--paths",required=True,type=Path);parser.add_argument("--config",default=Path("configs/confident_labels_v1.yaml"),type=Path);parser.add_argument("--finalization",default=Path("configs/supervised_finalization_v1.yaml"),type=Path);parser.add_argument("--bands",default=Path("configs/spectral_bands.yaml"),type=Path);parser.add_argument("--mode",choices=["full","worker","aggregate"],required=True);parser.add_argument("--outer",type=int,default=1);parser.add_argument("--physical-gpu",type=int,default=0);args=parser.parse_args()
    if args.mode=="worker":worker(args)
    elif args.mode=="aggregate":aggregate(args)
    else:scheduler(args)


if __name__=="__main__":main()
