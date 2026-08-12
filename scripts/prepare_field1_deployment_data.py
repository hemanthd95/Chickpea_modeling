#!/usr/bin/env python
"""Freeze all-eligible-primary-Field-1 samples and normalization for deployment fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_band_indices, load_records
from chickpea_ssl.interpretability import sha256
from scripts.prepare_confident_supervised_data import (
    deduplicate_and_thin,
    frame_for_labels,
    no_alley_variant,
    valid_spectra,
    write_samples,
)


BENCHMARK_SHA = "da95fd59b7dc5d8ba4add96ed43b398bb7c93749"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--paths",required=True,type=Path);parser.add_argument("--config",default=Path("configs/confident_labels_v1.yaml"),type=Path);parser.add_argument("--finalization",default=Path("configs/supervised_finalization_v1.yaml"),type=Path);parser.add_argument("--bands",default=Path("configs/spectral_bands.yaml"),type=Path);args=parser.parse_args()
    paths=yaml.safe_load(args.paths.read_text());confident=yaml.safe_load(args.config.read_text());finalization=yaml.safe_load(args.finalization.read_text())
    if finalization["benchmark_commit"]!=BENCHMARK_SHA or not finalization["field2_locked"]:raise ValueError("Deployment preparation requires frozen da95fd5 and Field 2 lock")
    project=Path(paths["project_root"]);contract_root=project/"metadata/local/contracts/supervised_finalization_v1";contract_root.mkdir(parents=True,exist_ok=True)
    benchmark_path=contract_root/"field1_confident_supervised_benchmark_contract.yaml";benchmark=yaml.safe_load(benchmark_path.read_text())
    if benchmark["status"]!="field1_confident_supervised_benchmark_frozen" or benchmark["field2_accessed"] is not False:raise ValueError("Benchmark is not frozen")
    deploy=finalization["deployment"];primary=set(deploy["primary_field1_cubes"]);prohibited=set(deploy["prohibited_primary_fit_cubes"])
    if primary&prohibited or {"field1_cube24","field1_cube28"}&primary:raise ValueError("Invalid deployment cube roles")
    records={r.cube_id:r for r in load_records(args.paths,project/"metadata/local/authoritative_manifest.csv")}
    if not primary<=records.keys():raise ValueError(f"Missing primary records: {sorted(primary-records.keys())}")
    bands=load_band_indices(args.bands,confident["normalization"]["band_section"])
    fold_contract=yaml.safe_load((project/"metadata/local/contracts/field1_spatial_fold_contract.yaml").read_text())
    folds=pd.read_csv(project/"metadata/local/contracts/field1_spatial_folds.csv")
    fold_lookup={(int(r.block_x),int(r.block_y)):int(r.fold) for r in folds.itertuples()}
    parts=[]
    for number,cube_id in enumerate(sorted(primary),1):
        label_root=project/confident["output_root"]/"canonical_outer_evaluation"/cube_id
        with rasterio.open(label_root/"confident_labels.tif") as dataset:labels=dataset.read(1);transform=dataset.transform;crs=dataset.crs.to_string()
        with rasterio.open(label_root/"provenance.tif") as dataset:provenance=dataset.read(1)
        cube=envi.open(str(records[cube_id].header),str(records[cube_id].data)).open_memmap()
        frame=frame_for_labels(cube_id,labels,provenance,valid_spectra(cube,bands),transform,crs,fold_lookup,
            fold_contract["block_size_m"],confident["sampling"]["patch_size_pixels"],deploy["minimum_separation_m"],
            deploy["ground_cell_resolution_m"],confident["sampling"]["seed"])
        parts.append(frame);print(f"[{number}/{len(primary)}] {cube_id}: {len(frame):,} patch-safe candidates",flush=True)
    full=pd.concat(parts,ignore_index=True)
    rows=[];sample_paths=[];normalization=[]
    for variant,population in (("with_alley",full),("without_alley",no_alley_variant(full))):
        selected=deduplicate_and_thin(population,2**31-1)
        if set(selected.cube_id)&prohibited:raise ValueError("Prohibited cube entered deployment fitting")
        if selected.duplicated(["ground_x","ground_y"]).any():raise ValueError("Duplicate ground cell in deployment data")
        selected["dataset_variant"]=variant;selected["nested_role"]="deployment_fit"
        path=contract_root/f"field1_deployment_{variant}_samples.csv";write_samples(path,selected);sample_paths.append(path)
        sums=np.zeros(len(bands),np.float64);squares=np.zeros(len(bands),np.float64);count=0
        for cube_id,one in selected.groupby("cube_id"):
            cube=envi.open(str(records[cube_id].header),str(records[cube_id].data)).open_memmap()
            values=np.asarray(cube[one.row.to_numpy(np.int64),one.column.to_numpy(np.int64)][:,bands],np.float64)
            sums+=values.sum(0);squares+=(values*values).sum(0);count+=len(values)
        mean=sums/count;variance=np.maximum(squares/count-mean*mean,0);std=np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError(f"Invalid deployment normalization: {variant}")
        for index,band in enumerate(bands):normalization.append({"variant":variant,"band_index":int(band),"mean":mean[index],"standard_deviation":std[index],"fitting_centers":count})
        for (class_id,cube_id,provenance_name),amount in selected.groupby(["class_id","cube_id","provenance"]).size().items():rows.append({"variant":variant,"class_id":int(class_id),"cube_id":cube_id,"provenance":provenance_name,"selected":int(amount)})
        print(f"{variant}: {len(selected):,} all-eligible thinned centers; classes={selected.groupby('class_id').size().to_dict()}",flush=True)
    normalization_path=contract_root/"field1_deployment_normalization.csv";pd.DataFrame(normalization).to_csv(normalization_path,index=False)
    support_path=project/"metadata/local/reports/supervised_finalization_v1/deployment_support.csv";support_path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(support_path,index=False)
    training=pd.read_csv(project/confident["report_root"]/"evaluation/training_run_summary.csv")
    observed={variant:int(training[training.variant==variant].best_epoch.median()) for variant in deploy["variants"]}
    if observed!={key:int(value) for key,value in deploy["fixed_epochs"].items()}:raise ValueError(f"Frozen median epoch rule mismatch: {observed}")
    contract={"status":"field1_deployment_data_frozen","benchmark_commit":BENCHMARK_SHA,"field":"Field 1","field2_accessed":False,
              "primary_cubes":sorted(primary),"prohibited_cubes":sorted(prohibited),"cubes24_28_used_as_blind_data":False,
              "patch_size_pixels":confident["sampling"]["patch_size_pixels"],"minimum_separation_m":deploy["minimum_separation_m"],
              "ground_cell_resolution_m":deploy["ground_cell_resolution_m"],"normalization_scope":deploy["normalization_scope"],
              "fixed_epochs":observed,"epoch_rule":deploy["epoch_rule"],"calibration":deploy["calibration"],
              "sample_sha256":{path.name:sha256(path) for path in sample_paths},"normalization_sha256":sha256(normalization_path),
              "support_report_sha256":sha256(support_path),"benchmark_contract_sha256":sha256(benchmark_path)}
    (contract_root/"field1_deployment_data_contract.yaml").write_text(yaml.safe_dump(contract,sort_keys=False))


if __name__=="__main__":main()
