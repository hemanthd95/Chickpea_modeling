#!/usr/bin/env python
"""Freeze a no-access Field 2 compatibility stop and required-product checklist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

import pandas as pd
import yaml

from chickpea_ssl.interpretability import sha256


BENCHMARK_SHA="da95fd59b7dc5d8ba4add96ed43b398bb7c93749"


REQUIRED_PRODUCTS = [
    ("cube_manifest", "required", "CSV or YAML", "One row per Field 2 cube with immutable cube ID, acquisition date, sensor, processing version, and relative paths."),
    ("georectified_envi_header", "required_per_cube", ".hdr", "ENVI header paired one-to-one with the cube binary; must declare samples, lines, 150 bands, BIP interleave, data type, byte order, wavelengths, wavelength units, map info, and NoData semantics."),
    ("georectified_envi_binary", "required_per_cube", ".bip", "Georectified Pika-L hyperspectral cube matching its header byte size and grid; no reflectance-only ungeorectified substitute."),
    ("wavelength_vector", "required_per_cube", "header metadata plus QC CSV", "Strictly increasing nanometers, compatible band order, and coverage of model indices 3–113 (111 channels; 401.84–869.34 nm). Record tolerance and any resampling decision before opening labels."),
    ("radiometric_provenance", "required_per_cube", "metadata document", "Processing/calibration method, units, scale factor, dark/white reference handling, saturation encoding, and confirmation of compatibility with Field 1. The current Field 1 ENVI headers do not identify a unit-reflectance scale."),
    ("georeferencing", "required_per_cube", "header map info and QC CSV", "Projected CRS identifier/WKT, affine transform, pixel size, bounds, and georectification residual/error summary. Target compatibility is Field 1 EPSG:32617 and nominal 0.015 m primary GSD; deviations require a frozen transfer decision."),
    ("valid_data_definition", "required_per_cube", "header metadata or aligned mask", "Explicit NoData/saturation values and an aligned valid-support mask or a verified rule. It must permit 15×15 fully observed patch screening."),
    ("footprint_index", "required", "GeoPackage/GeoJSON/CSV", "Per-cube footprint in the declared CRS, overlap relationships, and unique ground-coverage identifiers; no class labels or predictions."),
    ("source_checksum_manifest", "required", "CSV", "Relative path, byte size, and SHA-256 for every supplied header, binary, metadata, mask, footprint, and preview source."),
    ("false_color_source", "required_for_annotation", "aligned GeoTIFF/PNG recipe", "Prediction-free NIR-red-green and natural-color source views, or sufficient georectified bands plus a frozen recipe to generate them."),
    ("pca_preview_source", "required_for_annotation", "aligned GeoTIFF/PNG recipe", "Prediction-free PCA components and, if used, first/second spectral-difference PCA views with fit scope and sign/orientation provenance."),
    ("annotation_boundary", "required_for_annotation", "GeoPackage/GeoJSON", "Authorized Field 2 annotation extent and exclusions in the cube CRS, independent of supervised predictions."),
    ("investigator_annotation_output", "required_after_compatibility", "immutable JSON/CSV/GeoJSON", "Point ID, cube ID, coordinates/pixel location, soil/chickpea/weed/uncertain/invalid label, confidence, note, blinded duplicate ID, inclusion probability, and checksum. This is not required to pass imagery compatibility but is required before evaluation."),
]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--paths",required=True,type=Path);parser.add_argument("--finalization",default=Path("configs/supervised_finalization_v1.yaml"),type=Path);args=parser.parse_args()
    paths=yaml.safe_load(args.paths.read_text());finalization=yaml.safe_load(args.finalization.read_text());project=Path(paths["project_root"])
    if finalization["benchmark_commit"]!=BENCHMARK_SHA or not finalization["field2_locked"]:raise ValueError("Field 2 stop requires frozen da95fd5 and active lock")
    field2=paths.get("field2",{});imagery=str(field2.get("imagery","")).strip();labels=str(field2.get("tall_grass_rtk","")).strip()
    if imagery or labels or field2.get("locked") is not True:raise ValueError("Field 2 paths are unexpectedly populated or unlocked; refuse no-access freeze")
    manifest=pd.read_csv(project/"metadata/local/authoritative_manifest.csv",dtype=str).fillna("")
    field2_rows=manifest[manifest.astype(str).apply(lambda row:row.str.contains("field2",case=False,regex=False).any(),axis=1)]
    if len(field2_rows):raise ValueError("Authorized manifest unexpectedly contains Field 2")
    report=project/"metadata/local/reports/supervised_finalization_v1";report.mkdir(parents=True,exist_ok=True)
    checklist_path=report/"field2_georectified_product_checklist.csv"
    pd.DataFrame(REQUIRED_PRODUCTS,columns=["product","requirement","format","acceptance_check"]).to_csv(checklist_path,index=False)
    deployment_path=project/"metadata/local/contracts/supervised_finalization_v1/field1_deployment_ensemble_contract.yaml"
    deployment=yaml.safe_load(deployment_path.read_text())
    if deployment["status"]!="field1_deployment_ensembles_frozen_before_field2_compatibility" or deployment["field2_accessed"] is not False:raise ValueError("Deployment ensemble is not safely frozen")
    contract={"status":"blocked_not_georectified_or_authorized","decision":"stop_before_field2_open_or_prediction","benchmark_commit":BENCHMARK_SHA,
              "field2_accessed":False,"field2_scientific_arrays_opened":False,"field2_headers_opened":False,"field2_coordinates_or_labels_opened":False,
              "field2_predictions_generated":False,"authorized_field2_imagery_path":"","authorized_field2_label_path":"","authorized_manifest_field2_rows":0,
              "reason":"Field 2 is reflectance-only and no georectified product is available in the authorized data section.",
              "locked_archive_metadata_is_not_authorized_scientific_input":True,"cubes24_28_substitution":"prohibited_previously_inspected_field1_sensitivity_only",
              "compatibility_checks_not_run":["spectral_wavelength_compatibility","band_order","radiometric_units_and_scale","crs_and_transform","nodata","spatial_resolution","preview_availability"],
              "required_product_checklist":str(checklist_path.relative_to(project)),"required_product_checklist_sha256":sha256(checklist_path),
              "deployment_ensemble_contract_sha256":sha256(deployment_path),
              "resume_rule":"Populate a new authorized Field 2 manifest only after all required georectified products and provenance are supplied; then run read-only compatibility QC before annotation. Do not generate predictions."}
    root=project/"metadata/local/contracts/supervised_finalization_v1";(root/"field2_compatibility_stop_contract.yaml").write_text(yaml.safe_dump(contract,sort_keys=False))
    print(yaml.safe_dump(contract,sort_keys=False))


if __name__=="__main__":main()
