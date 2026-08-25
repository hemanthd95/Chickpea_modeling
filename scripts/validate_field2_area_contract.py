#!/usr/bin/env python
"""Independently validate and deterministically recompile the frozen Field 2 area contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.field2_area_review import Field2AreaAnnotationStore, ZONE_CODES
from chickpea_ssl.field2_blind_review import validate_frozen_sampling_contract
from chickpea_ssl.field2_readiness import sha256
from scripts.freeze_field2_area_contract import (
    directory_hashes, materialize_freeze_products, raw_artifact_hashes, validate_ready_payload,
)
from scripts.run_field2_area_annotator import validate_config_inputs


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_area_contract(project: Path, area_config: dict, field2_config: dict, recompile: bool = True) -> dict:
    validate_config_inputs(project, area_config)
    contract_path = project / area_config["freeze"]["contract"]
    require(contract_path.is_file(), "Frozen Field 2 area contract is missing")
    contract = json.loads(contract_path.read_text())
    require(contract.get("version") == "field2_area_zone_contract_v2", "Unknown area contract version")
    require(contract.get("status") == "field2_area_zone_contract_frozen", "Area contract is not frozen")
    require(contract.get("precedence_rule") == "outside_research_field > alley > uncertain_boundary > research_crop_area > unassigned_valid_support", "Area precedence changed")
    require(contract.get("domain_codes") == {
        "0": "invalid_outside_valid_support", "1": "research_crop_area", "2": "alley",
        "3": "outside_research_field", "4": "uncertain_boundary", "5": "unassigned_valid_support",
    }, "Area domain codes changed")
    safety = contract.get("prediction_free_status", {})
    require(safety and not any(safety.values()), "Area contract is not prediction-free")
    require(contract.get("reserve_coordinates_exposed") is False, "Reserve coordinate protection changed")

    store = Field2AreaAnnotationStore(
        project, project / area_config["inputs"]["review_manifest"]["path"],
        project / area_config["inputs"]["natural_rgb_manifest"]["path"],
        project / area_config["inputs"]["valid_support_manifest"]["path"],
        project / area_config["annotation"]["output_root"],
    )
    payload = store.load(); validate_ready_payload(store, payload)
    require(raw_artifact_hashes(store) == contract.get("raw_annotation_inputs"), "Raw area annotations changed after freeze")
    require(contract.get("annotation_revision") == payload.get("revision"), "Raw area revision changed after freeze")
    require(contract.get("raw_polygon_count") == 438, "Frozen raw polygon inventory changed")
    require(contract.get("raw_vertex_count") == 2778, "Frozen raw vertex inventory changed")
    for name, item in contract.get("frozen_input_contracts", {}).items():
        require((project / item["path"]).is_file(), f"Missing frozen input: {name}")
        require(sha256(project / item["path"]) == item["sha256"], f"Frozen input changed: {name}")
    for name, item in contract.get("frozen_outputs", {}).items():
        require((project / item["path"]).is_file(), f"Missing frozen area output: {name}")
        require(sha256(project / item["path"]) == item["sha256"], f"Frozen area output changed: {name}")

    output_root = project / area_config["freeze"]["output_root"]
    manifest_path = output_root / area_config["freeze"]["mask_manifest"]
    require(sha256(manifest_path) == contract["operational_mask_manifest_sha256"], "Operational mask manifest changed")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    require(len(manifest) == 40 and manifest.cube_id.nunique() == 40, "Operational mask inventory is not 40 unique cubes")
    for row in manifest.itertuples(index=False):
        support_path = project / row.support_mask_path
        domain_path = output_root / row.domain_mask_path
        raw_path = output_root / row.raw_membership_mask_path
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(support_path) as support_source, rasterio.open(domain_path) as domain_source, rasterio.open(raw_path) as raw_source:
                require(domain_source.width == support_source.width and domain_source.height == support_source.height, f"Domain dimensions changed: {row.cube_id}")
                require(domain_source.crs == support_source.crs and domain_source.transform == support_source.transform and domain_source.bounds == support_source.bounds, f"Domain grid changed: {row.cube_id}")
                require(raw_source.crs == support_source.crs and raw_source.transform == support_source.transform and raw_source.shape == support_source.shape, f"Raw membership grid changed: {row.cube_id}")
                support = support_source.read(1) == 1; domain = domain_source.read(1)
        require(set(np.unique(domain)).issubset(set(ZONE_CODES.values())), f"Unexpected domain code: {row.cube_id}")
        require(np.all(domain[~support] == 0) and np.all(domain[support] != 0), f"Domain/support mismatch: {row.cube_id}")

    sampling = validate_frozen_sampling_contract(project, field2_config, project / field2_config["sampling"]["sampling_contract"])
    require(sampling["contract"]["reserve_release_policy"]["status"] == "locked_not_released_for_annotation", "Reserve is not locked")
    main = pd.read_csv(output_root / area_config["freeze"]["main_membership"], keep_default_na=False)
    reserve = pd.read_csv(output_root / area_config["freeze"]["reserve_membership_locked"], keep_default_na=False)
    prohibited = {"row", "column", "x", "y", "longitude", "latitude"}
    require(prohibited.isdisjoint(main.columns) and prohibited.isdisjoint(reserve.columns), "Membership output exposes coordinates")
    require(len(main) == 800 and len(reserve) == 396, "Area sample membership counts changed")
    require(main.sample_id.tolist() == sampling["main"].sample_id.astype(str).tolist(), "Main sample identity/order changed")
    require(reserve.sample_id.tolist() == sampling["reserve"].sample_id.astype(str).tolist(), "Reserve sample identity/order changed")
    require(not set(main.sample_id) & set(reserve.sample_id), "Main/reserve membership intersection")
    require(main.inside_valid_support.astype(str).str.lower().eq("true").all(), "Main contains invalid-support membership")
    require(reserve.inside_valid_support.astype(str).str.lower().eq("true").all(), "Reserve contains invalid-support membership")
    require(main.domain_code.between(1, 5).all() and reserve.domain_code.between(1, 5).all(), "Sample domain is not uniquely assigned")

    if recompile:
        temporary = Path(tempfile.mkdtemp(prefix="field2-area-contract-recompile-"))
        try:
            materialize_freeze_products(project, area_config, store, payload, sampling["main"], sampling["reserve"], temporary / "products")
            require(directory_hashes(temporary / "products") == directory_hashes(output_root), "Deterministic area recompile differs from frozen products")
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        "contract_path": str(contract_path), "contract_sha256": sha256(contract_path),
        "raw_polygons": contract["raw_polygon_count"], "raw_vertices": contract["raw_vertex_count"],
        "main_count": len(main), "reserve_count": len(reserve),
        "main_domain_counts": main.domain_name.value_counts().sort_index().to_dict(),
        "reserve_domain_counts": reserve.domain_name.value_counts().sort_index().to_dict(),
        "deterministic_recompile": recompile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--area-config", default=Path("configs/field2_area_annotation.yaml"), type=Path)
    parser.add_argument("--field2-config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    parser.add_argument("--skip-recompile", action="store_true")
    args = parser.parse_args()
    project = Path(yaml.safe_load(args.paths.read_text())["project_root"]).resolve()
    result = validate_area_contract(
        project, yaml.safe_load(args.area_config.read_text()), yaml.safe_load(args.field2_config.read_text()),
        recompile=not args.skip_recompile,
    )
    print("Frozen Field 2 area contract validation passed")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Reserve records/coordinates served: 0; prediction/model/checkpoint use: false")


if __name__ == "__main__":
    main()
