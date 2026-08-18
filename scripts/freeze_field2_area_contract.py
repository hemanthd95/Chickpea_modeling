#!/usr/bin/env python
"""Freeze reviewed Field 2 area zones without assigning biological labels."""

from __future__ import annotations

import argparse
from collections import Counter
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

from chickpea_ssl.field2_area_review import (
    Field2AreaAnnotationStore, utc_now, zone_name_from_code,
)
from chickpea_ssl.field2_area_operational import (
    PRECEDENCE_RULE, RAW_ZONE_BITS, ZONE_PRECEDENCE,
    compile_operational_membership, raw_memberships_from_bits,
)
from chickpea_ssl.field2_blind_review import atomic_write_json, require_main_sampling_frame
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_area_annotator import validate_config_inputs


def membership_rows(frame: pd.DataFrame, compiled: dict[str, dict], sampling_frame: str) -> list[dict]:
    rows = []
    for sample in frame.itertuples(index=False):
        cube = compiled[str(sample.cube_id)]; row, column = int(sample.row), int(sample.column)
        code = int(cube["winning_membership"][row, column])
        zone = zone_name_from_code(code)
        raw_memberships = raw_memberships_from_bits(int(cube["raw_membership_bitmask"][row, column]))
        domain = {
            "research_crop_area": "research_field",
            "alley": "alley",
            "outside_research_field": "outside_research_field",
            "uncertain_boundary": "uncertain_boundary",
            "unassigned_valid_support": "unassigned_valid_support",
        }[zone]
        rows.append({
            "sample_id": str(sample.sample_id), "cube_id": str(sample.cube_id),
            "cube_evaluation_role": str(sample.cube_evaluation_role),
            "sampling_frame": sampling_frame, "zone_type": zone, "domain": domain,
            "raw_zone_memberships": "|".join(raw_memberships) or "unassigned_valid_support",
            "winning_operational_membership": zone,
            "precedence_rule_applied": PRECEDENCE_RULE if bool(cube["precedence_applied"][row, column]) else "single_or_unassigned_membership",
            "effective_operational_mode": cube["effective_coverage_mode"],
            "primary_external_validation_eligible": zone == "research_crop_area",
            "supplementary_or_domain_shift": zone != "research_crop_area",
        })
    return rows


def validate_ready_payload(store: Field2AreaAnnotationStore, payload: dict) -> None:
    issues = store.validate(payload)
    invalid = {cube_id: values for cube_id, values in issues.items() if values}
    if invalid:
        raise ValueError("Area annotations are not freeze-ready: " + json.dumps(invalid, sort_keys=True))
    if not all(record["reviewed"] for record in payload["annotations"].values()):
        raise ValueError("Area freeze requires all 40 cubes reviewed")
    for cube_id, record in payload["annotations"].items():
        compile_operational_membership(record, store.support_mask(cube_id))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--area-config", default=Path("configs/field2_area_annotation.yaml"), type=Path)
    parser.add_argument("--field2-config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text()); area_config = yaml.safe_load(args.area_config.read_text())
    field2_config = yaml.safe_load(args.field2_config.read_text()); project = Path(paths["project_root"]).resolve()
    validate_config_inputs(project, area_config)
    contract_path = project / area_config["freeze"]["contract"]
    output_root = project / area_config["freeze"]["output_root"]
    if contract_path.exists() or output_root.exists():
        raise FileExistsError("Frozen area contract/output already exists; overwrite is prohibited")
    released = require_main_sampling_frame(project, field2_config)
    reserve_path = project / field2_config["sampling"]["reserve_frame"]
    reserve = pd.read_csv(reserve_path).fillna("")
    if len(released["main"]) != 800 or len(reserve) != 396 or set(reserve.sampling_frame) != {"reserve"}:
        raise ValueError("Frozen sampling inventory changed")
    store = Field2AreaAnnotationStore(
        project, project / area_config["inputs"]["review_manifest"]["path"],
        project / area_config["inputs"]["natural_rgb_manifest"]["path"],
        project / area_config["inputs"]["valid_support_manifest"]["path"],
        project / area_config["annotation"]["output_root"],
    )
    payload = store.load(); validate_ready_payload(store, payload)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".field2-area-freeze-", dir=output_root.parent))
    try:
        mask_root = temporary / area_config["freeze"]["mask_root"]; mask_root.mkdir(parents=True)
        compiled = {}
        mask_paths = {}
        raw_mask_paths = {}
        raw_mask_root = temporary / area_config["freeze"]["raw_membership_mask_root"]; raw_mask_root.mkdir(parents=True)
        precedence_rows = []
        for cube_id in store.cube_order:
            support_path = project / str(store.support[cube_id].mask_path)
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(support_path, "r") as source:
                    support = source.read(1) == 1; profile = source.profile.copy()
            cube = compile_operational_membership(payload["annotations"][cube_id], support); compiled[cube_id] = cube
            mask = cube["winning_membership"]
            profile.update(dtype="uint8", count=1, nodata=0, compress="DEFLATE", predictor=2, zlevel=9)
            mask_path = mask_root / f"{cube_id}_area_zone_membership.tif"
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(mask_path, "w", **profile) as target: target.write(mask, 1)
            mask_paths[cube_id] = mask_path
            raw_mask_path = raw_mask_root / f"{cube_id}_raw_zone_membership_bits.tif"
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(raw_mask_path, "w", **profile) as target: target.write(cube["raw_membership_bitmask"], 1)
            raw_mask_paths[cube_id] = raw_mask_path
            for bits, pixel_count in zip(*np.unique(cube["raw_membership_bitmask"][support], return_counts=True)):
                raw = raw_memberships_from_bits(int(bits))
                winner = next((zone for zone in reversed(ZONE_PRECEDENCE) if int(bits) & RAW_ZONE_BITS[zone]), "unassigned_valid_support")
                precedence_rows.append({
                    "cube_id": cube_id, "raw_membership_bits": int(bits),
                    "raw_zone_memberships": "|".join(raw) or "unassigned_valid_support",
                    "winning_operational_membership": winner,
                    "precedence_rule_applied": PRECEDENCE_RULE if len(raw) > 1 else "single_or_unassigned_membership",
                    "pixel_count": int(pixel_count),
                })
        main_rows = membership_rows(released["main"], compiled, "main")
        reserve_rows = membership_rows(reserve, compiled, "reserve")
        main_path = temporary / area_config["freeze"]["main_membership"]
        reserve_membership_path = temporary / area_config["freeze"]["reserve_membership_locked"]
        counts_path = temporary / area_config["freeze"]["counts"]
        precedence_path = temporary / area_config["freeze"]["precedence_summary"]
        pd.DataFrame(main_rows).to_csv(main_path, index=False)
        pd.DataFrame(reserve_rows).to_csv(reserve_membership_path, index=False)
        counts = pd.concat([pd.DataFrame(main_rows), pd.DataFrame(reserve_rows)]).groupby(
            ["sampling_frame", "cube_id", "cube_evaluation_role", "zone_type"], as_index=False,
        ).size().rename(columns={"size": "point_count"})
        counts.to_csv(counts_path, index=False)
        pd.DataFrame(precedence_rows).to_csv(precedence_path, index=False)
        output_root_files = {
            "main_membership": main_path, "reserve_membership_locked": reserve_membership_path,
            "zone_counts": counts_path, "precedence_summary": precedence_path,
        }
        output_root_files.update({f"mask_{cube_id}": path for cube_id, path in mask_paths.items()})
        output_root_files.update({f"raw_membership_mask_{cube_id}": path for cube_id, path in raw_mask_paths.items()})
        temporary.replace(output_root)
        frozen_outputs = {
            name: {"path": str((output_root / path.relative_to(temporary)).relative_to(project)),
                   "sha256": sha256(output_root / path.relative_to(temporary))}
            for name, path in output_root_files.items()
        }
        contract = {
            "version": "field2_area_zone_contract_v1", "status": "field2_area_zone_contract_frozen",
            "freeze_timestamp_utc": utc_now(), "annotation_path": str(store.json_path.relative_to(project)),
            "annotation_sha256": sha256(store.json_path), "annotation_revision": payload["revision"],
            "reviewed_cube_count": 40, "main_count": 800, "reserve_count": 396,
            "reserve_coordinates_exposed": False, "biological_labels_assigned": False,
            "predictions_or_probabilities_used": False, "morphology_or_segmentation_used": False,
            "raw_annotation_vertices_preserved": True,
            "operational_geometry_policy": {
                "terminal_reconciliation_applied_to_operational_only": True,
                "validity_repair": "deterministic_raster_polygonize_all_components",
                "precedence_rule": PRECEDENCE_RULE,
                "self_intersections_overlaps_and_edge_extensions": "audit_warnings",
                "blank_mode_with_polygons": "mixed_manual_boundaries",
            },
            "zone_codes": {"0": "unassigned_valid_support", "1": "research_crop_area", "2": "alley", "3": "outside_research_field", "4": "uncertain_boundary"},
            "frozen_input_sha256": {name: item["sha256"] for name, item in area_config["inputs"].items()},
            "frozen_outputs": frozen_outputs,
            "zone_totals": dict(Counter(row["zone_type"] for row in main_rows)),
        }
        atomic_write_json(contract_path, contract)
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise
    print("Field 2 area-zone contract frozen")
    print(f"Contract: {contract_path}")
    print("Main memberships: 800; locked reserve memberships: 396; reserve coordinates omitted")
    print("Biological labels assigned: 0; predictions/probabilities used: false")


if __name__ == "__main__": main()
