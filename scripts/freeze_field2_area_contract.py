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
    Field2AreaAnnotationStore, compile_zone_mask, incompatible_overlap_rows,
    polygon_self_intersects, utc_now, zone_name_from_code,
)
from chickpea_ssl.field2_blind_review import atomic_write_json, require_main_sampling_frame
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_area_annotator import validate_config_inputs


def membership_rows(frame: pd.DataFrame, masks: dict[str, np.ndarray], sampling_frame: str) -> list[dict]:
    rows = []
    for sample in frame.itertuples(index=False):
        code = int(masks[str(sample.cube_id)][int(sample.row), int(sample.column)])
        zone = zone_name_from_code(code)
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
        if any(polygon_self_intersects(polygon["vertices_pixel"]) for polygon in record["polygons"]):
            raise ValueError(f"Self-intersection must be resolved before freeze: {cube_id}")
        if incompatible_overlap_rows(record["polygons"], store.support_mask(cube_id)):
            raise ValueError(f"Incompatible zone overlap must be resolved before freeze: {cube_id}")


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
        masks = {}
        mask_paths = {}
        for cube_id in store.cube_order:
            support_path = project / str(store.support[cube_id].mask_path)
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(support_path, "r") as source:
                    support = source.read(1) == 1; profile = source.profile.copy()
            mask = compile_zone_mask(payload["annotations"][cube_id], support); masks[cube_id] = mask
            profile.update(dtype="uint8", count=1, nodata=0, compress="DEFLATE", predictor=2, zlevel=9)
            mask_path = mask_root / f"{cube_id}_area_zone_membership.tif"
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(mask_path, "w", **profile) as target: target.write(mask, 1)
            mask_paths[cube_id] = mask_path
        main_rows = membership_rows(released["main"], masks, "main")
        reserve_rows = membership_rows(reserve, masks, "reserve")
        main_path = temporary / area_config["freeze"]["main_membership"]
        reserve_membership_path = temporary / area_config["freeze"]["reserve_membership_locked"]
        counts_path = temporary / area_config["freeze"]["counts"]
        pd.DataFrame(main_rows).to_csv(main_path, index=False)
        pd.DataFrame(reserve_rows).to_csv(reserve_membership_path, index=False)
        counts = pd.concat([pd.DataFrame(main_rows), pd.DataFrame(reserve_rows)]).groupby(
            ["sampling_frame", "cube_id", "cube_evaluation_role", "zone_type"], as_index=False,
        ).size().rename(columns={"size": "point_count"})
        counts.to_csv(counts_path, index=False)
        output_root_files = {
            "main_membership": main_path, "reserve_membership_locked": reserve_membership_path,
            "zone_counts": counts_path,
        }
        output_root_files.update({f"mask_{cube_id}": path for cube_id, path in mask_paths.items()})
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
