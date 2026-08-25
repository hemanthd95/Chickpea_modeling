#!/usr/bin/env python
"""Materialize review-only polygon-guided soil/non-soil support layers.

Investigator polygons define where chickpea may occur, but they are not class
labels. Existing cubes retain their provenance-tracked soil masks. Only the
three label-expansion cubes without soil masks use the separately validated,
uninterpreted Spectronon scalar-index rule. Outputs are candidates for human
review and later weed/chickpea separation; authoritative masks and model inputs
are never changed by this script.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
import yaml

from chickpea_ssl.data import load_records
from scripts.audit_chickpea_region_annotations import (
    disk,
    metres_per_pixel,
    normalized_ring,
    observed_support,
    planter_alley_features,
    reconcile_exports,
)
from scripts.audit_polygon_soil_sources import product_array, source_root


# 0 is observed background outside the spatial prior; 255 is source NoData.
CLASS_NAMES = {
    0: "outside_polygon_prior",
    1: "polygon_core_soil",
    2: "polygon_core_nonsoil_candidate",
    3: "polygon_edge_soil_uncertain",
    4: "polygon_edge_nonsoil_uncertain",
    5: "explicit_alley_excluded",
    255: "nodata_or_missing_source",
}
CMAP = ListedColormap([
    "#000000", "#8B5A2B", "#16A34A", "#D6A86A",
    "#86EFAC", "#F59E0B", "#DC2626",
])
NORM = BoundaryNorm(np.arange(-0.5, 7.5, 1), CMAP.N)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_candidate_support(
    observed: np.ndarray,
    core: np.ndarray,
    edge: np.ndarray,
    alley: np.ndarray,
    soil: np.ndarray,
    source_valid: np.ndarray,
) -> np.ndarray:
    """Return mutually exclusive review classes with alley precedence."""
    arrays = (core, edge, alley, soil, source_valid)
    if any(array.shape != observed.shape for array in arrays):
        raise ValueError("Candidate-support arrays must share one raster shape")
    result = np.zeros(observed.shape, dtype=np.uint8)
    spatial = (core | edge) & observed
    missing = spatial & ~source_valid
    result[core & source_valid & soil] = 1
    result[core & source_valid & ~soil] = 2
    result[edge & source_valid & soil] = 3
    result[edge & source_valid & ~soil] = 4
    result[~observed | missing] = 255
    # Investigator-drawn alleys are explicit exclusion evidence and therefore
    # override a missing scalar-index value wherever the cube itself is observed.
    result[spatial & alley] = 5
    return result


def display_classes(candidate: np.ndarray) -> np.ndarray:
    result = candidate.copy()
    result[result == 255] = 6
    return result


def predict_soil(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "lower_values_indicate_soil":
        return values <= threshold
    if direction == "higher_values_indicate_soil":
        return values >= threshold
    raise ValueError(f"Unsupported frozen index direction: {direction}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--annotations-root", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config.get("polygon_guided_candidate_materialization", {})
    if bool(policy.get("automatically_replace_authoritative_masks", False)):
        raise ValueError("Candidate materialization cannot replace authoritative masks")
    edge_tolerance = float(policy.get("outward_uncertainty_tolerance_m", 0.05))
    seam_snap = float(config["chickpea_region_polygon_audit"].get(
        "seam_snap_maximum_m", 0.30
    ))
    expansion = set(config["investigator_guidance"]["optional_mask_expansion"]["cubes"])

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "chickpea_regions"
    json_path = annotation_root / "field1_chickpea_region_annotations.json"
    csv_path = annotation_root / "field1_chickpea_region_vertices.csv"
    geojson_path = annotation_root / "field1_chickpea_region_annotations.geojson"
    _, geojson, _ = reconcile_exports(json_path, csv_path, geojson_path)

    polygons: dict[str, list[dict]] = defaultdict(list)
    for feature in geojson["features"]:
        cube_id = str(feature["properties"]["cube_id"])
        coordinates = feature["geometry"]["coordinates"][0]
        if coordinates and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        ring, _, _ = normalized_ring(coordinates, seam_snap)
        polygons[cube_id].append({"type": "Polygon", "coordinates": [ring]})

    alley_path = (
        local / "annotations" / "planter_tracks"
        / "field1_planter_track_annotations.geojson"
    )
    alleys = planter_alley_features(alley_path, seam_snap)
    manifest_path = local / "authoritative_manifest.csv"
    records = {
        record.cube_id: record
        for record in load_records(args.paths, manifest_path)
        if record.cube_id in polygons
    }
    missing_records = sorted(set(polygons) - set(records))
    if missing_records:
        raise ValueError(f"Annotated cubes missing from manifest: {missing_records}")

    layer_manifest_path = (
        local / "reports" / "chickpea_region_annotation"
        / "chickpea_region_annotation_layer_manifest.csv"
    )
    roles = pd.read_csv(layer_manifest_path).set_index("cube_id")["analysis_role"]

    soil_contract_path = (
        local / "contracts" / "field1_stored_index_soil_separability_contract.yaml"
    )
    soil_contract = yaml.safe_load(soil_contract_path.read_text())
    if not bool(soil_contract.get("validation_gate_passed", False)):
        raise ValueError("Stored-index soil validation gate did not pass")
    expected_interpretation = (
        "uninterpreted_scalar_index_not_assumed_to_be_documented_ndvi"
    )
    if soil_contract.get("source_interpretation") != expected_interpretation:
        raise ValueError("Unexpected stored-index source interpretation")
    direction = str(soil_contract["final_direction"])
    threshold = float(soil_contract["final_threshold"])

    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    index_catalog = catalog[
        catalog["file_role"].eq("ndvi")
        & catalog["extension"].isin([".bip", ".bil", ".bsq", ".tif", ".tiff"])
    ]

    output_root = project / str(policy.get(
        "output_root_relative", "data/processed/polygon_guided_mask_candidates"
    ))
    reports = (
        local / "reports" / "mask_refinement"
        / "polygon_guided_candidate_materialization"
    )
    individual = reports / "individual"
    output_root.mkdir(parents=True, exist_ok=True)
    individual.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    output_paths: list[Path] = []
    overview = []
    ordered = sorted(records, key=lambda item: int(item.split("cube")[-1]))
    for number, cube_id in enumerate(ordered, 1):
        record = records[cube_id]
        role = str(roles.loc[cube_id])
        with rasterio.open(record.data) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            crs = reference.crs
            gsd = metres_per_pixel(transform)
            observed = observed_support(reference)

        core = rasterize(
            [(geometry, 1) for geometry in polygons[cube_id]],
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
            all_touched=False,
        ).astype(bool) & observed
        radius = int(math.ceil(edge_tolerance / gsd))
        tolerant = binary_dilation(core, structure=disk(radius)) & observed
        edge = tolerant & ~core
        alley_shapes = [(geometry, 1) for geometry in alleys.get(cube_id, [])]
        alley = (
            rasterize(
                alley_shapes, out_shape=shape, transform=transform, fill=0,
                dtype="uint8", all_touched=True,
            ).astype(bool)
            if alley_shapes else np.zeros(shape, dtype=bool)
        )

        if record.soil_mask is not None:
            with rasterio.open(record.soil_mask) as dataset:
                if (dataset.height, dataset.width) != shape:
                    raise ValueError(f"Soil-mask shape mismatch for {cube_id}")
                soil = dataset.read(1) > 0
            source_valid = observed.copy()
            soil_source = "provenance_tracked_authoritative_soil_mask"
            source_path = record.soil_mask
        elif cube_id in expansion:
            candidates = index_catalog[index_catalog["cube_id_inferred"].eq(cube_id)]
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one validated stored index for {cube_id}, found {len(candidates)}"
                )
            row = candidates.iloc[0]
            source_path = source_root(paths, row) / row["relative_path"]
            values, metadata = product_array(source_path, shape)
            if values is None or not metadata["plausible_ndvi_scale"]:
                raise ValueError(f"Stored index unavailable for {cube_id}: {metadata['read_status']}")
            source_valid = observed & np.isfinite(values)
            soil = np.zeros(shape, dtype=bool)
            soil[source_valid] = predict_soil(values[source_valid], direction, threshold)
            soil_source = "validated_uninterpreted_scalar_index_review_rule"
        else:
            raise ValueError(f"No approved review-only soil source for {cube_id}")

        candidate = classify_candidate_support(
            observed, core, edge, alley, soil, source_valid
        )
        cube_root = output_root / cube_id
        cube_root.mkdir(parents=True, exist_ok=True)
        candidate_path = cube_root / "polygon_guided_candidate_support.tif"
        profile = {
            "driver": "GTiff", "height": shape[0], "width": shape[1],
            "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
            "nodata": 255, "compress": "deflate", "predictor": 2,
        }
        with rasterio.open(candidate_path, "w", **profile) as dataset:
            dataset.write(candidate, 1)
            dataset.update_tags(
                status="review_only_candidate_not_authoritative",
                soil_source=soil_source,
                index_direction=direction if cube_id in expansion else "not_applicable",
                index_threshold=(f"{threshold:.9f}" if cube_id in expansion else "not_applicable"),
                class_legend=json.dumps(CLASS_NAMES, sort_keys=True),
            )
        output_paths.append(candidate_path)

        counts = {value: int((candidate == value).sum()) for value in CLASS_NAMES}
        eligible = counts[1] + counts[2]
        summary_rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "soil_source": soil_source,
            "source_path": str(source_path),
            "polygon_count": len(polygons[cube_id]),
            "edge_tolerance_m": edge_tolerance,
            "edge_tolerance_pixels": radius,
            "core_soil_pixels": counts[1],
            "core_nonsoil_candidate_pixels": counts[2],
            "core_soil_fraction": counts[1] / eligible if eligible else np.nan,
            "edge_soil_uncertain_pixels": counts[3],
            "edge_nonsoil_uncertain_pixels": counts[4],
            "alley_excluded_pixels": counts[5],
            "nodata_or_missing_source_pixels": counts[255],
            "candidate_path": str(candidate_path),
            "authoritative_mask_modified": False,
        })

        step = max(1, math.ceil(max(shape) / 600))
        tile = display_classes(candidate[::step, ::step])
        preview_path = individual / f"{cube_id}_polygon_guided_candidate_support.png"
        figure, axis = plt.subplots(figsize=(6, 8), constrained_layout=True)
        axis.imshow(tile, cmap=CMAP, norm=NORM, interpolation="nearest")
        axis.axis("off")
        axis.set_title(
            f"{cube_id} — {role}\nsoil={counts[1]:,}; "
            f"core non-soil={counts[2]:,}; source={soil_source}", fontsize=10,
        )
        figure.savefig(preview_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(preview_path)
        overview.append((cube_id, role, soil_source, tile, counts[1], counts[2]))
        print(
            f"Materialized {number}/{len(ordered)} {cube_id}: "
            f"soil={counts[1]:,}; core non-soil={counts[2]:,}; source={soil_source}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "polygon_guided_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    output_paths.append(summary_path)

    columns = 4
    rows = math.ceil(len(overview) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, role, source, tile, soil_count, nonsoil_count) in zip(flat, overview):
        axis.imshow(tile, cmap=CMAP, norm=NORM, interpolation="nearest")
        short_source = "existing soil" if source.startswith("provenance") else "calibrated index"
        axis.set_title(
            f"{cube_id} | {role} | {short_source}\n"
            f"soil {soil_count:,}; core non-soil {nonsoil_count:,}", fontsize=8,
        )
    figure.suptitle(
        "Field 1 review-only polygon-guided candidate support\n"
        "brown=soil; green=core non-soil candidate; pale colors=5 cm uncertain edge; "
        "orange=alley; red=NoData",
        fontsize=14,
    )
    overview_path = reports / "polygon_guided_candidate_materialization_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    output_paths.append(overview_path)

    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "review_only_polygon_guided_candidates_materialized",
        "field": "Field 1",
        "class_legend": CLASS_NAMES,
        "spatial_authority": "investigator_chickpea_region_polygons",
        "existing_cube_soil_source": "provenance_tracked_authoritative_soil_mask",
        "expansion_cube_soil_source": "validated_uninterpreted_scalar_index_review_rule",
        "expansion_cubes": sorted(expansion),
        "frozen_index_direction": direction,
        "frozen_index_threshold": threshold,
        "outward_edge_tolerance_m": edge_tolerance,
        "explicit_alley_precedence": True,
        "core_nonsoil_status": "candidate_pending_observed_weed_chickpea_separation",
        "edge_status": "uncertain_not_training_ready",
        "sensitivity_cubes_remain_excluded_from_primary_analysis": [
            cube for cube in ordered if str(roles.loc[cube]) == "sensitivity_only"
        ],
        "candidate_geotiffs_are_authoritative": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "planter_annotations_geojson": sha256(alley_path),
            "authoritative_manifest": sha256(manifest_path),
            "annotation_layer_manifest": sha256(layer_manifest_path),
            "stored_index_soil_contract": sha256(soil_contract_path),
        },
        "report_hashes": {
            str(path.relative_to(reports)): sha256(path)
            for path in output_paths if reports in path.parents
        },
    }
    contract_path = contracts / "field1_polygon_guided_candidate_materialization_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Candidate GeoTIFF root: {output_root}")
    print(f"Summary: {summary_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print(
        "Review only: authoritative masks and model inputs were not changed; "
        "no model was retrained and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
