#!/usr/bin/env python
"""Build leakage-safe, versioned Field 1 confident-label rasters.

The script never edits source masks.  It reconstructs the frozen investigator
reference spectra, fits one separator per nested outer-fold fitting role, and
writes both fold-specific label products and a canonical outer-evaluation view.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_erosion
from spectral.io import envi
import yaml

from chickpea_ssl.confident_labels import (
    CHICKPEA, PROVENANCE, UNRESOLVED, apply_confirmed_points,
    compose_confident_labels,
)
from chickpea_ssl.data import authoritative_class_map, load_band_indices, load_records
from chickpea_ssl.spatial import map_block_indices, spatial_group_id
from scripts.audit_chickpea_region_annotations import disk, metres_per_pixel, planter_alley_features
from scripts.audit_investigator_vegetation_references import fit_separator
from scripts.build_investigator_chickpea_probability_maps import (
    dense_neighbourhood_features, map_disk_offsets, reconstruct_frozen_features,
)


CLASS_NAMES = {0: "soil", 1: "chickpea", 2: "weed", 255: "unresolved"}
COLORS = np.asarray([[139, 90, 43], [22, 163, 74], [109, 40, 217], [190, 190, 190]], np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen {label} hash mismatch: {path}; {actual} != {expected}")


def project_path(project: Path, path: Path) -> Path:
    return path if path.is_absolute() else project / path


def raster_valid_spectra(cube: np.ndarray, bands: np.ndarray, rows_per_chunk: int = 128) -> np.ndarray:
    valid = np.zeros(cube.shape[:2], dtype=bool)
    for start in range(0, cube.shape[0], rows_per_chunk):
        values = np.asarray(cube[start:start + rows_per_chunk, :, :][..., bands])
        valid[start:start + len(values)] = np.isfinite(values).all(axis=2) & np.any(values > 0, axis=2)
    return valid


def rasterize_alleys(alley_geometries, shape_, transform, inward_m: float):
    if not alley_geometries:
        empty = np.zeros(shape_, bool)
        return empty, empty.copy()
    full_shapes = [(geometry, 1) for geometry in alley_geometries]
    full = rasterize(full_shapes, out_shape=shape_, transform=transform, fill=0,
                     dtype="uint8", all_touched=True).astype(bool)
    if inward_m <= 0:
        core = full.copy()
    else:
        radius = int(math.ceil(inward_m / metres_per_pixel(transform)))
        core = binary_erosion(full, structure=disk(radius), border_value=0)
    return full, core & full


def fold_for_points(points: pd.DataFrame, folds: pd.DataFrame, block_size: float) -> pd.Series:
    lookup = {(int(row.block_x), int(row.block_y)): int(row.fold) for row in folds.itertuples()}
    bx, by = map_block_indices(points.map_x.to_numpy(), points.map_y.to_numpy(), block_size, 0, 0)
    result = pd.Series([lookup.get((int(x), int(y)), 0) for x, y in zip(bx, by)], index=points.index)
    if (result == 0).any():
        raise ValueError("Investigator reference point outside frozen spatial-fold groups")
    return result


def raster_folds(shape_, transform, fold_lookup, block_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(shape_[0])
    columns = np.arange(shape_[1])
    x = transform.a * (columns + 0.5) + transform.c
    y = transform.e * (rows + 0.5) + transform.f
    bx = np.floor(x / block_size).astype(np.int64)
    by = np.floor(y / block_size).astype(np.int64)
    output = np.zeros(shape_, dtype=np.uint8)
    for one_x in np.unique(bx):
        column_mask = bx == one_x
        for one_y in np.unique(by):
            fold = fold_lookup.get((int(one_x), int(one_y)), 0)
            if fold:
                output[np.ix_(by == one_y, column_mask)] = fold
    return output, bx, by


def write_raster(path: Path, values: np.ndarray, reference_profile: dict, nodata: int, tags: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference_profile.copy()
    profile.update(driver="GTiff", count=1, dtype=str(values.dtype), nodata=nodata,
                   compress="deflate", predictor=2)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(values, 1)
        dataset.update_tags(**{key: str(value) for key, value in tags.items()})


def preview(path: Path, labels: np.ndarray, provenance: np.ndarray, alley: np.ndarray,
            polygon: np.ndarray, historical: np.ndarray | None, title: str) -> None:
    step = max(1, math.ceil(max(labels.shape) / 900))
    sampled = labels[::step, ::step]
    rgb = np.zeros((*sampled.shape, 3), dtype=np.uint8)
    for index, value in enumerate((0, 1, 2, 255)):
        rgb[sampled == value] = COLORS[index]
    alley_edge = alley & ~binary_erosion(alley)
    polygon_edge = polygon & ~binary_erosion(polygon)
    rgb[alley_edge[::step, ::step]] = [245, 158, 11]
    rgb[polygon_edge[::step, ::step]] = [255, 255, 255]
    if historical is not None:
        disagreement = (historical >= 0) & (labels != 255) & (historical != labels)
        rgb[disagreement[::step, ::step]] = [220, 38, 38]
    figure, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    axes[0].imshow(rgb); axes[0].set_title(title); axes[0].axis("off")
    axes[1].imshow(provenance[::step, ::step], cmap="tab10", vmin=0, vmax=9,
                   interpolation="nearest")
    axes[1].set_title("Provenance codes 0–9"); axes[1].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/confident_labels_v1.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config.get("field2_locked") or not yaml.safe_load(Path("configs/data_decisions.yaml").read_text())["rules"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    source_contract_names = [
        "field1_investigator_vegetation_reference_contract.yaml",
        "field1_investigator_chickpea_probability_review_contract.yaml",
        "field1_investigator_probability_threshold_audit_contract.yaml",
        "field1_polygon_guided_candidate_materialization_contract.yaml",
        "field1_chickpea_region_annotation_layers_contract.yaml",
        "field1_planter_annotation_qc_contract.yaml",
        "field1_spatial_fold_contract.yaml",
        "field1_supervised_primary_evaluation_protocol.yaml",
        "field1_nested_model_samples_contract.yaml",
        "field1_nested_normalization_contract.yaml",
        "field1_nested_supervised_checkpoints_contract.yaml",
        "field1_supervised_outer_evaluation_contract.yaml",
    ]
    source_contract_paths = [contracts / name for name in source_contract_names]
    missing = [path for path in source_contract_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required contracts: " + ", ".join(map(str, missing)))

    reference_contract = yaml.safe_load(source_contract_paths[0].read_text())
    threshold_contract = yaml.safe_load(source_contract_paths[2].read_text())
    candidate_contract = yaml.safe_load(source_contract_paths[3].read_text())
    fold_contract = yaml.safe_load(source_contract_paths[6].read_text())
    protocol = yaml.safe_load(source_contract_paths[7].read_text())
    if reference_contract.get("reference_points_total") != 354 or reference_contract.get("usable_reference_points") != 348:
        raise ValueError("Frozen investigator reference counts changed")
    if not reference_contract.get("acceptance", {}).get("gate_passed"):
        raise ValueError("Investigator vegetation reference gate failed")
    if not threshold_contract.get("point_reference_gate_passed"):
        raise ValueError("Probability threshold gate failed")
    frozen_thresholds = threshold_contract["display_thresholds"]
    probability_config = config["probability"]
    if not np.isclose(float(frozen_thresholds["weed_maximum_probability"]), probability_config["weed_maximum"]):
        raise ValueError("Weed threshold differs from frozen audit")
    if not np.isclose(float(frozen_thresholds["chickpea_minimum_probability"]), probability_config["chickpea_minimum"]):
        raise ValueError("Chickpea threshold differs from frozen audit")
    if not np.isclose(candidate_contract["outward_edge_tolerance_m"], config["geometry"]["polygon_edge_tolerance_m"]):
        raise ValueError("Polygon edge tolerance differs from frozen candidate contract")

    folds_path = contracts / "field1_spatial_folds.csv"
    roles_path = contracts / "field1_nested_fold_roles.csv"
    require_hash(folds_path, fold_contract["assignment_sha256"], "spatial fold")
    require_hash(roles_path, protocol["nested_role_table_sha256"], "nested role table")
    frozen_files = {
        "reference_points": contracts / "field1_investigator_vegetation_reference_points.csv",
        "point_qc": contracts / "field1_investigator_vegetation_reference_qc.csv",
        "loco": contracts / "field1_investigator_vegetation_reference_loco_validation.csv",
        "spectral_profile": contracts / "field1_investigator_vegetation_reference_spectral_profile.csv",
        "visual_qc": contracts / "field1_investigator_vegetation_reference_overview.png",
    }
    for name, path in frozen_files.items():
        require_hash(path, reference_contract["frozen_artifact_hashes"][name], name)

    folds = pd.read_csv(folds_path)
    roles = pd.read_csv(roles_path)
    fold_lookup = {(int(row.block_x), int(row.block_y)): int(row.fold) for row in folds.itertuples()}
    block_size = float(fold_contract["block_size_m"])
    inner_by_outer = {
        int(outer): int(roles[(roles.outer_evaluation_fold == outer) & (roles.role == "inner_validation")].spatial_fold.iloc[0])
        for outer in sorted(roles.outer_evaluation_fold.unique())
    }
    points = pd.read_csv(frozen_files["reference_points"])
    point_qc = pd.read_csv(frozen_files["point_qc"])
    points["fold"] = fold_for_points(points, folds, block_size)
    manifest_path = local / "authoritative_manifest.csv"
    records = {record.cube_id: record for record in load_records(args.paths, manifest_path)}
    bands = load_band_indices(args.bands, probability_config["reference_band_section"])
    features, targets, reference_cubes = reconstruct_frozen_features(records, points, point_qc, bands)
    usable_ids = set(point_qc.loc[point_qc.usable_reference.astype(str).str.lower().eq("true"), "annotation_id"])
    usable_points = points[points.annotation_id.isin(usable_ids)].copy()
    if len(usable_points) != len(features):
        raise ValueError("Usable point order/count changed")
    usable_folds = usable_points.fold.to_numpy()

    separators = {}
    reference_rows = []
    for outer, inner in inner_by_outer.items():
        fit = ~np.isin(usable_folds, [outer, inner])
        if fit.sum() < 20 or len(np.unique(targets[fit])) != 2:
            raise ValueError(f"Outer fold {outer} lacks sufficient fitting references")
        scaler, model = fit_separator(features[fit], targets[fit], probability_config["logistic_regularization_c"],
                                      probability_config["maximum_iterations"], probability_config["seed"])
        separators[outer] = (scaler, model)
        for point in usable_points.loc[fit].itertuples():
            reference_rows.append({"outer_fold": outer, "inner_validation_fold": inner,
                                   "annotation_id": point.annotation_id, "cube_id": point.cube_id,
                                   "reference_spatial_fold": int(point.fold), "kind": point.kind,
                                   "use": "separator_fitting"})
    reference_use = pd.DataFrame(reference_rows)

    role_manifest_path = local / "reports/mask_refinement/polygon_guided_candidate_materialization/polygon_guided_candidate_summary.csv"
    role_manifest = pd.read_csv(role_manifest_path).set_index("cube_id")
    candidate_root = project / "data/processed/polygon_guided_mask_candidates"
    output_root = project / config["output_root"]
    report_root = project / config["report_root"]
    contract_root = project / config["contract_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    contract_root.mkdir(parents=True, exist_ok=True)
    reference_path = contract_root / "fold_specific_reference_use.csv"
    reference_use.to_csv(reference_path, index=False)

    alley_path = local / "annotations/planter_tracks/field1_planter_track_annotations.geojson"
    alleys = planter_alley_features(alley_path, config["geometry"]["annotation_seam_snap_m"])
    sensitivity = set(config["cube_roles"]["sensitivity_only"])
    expansion = set(config["cube_roles"]["expansion_only"])
    warning = set(config["cube_roles"]["transfer_warning_dense_sensitivity_only"])
    expected_roles = {"field1_cube24", "field1_cube28"} == sensitivity and {"field1_cube12", "field1_cube14", "field1_cube15"} == expansion and warning == {"field1_cube32"}
    if not expected_roles:
        raise ValueError("Configured cube roles disagree with frozen investigator decisions")

    summary_rows, provenance_rows, group_rows, conflict_rows, buffer_rows = [], [], [], [], []
    output_paths = [reference_path]
    ordered_cubes = sorted(role_manifest.index, key=lambda value: int(value.split("cube")[-1]))
    for cube_number, cube_id in enumerate(ordered_cubes, 1):
        record = records[cube_id]
        candidate_path = candidate_root / cube_id / "polygon_guided_candidate_support.tif"
        with rasterio.open(candidate_path) as dataset:
            candidate = dataset.read(1)
            profile = dataset.profile.copy()
            transform, crs = dataset.transform, dataset.crs
        image = envi.open(str(record.header), str(record.data))
        cube = image.open_memmap()
        valid = raster_valid_spectra(cube, bands)
        polygon_core = np.isin(candidate, [1, 2]) & valid
        polygon_edge = np.isin(candidate, [3, 4]) & valid
        alley_full, alley_core = rasterize_alleys(
            alleys.get(cube_id, []), valid.shape, transform,
            float(config["geometry"]["alley_inward_buffer_m"]),
        )
        alley_full &= valid; alley_core &= valid

        authoritative_soil = record.soil_mask is not None
        if authoritative_soil:
            with rasterio.open(record.soil_mask) as dataset:
                soil = dataset.read(1) > 0
            soil_valid = valid.copy()
        elif cube_id in expansion:
            soil = np.isin(candidate, [1, 3])
            soil_valid = np.isin(candidate, [1, 2, 3, 4]) & valid
        else:
            raise ValueError(f"No approved soil evidence for {cube_id}")

        coordinates = np.argwhere((candidate == 2) & valid).astype(np.int32)
        if authoritative_soil:
            spatially_allowed = valid
        else:
            spatially_allowed = np.isin(candidate, [1, 2, 3, 4]) & valid
        offsets = map_disk_offsets(transform, probability_config["neighbourhood_radius_m"])
        feature_chunks = []
        chunk_size = int(probability_config["prediction_chunk_size"])
        for start in range(0, len(coordinates), chunk_size):
            chunk_features, _ = dense_neighbourhood_features(
                cube, coordinates[start:start + chunk_size], bands, soil,
                spatially_allowed, offsets, probability_config["minimum_usable_pixels"],
            )
            feature_chunks.append(chunk_features)
        dense_features = np.concatenate(feature_chunks) if feature_chunks else np.empty((0, len(bands)), np.float32)
        feature_valid = np.isfinite(dense_features).all(axis=1)
        label_by_outer, source_by_outer, conflict_by_outer = {}, {}, {}
        for outer, (scaler, model) in separators.items():
            probability = np.full(valid.shape, np.nan, np.float32)
            scores = np.full(len(coordinates), np.nan, np.float32)
            if feature_valid.any():
                scores[feature_valid] = model.predict_proba(scaler.transform(dense_features[feature_valid]))[:, 1]
            probability[coordinates[:, 0], coordinates[:, 1]] = scores
            labels, provenance, conflicts = compose_confident_labels(
                valid=valid, polygon_core=polygon_core, polygon_edge=polygon_edge,
                alley_full=alley_full, alley_core=alley_core, soil=soil,
                soil_evidence_valid=soil_valid, probability=probability,
                authoritative_soil=authoritative_soil,
                weed_maximum_probability=probability_config["weed_maximum"],
                chickpea_minimum_probability=probability_config["chickpea_minimum"],
                allow_dense_probability=cube_id not in warning, enrich_alleys=True,
            )
            # Sparse Cube 32 points remain traceable; dense separator decisions do not.
            point_conflicts = []
            if cube_id in warning:
                cube_points = usable_points[usable_points.cube_id == cube_id]
                point_conflicts = apply_confirmed_points(
                    labels, provenance,
                    [(int(row.original_row), int(row.original_column), 1 if row.kind == "confirmed_chickpea" else 2)
                     for row in cube_points.itertuples()],
                    valid=valid, alley_full=alley_full, soil=soil,
                )
            label_by_outer[outer], source_by_outer[outer], conflict_by_outer[outer] = labels, provenance, conflicts
            cube_dir = output_root / f"outer_fold{outer}" / cube_id
            label_path = cube_dir / "confident_labels.tif"
            source_path = cube_dir / "provenance.tif"
            unresolved_path = cube_dir / "unresolved.tif"
            write_raster(label_path, labels, profile, 255, {"version": config["version"], "outer_fold": outer,
                         "class_legend": json.dumps(CLASS_NAMES), "candidate_not_authoritative": True})
            write_raster(source_path, provenance, profile, 0, {"provenance_legend": json.dumps(PROVENANCE), "outer_fold": outer})
            write_raster(unresolved_path, (labels == UNRESOLVED).astype(np.uint8), profile, 255,
                         {"meaning": "1=unresolved,0=resolved"})
            output_paths.extend([label_path, source_path, unresolved_path])
            for item in point_conflicts:
                conflict_rows.append({"cube_id": cube_id, "outer_fold": outer, "conflict_type": item["reason"],
                                      "pixels": 1, "row": item["row"], "column": item["column"]})
            for bit, name in ((1, "probable_chickpea_inside_alley"), (2, "soil_probability_disagreement"),
                              (4, "alley_core_missing_soil_evidence")):
                conflict_rows.append({"cube_id": cube_id, "outer_fold": outer, "conflict_type": name,
                                      "pixels": int(((conflicts & bit) > 0).sum()), "row": "", "column": ""})

        fold_raster, bx, by = raster_folds(valid.shape, transform, fold_lookup, block_size)
        canonical = np.full(valid.shape, UNRESOLVED, np.uint8)
        canonical_source = np.zeros(valid.shape, np.uint8)
        for outer in separators:
            choose = fold_raster == outer
            canonical[choose] = label_by_outer[outer][choose]
            canonical_source[choose] = source_by_outer[outer][choose]
        if np.any((fold_raster == 0) & np.logical_or.reduce([values != UNRESOLVED for values in label_by_outer.values()])):
            raise ValueError(f"Confident labels outside frozen spatial groups: {cube_id}")
        canonical_dir = output_root / "canonical_outer_evaluation" / cube_id
        canonical_path = canonical_dir / "confident_labels.tif"
        canonical_source_path = canonical_dir / "provenance.tif"
        unresolved_path = canonical_dir / "unresolved.tif"
        write_raster(canonical_path, canonical, profile, 255, {"version": config["version"],
                     "semantics": "each pixel labeled by its own outer-fold separator", "candidate_not_authoritative": True})
        write_raster(canonical_source_path, canonical_source, profile, 0, {"provenance_legend": json.dumps(PROVENANCE)})
        write_raster(unresolved_path, (canonical == UNRESOLVED).astype(np.uint8), profile, 255,
                     {"meaning": "1=unresolved,0=resolved"})
        output_paths.extend([canonical_path, canonical_source_path, unresolved_path])

        historical = authoritative_class_map(record) if record.chickpea_mask is not None else None
        removed = int(((historical == 1) & alley_full).sum()) if historical is not None else 0
        role = ("sensitivity_only" if cube_id in sensitivity else "expansion_only" if cube_id in expansion
                else "transfer_warning_sparse_points_only" if cube_id in warning else "primary")
        for class_id, class_name in CLASS_NAMES.items():
            summary_rows.append({"cube_id": cube_id, "dataset_role": role, "class_id": class_id,
                                 "class_name": class_name, "pixels": int((canonical == class_id).sum()),
                                 "historical_chickpea_removed_by_alley": removed if class_id == 1 else 0})
        for source_code, source_name in PROVENANCE.items():
            provenance_rows.append({"cube_id": cube_id, "dataset_role": role, "provenance_code": source_code,
                                    "provenance": source_name, "pixels": int((canonical_source == source_code).sum())})
        selected = canonical != UNRESOLVED
        rows_, columns_ = np.nonzero(selected)
        if len(rows_):
            blocks_x = bx[columns_]; blocks_y = by[rows_]
            frame = pd.DataFrame({"block_x": blocks_x, "block_y": blocks_y,
                                  "fold": fold_raster[rows_, columns_], "class_id": canonical[rows_, columns_],
                                  "provenance_code": canonical_source[rows_, columns_]})
            grouped = frame.groupby(["block_x", "block_y", "fold", "class_id", "provenance_code"]).size()
            for keys, count in grouped.items():
                one_x, one_y, fold, class_id, source_code = keys
                group_rows.append({"cube_id": cube_id, "dataset_role": role,
                                   "spatial_group_id": spatial_group_id(crs.to_string(), int(one_x), int(one_y)),
                                   "fold": int(fold), "class_id": int(class_id),
                                   "class_name": CLASS_NAMES[int(class_id)], "provenance_code": int(source_code),
                                   "provenance": PROVENANCE[int(source_code)], "pixels": int(count)})
        for inward in config["geometry"]["alley_inward_buffer_sensitivity_m"]:
            _, one_core = rasterize_alleys(alleys.get(cube_id, []), valid.shape, transform, float(inward))
            buffer_rows.append({"cube_id": cube_id, "inward_buffer_m": float(inward),
                                "full_alley_pixels": int(alley_full.sum()), "core_pixels": int((one_core & valid).sum()),
                                "core_soil_pixels": int((one_core & valid & soil & soil_valid).sum()),
                                "core_weed_pixels": int((one_core & valid & ~soil & soil_valid).sum())})
        visual_path = report_root / "overlays" / f"{cube_id}_confident_label_overlay.png"
        preview(visual_path, canonical, canonical_source, alley_full, polygon_core | polygon_edge, historical,
                f"{cube_id} | {role} | canonical leakage-safe labels")
        output_paths.append(visual_path)
        print(f"[{cube_number}/{len(ordered_cubes)}] {cube_id}: resolved={int(selected.sum()):,}; "
              f"soil={int((canonical == 0).sum()):,}; chickpea={int((canonical == 1).sum()):,}; "
              f"weed={int((canonical == 2).sum()):,}; removed historical chickpea={removed:,}", flush=True)

    tables = {
        "class_totals_by_cube.csv": pd.DataFrame(summary_rows),
        "totals_by_provenance.csv": pd.DataFrame(provenance_rows),
        "totals_by_spatial_group_fold_provenance.csv": pd.DataFrame(group_rows),
        "conflicts.csv": pd.DataFrame(conflict_rows),
        "alley_inward_buffer_sensitivity.csv": pd.DataFrame(buffer_rows),
    }
    for name, frame in tables.items():
        path = report_root / name; frame.to_csv(path, index=False); output_paths.append(path)
    totals = pd.DataFrame(summary_rows).groupby(["dataset_role", "class_name"]).pixels.sum().reset_index()
    totals_path = report_root / "class_totals_by_dataset_role.csv"; totals.to_csv(totals_path, index=False); output_paths.append(totals_path)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, text=True, capture_output=True, check=True).stdout.strip()
    annotation_paths = [
        local / "annotations/chickpea_regions/field1_chickpea_region_annotations.json",
        local / "annotations/chickpea_regions/field1_chickpea_region_vertices.csv",
        local / "annotations/chickpea_regions/field1_chickpea_region_annotations.geojson",
        local / "annotations/planter_tracks/field1_planter_track_annotations.json",
        local / "annotations/planter_tracks/field1_planter_track_vertices.csv",
        alley_path,
    ]
    contract = {
        "status": "candidate_pending_validation_gates", "version": config["version"], "field": "Field 1",
        "field2_accessed": False, "authoritative_masks_modified": False,
        "class_mapping": config["class_mapping"], "provenance_codes": PROVENANCE,
        "thresholds": probability_config, "precedence": ["nodata", "full_alley_chickpea_prohibition",
            "authoritative_soil", "validated_expansion_soil", "alley_core", "polygon_core_probability",
            "polygon_edge_unresolved", "unresolved_conflict"],
        "geometry": config["geometry"], "cube_roles": config["cube_roles"],
        "fold_specific_separator": {"outer_and_inner_references_excluded": True,
                                     "reference_use_table": str(reference_path.relative_to(project))},
        "code_commit_at_generation": commit,
        "input_hashes": {
            str(project_path(project, path).relative_to(project)): sha256(project_path(project, path))
            for path in [manifest_path, args.config, args.bands, *source_contract_paths, *annotation_paths]
        },
        "output_hashes": {str(path.relative_to(project)): sha256(path) for path in output_paths},
    }
    contract_path = contract_root / "field1_confident_label_candidate_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Candidate contract: {contract_path}")
    print("Status remains candidate_pending_validation_gates; no source mask was changed.")


if __name__ == "__main__":
    main()
