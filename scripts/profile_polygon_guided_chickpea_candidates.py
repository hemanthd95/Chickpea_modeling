#!/usr/bin/env python
"""Profile polygon-first chickpea candidates using observed NDVI and spectra.

Investigator row polygons are the primary spatial authority.  The historical
chickpea mask is comparison-only.  This read-only stage removes internal NoData,
identifies soil with the predeclared NDVI threshold, applies explicit alley
precedence, and profiles real spectral features before any weed/chickpea
separator is frozen or any replacement mask is written.
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
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
from spectral.io import envi
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records
from scripts.audit_chickpea_region_annotations import (
    disk,
    metres_per_pixel,
    normalized_ring,
    observed_support,
    planter_alley_features,
    reconcile_exports,
)


MAP_CMAP = ListedColormap([
    "#000000",  # background / outside prior
    "#8B5A2B",  # polygon soil by NDVI
    "#20C76F",  # vegetation in polygon core
    "#A7F3D0",  # vegetation in outward uncertainty belt
    "#F59E0B",  # investigator alley exclusion
    "#D946EF",  # historical chickpea outside polygon prior (comparison only)
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.nanargmin(np.abs(wavelengths - target_nm)))


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    usable = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    result[usable] = numerator[usable] / denominator[usable]
    return result


def feature_summary(
    cube_id: str,
    role: str,
    zone: str,
    mask: np.ndarray,
    features: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for name, values in features.items():
        sample = values[mask & np.isfinite(values)]
        if sample.size:
            quantiles = np.quantile(sample, [0.05, 0.25, 0.50, 0.75, 0.95])
            mean, standard_deviation = float(sample.mean()), float(sample.std())
        else:
            quantiles = [np.nan] * 5
            mean = standard_deviation = np.nan
        rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "zone": zone,
            "feature": name,
            "pixels": int(sample.size),
            "mean": mean,
            "standard_deviation": standard_deviation,
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
        })
    return rows


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
    policy = config.get("polygon_guided_candidate_profile", {})
    edge_tolerance = float(policy.get("outward_uncertainty_tolerance_m", 0.05))
    soil_ndvi_threshold = float(policy.get("soil_ndvi_threshold", 0.30))
    seam_snap = float(config.get("chickpea_region_polygon_audit", {}).get(
        "seam_snap_maximum_m", 0.30
    ))

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "chickpea_regions"
    json_path = annotation_root / "field1_chickpea_region_annotations.json"
    csv_path = annotation_root / "field1_chickpea_region_vertices.csv"
    geojson_path = annotation_root / "field1_chickpea_region_annotations.geojson"
    _, geojson, _ = reconcile_exports(json_path, csv_path, geojson_path)

    polygons_by_cube: dict[str, list[dict]] = defaultdict(list)
    for feature in geojson["features"]:
        properties = feature["properties"]
        coordinates = feature["geometry"]["coordinates"][0]
        if coordinates and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        ring, _, _ = normalized_ring(coordinates, seam_snap)
        polygons_by_cube[str(properties["cube_id"])].append(
            {"type": "Polygon", "coordinates": [ring]}
        )

    alley_path = (
        local / "annotations" / "planter_tracks"
        / "field1_planter_track_annotations.geojson"
    )
    alleys_by_cube = planter_alley_features(alley_path, seam_snap)
    layer_manifest_path = (
        local / "reports" / "chickpea_region_annotation"
        / "chickpea_region_annotation_layer_manifest.csv"
    )
    layer_manifest = pd.read_csv(layer_manifest_path).set_index("cube_id")
    records = {
        record.cube_id: record for record in load_records(
            args.paths, local / "authoritative_manifest.csv"
        ) if record.cube_id in polygons_by_cube
    }
    missing = sorted(set(polygons_by_cube) - set(records))
    if missing:
        raise ValueError(f"Annotated cubes are absent from the manifest: {missing}")

    reports = local / "reports" / "mask_refinement" / "polygon_guided_candidate_profile"
    individual = reports / "individual_cubes"
    individual.mkdir(parents=True, exist_ok=True)
    support_rows = []
    spectral_rows = []
    overview = []
    output_paths: list[Path] = []

    for number, cube_id in enumerate(
        sorted(records, key=lambda item: int(item.split("cube")[-1])), start=1
    ):
        record = records[cube_id]
        role = str(layer_manifest.loc[cube_id, "analysis_role"])
        image = envi.open(str(record.header), str(record.data))
        wavelengths = np.asarray(image.metadata["wavelength"], dtype=np.float32)
        green_index = nearest_band(wavelengths, 550.0)
        red_index = nearest_band(wavelengths, 670.0)
        red_edge_index = nearest_band(wavelengths, 720.0)
        nir_index = nearest_band(wavelengths, 800.0)

        with rasterio.open(record.data) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            gsd = metres_per_pixel(transform)
            observed = observed_support(reference)
            selected = reference.read([
                green_index + 1, red_index + 1, red_edge_index + 1, nir_index + 1
            ]).astype(np.float32)
        green, red, red_edge, nir = selected
        selected[:, ~observed] = np.nan
        denominator = nir + red
        ndvi = safe_ratio(nir - red, denominator)
        features = {
            "ndvi": ndvi,
            "green_stored_value": green,
            "red_stored_value": red,
            "red_edge_stored_value": red_edge,
            "nir_stored_value": nir,
            "nir_to_green_ratio": safe_ratio(nir, green),
            "green_to_red_ratio": safe_ratio(green, red),
            "red_edge_to_red_ratio": safe_ratio(red_edge, red),
            "nir_minus_red_edge": nir - red_edge,
        }

        core_raw = rasterize(
            [(geometry, 1) for geometry in polygons_by_cube[cube_id]],
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
            all_touched=False,
        ).astype(bool)
        radius = int(math.ceil(edge_tolerance / gsd))
        tolerant = binary_dilation(core_raw, structure=disk(radius)) & observed
        core = core_raw & observed
        edge = tolerant & ~core
        alley_shapes = [(geometry, 1) for geometry in alleys_by_cube.get(cube_id, [])]
        alley = (
            rasterize(
                alley_shapes, out_shape=shape, transform=transform, fill=0,
                dtype="uint8", all_touched=True,
            ).astype(bool)
            if alley_shapes else np.zeros(shape, dtype=bool)
        )
        eligible_core = core & ~alley
        eligible_edge = edge & ~alley
        core_soil = eligible_core & np.isfinite(ndvi) & (ndvi < soil_ndvi_threshold)
        core_vegetation = eligible_core & np.isfinite(ndvi) & (ndvi >= soil_ndvi_threshold)
        edge_soil = eligible_edge & np.isfinite(ndvi) & (ndvi < soil_ndvi_threshold)
        edge_vegetation = eligible_edge & np.isfinite(ndvi) & (ndvi >= soil_ndvi_threshold)
        outside_vegetation = (
            observed & ~tolerant & np.isfinite(ndvi) & (ndvi >= soil_ndvi_threshold)
        )
        labels = authoritative_class_map(record) if record.chickpea_mask else None
        historical = labels == 1 if labels is not None else np.zeros(shape, dtype=bool)

        zones = {
            "polygon_core_vegetation": core_vegetation,
            "polygon_edge_vegetation": edge_vegetation,
            "outside_polygon_vegetation": outside_vegetation,
        }
        if labels is not None:
            zones.update({
                "historical_chickpea_in_core_vegetation": core_vegetation & historical,
                "historical_nonchickpea_in_core_vegetation": core_vegetation & ~historical,
            })
        for zone, mask in zones.items():
            spectral_rows.extend(feature_summary(cube_id, role, zone, mask, features))

        support_rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "polygons": len(polygons_by_cube[cube_id]),
            "gsd_m": gsd,
            "edge_tolerance_m": edge_tolerance,
            "edge_tolerance_pixels": radius,
            "observed_pixels": int(observed.sum()),
            "polygon_core_observed_pixels": int(core.sum()),
            "polygon_core_alley_excluded_pixels": int((core & alley).sum()),
            "polygon_core_soil_pixels_ndvi_below_threshold": int(core_soil.sum()),
            "polygon_core_vegetation_pixels_ndvi_at_or_above_threshold": int(core_vegetation.sum()),
            "polygon_core_soil_fraction": (
                float(core_soil.sum() / eligible_core.sum()) if eligible_core.any() else np.nan
            ),
            "edge_vegetation_uncertain_pixels": int(edge_vegetation.sum()),
            "edge_soil_pixels": int(edge_soil.sum()),
            "historical_chickpea_pixels_comparison_only": int(historical.sum()),
            "historical_chickpea_inside_core_vegetation": int((historical & core_vegetation).sum()),
            "historical_chickpea_outside_tolerant_prior": int((historical & ~tolerant).sum()),
        })

        qc = np.zeros(shape, dtype=np.uint8)
        qc[core_soil] = 1
        qc[core_vegetation] = 2
        qc[edge_vegetation] = 3
        qc[tolerant & alley] = 4
        qc[historical & ~tolerant] = 5
        step = max(1, math.ceil(max(shape) / 500))
        tile = qc[::step, ::step]
        overview.append((cube_id, role, tile, int(core_vegetation.sum()), int(core_soil.sum())))

        image_path = individual / f"{cube_id}_polygon_guided_candidate_profile.png"
        figure, axis = plt.subplots(figsize=(6, 8), constrained_layout=True)
        axis.imshow(tile, cmap=MAP_CMAP, vmin=0, vmax=5, interpolation="nearest")
        axis.axis("off")
        axis.set_title(
            f"{cube_id} — {role}\ncore vegetation={int(core_vegetation.sum()):,}; "
            f"core soil={int(core_soil.sum()):,}", fontsize=12,
        )
        figure.savefig(image_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(image_path)
        print(
            f"Profiled {number}/{len(records)} {cube_id}: "
            f"core vegetation={int(core_vegetation.sum()):,}; "
            f"core soil={int(core_soil.sum()):,}; edge uncertain={int(edge_vegetation.sum()):,}",
            flush=True,
        )

    support = pd.DataFrame(support_rows)
    support_path = reports / "polygon_guided_pixel_support.csv"
    spectral_path = reports / "polygon_guided_spectral_profile.csv"
    support.to_csv(support_path, index=False)
    pd.DataFrame(spectral_rows).to_csv(spectral_path, index=False)
    output_paths.extend([support_path, spectral_path])

    columns = 4
    rows = math.ceil(len(overview) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, role, tile, vegetation, soil) in zip(flat, overview):
        axis.imshow(tile, cmap=MAP_CMAP, vmin=0, vmax=5, interpolation="nearest")
        axis.set_title(
            f"{cube_id} | {role}\ncore vegetation {vegetation:,}; soil {soil:,}", fontsize=9
        )
    figure.suptitle(
        "Field 1 polygon-first observed candidate profile\n"
        "green=core vegetation; pale green=5 cm uncertain edge; brown=NDVI soil; "
        "orange=alley; magenta=historical mask outside prior (comparison only)",
        fontsize=15,
    )
    overview_path = reports / "polygon_guided_candidate_profile_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white")
    plt.close(figure)
    output_paths.append(overview_path)

    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "polygon_first_candidate_profile_complete_no_mask_change",
        "field": "Field 1",
        "spatial_authority": "investigator_chickpea_region_polygons",
        "historical_chickpea_mask_role": "comparison_only_not_a_veto",
        "soil_rule": f"observed NDVI < {soil_ndvi_threshold}",
        "polygon_core_vegetation_status": "candidate_pending_spectral_weed_separation",
        "outward_edge_belt_status": "uncertain_pending_stronger_spectral_evidence",
        "outward_edge_tolerance_m": edge_tolerance,
        "explicit_alley_precedence": True,
        "tyre_tracks_change_labels": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "planter_annotations_geojson": sha256(alley_path),
            "layer_manifest": sha256(layer_manifest_path),
        },
        "output_hashes": {
            str(path.relative_to(reports)): sha256(path) for path in output_paths
        },
    }
    contract_path = contracts / "field1_polygon_guided_candidate_profile_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Pixel support: {support_path}")
    print(f"Spectral profile: {spectral_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Profile only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
