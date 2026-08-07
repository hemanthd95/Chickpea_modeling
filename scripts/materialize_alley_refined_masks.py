#!/usr/bin/env python
"""Materialize reversible Field 1 mask candidates from investigator alleys.

Only current chickpea pixels inside investigator-drawn alley footprints are
reclassified to weed. Tyre-track footprints remain geometry evidence and do
not alter a class. Source and authoritative masks are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
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
import yaml

from audit_planter_track_annotations import polygon_geometry, reconcile_exports
from chickpea_ssl.data import authoritative_class_map, load_records


REVIEW_CMAP = ListedColormap(["#000000", "#20C76F", "#F59E0B", "#E83E8C"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_raster(path: Path, array: np.ndarray, profile: dict, tags: dict[str, str]) -> None:
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=0,
        compress="deflate", predictor=2,
    )
    for key in ("interleave", "blockxsize", "blockysize", "tiled"):
        output_profile.pop(key, None)
    with rasterio.open(path, "w", **output_profile) as dataset:
        dataset.write(array.astype(np.uint8), 1)
        dataset.update_tags(**tags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--decisions", default=Path("configs/data_decisions.yaml"), type=Path)
    parser.add_argument("--annotations-root", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    decisions = yaml.safe_load(args.decisions.read_text())
    if not decisions["rules"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked during Field 1 mask refinement")
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "planter_tracks"
    json_path = annotation_root / "field1_planter_track_annotations.json"
    csv_path = annotation_root / "field1_planter_track_vertices.csv"
    geojson_path = annotation_root / "field1_planter_track_annotations.geojson"
    annotations, geojson = reconcile_exports(json_path, csv_path, geojson_path)

    alleys_by_cube: dict[str, list[dict[str, object]]] = defaultdict(list)
    for feature in geojson["features"]:
        properties = feature["properties"]
        if properties["kind"] != "alley_boundary":
            continue
        alleys_by_cube[str(properties["cube_id"])].append(
            polygon_geometry(feature["geometry"]["coordinates"])
        )

    manifest = local / "authoritative_manifest.csv"
    records = sorted(
        [
            record for record in load_records(args.paths, manifest)
            if record.chickpea_mask and record.cube_id in alleys_by_cube
        ],
        key=lambda record: int(record.cube_id.split("cube")[-1]),
    )
    if not records:
        raise ValueError("No authoritative chickpea cubes have alley annotations")

    candidates_root = project / "data" / "processed" / "chickpea_alley_refinement_candidates"
    if candidates_root.exists() and any(candidates_root.iterdir()):
        raise SystemExit(
            f"Candidate destination is not empty: {candidates_root}\n"
            "No file was overwritten. Move the existing candidate directory after review."
        )
    candidates_root.mkdir(parents=True, exist_ok=True)
    reports = local / "reports" / "mask_refinement" / "alley_refined_candidates"
    individual = reports / "individual_cubes"
    individual.mkdir(parents=True, exist_ok=True)
    sensitivity_only = set(decisions.get("investigator_phenology_exclusions", {}))

    summary_rows: list[dict[str, object]] = []
    output_paths: list[Path] = []
    overview_tiles: list[tuple[str, np.ndarray, int, float, str]] = []
    total_reclassified = 0
    total_source_chickpea = 0

    for number, record in enumerate(records, start=1):
        labels = authoritative_class_map(record)
        source_chickpea = labels == 1
        source_weed = labels == 2
        source_soil = labels == 0
        with rasterio.open(record.data) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            profile = reference.profile
        if labels.shape != shape:
            raise ValueError(f"Mask/reference shape mismatch for {record.cube_id}")

        alley_mask = rasterize(
            ((geometry, 1) for geometry in alleys_by_cube[record.cube_id]),
            out_shape=shape, transform=transform, fill=0,
            dtype="uint8", all_touched=True,
        ).astype(bool)
        reclassified = source_chickpea & alley_mask
        refined_chickpea = source_chickpea & ~alley_mask
        refined_weed = source_weed | reclassified
        refined_soil = source_soil.copy()
        if np.any(refined_chickpea & refined_weed) or np.any(refined_chickpea & refined_soil) or np.any(refined_weed & refined_soil):
            raise RuntimeError(f"Refined masks overlap for {record.cube_id}")
        if not np.array_equal(refined_soil, source_soil):
            raise RuntimeError(f"Soil changed unexpectedly for {record.cube_id}")
        if np.any(refined_chickpea & ~source_chickpea):
            raise RuntimeError(f"Refinement added chickpea pixels for {record.cube_id}")
        if not np.array_equal(refined_weed & ~source_weed, reclassified):
            raise RuntimeError(f"Unexpected weed additions for {record.cube_id}")

        class_map = np.zeros(shape, dtype=np.uint8)
        class_map[refined_soil] = 1
        class_map[refined_chickpea] = 2
        class_map[refined_weed] = 3
        role = "sensitivity_only" if record.cube_id in sensitivity_only else "primary_candidate"
        destination = candidates_root / record.cube_id
        destination.mkdir(parents=True, exist_ok=False)
        tags = {
            "cube_id": record.cube_id,
            "candidate_status": "investigator_review_required",
            "analysis_role": role,
            "transformation": "current chickpea inside investigator alley footprint -> weed",
            "tyre_track_policy": "no class change",
            "source_annotations": str(geojson_path),
            "label_mapping": "0=unlabeled,1=soil,2=chickpea,3=weed",
        }
        products = {
            "class_map": class_map,
            "soil_mask": refined_soil.astype(np.uint8),
            "chickpea_mask": refined_chickpea.astype(np.uint8),
            "weed_mask": refined_weed.astype(np.uint8),
            "reclassified_chickpea_to_weed": reclassified.astype(np.uint8),
            "investigator_alley_footprint": alley_mask.astype(np.uint8),
        }
        for name, array in products.items():
            path = destination / f"{name}.tif"
            write_raster(path, array, profile, tags)
            output_paths.append(path)

        removed = int(reclassified.sum())
        source_count = int(source_chickpea.sum())
        fraction = removed / source_count if source_count else 0.0
        total_reclassified += removed if role == "primary_candidate" else 0
        total_source_chickpea += source_count if role == "primary_candidate" else 0
        summary_rows.append({
            "cube_id": record.cube_id,
            "analysis_role": role,
            "alley_annotations": len(alleys_by_cube[record.cube_id]),
            "source_chickpea_pixels": source_count,
            "refined_chickpea_pixels": int(refined_chickpea.sum()),
            "chickpea_pixels_reclassified_to_weed": removed,
            "chickpea_reclassified_fraction": fraction,
            "source_weed_pixels": int(source_weed.sum()),
            "refined_weed_pixels": int(refined_weed.sum()),
            "soil_pixels_unchanged": int(refined_soil.sum()),
            "candidate_directory": str(destination.relative_to(project)),
        })

        review = np.zeros(shape, dtype=np.uint8)
        review[refined_chickpea] = 1
        review[reclassified] = 2
        step = max(1, math.ceil(max(shape) / 500))
        overview_tiles.append((record.cube_id, review[::step, ::step], removed, fraction, role))

        image_path = individual / f"{record.cube_id}_alley_refinement_candidate.png"
        figure, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
        axes[0].imshow(source_chickpea, cmap=ListedColormap(["#000000", "#20C76F"]), vmin=0, vmax=1)
        axes[0].set_title(f"Current chickpea\n{source_count:,} pixels")
        axes[1].imshow(review, cmap=REVIEW_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[1].set_title(f"Candidate: orange → weed\n{removed:,} pixels ({fraction:.2%})")
        axes[2].imshow(refined_chickpea, cmap=ListedColormap(["#000000", "#20C76F"]), vmin=0, vmax=1)
        axes[2].set_title(f"Refined chickpea\n{int(refined_chickpea.sum()):,} pixels")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{record.cube_id} investigator-alley candidate — {role}\n"
            "Source masks remain unchanged",
            fontsize=15,
        )
        figure.savefig(image_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(image_path)
        print(
            f"Materialized {number}/{len(records)} {record.cube_id}: "
            f"{removed:,} chickpea pixels proposed as weed ({fraction:.2%}) [{role}]",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "alley_refined_mask_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    output_paths.append(summary_path)

    columns = 4
    rows = math.ceil(len(overview_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.8 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, tile, removed, fraction, role) in zip(flat, overview_tiles):
        axis.imshow(tile, cmap=REVIEW_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(
            f"{cube_id} | proposed weed {removed:,} ({fraction:.1%})\n{role}",
            fontsize=10,
        )
    figure.suptitle(
        "Field 1 reversible investigator-alley mask candidates\n"
        "green=retained chickpea; orange=chickpea proposed for weed; tyre tracks make no class change",
        fontsize=16,
    )
    overview_path = reports / "alley_refined_mask_candidate_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white")
    plt.close(figure)
    output_paths.append(overview_path)

    contract = {
        "status": "reversible_alley_refined_mask_candidates_materialized",
        "field": "Field 1",
        "field2_accessed": False,
        "source_masks_modified": False,
        "authoritative_manifest_modified": False,
        "model_retrained": False,
        "rule": "authoritative chickpea inside investigator alley footprint becomes weed",
        "tyre_track_policy": "geometry evidence only; no automatic class change",
        "primary_candidate_cubes": [
            row["cube_id"] for row in summary_rows if row["analysis_role"] == "primary_candidate"
        ],
        "sensitivity_only_cubes": sorted(sensitivity_only),
        "primary_source_chickpea_pixels": total_source_chickpea,
        "primary_reclassified_pixels": total_reclassified,
        "primary_reclassified_fraction": (
            total_reclassified / total_source_chickpea if total_source_chickpea else 0.0
        ),
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "authoritative_manifest": sha256(manifest),
        },
        "output_hashes": {
            str(path.relative_to(project)): sha256(path) for path in output_paths
        },
        "notes": [
            "All products are review candidates and are not referenced by the authoritative manifest.",
            "Cubes 24 and 28 are materialized for documented sensitivity review only.",
            "No synthetic scientific observations were created.",
        ],
    }
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract_path = contracts / "field1_alley_refined_mask_candidate_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Candidate root: {candidates_root}")
    print(f"Summary: {summary_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print(
        f"Primary candidates: {total_reclassified:,}/{total_source_chickpea:,} "
        f"chickpea pixels proposed as weed "
        f"({total_reclassified / total_source_chickpea:.2%})."
    )
    print("Candidates only: authoritative masks and manifest remain unchanged; Field 2 stayed locked.")


if __name__ == "__main__":
    main()
