"""Capacity-aware, prediction-free Field 2 area-stratified sampling v2."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from scipy.spatial import cKDTree
import yaml

from chickpea_ssl.field2_area_operational import PRECEDENCE_RULE, raw_memberships_from_bits
from chickpea_ssl.field2_area_review import ZONE_CODES
from chickpea_ssl.field2_blind_review import (
    _block_balanced_selection,
    atomic_write_bytes,
    atomic_write_yaml,
    minimum_separation_thin,
    require_frozen_role_contract,
    stable_selection_rank,
    validate_frozen_role_contract,
)
from chickpea_ssl.field2_readiness import ReadOnlySourceGuard, open_envi_memmap, sha256
from scripts.build_field2_blind_annotation_sampling_frame import (
    collapsed_cube_candidates,
    finalize_frame,
    nearest_neighbor_summary,
    save_sampling_map,
)
from scripts.validate_field2_area_contract import validate_area_contract


DOMAINS = (
    "research_crop_area",
    "alley",
    "outside_research_field",
    "unassigned_valid_support",
)
DOMAIN_REPORTING = {
    "research_crop_area": "primary_in_field",
    "alley": "supplementary_alley",
    "outside_research_field": "outside_domain_ood",
    "unassigned_valid_support": "separately_reported_unassigned",
}
OUTPUT_COLUMNS = [
    "sample_id", "sampling_frame", "cube_id", "row", "column", "x", "y",
    "longitude", "latitude", "crs", "valid_support_status", "valid_context_status",
    "sampling_domain", "domain_code", "domain_name", "domain_reporting_stratum",
    "primary_external_validation_eligible", "raw_polygon_memberships",
    "precedence_applied", "precedence_rule", "crop_sampling_eligible",
    "spatial_group_id", "geographic_ground_cell_id", "scalar_index_value",
    "scalar_index_rank_stratum", "cube_evaluation_role", "cube_challenge_flags",
    "cube_investigator_notes", "within_cube_ground_cell_pixel_count",
    "cross_cube_ground_cell_view_count", "first_stage_inclusion_probability",
    "sampling_stratum_block_population", "sampling_stratum_block_quota",
    "cube_rank_stratum_requested", "final_inclusion_probability",
    "overall_inclusion_probability", "design_weight", "overlap_status",
    "deterministic_selection_rank", "source_reflectance_sha256", "support_mask_sha256",
    "source_preview_checksums", "v1_ground_cell_exclusion_status", "reserve_release_status",
]


def frame_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO()
    frame[OUTPUT_COLUMNS].to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO()
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode()


def resolve_configs(project: Path, v2_config: dict) -> tuple[dict, dict]:
    base = yaml.safe_load((project / v2_config["base_config"]).read_text())
    area = yaml.safe_load((project / v2_config["area_config"]).read_text())
    return base, area


def _unique_and_thin(candidates: pd.DataFrame, sampling: dict) -> pd.DataFrame:
    work = candidates.copy()
    work["cross_cube_ground_cell_view_count"] = work.groupby(
        "geographic_ground_cell_id"
    )["cube_id"].transform("size").astype(int)
    work["overlap_status"] = np.where(
        work.cross_cube_ground_cell_view_count > 1, "overlapping_view", "unique_view"
    )
    work = work.sort_values([
        "geographic_ground_cell_id", "deterministic_selection_rank", "cube_id",
        "sampling_domain", "row", "column",
    ])
    unique = work.groupby("geographic_ground_cell_id", sort=False).head(1).copy()
    unique.loc[unique.overlap_status == "overlapping_view", "overlap_status"] = "overlapping_view_selected"
    unique["first_stage_inclusion_probability"] = 1.0 / (
        unique.within_cube_ground_cell_pixel_count * unique.cross_cube_ground_cell_view_count
    )
    thinned = minimum_separation_thin(unique, float(sampling["minimum_separation_m"]))
    return thinned.sort_values([
        "cube_id", "sampling_domain", "deterministic_selection_rank"
    ]).reset_index(drop=True)


def build_eligible_candidates(
    project: Path, base: dict, area: dict, v2: dict, role_records: dict[str, dict],
    readiness_roots: Iterable[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build domain candidates from immutable masks and exclude every v1 cell."""
    expected = list(base["expected_cube_ids"])
    sampling = v2["sampling"]
    inventory = pd.read_csv(project / base["inputs"]["readiness_inventory"]).fillna("")
    support_manifest = pd.read_csv(project / base["inputs"]["valid_support_manifest"]).fillna("")
    review_manifest = pd.read_csv(project / base["review"]["package_manifest"]).fillna("")
    area_root = project / area["freeze"]["output_root"]
    area_manifest = pd.read_csv(area_root / area["freeze"]["mask_manifest"]).fillna("")
    v1 = pd.read_csv(project / base["sampling"]["combined_frame"], keep_default_na=False)
    v1_cells = set(v1.geographic_ground_cell_id.astype(str))
    preview_by_cube = review_manifest.set_index("cube_id")
    support_by_cube = support_manifest.set_index("cube_id")
    area_by_cube = area_manifest.set_index("cube_id")
    roots = tuple(Path(value) for value in readiness_roots)
    guard = ReadOnlySourceGuard(roots)
    parts: list[pd.DataFrame] = []
    raw_pixel_rows: list[dict] = []
    for cube_id in expected:
        support_row = support_by_cube.loc[cube_id]
        area_row = area_by_cube.loc[cube_id]
        domain_path = area_root / str(area_row.domain_mask_path)
        raw_path = area_root / str(area_row.raw_membership_mask_path)
        if sha256(domain_path) != str(area_row.domain_mask_sha256):
            raise ValueError(f"Frozen area-domain mask changed: {cube_id}")
        if sha256(raw_path) != str(area_row.raw_membership_mask_sha256):
            raise ValueError(f"Frozen raw-membership mask changed: {cube_id}")
        index_row = inventory[(inventory.cube_id == cube_id) & (inventory.product_type == "stored_index")].iloc[0]
        scalar, _ = open_envi_memmap(
            project / str(index_row.header_path), project / str(index_row.binary_path), guard
        )
        scalar = np.asarray(scalar[..., 0], dtype=np.float32)
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(domain_path, "r") as dataset:
                domain_mask, transform, crs = dataset.read(1), dataset.transform, str(dataset.crs)
            with rasterio.open(raw_path, "r") as dataset:
                raw_mask = dataset.read(1)
        review = preview_by_cube.loc[cube_id]
        crop_eligible = (
            role_records[cube_id]["primary_role"] == "primary_three_class"
            and bool(np.any(domain_mask == ZONE_CODES["research_crop_area"]))
        )
        for domain in DOMAINS:
            domain_code = ZONE_CODES[domain]
            domain_support = domain_mask == domain_code
            raw_pixel_rows.append({
                "cube_id": cube_id,
                "cube_evaluation_role": role_records[cube_id]["primary_role"],
                "sampling_domain": domain,
                "operational_pixel_count": int(domain_support.sum()),
                "crop_sampling_eligible": crop_eligible,
            })
            if not domain_support.any():
                continue
            candidate = collapsed_cube_candidates(
                cube_id, domain_support, scalar, transform, crs, role_records[cube_id],
                str(support_row.source_reflectance_sha256), str(support_row.mask_sha256),
                str(review.preview_sha256_json), int(review.preview_step),
                int(review.preview_width), int(review.preview_height), sampling,
            )
            candidate["sampling_domain"] = domain
            candidate["domain_code"] = domain_code
            candidate["domain_name"] = domain
            candidate["domain_reporting_stratum"] = DOMAIN_REPORTING[domain]
            candidate["primary_external_validation_eligible"] = domain == "research_crop_area"
            candidate["crop_sampling_eligible"] = crop_eligible
            candidate["raw_polygon_memberships"] = [
                "|".join(raw_memberships_from_bits(int(raw_mask[int(row), int(column)])))
                for row, column in zip(candidate.row, candidate.column)
            ]
            candidate["precedence_applied"] = [
                len(value.split("|")) > 1 if value else False
                for value in candidate.raw_polygon_memberships
            ]
            candidate["precedence_rule"] = PRECEDENCE_RULE
            candidate["deterministic_selection_rank"] = [
                stable_selection_rank(
                    int(sampling["seed"]), str(cell), cube_id, domain, int(row), int(column)
                )
                for cell, row, column in zip(
                    candidate.geographic_ground_cell_id, candidate.row, candidate.column
                )
            ]
            candidate["v1_ground_cell_exclusion_status"] = np.where(
                candidate.geographic_ground_cell_id.astype(str).isin(v1_cells),
                "excluded_v1", "not_in_v1",
            )
            parts.append(candidate[candidate.v1_ground_cell_exclusion_status == "not_in_v1"].copy())
    if not parts:
        raise RuntimeError("No v2 domain candidates were available")
    return _unique_and_thin(pd.concat(parts, ignore_index=True), sampling), pd.DataFrame(raw_pixel_rows)


def crop_eligibility_table(
    expected: Iterable[str], roles: dict[str, str], capacity: pd.DataFrame
) -> pd.DataFrame:
    crop = capacity[capacity.sampling_domain == "research_crop_area"].set_index("cube_id")
    rows = []
    for cube_id in expected:
        pixels = int(crop.loc[cube_id, "operational_pixel_count"])
        eligible = roles[cube_id] == "primary_three_class" and pixels > 0
        rows.append({
            "cube_id": cube_id, "frozen_cube_role": roles[cube_id],
            "operational_research_crop_area_pixels": pixels,
            "crop_sampling_eligible": eligible,
            "eligibility_rule": "primary_three_class AND operational research_crop_area support > 0",
        })
    return pd.DataFrame(rows)


def _capacity_order(cubes: Iterable[str], capacities: dict[str, int]) -> list[str]:
    return sorted(cubes, key=lambda cube: (-capacities.get(cube, 0), cube))


def _even_distribution(
    total: int, cubes: list[str], capacities: dict[str, int], roles: dict[str, str] | None = None
) -> dict[str, int]:
    if not cubes and total:
        raise RuntimeError("No eligible cubes for a nonzero domain allocation")
    quotas = {cube: 0 for cube in cubes}
    for _ in range(total):
        choices = [cube for cube in cubes if quotas[cube] < capacities.get(cube, 0)]
        if not choices:
            raise RuntimeError(f"Global domain allocation infeasible: requested={total}, achieved={sum(quotas.values())}")
        if roles:
            role_counts = defaultdict(int)
            role_sizes = defaultdict(int)
            for cube in cubes:
                role_sizes[roles[cube]] += 1
                role_counts[roles[cube]] += quotas[cube]
            choices.sort(key=lambda cube: (
                quotas[cube], role_counts[roles[cube]] / role_sizes[roles[cube]],
                -capacities[cube], cube,
            ))
        else:
            choices.sort(key=lambda cube: (quotas[cube], -capacities[cube], cube))
        quotas[choices[0]] += 1
    return quotas


def initial_quotas(
    expected: list[str], roles: dict[str, str], capacity: pd.DataFrame, v2: dict
) -> dict[str, dict[str, dict[str, int]]]:
    caps = {
        domain: capacity[capacity.sampling_domain == domain].set_index("cube_id")["feasible_capacity"].astype(int).to_dict()
        for domain in DOMAINS
    }
    result = {frame: {domain: {cube: 0 for cube in expected} for domain in DOMAINS} for frame in ("main", "reserve")}
    for frame in ("main", "reserve"):
        crop_cfg = v2["sampling"]["domains"]["research_crop_area"]
        crop_cubes = [cube for cube in expected if roles[cube] == crop_cfg["eligible_role"] and caps["research_crop_area"].get(cube, 0) > 0]
        base, remainder = int(crop_cfg[f"{frame}_base"]), int(crop_cfg[f"{frame}_remainder"])
        for cube in crop_cubes:
            result[frame]["research_crop_area"][cube] = base
        for cube in _capacity_order(crop_cubes, caps["research_crop_area"])[:remainder]:
            result[frame]["research_crop_area"][cube] += 1

        alley_cfg = v2["sampling"]["domains"]["alley"]
        alley_cubes = [cube for cube in expected if caps["alley"].get(cube, 0) > 0]
        base, remainder = int(alley_cfg[f"{frame}_base"]), int(alley_cfg[f"{frame}_remainder"])
        for cube in alley_cubes:
            result[frame]["alley"][cube] = base
        for cube in _capacity_order(alley_cubes, caps["alley"])[:remainder]:
            result[frame]["alley"][cube] += 1

        for domain in ("outside_research_field", "unassigned_valid_support"):
            domain_cfg = v2["sampling"]["domains"][domain]
            guarantees = {cube: int(values[frame]) for cube, values in domain_cfg["guaranteed_cubes"].items()}
            for cube, value in guarantees.items():
                result[frame][domain][cube] = value
            remaining = int(domain_cfg[f"{frame}_total"]) - sum(guarantees.values())
            other = [cube for cube in expected if cube not in guarantees and caps[domain].get(cube, 0) > 0]
            distributed = _even_distribution(
                remaining, other, caps[domain], roles if domain == "unassigned_valid_support" else None
            )
            result[frame][domain].update(distributed)
    return result


def reconcile_quotas(
    expected: list[str], roles: dict[str, str], capacity: pd.DataFrame,
    requested: dict[str, dict[str, dict[str, int]]], v2: dict,
) -> tuple[dict[str, dict[str, dict[str, int]]], pd.DataFrame]:
    caps = {
        domain: capacity[capacity.sampling_domain == domain].set_index("cube_id")["feasible_capacity"].astype(int).to_dict()
        for domain in DOMAINS
    }
    final = {frame: {domain: dict(requested[frame][domain]) for domain in DOMAINS} for frame in ("main", "reserve")}
    audit_rows = []
    used = {domain: {cube: 0 for cube in expected} for domain in DOMAINS}
    for frame in ("main", "reserve"):
        for domain in DOMAINS:
            desired = requested[frame][domain]
            achieved = {cube: min(desired[cube], max(0, caps[domain].get(cube, 0) - used[domain][cube])) for cube in expected}
            deficits = [(cube, desired[cube] - achieved[cube]) for cube in expected if desired[cube] > achieved[cube]]
            for deficient_cube, shortfall in deficits:
                for _ in range(shortfall):
                    choices = [
                        cube for cube in expected
                        if caps[domain].get(cube, 0) - used[domain][cube] - achieved[cube] > 0
                    ]
                    if not choices:
                        raise RuntimeError(f"STOP: global {frame} {domain} total is infeasible")
                    choices.sort(key=lambda cube: (
                        roles[cube] != roles[deficient_cube],
                        -(caps[domain][cube] - used[domain][cube] - achieved[cube]), cube,
                    ))
                    achieved[choices[0]] += 1
            for cube in expected:
                final[frame][domain][cube] = achieved[cube]
                used[domain][cube] += achieved[cube]
                audit_rows.append({
                    "sampling_frame": frame, "sampling_domain": domain, "cube_id": cube,
                    "frozen_cube_role": roles[cube], "feasible_capacity_before_frame": caps[domain].get(cube, 0) - (used[domain][cube] - achieved[cube]),
                    "initial_requested": desired[cube], "final_allocated": achieved[cube],
                    "shortfall_exported": max(0, desired[cube] - achieved[cube]),
                    "redistribution_received": max(0, achieved[cube] - desired[cube]),
                })
            expected_total = int(v2["sampling"]["domains"][domain][f"{frame}_total"])
            if sum(achieved.values()) != expected_total:
                raise RuntimeError(f"STOP: {frame} {domain} total differs from {expected_total}")
    return final, pd.DataFrame(audit_rows)


def _rank_quotas(population: pd.DataFrame, total: int, strata: list[str]) -> dict[str, int]:
    capacities = population.groupby("scalar_index_rank_stratum").size().astype(int).to_dict()
    return _even_distribution(total, strata, capacities)


def select_frames(
    eligible: pd.DataFrame, expected: list[str], quotas: dict, v2: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = list(v2["sampling"]["scalar_index_rank_strata"])
    seed = int(v2["sampling"]["seed"])
    remaining = eligible.copy()
    selected_frames = {}
    for frame_name in ("main", "reserve"):
        parts = []
        for domain in DOMAINS:
            for cube_id in expected:
                quota = int(quotas[frame_name][domain][cube_id])
                if not quota:
                    continue
                population = remaining[(remaining.cube_id == cube_id) & (remaining.sampling_domain == domain)]
                rank_quotas = _rank_quotas(population, quota, strata)
                group_parts = []
                for stratum in strata:
                    stratum_population = population[population.scalar_index_rank_stratum == stratum]
                    selected = _block_balanced_selection(
                        stratum_population, rank_quotas[stratum], seed,
                        frame_name, f"{cube_id}:{domain}", stratum,
                    )
                    selected["cube_rank_stratum_requested"] = rank_quotas[stratum]
                    group_parts.append(selected)
                group = pd.concat(group_parts)
                parts.append(group)
                remaining = remaining.drop(index=group.index)
        frame = pd.concat(parts, ignore_index=False).copy()
        frame["overall_inclusion_probability"] = frame.first_stage_inclusion_probability * frame.final_inclusion_probability
        frame["design_weight"] = 1.0 / frame.overall_inclusion_probability
        frame["sample_id"] = [
            "f2v2-" + hashlib.sha256(
                f"{seed}:{row.cube_id}:{row.row}:{row.column}".encode()
            ).hexdigest()[:20]
            for row in frame.itertuples(index=False)
        ]
        release = "released_for_main_annotation" if frame_name == "main" else "locked_pending_separate_authorization"
        selected_frames[frame_name] = finalize_frame(frame, release)
    return selected_frames["main"], selected_frames["reserve"]


def capacity_table(raw_pixels: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    feasible = eligible.groupby(["cube_id", "sampling_domain"]).size().rename("feasible_capacity").reset_index()
    result = raw_pixels.merge(feasible, on=["cube_id", "sampling_domain"], how="left")
    result["feasible_capacity"] = result.feasible_capacity.fillna(0).astype(int)
    result["capacity_rules"] = "valid support + operational domain + finite scalar + 0.10m cell + v1 exclusion + global 0.25m thinning"
    return result.sort_values(["cube_id", "sampling_domain"]).reset_index(drop=True)


def nearest_neighbor_rows(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crs, group in combined.groupby("crs"):
        distances, _ = cKDTree(group[["x", "y"]].to_numpy(float)).query(group[["x", "y"]].to_numpy(float), k=2)
        for sample_id, distance in zip(group.sample_id, distances[:, 1]):
            rows.append({"sample_id": sample_id, "crs": crs, "nearest_neighbor_distance_m": float(distance)})
    return pd.DataFrame(rows).sort_values("sample_id")


def crop_polygon_coverage(project: Path, area: dict, combined: pd.DataFrame) -> pd.DataFrame:
    from matplotlib.path import Path as PlotPath

    path = project / area["freeze"]["output_root"] / area["freeze"]["operational_geometry"]
    payload = json.loads(path.read_text())
    crop = combined[combined.sampling_domain == "research_crop_area"]
    rows = []
    for feature in payload["features"]:
        props = feature["properties"]
        if props["zone_type"] != "research_crop_area":
            continue
        rings = feature["geometry"]["coordinates"]
        exterior = np.asarray(rings[0], dtype=float)
        holes = [np.asarray(ring, dtype=float) for ring in rings[1:]]
        cube = crop[crop.cube_id == props["cube_id"]]
        points = np.column_stack((cube.column.to_numpy(float) + .5, cube.row.to_numpy(float) + .5))
        matched = PlotPath(exterior).contains_points(points, radius=1e-9)
        for hole in holes:
            matched &= ~PlotPath(hole).contains_points(points, radius=-1e-9)
        def ring_area(ring: np.ndarray) -> float:
            return abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], -1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], -1)))) / 2
        area_pixels = ring_area(exterior) - sum(ring_area(hole) for hole in holes)
        rows.append({
            "cube_id": props["cube_id"], "polygon_id": props["polygon_id"],
            "component_index": props["component_index"], "operational_area_pixels": area_pixels,
            "main_points": int(((cube.sampling_frame == "main").to_numpy() & matched).sum()),
            "reserve_points": int(((cube.sampling_frame == "reserve").to_numpy() & matched).sum()),
            "original_vertices_preserved": bool(props["source_vertices_preserved"]),
        })
    return pd.DataFrame(rows).sort_values(["cube_id", "polygon_id", "component_index"])


def load_verified_v2(project: Path, v2: dict) -> dict:
    """Independently validate the immutable v2 contract without exposing reserve via API."""
    base, area = resolve_configs(project, v2)
    contract_path = project / v2["outputs"]["contract"]
    contract = yaml.safe_load(contract_path.read_text())
    if contract.get("version") != "field2_area_stratified_sampling_contract_v2" or contract.get("status") != "frozen":
        raise ValueError("Unknown or unfinished Field 2 v2 sampling contract")
    if contract.get("seed") != int(v2["sampling"]["seed"]):
        raise ValueError("v2 sampling seed changed")
    provenance = contract.get("provenance", {})
    required_false = (
        "checkpoint_loaded", "predictions_used", "model_probabilities_used", "embeddings_used",
        "field1_outputs_used", "pseudo_labels_used", "biological_labels_preselected",
    )
    if provenance.get("prediction_free") is not True or any(provenance.get(key) is not False for key in required_false):
        raise ValueError("v2 provenance is not prediction-free")
    if sha256(project / base["sampling"]["sampling_contract"]) != contract["inputs"]["v1_sampling_contract_sha256"]:
        raise ValueError("Frozen v1 contract changed")
    if sha256(project / area["freeze"]["contract"]) != contract["inputs"]["area_contract_sha256"]:
        raise ValueError("Frozen area contract changed")
    amendment_ref = contract.get("amendment", {})
    amendment_path = project / str(amendment_ref.get("path", ""))
    if not amendment_path.is_file() or sha256(amendment_path) != amendment_ref.get("sha256"):
        raise ValueError("v1 supersession amendment changed")
    amendment = yaml.safe_load(amendment_path.read_text())
    if amendment.get("status") != "immutable_supersession_before_biological_annotation" or not all(amendment.get("statements", {}).values()):
        raise ValueError("v1 supersession amendment is incomplete")
    verified = {}
    for name, item in contract["frozen_outputs"].items():
        path = project / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"v2 frozen output changed: {name}")
        verified[name] = path
    main = pd.read_csv(verified["main_frame"], keep_default_na=False)
    reserve = pd.read_csv(verified["reserve_frame"], keep_default_na=False)
    combined = pd.read_csv(verified["combined_frame"], keep_default_na=False)
    if len(main) != 800 or len(reserve) != 400 or not combined.equals(pd.concat([main, reserve], ignore_index=True)):
        raise ValueError("v2 main/reserve totals or combined ordering changed")
    expected_totals = {
        "main": {domain: int(v2["sampling"]["domains"][domain]["main_total"]) for domain in DOMAINS},
        "reserve": {domain: int(v2["sampling"]["domains"][domain]["reserve_total"]) for domain in DOMAINS},
    }
    for name, frame in (("main", main), ("reserve", reserve)):
        if frame.sampling_domain.value_counts().to_dict() != expected_totals[name]:
            raise ValueError(f"v2 {name} domain totals changed")
    if set(main.sample_id) & set(reserve.sample_id) or combined.geographic_ground_cell_id.nunique() != len(combined):
        raise ValueError("v2 main/reserve intersects or duplicates a ground cell")
    v1 = pd.read_csv(project / base["sampling"]["combined_frame"], keep_default_na=False)
    if set(v1.geographic_ground_cell_id) & set(combined.geographic_ground_cell_id):
        raise ValueError("v2 intersects a v1 ground cell despite feasible exclusion")
    expected_ids = [
        "f2v2-" + hashlib.sha256(f"{contract['seed']}:{row.cube_id}:{row.row}:{row.column}".encode()).hexdigest()[:20]
        for row in combined.itertuples(index=False)
    ]
    if combined.sample_id.tolist() != expected_ids:
        raise ValueError("v2 stable sample IDs changed")
    numeric = combined[["first_stage_inclusion_probability", "final_inclusion_probability", "overall_inclusion_probability", "design_weight"]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all() or not (numeric.to_numpy() > 0).all():
        raise ValueError("v2 probabilities or weights are invalid")
    if not np.allclose(numeric.overall_inclusion_probability, numeric.first_stage_inclusion_probability * numeric.final_inclusion_probability) or not np.allclose(numeric.design_weight, 1 / numeric.overall_inclusion_probability):
        raise ValueError("v2 probability/weight definitions changed")
    if set(reserve.reserve_release_status) != {"locked_pending_separate_authorization"}:
        raise ValueError("v2 reserve is not locked")
    if contract.get("audit", {}).get("reserve_coordinates_in_sampling_map") is not False:
        raise ValueError("v2 reserve coordinates are not explicitly omitted from previews")
    separation = nearest_neighbor_summary(combined)
    if separation["minimum_m"] + 1e-9 < float(v2["sampling"]["minimum_separation_m"]):
        raise ValueError("v2 minimum separation changed")
    eligibility = pd.read_csv(verified["crop_eligibility"], keep_default_na=False)
    if int(eligibility.crop_sampling_eligible.astype(str).str.lower().eq("true").sum()) != 14:
        raise ValueError("v2 crop-sampling eligibility is not exactly 14 derived cubes")
    all_primary = eligibility[eligibility.frozen_cube_role == "primary_three_class"]
    if len(all_primary) != 20:
        raise ValueError("v2 redefined the 20 frozen primary cubes")
    area_root = project / area["freeze"]["output_root"]
    mask_manifest = pd.read_csv(
        area_root / area["freeze"]["mask_manifest"], keep_default_na=False
    ).set_index("cube_id")
    for cube_id, group in combined.groupby("cube_id"):
        mask_row = mask_manifest.loc[cube_id]
        domain_path = area_root / str(mask_row.domain_mask_path)
        if sha256(domain_path) != str(mask_row.domain_mask_sha256):
            raise ValueError(f"v2 domain mask changed: {cube_id}")
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(domain_path, "r") as dataset:
                values = dataset.read(1)[group.row.astype(int), group.column.astype(int)]
        if not np.array_equal(values.astype(int), group.domain_code.astype(int).to_numpy()):
            raise ValueError(f"v2 selected point/domain mismatch: {cube_id}")
        if not group.valid_support_status.astype(str).str.lower().eq("true").all():
            raise ValueError(f"v2 selected point is outside valid support: {cube_id}")
    return {"contract": contract, "main": main, "reserve": reserve, "combined": combined, "verified": verified}


def require_v2_main_frame(project: Path, v2: dict) -> dict:
    result = load_verified_v2(project, v2)
    if result["contract"]["reserve_release"]["status"] != "locked_not_exposed":
        raise ValueError("v2 reserve protection is not active")
    return {
        "contract": result["contract"], "main": result["main"],
        "main_frame_sha256": result["contract"]["frozen_outputs"]["main_frame"]["sha256"],
    }


def materialize(project: Path, v2: dict, paths_config: dict | None = None) -> dict:
    base, area = resolve_configs(project, v2)
    outputs = v2["outputs"]
    output_root, report_root = project / outputs["root"], project / outputs["report_root"]
    targets = {
        "main_frame": output_root / outputs["main_frame"],
        "reserve_frame": output_root / outputs["reserve_frame"],
        "combined_frame": output_root / outputs["combined_frame"],
        "allocation_audit": report_root / outputs["allocation_audit"],
        "crop_eligibility": report_root / outputs["crop_eligibility"],
        "capacity_audit": report_root / outputs["capacity_audit"],
        "redistribution_audit": report_root / outputs["redistribution_audit"],
        "v1_intersection_audit": report_root / outputs["v1_intersection_audit"],
        "main_reserve_intersection_audit": report_root / outputs["main_reserve_intersection_audit"],
        "compliance_audit": report_root / outputs["compliance_audit"],
        "nearest_neighbor_audit": report_root / outputs["nearest_neighbor_audit"],
        "crop_polygon_coverage": report_root / outputs["crop_polygon_coverage"],
        "sampling_audit": report_root / outputs["sampling_audit"],
        "sampling_map": report_root / outputs["sampling_map"],
        "contract": project / outputs["contract"],
        "amendment": project / outputs["amendment"],
        "sha256_manifest": project / outputs["sha256_manifest"],
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise RuntimeError(f"REFUSED: immutable v2 output already exists: {existing}")
    role_path = project / base["review"]["role_contract"]
    role_contract = require_frozen_role_contract(role_path)
    validate_frozen_role_contract(project, base, role_path)
    validate_area_contract(project, area, base, recompile=False)
    expected = list(base["expected_cube_ids"])
    role_records = {row["cube_id"]: row for row in role_contract["cube_roles"]}
    roles = {cube: role_records[cube]["primary_role"] for cube in expected}
    if paths_config is None:
        paths_config = yaml.safe_load((project / "configs/paths.local.yaml").read_text())
    eligible, raw_pixels = build_eligible_candidates(
        project, base, area, v2, role_records, paths_config["field2"]["readiness_roots"]
    )
    capacity = capacity_table(raw_pixels, eligible)
    eligibility = crop_eligibility_table(expected, roles, capacity)
    eligible_crop = eligibility[eligibility.crop_sampling_eligible]
    zero_crop_primary = eligibility[(eligibility.frozen_cube_role == "primary_three_class") & ~eligibility.crop_sampling_eligible]
    if len(eligible_crop) != 14 or len(zero_crop_primary) != 6:
        raise RuntimeError("Derived crop eligibility differs from the approved 14/6 decision")
    requested = initial_quotas(expected, roles, capacity, v2)
    quotas, redistribution = reconcile_quotas(expected, roles, capacity, requested, v2)
    main, reserve = select_frames(eligible, expected, quotas, v2)
    repeated_main, repeated_reserve = select_frames(eligible, expected, quotas, v2)
    if frame_bytes(main) != frame_bytes(repeated_main) or frame_bytes(reserve) != frame_bytes(repeated_reserve):
        raise RuntimeError("Deterministic v2 byte reproduction failed")
    combined = pd.concat([main, reserve], ignore_index=True)
    if len(main) != 800 or len(reserve) != 400:
        raise RuntimeError("STOP: global v2 main/reserve totals are infeasible")
    v1 = pd.read_csv(project / base["sampling"]["combined_frame"], keep_default_na=False)
    v1_intersection = set(v1.geographic_ground_cell_id) & set(combined.geographic_ground_cell_id)
    if v1_intersection:
        raise RuntimeError("v1 ground-cell exclusion was feasible but not preserved")
    allocation = combined.groupby([
        "sampling_frame", "cube_id", "cube_evaluation_role", "sampling_domain",
        "scalar_index_rank_stratum", "spatial_group_id",
    ], sort=True).agg(
        points=("sample_id", "size"), population=("sampling_stratum_block_population", "first"),
        inclusion_probability=("overall_inclusion_probability", "mean"), design_weight=("design_weight", "mean"),
    ).reset_index()
    intersection_audit = pd.DataFrame([{
        "v1_count": len(v1), "v2_count": len(combined),
        "intersecting_ground_cells": len(v1_intersection), "status": "pass_disjoint",
    }])
    main_reserve = set(main.geographic_ground_cell_id) & set(reserve.geographic_ground_cell_id)
    main_reserve_audit = pd.DataFrame([{
        "main_count": len(main), "reserve_count": len(reserve),
        "intersecting_ground_cells": len(main_reserve), "reserve_exposed_to_api": False,
    }])
    compliance = combined.groupby(["sampling_frame", "sampling_domain"], sort=True).agg(
        points=("sample_id", "size"), valid_support=("valid_support_status", "sum"),
        valid_context=("valid_context_status", "sum"),
    ).reset_index()
    compliance["matching_domain"] = compliance["points"]
    nn = nearest_neighbor_rows(combined)
    coverage = crop_polygon_coverage(project, area, combined)
    separation = nearest_neighbor_summary(combined)
    audit = {
        "status": "field2_area_sampling_v2_passed", "main_count": 800, "reserve_count": 400,
        "domain_totals": {
            frame: {domain: int((data.sampling_domain == domain).sum()) for domain in DOMAINS}
            for frame, data in (("main", main), ("reserve", reserve))
        },
        "all_primary_three_class_cube_count": 20,
        "crop_sampling_eligible_primary_cube_count": 14,
        "zero_crop_support_primary_cube_count": 6,
        "zero_crop_support_primary_cubes": zero_crop_primary.cube_id.tolist(),
        "v1_v2_ground_cell_intersection": 0, "main_reserve_intersection": 0,
        "combined_nearest_neighbor_separation_m": separation,
        "deterministic_reproduction": "pass_byte_identical_main_and_reserve",
        "reserve_coordinates_in_sampling_map": False,
        "redistribution_rows_with_shortfall": int((redistribution.shortfall_exported > 0).sum()),
        "scalar_use": "prediction-free cube-domain empirical rank balancing only",
        "model_or_biological_output_used": False,
    }
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(targets["main_frame"], frame_bytes(main))
    atomic_write_bytes(targets["reserve_frame"], frame_bytes(reserve))
    atomic_write_bytes(targets["combined_frame"], frame_bytes(combined))
    for name, frame in (
        ("allocation_audit", allocation), ("crop_eligibility", eligibility),
        ("capacity_audit", capacity), ("redistribution_audit", redistribution),
        ("v1_intersection_audit", intersection_audit),
        ("main_reserve_intersection_audit", main_reserve_audit),
        ("compliance_audit", compliance), ("nearest_neighbor_audit", nn),
        ("crop_polygon_coverage", coverage),
    ):
        atomic_write_bytes(targets[name], csv_bytes(frame))
    atomic_write_yaml(targets["sampling_audit"], audit)
    review_manifest = pd.read_csv(project / base["review"]["package_manifest"]).fillna("")
    save_sampling_map(targets["sampling_map"], review_manifest, main, project, include_reserve=False)
    frozen_output_names = [name for name in targets if name not in {"contract", "amendment", "sha256_manifest"}]
    output_refs = {
        name: {"path": str(targets[name].relative_to(project)), "sha256": sha256(targets[name])}
        for name in frozen_output_names
    }
    manifest = pd.DataFrame([
        {"artifact": name, **item} for name, item in output_refs.items()
    ]).sort_values("artifact")
    atomic_write_bytes(targets["sha256_manifest"], csv_bytes(manifest))
    output_refs["sha256_manifest"] = {
        "path": str(targets["sha256_manifest"].relative_to(project)),
        "sha256": sha256(targets["sha256_manifest"]),
    }
    v1_contract_path = project / base["sampling"]["sampling_contract"]
    old_annotation_path = project / base["point_annotation"]["output_root"] / "field2_blind_main_point_annotations.json"
    v1_area_membership = pd.read_csv(
        project / area["freeze"]["output_root"] / area["freeze"]["main_membership"],
        keep_default_na=False,
    )
    if int((v1_area_membership.domain_name == "research_crop_area").sum()) != 5:
        raise RuntimeError("The v1 crop-domain main-point finding changed")
    old_annotations = json.loads(old_annotation_path.read_text())
    if any(record.get("selected_label") or record.get("reviewed") for record in old_annotations["annotations"].values()):
        raise RuntimeError("v1 contains biological annotations and cannot be superseded under this amendment")
    amendment = {
        "version": "field2_blind_sampling_v1_supersession_amendment_v1",
        "status": "immutable_supersession_before_biological_annotation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v1_sampling_contract": {"path": str(v1_contract_path.relative_to(project)), "sha256": sha256(v1_contract_path)},
        "v1_main_crop_domain_points": 5,
        "statements": {
            "v1_frozen_before_area_annotations": True,
            "v1_superseded_before_biological_annotation": True,
            "v1_never_used_for_field2_model_evaluation": True,
            "v2_uses_only_prediction_free_investigator_geometry_and_stored_scalar_ranks": True,
            "no_biological_labels_or_model_outputs_existed_when_v2_frozen": True,
        },
        "preserved_v1_point_annotation": {"path": str(old_annotation_path.relative_to(project)), "sha256": sha256(old_annotation_path)},
        "v2_contract_path": str(targets["contract"].relative_to(project)),
    }
    atomic_write_yaml(targets["amendment"], amendment)
    contract = {
        "version": "field2_area_stratified_sampling_contract_v2", "status": "frozen",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "materialization_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "seed": int(v2["sampling"]["seed"]),
        "inputs": {
            "v1_sampling_contract_sha256": sha256(v1_contract_path),
            "v1_main_frame_sha256": sha256(project / base["sampling"]["main_frame"]),
            "v1_reserve_frame_sha256": sha256(project / base["sampling"]["reserve_frame"]),
            "v1_point_annotation_sha256": sha256(old_annotation_path),
            "area_contract_sha256": sha256(project / area["freeze"]["contract"]),
            "cube_role_contract_sha256": sha256(role_path),
            "valid_support_contract_sha256": sha256(project / base["inputs"]["valid_support_contract"]),
        },
        "allocation_policy": v2["sampling"]["domains"],
        "derived_crop_sampling_eligibility_rule": "primary_three_class AND operational research_crop_area support > 0",
        "frozen_primary_role_unchanged": True,
        "main_count": 800, "reserve_count": 400, "frozen_outputs": output_refs,
        "amendment": {"path": str(targets["amendment"].relative_to(project)), "sha256": sha256(targets["amendment"])},
        "reserve_release": {"status": "locked_not_exposed", "authorization_required": "separate immutable authorization"},
        "audit": audit,
        "provenance": {
            "prediction_free": True, "checkpoint_loaded": False, "predictions_used": False,
            "model_probabilities_used": False, "embeddings_used": False,
            "field1_outputs_used": False, "pseudo_labels_used": False,
            "biological_labels_preselected": False,
        },
    }
    atomic_write_yaml(targets["contract"], contract)
    return {"contract": contract, "main": main, "reserve": reserve, "capacity": capacity, "eligibility": eligibility, "redistribution": redistribution}
