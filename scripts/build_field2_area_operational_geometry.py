#!/usr/bin/env python
"""Materialize derived Field 2 operational geometry without editing raw annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from chickpea_ssl.field2_area_operational import (
    PRECEDENCE_RULE,
    ZONE_PRECEDENCE,
    compile_operational_membership,
)
from chickpea_ssl.field2_area_review import ZONE_CODES, Field2AreaAnnotationStore
from chickpea_ssl.field2_blind_review import atomic_write_bytes
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_area_annotator import validate_config_inputs


def _csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader(); writer.writerows(rows); return stream.getvalue().encode()


def materialize(store: Field2AreaAnnotationStore, payload: dict, output_root: Path) -> dict:
    source_hash = sha256(store.json_path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite derived operational geometry: {output_root}")
    output_root.mkdir(parents=True)
    audits, features, summaries = [], [], []
    methods = Counter()
    for cube_id in store.cube_order:
        record = payload["annotations"][cube_id]
        support = store.support_mask(cube_id)
        compiled = compile_operational_membership(record, support)
        for item in compiled["polygon_audits"]:
            methods[item["repair_method"]] += 1
            audits.append({
                "cube_id": cube_id, "polygon_id": item["polygon_id"], "zone_type": item["zone_type"],
                "repair_method": item["repair_method"], "raw_vertex_count": item["raw_vertex_count"],
                "operational_vertex_count": item["operational_vertex_count"], "component_count": item["component_count"],
                "first_to_last_distance_pixels": item["first_to_last_distance_pixels"],
                "raw_self_intersection": item["raw_self_intersection"],
                "operational_self_intersection": item["operational_self_intersection"],
                "raw_area_pixels2": item["raw_area_pixels2"],
                "operational_area_pixels2": item["operational_area_pixels2"],
                "absolute_area_change_pixels2": item["absolute_area_change_pixels2"],
                "percentage_area_change": item["percentage_area_change"],
                "rasterized_pixel_count": item["rasterized_pixel_count"],
                "warnings": "|".join(item["warnings"]), "source_vertices_preserved": True,
                "source_annotation_sha256": source_hash,
            })
            for component in item["components"]:
                exterior = [[point["x"], point["y"]] for point in component["exterior_vertices_pixel"]]
                rings = [[*exterior, exterior[0]]]
                for hole in component.get("holes_vertices_pixel", []):
                    ring = [[point["x"], point["y"]] for point in hole]
                    rings.append([*ring, ring[0]])
                features.append({
                    "type": "Feature",
                    "properties": {
                        "cube_id": cube_id, "polygon_id": item["polygon_id"],
                        "zone_type": item["zone_type"], "component_index": component["component_index"],
                        "repair_method": item["repair_method"], "source_annotation_sha256": source_hash,
                        "source_vertices_preserved": True,
                    },
                    "geometry": {"type": "Polygon", "coordinates": rings},
                })
        summary = {
            "cube_id": cube_id, "raw_coverage_mode": record.get("coverage_mode", ""),
            "effective_operational_mode": compiled["effective_coverage_mode"],
            "raw_polygon_count": len(record.get("polygons", [])),
            "raw_overlap_pixel_count": int(np.count_nonzero(compiled["precedence_applied"])),
            "precedence_rule": PRECEDENCE_RULE, "source_annotation_sha256": source_hash,
        }
        for zone in ZONE_PRECEDENCE:
            summary[f"raw_{zone}_pixels"] = int(np.count_nonzero(compiled["raw_zone_masks"][zone]))
            summary[f"winning_{zone}_pixels"] = int(np.count_nonzero(
                compiled["winning_membership"] == ZONE_CODES[zone]
            ))
        summaries.append(summary)

    audit_columns = list(audits[0]) if audits else []
    summary_columns = list(summaries[0]) if summaries else []
    atomic_write_bytes(output_root / "field2_area_operational_geometry.geojson", json.dumps({
        "type": "FeatureCollection", "pixel_domain": True, "source_annotation_sha256": source_hash,
        "precedence_rule": PRECEDENCE_RULE, "features": features,
    }, indent=2).encode())
    atomic_write_bytes(output_root / "field2_area_operational_geometry_audit.csv", _csv_bytes(audits, audit_columns))
    atomic_write_bytes(output_root / "field2_area_operational_membership_summary.csv", _csv_bytes(summaries, summary_columns))
    return {
        "source_annotation_sha256": source_hash, "raw_polygon_count": len(audits),
        "operational_component_count": len(features), "repair_methods": dict(methods),
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_area_annotation.yaml"), type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    paths, config = yaml.safe_load(args.paths.read_text()), yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve(); validate_config_inputs(project, config)
    store = Field2AreaAnnotationStore(
        project, project / config["inputs"]["review_manifest"]["path"],
        project / config["inputs"]["natural_rgb_manifest"]["path"],
        project / config["inputs"]["valid_support_manifest"]["path"],
        project / config["annotation"]["output_root"],
    )
    payload = store.load(); source_hash = sha256(store.json_path)
    output_root = args.output_root or (
        project / config["operational_geometry"]["output_root"] / source_hash[:16]
    )
    result = materialize(store, payload, output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Raw annotation vertices modified: 0; area geometry frozen: false")


if __name__ == "__main__":
    main()
