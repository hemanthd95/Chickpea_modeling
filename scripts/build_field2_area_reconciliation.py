#!/usr/bin/env python
"""Build a review-only terminal-vertex reconciliation table; never edit annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chickpea_ssl.field2_area_review import (
    distinct_vertex_count,
    normalize_vertices,
    polygon_area,
    polygon_self_intersects,
)
from chickpea_ssl.field2_blind_review import atomic_write_bytes
from chickpea_ssl.field2_readiness import sha256


COLUMNS = (
    "cube_id", "polygon_id", "zone_type", "original_vertex_count",
    "proposed_operational_vertex_count", "first_to_last_distance_pixels",
    "original_self_intersection_status", "proposed_self_intersection_status",
    "area_before_pixels2", "area_after_pixels2",
    "absolute_area_difference_pixels2", "percentage_area_difference",
    "repair_reason", "proposed_action", "proposal_applied",
    "investigator_decision", "reviewer_identifier", "review_timestamp",
    "source_annotation_sha256",
)


def reconciliation_rows(payload: dict, source_hash: str, terminal_distance_pixels: float = 12.0) -> list[dict]:
    """Propose, but never apply, removal of one apparent tool-generated terminal vertex."""
    rows = []
    for cube_id in payload["cube_order"]:
        for polygon in payload["annotations"][cube_id]["polygons"]:
            original = polygon["vertices_pixel"]
            if not polygon_self_intersects(original):
                continue
            proposed = original[:-1]
            distance = math.dist(
                (float(original[0]["x"]), float(original[0]["y"])),
                (float(original[-1]["x"]), float(original[-1]["y"])),
            )
            before = polygon_area(normalize_vertices(original))
            candidate_is_safe = (
                distance <= terminal_distance_pixels
                and distinct_vertex_count(proposed) >= 3
                and polygon_area(normalize_vertices(proposed)) > 0
                and not polygon_self_intersects(proposed)
            )
            operational = proposed if candidate_is_safe else original
            after = polygon_area(normalize_vertices(operational))
            difference = abs(after - before)
            rows.append({
                "cube_id": cube_id,
                "polygon_id": polygon["polygon_id"],
                "zone_type": polygon["zone_type"],
                "original_vertex_count": len(original),
                "proposed_operational_vertex_count": len(operational),
                "first_to_last_distance_pixels": distance,
                "original_self_intersection_status": True,
                "proposed_self_intersection_status": polygon_self_intersects(operational),
                "area_before_pixels2": before,
                "area_after_pixels2": after,
                "absolute_area_difference_pixels2": difference,
                "percentage_area_difference": (100.0 * difference / before) if before else 0.0,
                "repair_reason": (
                    "apparent tool-generated terminal near-duplicate; remove only final operational vertex after investigator confirmation"
                    if candidate_is_safe else
                    "manual geometry review required; safe terminal-only repair not proposed"
                ),
                "proposed_action": "remove_terminal_operational_vertex" if candidate_is_safe else "manual_review_no_change",
                "proposal_applied": False,
                "investigator_decision": "",
                "reviewer_identifier": "",
                "review_timestamp": "",
                "source_annotation_sha256": source_hash,
            })
    return rows


def csv_bytes(rows: list[dict]) -> bytes:
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("metadata/local/annotations/field2_area_zones/field2_area_annotations.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("metadata/local/annotations/field2_area_zones/field2_area_terminal_vertex_reconciliation.csv"),
    )
    parser.add_argument("--terminal-distance-pixels", type=float, default=12.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reconciliation workflow: {args.output}")
    source_hash = sha256(args.annotations)
    rows = reconciliation_rows(json.loads(args.annotations.read_text()), source_hash, args.terminal_distance_pixels)
    atomic_write_bytes(args.output, csv_bytes(rows))
    candidates = sum(row["proposed_action"] == "remove_terminal_operational_vertex" for row in rows)
    print(f"Wrote {len(rows)} flagged polygons to {args.output}")
    print(f"Terminal-only candidates: {candidates}; automatically applied: 0")
    print(f"Source annotation SHA-256: {source_hash}")


if __name__ == "__main__":
    main()
