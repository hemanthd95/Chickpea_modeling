"""Minimal read-only ESRI polygon shapefile parser for project geometry."""

from __future__ import annotations

import struct
from pathlib import Path


def read_polygon_geometries(path: Path) -> list[dict[str, object]]:
    geometries: list[dict[str, object]] = []
    with path.open("rb") as stream:
        header = stream.read(100)
        if len(header) != 100 or struct.unpack(">i", header[:4])[0] != 9994:
            raise ValueError(f"Invalid shapefile header: {path}")
        while record_header := stream.read(8):
            if len(record_header) != 8:
                raise ValueError(f"Truncated record header: {path}")
            _, words = struct.unpack(">2i", record_header)
            content = stream.read(words * 2)
            if len(content) != words * 2:
                raise ValueError(f"Truncated record: {path}")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in {5, 15, 25}:
                raise ValueError(f"Expected polygon, found shape type {shape_type}: {path}")
            number_parts, number_points = struct.unpack("<2i", content[36:44])
            part_offset = 44
            parts = list(struct.unpack(
                f"<{number_parts}i",
                content[part_offset:part_offset + 4 * number_parts],
            ))
            point_offset = part_offset + 4 * number_parts
            coordinates = [
                struct.unpack("<2d", content[point_offset + index * 16:point_offset + (index + 1) * 16])
                for index in range(number_points)
            ]
            rings = []
            ends = parts[1:] + [number_points]
            for start, end in zip(parts, ends):
                ring = coordinates[start:end]
                if ring and ring[0] != ring[-1]:
                    ring.append(ring[0])
                rings.append(ring)
            geometries.append({"type": "Polygon", "coordinates": rings})
    return geometries
