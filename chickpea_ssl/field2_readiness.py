"""Read-only guards and ENVI helpers for the Field 2 readiness audit."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from spectral.io import envi

ENVI_DTYPES = {
    1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 6: "c8",
    9: "c16", 12: "u2", 13: "u4", 14: "i8", 15: "u8",
}


def yaml_safe(value):
    """Convert NumPy/path containers into objects accepted by safe_dump."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [yaml_safe(item) for item in value]
    return value


def dump_yaml(value: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(yaml_safe(value), sort_keys=False))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_inside(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


@dataclass(frozen=True)
class ReadOnlySourceGuard:
    source_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        roots = tuple(root.resolve(strict=True) for root in self.source_roots)
        object.__setattr__(self, "source_roots", roots)
        if len(set(roots)) != len(roots):
            raise ValueError("Authorized source roots must be unique")

    def source(self, path: Path, mode: str = "r") -> Path:
        candidate = path.resolve(strict=True)
        if not is_inside(candidate, self.source_roots):
            raise ValueError(f"Source is outside authorized roots: {candidate}")
        normalized = mode.replace("b", "").replace("t", "")
        if normalized != "r" or any(token in mode for token in ("+", "w", "a", "x")):
            raise ValueError(f"Source update/write mode rejected: {mode}")
        return candidate

    def output(self, path: Path) -> Path:
        candidate = path.resolve()
        if is_inside(candidate, self.source_roots):
            raise ValueError(f"Output path is inside an authorized source root: {candidate}")
        return candidate

    def memmap(self, path: Path, dtype, shape, offset: int = 0, order: str = "C") -> np.memmap:
        candidate = self.source(path, "r")
        return np.memmap(candidate, dtype=dtype, mode="r", shape=shape, offset=offset, order=order)


def source_snapshot(guard: ReadOnlySourceGuard, project: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in sorted(guard.source_roots, key=str):
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=str):
            source = guard.source(path)
            details = source.stat()
            rows.append({
                "relative_path": str(source.relative_to(project)),
                "byte_size": int(details.st_size),
                "mtime_ns": int(details.st_mtime_ns),
                "file_mode": stat.filemode(details.st_mode),
                "mode_octal": format(stat.S_IMODE(details.st_mode), "04o"),
                "sha256": sha256(source),
            })
    return rows


def compare_snapshots(before: list[dict], after: list[dict]) -> list[str]:
    keyed_before = {str(row["relative_path"]): row for row in before}
    keyed_after = {str(row["relative_path"]): row for row in after}
    issues: list[str] = []
    for path in sorted(keyed_before.keys() - keyed_after.keys()):
        issues.append(f"disappeared:{path}")
    for path in sorted(keyed_after.keys() - keyed_before.keys()):
        issues.append(f"appeared:{path}")
    for path in sorted(keyed_before.keys() & keyed_after.keys()):
        for field in ("byte_size", "mtime_ns", "sha256"):
            if keyed_before[path][field] != keyed_after[path][field]:
                issues.append(f"changed_{field}:{path}")
    return issues


def parse_envi_header(path: Path, guard: ReadOnlySourceGuard) -> dict:
    return envi.read_envi_header(str(guard.source(path, "r")))


def scalar_int(metadata: dict, key: str, default: int = 0) -> int:
    return int(str(metadata.get(key, default)).strip())


def resolve_envi_payload(header: Path, guard: ReadOnlySourceGuard) -> Path:
    header = guard.source(header)
    candidates: set[Path] = set()
    if header.name.lower().endswith(".hdr"):
        direct = Path(str(header)[:-4])
        if direct.is_file():
            candidates.add(direct.resolve())
    metadata = parse_envi_header(header, guard)
    declared = str(metadata.get("data file", "")).strip()
    if declared:
        declared_path = header.parent / declared
        if declared_path.is_file():
            candidates.add(declared_path.resolve())
    stem = Path(str(header)[:-4]) if header.name.lower().endswith(".hdr") else header
    for suffix in ("", ".bip", ".bil", ".bsq", ".dat", ".img", ".raw"):
        candidate = Path(str(stem) + suffix)
        if candidate.is_file() and candidate.resolve() != header:
            candidates.add(candidate.resolve())
    authorized = sorted((guard.source(candidate) for candidate in candidates), key=str)
    if len(authorized) != 1:
        raise ValueError(f"Expected exactly one ENVI payload for {header}; found {authorized}")
    return authorized[0]


def normalized_cube_id(path: Path | str) -> str:
    match = re.search(r"Pika_L-GigE[_ -]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot normalize Field 2 cube identifier: {path}")
    return f"field2_cube{int(match.group(1)):02d}"


def envi_dtype(metadata: dict) -> np.dtype:
    code = scalar_int(metadata, "data type")
    if code not in ENVI_DTYPES:
        raise ValueError(f"Unsupported ENVI data type: {code}")
    endian = "<" if scalar_int(metadata, "byte order") == 0 else ">"
    base = ENVI_DTYPES[code]
    return np.dtype(base if base == "u1" else endian + base)


def expected_payload_bytes(metadata: dict) -> int:
    shape = [scalar_int(metadata, key) for key in ("lines", "samples", "bands")]
    return int(np.prod(shape, dtype=np.int64) * envi_dtype(metadata).itemsize + scalar_int(metadata, "header offset"))


def open_envi_memmap(header: Path, payload: Path, guard: ReadOnlySourceGuard) -> tuple[np.ndarray, dict]:
    metadata = parse_envi_header(header, guard)
    lines, samples, bands = (scalar_int(metadata, key) for key in ("lines", "samples", "bands"))
    interleave = str(metadata.get("interleave", "")).strip().lower()
    shapes = {"bip": (lines, samples, bands), "bil": (lines, bands, samples), "bsq": (bands, lines, samples)}
    if interleave not in shapes:
        raise ValueError(f"Unsupported ENVI interleave: {interleave}")
    raw = guard.memmap(payload, envi_dtype(metadata), shapes[interleave], scalar_int(metadata, "header offset"))
    standard = raw if interleave == "bip" else (np.moveaxis(raw, 1, 2) if interleave == "bil" else np.moveaxis(raw, 0, 2))
    if standard.flags.writeable:
        raise RuntimeError(f"Read-only ENVI memmap unexpectedly writable: {payload}")
    return standard, metadata


def parse_wavelengths(metadata: dict) -> np.ndarray:
    raw = metadata.get("wavelength", [])
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.strip("{}").split(",") if item.strip()]
    return np.asarray([float(value) for value in raw], dtype=np.float64)


def match_wavelengths(expected: np.ndarray, observed: np.ndarray, tolerance: float) -> list[dict[str, object]]:
    rows = []
    increasing = bool(len(observed) > 1 and np.all(np.diff(observed) > 0))
    for position, wavelength in enumerate(expected):
        if not len(observed):
            index, value, difference = -1, np.nan, np.nan
            status = "missing"
        else:
            index = int(np.argmin(np.abs(observed - wavelength)))
            value = float(observed[index])
            difference = abs(value - float(wavelength))
            status = "matched" if difference <= tolerance else "outside_tolerance"
        rows.append({
            "field1_band_index": int(position + 3),
            "expected_wavelength_nm": float(wavelength),
            "field2_matched_band_index": index,
            "field2_wavelength_nm": value,
            "absolute_difference_nm": float(difference),
            "match_status": status,
            "tolerance_nm": float(tolerance),
            "ordering_status": "strictly_increasing" if increasing else "not_strictly_increasing",
        })
    return rows


def deterministic_coordinates(height: int, width: int, count: int, seed: int, cube_id: str) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{cube_id}".encode()).digest()
    cube_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(cube_seed)
    total = height * width
    flat = rng.choice(total, size=min(count, total), replace=False)
    return np.column_stack(np.divmod(flat, width)).astype(np.int64)


def affine_comparison(reference, candidate, exact_tolerance: float, numerical_tolerance: float) -> str:
    first = np.asarray(tuple(reference), dtype=float)
    second = np.asarray(tuple(candidate), dtype=float)
    difference = float(np.max(np.abs(first - second)))
    if difference <= exact_tolerance:
        return "exact_alignment"
    if difference <= numerical_tolerance:
        return "numerically_equivalent"
    return "potentially_repairable"
