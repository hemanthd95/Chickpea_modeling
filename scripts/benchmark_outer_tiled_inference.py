#!/usr/bin/env python
"""Benchmark real-data GPU patch gathering for frozen outer-test inference.

This engineering gate generates predictions but intentionally never reads or
reports target classes. It therefore cannot be used as a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from chickpea_ssl.data import EnviCube, load_band_indices, load_records
from chickpea_ssl.model import build_supervised_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def nested_strings(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from nested_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from nested_strings(child)
    elif isinstance(value, str):
        yield value


def discover_checkpoints(
    project: Path, contract: dict, outer_fold: int, architecture: str,
    expected_seeds: list[int],
) -> list[tuple[int, Path, dict]]:
    strings = list(nested_strings(contract))
    allowed_hashes = {
        value.lower() for value in strings
        if re.fullmatch(r"[0-9a-fA-F]{64}", value)
    }
    candidates: set[Path] = set()
    for value in strings:
        if value.endswith(".pt"):
            path = Path(value)
            candidates.add(path if path.is_absolute() else project / path)
    if not candidates:
        roots = [
            project / "artifacts", project / "outputs", project / "models",
            project / "metadata" / "local" / "reports" / "models",
        ]
        for root in roots:
            if root.exists():
                candidates.update(root.rglob("*_best.pt"))

    selected: dict[int, tuple[Path, dict]] = {}
    for path in sorted(candidates):
        if not path.exists():
            continue
        digest = sha256(path)
        if allowed_hashes and digest not in allowed_hashes:
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        checkpoint_outer = int(config.get("heldout_fold", -1))
        checkpoint_architecture = checkpoint.get("architecture")
        seed = int(checkpoint.get("seed", -1))
        if checkpoint_outer == outer_fold and checkpoint_architecture == architecture:
            if seed in selected:
                raise ValueError(f"Duplicate checkpoint for {architecture}, seed {seed}")
            selected[seed] = (path, checkpoint)
    missing = sorted(set(expected_seeds) - set(selected))
    if missing:
        raise FileNotFoundError(
            f"Could not discover frozen {architecture} checkpoints for outer fold "
            f"{outer_fold}, seeds {missing}. Contract/checkpoint paths need review."
        )
    return [(seed, *selected[seed]) for seed in expected_seeds]


def gather_patches(
    cube: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor, radius: int
) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, device=cube.device)
    rr = rows[:, None, None] + offsets[None, :, None]
    cc = columns[:, None, None] + offsets[None, None, :]
    return cube[:, rr, cc].permute(1, 0, 2, 3).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/outer_inference.yaml"), type=Path)
    parser.add_argument("--protocol", default=Path("configs/supervised_primary_evaluation.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    parser.add_argument("--outer-fold", default=1, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    protocol_config = yaml.safe_load(args.protocol.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "outer_inference_pilot"
    reports.mkdir(parents=True, exist_ok=True)

    checkpoint_contract_path = contracts / "field1_nested_supervised_checkpoints_contract.yaml"
    support_contract_path = contracts / "field1_outer_test_support_contract.yaml"
    normalization_contract_path = contracts / "field1_nested_normalization_contract.yaml"
    normalization_path = contracts / "field1_nested_normalization.csv"
    for required in (
        checkpoint_contract_path, support_contract_path,
        normalization_contract_path, normalization_path,
        local / "authoritative_manifest.csv",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    checkpoints_contract = yaml.safe_load(checkpoint_contract_path.read_text())
    support_contract = yaml.safe_load(support_contract_path.read_text())
    normalization_contract = yaml.safe_load(normalization_contract_path.read_text())
    if checkpoints_contract.get("status") != "nested_supervised_checkpoints_frozen":
        raise ValueError("Nested checkpoint contract is not frozen")
    if support_contract.get("status") != "outer_test_support_audited":
        raise ValueError("Corrected outer-test support contract is not present")
    if support_contract.get("outer_test_predictions_generated") is not False:
        raise ValueError("Support contract indicates predictions already existed")
    if sha256(normalization_path) != normalization_contract["normalization_csv_sha256"]:
        raise ValueError("Nested normalization hash mismatch")

    architecture = str(config["pilot"]["architecture"])
    seeds = [int(value) for value in protocol_config["model_seeds"]]
    checkpoint_entries = discover_checkpoints(
        project, checkpoints_contract, args.outer_fold, architecture, seeds
    )
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {args.gpu} is unavailable")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    band_indices = load_band_indices(args.bands, normalization_contract["band_section"])
    statistics = pd.read_csv(normalization_path)
    statistics = (
        statistics[statistics["outer_fold"] == args.outer_fold]
        .set_index("band_index").loc[band_indices]
    )
    mean = statistics["mean"].to_numpy(np.float32)
    std = np.maximum(statistics["standard_deviation"].to_numpy(np.float32), 1e-6)

    support = pd.read_csv(
        local / "reports" / "outer_evaluation_support" /
        "outer_test_support_by_cube_class.csv"
    )
    support = support[support["outer_fold"] == args.outer_fold]
    cube_totals = support.groupby("cube_id")["eligible_outer_test_centers"].sum()
    cube_id = str(cube_totals[cube_totals > 0].sort_values().index[0])
    records = {record.cube_id: record for record in load_records(
        args.paths, local / "authoritative_manifest.csv"
    )}
    cube = EnviCube(records[cube_id], band_indices)
    radius = int(protocol_config["training"]["patch_size_pixels"]) // 2

    observed = np.any(cube.array > 0, axis=2)
    valid = observed.copy()
    valid[:radius] = False; valid[-radius:] = False
    valid[:, :radius] = False; valid[:, -radius:] = False
    rows, columns = np.nonzero(valid)
    maximum = int(config["pilot"]["maximum_observed_centers"])
    if len(rows) > maximum:
        indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
        rows, columns = rows[indices], columns[indices]

    raw = np.asarray(cube.array[..., band_indices], dtype=np.float32)
    normalized = (raw - mean[None, None, :]) / std[None, None, :]
    cube_tensor = torch.from_numpy(np.moveaxis(normalized, -1, 0).copy()).to(device)
    row_tensor = torch.from_numpy(rows.astype(np.int64)).to(device)
    column_tensor = torch.from_numpy(columns.astype(np.int64)).to(device)
    del raw, normalized

    models = []
    checkpoint_rows = []
    for seed, path, checkpoint in checkpoint_entries:
        model = build_supervised_model(architecture, len(band_indices), 3).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
        checkpoint_rows.append({
            "outer_fold": args.outer_fold, "architecture": architecture,
            "seed": seed, "checkpoint": str(path), "sha256": sha256(path),
        })
    pd.DataFrame(checkpoint_rows).to_csv(reports / "pilot_checkpoint_verification.csv", index=False)

    rows_out = []
    amp = bool(config["pilot"]["mixed_precision"])
    for batch_size in [int(value) for value in config["pilot"]["batch_sizes"]]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        prediction_count = 0
        probability_sum_error = 0.0
        try:
            with torch.inference_mode():
                for start in range(0, len(row_tensor), batch_size):
                    stop = min(start + batch_size, len(row_tensor))
                    patches = gather_patches(
                        cube_tensor, row_tensor[start:stop],
                        column_tensor[start:stop], radius,
                    )
                    with torch.amp.autocast(device_type="cuda", enabled=amp):
                        probability = torch.stack([
                            model(patches).softmax(dim=1) for model in models
                        ]).mean(dim=0)
                    probability_sum_error = max(
                        probability_sum_error,
                        float((probability.sum(dim=1) - 1).abs().max().item()),
                    )
                    prediction_count += len(probability)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            rows_out.append({
                "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(device),
                "outer_fold": args.outer_fold, "cube_id": cube_id,
                "architecture": architecture, "ensemble_seeds": len(models),
                "batch_size": batch_size, "observed_predictions": prediction_count,
                "elapsed_seconds": elapsed,
                "observed_predictions_per_second": prediction_count / elapsed,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "maximum_probability_sum_error": probability_sum_error,
                "status": "passed",
            })
            print(
                f"batch={batch_size}: {prediction_count:,} observed predictions, "
                f"{prediction_count / elapsed:,.0f}/s, "
                f"peak={rows_out[-1]['peak_memory_gib']:.2f} GiB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows_out.append({
                "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(device),
                "outer_fold": args.outer_fold, "cube_id": cube_id,
                "architecture": architecture, "ensemble_seeds": len(models),
                "batch_size": batch_size, "observed_predictions": 0,
                "elapsed_seconds": np.nan,
                "observed_predictions_per_second": 0.0,
                "peak_memory_gib": np.nan,
                "maximum_probability_sum_error": np.nan,
                "status": "out_of_memory",
            })
            print(f"batch={batch_size}: out of memory (safely skipped)", flush=True)

    frame = pd.DataFrame(rows_out)
    report = reports / "outer_inference_pilot_benchmark.csv"
    frame.to_csv(report, index=False)
    passed = frame[frame["status"] == "passed"]
    if passed.empty:
        raise RuntimeError("No pilot batch size passed")
    safe = passed[passed["peak_memory_gib"] <= float(config["pilot"]["maximum_peak_memory_gib"])]
    if safe.empty:
        raise RuntimeError("No batch size passed the configured VRAM safety gate")
    chosen = safe.sort_values(
        ["observed_predictions_per_second", "batch_size"], ascending=False
    ).iloc[0]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(passed["batch_size"].astype(str), passed["observed_predictions_per_second"])
    axes[0].set_xlabel("Batch size"); axes[0].set_ylabel("Observed predictions/s")
    axes[0].set_title("Three-seed ensemble throughput")
    axes[1].bar(passed["batch_size"].astype(str), passed["peak_memory_gib"], color="#E6863B")
    axes[1].axhline(float(config["pilot"]["maximum_peak_memory_gib"]), color="black", linestyle="--", label="Safety gate")
    axes[1].set_xlabel("Batch size"); axes[1].set_ylabel("Peak allocated VRAM (GiB)")
    axes[1].set_title("GPU memory safety"); axes[1].legend()
    figure.suptitle(
        f"Observed Field 1 outer-inference engineering pilot\n"
        f"{cube_id}; outer fold {args.outer_fold}; no target metrics computed"
    )
    preview = reports / "outer_inference_pilot_overview.png"
    figure.savefig(preview, dpi=200); plt.close(figure)

    contract = {
        "status": "outer_inference_engineering_pilot_passed",
        "field": "Field 1", "field2_accessed": False,
        "synthetic_scientific_observations": False,
        "target_labels_evaluated": False,
        "outer_test_predictions_generated": True,
        "scientific_results_generated": False,
        "outer_fold": args.outer_fold, "cube_id": cube_id,
        "architecture": architecture, "ensemble_seeds": seeds,
        "selected_batch_size": int(chosen["batch_size"]),
        "selected_predictions_per_second": float(chosen["observed_predictions_per_second"]),
        "selected_peak_memory_gib": float(chosen["peak_memory_gib"]),
        "benchmark_sha256": sha256(report), "visual_qc_sha256": sha256(preview),
    }
    contract_path = contracts / "field1_outer_inference_pilot_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Selected batch size: {contract['selected_batch_size']}")
    print(f"Report: {report}")
    print(f"Visual QC: {preview}")
    print(f"Contract: {contract_path}")
    print("Engineering pilot only: no target metrics or Field 2 data were accessed.")


if __name__ == "__main__":
    main()
