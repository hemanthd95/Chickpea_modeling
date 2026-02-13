#!/usr/bin/env python3
"""Train a GPU-accelerated classifier for Chickpea/weed/soil CSV datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb


VALID_EXTENSIONS = {".csv", ".CSV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a supervised multiclass model to classify Chickpea, weeds, and soil "
            "from many large CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--label-column",
        required=True,
        help="Column name containing class labels (e.g., chickpea/weeds/soil).",
    )
    parser.add_argument(
        "--target-map",
        type=Path,
        default=Path("label_mapping.json"),
        help="Path to store label-to-index mapping JSON.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("xgb_chickpea_model.json"),
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Dask workers.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=8,
        help="CPU threads per worker for ingestion/preprocessing.",
    )
    parser.add_argument(
        "--memory-limit",
        default="0",
        help=(
            "Per-worker memory limit (e.g., '24GB'). Use '0' to disable limits and rely "
            "on system memory."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Pattern used to find CSV files inside data-dir.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="XGBoost max_depth.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="XGBoost number of trees.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning rate.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="XGBoost subsample.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="XGBoost colsample_bytree.",
    )
    return parser.parse_args()


def discover_csvs(data_dir: Path, pattern: str) -> list[str]:
    files = sorted(
        file
        for file in data_dir.glob(pattern)
        if file.is_file() and file.suffix in VALID_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} with pattern '{pattern}'.")
    return [str(path) for path in files]


def convert_labels_to_ids(series: dd.Series) -> tuple[dd.Series, dict[str, int]]:
    unique_labels = series.dropna().unique().compute().tolist()
    unique_labels = sorted(str(label) for label in unique_labels)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    converted = series.astype(str).map(label_to_id, meta=(series.name, "int64"))
    return converted, label_to_id


def prepare_features(df: dd.DataFrame, label_column: str) -> tuple[dd.DataFrame, dd.Series, dict[str, int]]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {list(df.columns)}")

    y, label_to_id = convert_labels_to_ids(df[label_column])
    x = df.drop(columns=[label_column])

    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    x_num = x[numeric_columns]
    if categorical_columns:
        x_cat = x[categorical_columns].categorize()
        x_cat = dd.get_dummies(x_cat, dummy_na=True)
        x_prepared = dd.concat([x_num, x_cat], axis=1)
    else:
        x_prepared = x_num

    return x_prepared.fillna(0), y, label_to_id


def main() -> None:
    args = parse_args()

    csv_files = discover_csvs(args.data_dir, args.glob)
    print(f"Discovered {len(csv_files)} files.")

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        processes=True,
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    df = dd.read_csv(csv_files, assume_missing=True, blocksize="128MB")

    x, y, label_to_id = prepare_features(df, args.label_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    clf = xgb.dask.DaskXGBClassifier(
        objective="multi:softprob",
        num_class=len(label_to_id),
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    clf.client = client
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test).compute()
    truth = y_test.compute().to_numpy()

    inverse_map = {value: key for key, value in label_to_id.items()}
    target_names = [inverse_map[idx] for idx in range(len(inverse_map))]

    print("Classification report:")
    print(classification_report(truth, preds, target_names=target_names, digits=4, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(truth, preds))

    booster = clf.get_booster()
    booster.save_model(args.output_model)
    args.target_map.write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")

    print(f"Model saved to: {args.output_model}")
    print(f"Label mapping saved to: {args.target_map}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
