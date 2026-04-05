"""train_initial.py

MLOps Unit 6: initial training on a reference month.

Example:
  python train_initial.py --data-parquet TLC_data/green_tripdata_2020-01.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.sources import LocalArtifactDatasetSource

from green_taxi_drift_lib import (
    load_taxi_table,
    make_tip_frame,
    run_integrity_checks,
    cast_ints_to_float,
    resolve_input_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="6_green_taxi_drift")
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--data-parquet",
        type=Path,
        required=True,
        help="Path to the reference slice parquet (relative to cwd or this unit folder).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=200)
    p.add_argument("--val-size", type=float, default=0.2)
    return p.parse_args()


def build_model(random_state: int = 0, max_depth: int = 8, min_samples_leaf: int = 200) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeRegressor(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )),
    ])


def main() -> None:
    args = parse_args()
    data_parquet = resolve_input_path(args.data_parquet)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or "initial"

    with mlflow.start_run(run_name=run_name):
        # Log run identity as early as possible (so failures still have context)
        mlflow.set_tags({
            "pipeline_step": "train",
            "slice": data_parquet.stem,
        })
        mlflow.log_params({
            "seed": args.seed,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "val_size": args.val_size,
            "data_path": str(data_parquet),
        })

        # --- Datasets (lineage) ---
        raw_source = LocalArtifactDatasetSource(str(data_parquet))

        df_raw = load_taxi_table(data_parquet)
        raw_ds = mlflow.data.from_pandas(
            df_raw,
            source=raw_source,
            name=f"green_taxi_raw_{data_parquet.stem}",
        )
        mlflow.log_input(raw_ds, context="raw_reference")

        # --- Integrity checks on raw slice ---
        chk = run_integrity_checks(df_raw)

        # Log all check tables (each is small and UI-friendly)
        for name, tbl in chk.tables.items():
            mlflow.log_table(tbl, artifact_file=f"checks/{name}.json")


        # --- Build modeling frame ---
        X, y, feature_cols = make_tip_frame(df_raw, credit_card_only=True)
        mlflow.log_dict({"feature_cols": feature_cols}, "feature_cols.json")

        X = cast_ints_to_float(X)

        # --- Split ---
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=args.val_size, random_state=args.seed
        )

        # --- Fit ---
        model = build_model(
            random_state=args.seed,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )
        model.fit(X_tr, y_tr)

        # --- Log model ---
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_tr.head(5),
        )

        mlflow.log_text(model_info.model_uri, "model_uri.txt")
        mlflow.set_tag("model_uri", model_info.model_uri)

        # --- Evaluate ---
        eval_df = X_va.copy()
        eval_df["target"] = y_va


        res = mlflow.models.evaluate(
            model=model_info.model_uri,
            data=eval_df,          
            targets="target",
            model_type="regressor",
        )

if __name__ == "__main__":
    main()
