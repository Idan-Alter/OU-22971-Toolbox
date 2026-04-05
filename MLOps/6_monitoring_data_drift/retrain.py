"""retrain.py

MLOps Unit 6: retrain workflow (short + clean).

- Finds baseline model from latest FINISHED train/retrain run
- Trains a new simple model on a wider training window
- Evaluates baseline and new model on eval month
- Logs new model and sets tag new_model_wins=true/false

Example:
  python retrain.py --train-parquets TLC_data/green_tripdata_2020-01.parquet TLC_data/green_tripdata_2020-04.parquet --eval-parquet TLC_data/green_tripdata_2020-08.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.sources import LocalArtifactDatasetSource
from mlflow.tracking import MlflowClient

from green_taxi_drift_lib import (
    align_feature_frame,
    load_taxi_table,
    make_tip_frame,
    cast_ints_to_float,
    latest_model_uri,
    load_feature_cols_from_run,
    resolve_input_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", type=str, default="http://localhost:5000")
    p.add_argument("--experiment", type=str, default="6_green_taxi_drift")
    p.add_argument("--run-name", default=None)

    p.add_argument(
        "--train-parquets",
        nargs="+",
        required=True,
        help="One or more training slice parquets (relative to cwd or this unit folder).",
    )
    p.add_argument(
        "--eval-parquet",
        required=True,
        help="Evaluation slice parquet (relative to cwd or this unit folder).",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=200)
    return p.parse_args()


# -----------------------------
# Model (match train_initial)
# -----------------------------

def build_model(random_state: int = 0, max_depth: int = 8, min_samples_leaf: int = 200) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeRegressor(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )),
    ])





def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# Main
# -----------------------------

def main() -> None:

    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise SystemExit(f"Experiment not found: {args.experiment}")

    baseline_uri, baseline_run_id = latest_model_uri(client, exp.experiment_id)

    train_paths = [resolve_input_path(Path(p)) for p in args.train_parquets]
    eval_path = resolve_input_path(Path(args.eval_parquet))

    run_name = args.run_name or "retrain"

    # Load raw slices ONCE (and reuse for lineage + concat)
    train_slices = []
    for p in train_paths:
        df_slice = load_taxi_table(p)
        train_slices.append((p, df_slice))

    df_train_raw = pd.concat([df for _, df in train_slices], ignore_index=True)
    df_eval_raw = load_taxi_table(eval_path)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "pipeline_step": "retrain",
            "eval_slice": eval_path.stem,
            "baseline_model_uri": baseline_uri,
            "baseline_source_run_id": baseline_run_id,
        })
        mlflow.log_params({
            "seed": args.seed,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "train_parquets": ",".join(str(p) for p in train_paths),
            "eval_parquet": str(eval_path),
        })

        # --- Raw lineage  ---
        for p, df_slice in train_slices:
            ds_slice = mlflow.data.from_pandas(
                df_slice,
                source=LocalArtifactDatasetSource(str(p)),
                name=f"green_taxi_raw_{p.stem}",
            )
            mlflow.log_input(ds_slice, context="raw_train_slice")


        ds_eval_raw = mlflow.data.from_pandas(
            df_eval_raw,
            source=LocalArtifactDatasetSource(str(eval_path)),
            name=f"green_taxi_raw_{eval_path.stem}",
        )
        mlflow.log_input(ds_eval_raw, context="raw_eval")

        # --- Model frames ---
        Xtr, ytr, feature_cols = make_tip_frame(df_train_raw, credit_card_only=True)
        Xev, yev, _ = make_tip_frame(df_eval_raw, credit_card_only=True)

        # Optional but recommended: avoid MLflow/SHAP dtype issues with pandas nullable Int64
        Xtr = cast_ints_to_float(Xtr).astype("float64")
        Xev = cast_ints_to_float(Xev).astype("float64")

        baseline_feature_cols = load_feature_cols_from_run(baseline_run_id)
        if baseline_feature_cols:
            Xtr = align_feature_frame(Xtr, baseline_feature_cols)
            Xev = align_feature_frame(Xev, baseline_feature_cols)
            feature_cols = baseline_feature_cols
            mlflow.set_tag("feature_alignment", "baseline_run_feature_cols")
        else:
            mlflow.set_tag("feature_alignment", "current_training_feature_cols")

        mlflow.log_dict({"feature_cols": feature_cols}, "feature_cols.json")

        eval_df = Xev.copy()
        eval_df["target"] = yev


        # --- Train + log new model ---
        model = build_model(
            random_state=args.seed,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
        )
        model.fit(Xtr, ytr)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=Xtr.head(5),
        )
        mlflow.set_tag("model_uri", model_info.model_uri)
        mlflow.log_text(model_info.model_uri, "model_uri.txt")

        # --- Compare baseline vs new (simple + robust) ---
        baseline_model = mlflow.pyfunc.load_model(baseline_uri)
        yhat_base = np.asarray(baseline_model.predict(Xev), dtype=float)
        yhat_new = np.asarray(model.predict(Xev), dtype=float)

        base_rmse = _rmse(yev, yhat_base)
        new_rmse = _rmse(yev, yhat_new)

        mlflow.log_metrics({
            "baseline_rmse": float(base_rmse),
            "new_rmse": float(new_rmse),
            "delta_rmse_new_minus_base": float(new_rmse - base_rmse),
        })

        mlflow.set_tag("new_model_wins", str(new_rmse < base_rmse).lower())



if __name__ == "__main__":
    main()
