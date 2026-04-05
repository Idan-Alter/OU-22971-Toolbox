"""check_drift.py

MLOps Unit 6: monitoring pass (integrity + drift + degradation).

- Loads reference and current parquets
- Optionally corrupts the current slice to simulate pipeline failures
- Runs integrity checks and drift report
- Loads latest model from experiment (by tag 'model_uri') unless --model-uri provided
- Evaluates model on current slice using mlflow.models.evaluate

Example:
  python check_drift.py --ref-parquet TLC_data/green_tripdata_2020-01.parquet --cur-parquet TLC_data/green_tripdata_2020-04.parquet --simulate-issues
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.data.sources import LocalArtifactDatasetSource

from green_taxi_drift_lib import (
    align_feature_frame,
    load_taxi_table,
    make_tip_frame,
    run_integrity_checks,
    compute_drift_report,
    log_violin_plots_ref_vs_cur,
    corrupt_current_slice,
    cast_ints_to_float,
    latest_model_uri,
    load_feature_cols_from_run,
    resolve_input_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="6_green_taxi_drift")
    p.add_argument("--run-name", default=None)
    
    p.add_argument("--model-uri", default=None)


    p.add_argument(
        "--ref-parquet",
        type=Path,
        required=True,
        help="Path to the reference slice parquet (relative to cwd or this unit folder).",
    )
    p.add_argument(
        "--cur-parquet",
        type=Path,
        required=True,
        help="Path to the current slice parquet (relative to cwd or this unit folder).",
    )

    p.add_argument("--simulate-issues", action="store_true")
    p.add_argument("--severity", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drift-bins", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref_parquet = resolve_input_path(args.ref_parquet)
    cur_parquet = resolve_input_path(args.cur_parquet)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {args.experiment}")

    df_ref_raw = load_taxi_table(ref_parquet)
    df_cur_raw = load_taxi_table(cur_parquet)

    if args.simulate_issues:
        df_cur_raw = corrupt_current_slice(df_cur_raw, seed=args.seed, severity=args.severity)

    if args.model_uri:
        model_uri = args.model_uri
        model_source_run_id = None
    else:
        model_uri, model_source_run_id = latest_model_uri(client, exp.experiment_id)

    model_feature_cols = (
        load_feature_cols_from_run(model_source_run_id)
        if model_source_run_id
        else None
    )

    run_name = args.run_name or "monitor"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "pipeline_step": "monitor",
            "simulated_issues": str(bool(args.simulate_issues)).lower(),
            "severity": args.severity,
            "model_uri_used": model_uri,
            **({"model_source_run_id": model_source_run_id} if model_source_run_id else {}),
        })

        ds_ref = mlflow.data.from_pandas(
            df_ref_raw,
            source=LocalArtifactDatasetSource(str(ref_parquet)),
            name="raw_reference",
        )
        ds_cur = mlflow.data.from_pandas(
            df_cur_raw,
            source=LocalArtifactDatasetSource(str(cur_parquet)),
            name="raw_current",
        )
        mlflow.log_input(ds_ref, context="raw_reference")
        mlflow.log_input(ds_cur, context="raw_current")

        # Integrity checks on current
        checks = run_integrity_checks(df_cur_raw)
        for name, table in checks.tables.items():
            mlflow.log_table(table, artifact_file=f"checks/{name}.json")

        # only log a small "dashboard" subset as metrics
        TOP5 = [
            "schema_missing_cols",
            "missing_frac_max",
            "range_worst_bad_frac",
            "domain_worst_bad_frac",
            "duration_neg_frac",
        ]
        mlflow.log_metrics(
            {
                f"integrity_{k}": checks.metrics.get(k, float("nan"))
                for k in TOP5
            }
        )


        # Drift in raw space (selected cols)
        raw_cols = [c for c in [
            "passenger_count", "trip_distance", "fare_amount", "payment_type",
            "RatecodeID", "PULocationID", "DOLocationID", "tip_amount", "total_amount",
        ] if c in df_ref_raw.columns and c in df_cur_raw.columns]

        drift_raw, drift_raw_metrics = compute_drift_report(
            df_ref_raw[raw_cols], df_cur_raw[raw_cols], bins=args.drift_bins
        )
        mlflow.log_metrics({f"drift_raw_{k}": v for k, v in drift_raw_metrics.items()})
        mlflow.log_table(drift_raw, artifact_file="drift/raw.json")


        log_violin_plots_ref_vs_cur(df_ref_raw,df_cur_raw)
        # Model feature space
        Xcur, ycur, feature_cols = make_tip_frame(df_cur_raw, credit_card_only=True)
        Xcur = cast_ints_to_float(Xcur)

        #Consider logging drift in feature space.

        # drift_feat, drift_feat_metrics = compute_drift_report(
        #     ref_frame[feature_cols], cur_frame[feature_cols], bins=args.drift_bins
        # )
        # mlflow.log_metrics({f"drift_feat_{k}": v for k, v in drift_feat_metrics.items()})
        # mlflow.log_table(drift_feat, artifact_file="drift/features.json")

        # Evaluate model on current with stable feature order when available.
        if model_feature_cols:
            Xeval = align_feature_frame(Xcur, model_feature_cols)
            mlflow.set_tag("feature_alignment", "baseline_run_feature_cols")
        else:
            Xeval = Xcur[feature_cols].copy()
            mlflow.set_tag("feature_alignment", "current_slice_feature_cols")

        eval_df = Xeval.copy()
        eval_df["target"] = ycur

        res = mlflow.models.evaluate(
            model=model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
        )

        # Also log a quick "degradation" number if we can find baseline RMSE from source run
        cur_rmse = res.metrics.get("root_mean_squared_error")
        if model_source_run_id and cur_rmse is not None:
            base = client.get_run(model_source_run_id)
            base_rmse = base.data.metrics.get("root_mean_squared_error")
            if base_rmse is not None and base_rmse > 0:
                mlflow.log_metric("rmse_increase_pct_vs_baseline", float((cur_rmse - base_rmse) / base_rmse))

        # Record decision hint for retrain script
        mlflow.log_dict(
            {
                "model_uri": model_uri,
                "cur_rmse": cur_rmse,
                "baseline_run": model_source_run_id,
            },
            artifact_file="monitor_summary.json",
        )


if __name__ == "__main__":
    main()
