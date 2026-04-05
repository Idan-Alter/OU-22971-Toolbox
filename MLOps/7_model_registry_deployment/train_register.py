"""
train_register.py

train + register + alias two simple sklearn regressors in MLflow Model Registry.

What this script does:
1) Loads a toy regression dataset from CSV.
2) Splits into train/test.
3) Trains two models:
   - DecisionTreeRegressor  (treated as the initial "production" model)
   - LinearSVR              (treated as the "candidate" model)
4) Logs each model to MLflow Tracking AND registers it into a single Registered Model family.
5) Immediately attaches model version tags and descriptions:
   - algo, rmse, r2, feature list
6) Sets registry aliases:
   - production -> tree version
   - candidate  -> svr version

Notes:
- `await_registration_for=300` makes `model_info.registered_model_version` reliably available
  immediately after `log_model` returns.
- Serving is separate:
    mlflow models serve -m "models:/<model-name>@production" -p 5001 --env-manager local

Example:
  python train_register.py --data data/toy_regression.csv --tracking-uri http://localhost:5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, yhat)))


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/toy_regression.csv"))
    p.add_argument("--tracking-uri", type=str, default="http://localhost:5000")
    p.add_argument("--experiment", type=str, default="7_registry_deployment_demo")
    p.add_argument("--model-name", type=str, default="toy_registry_demo_model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.25)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    y = df["y"].to_numpy(dtype=float)
    X = df.drop(columns=["y"])
    feature_cols = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    ensure_registered_model(client, args.model_name)
    client.set_registered_model_tag(args.model_name, "task", "toy_regression")
    client.set_registered_model_tag(args.model_name, "part", "MLOps")
    client.set_registered_model_tag(args.model_name, "unit", "7")

    # ----------------------------
    # Model A: DecisionTreeRegressor (production)
    # ----------------------------
    tree_algo = "decision_tree"
    tree = DecisionTreeRegressor(
        random_state=args.seed,
        max_depth=None,
        min_samples_leaf=10,
    )

    with mlflow.start_run(run_name="train_tree") as run:
        tree.fit(X_tr, y_tr)
        pred = tree.predict(X_te)

        rmse_score = rmse(y_te, pred)
        r2 = float(r2_score(y_te, pred))

        sig = infer_signature(X_tr, tree.predict(X_tr))

        mlflow.log_param("algo", tree_algo)
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("r2", r2)

        model_info = mlflow.sklearn.log_model(
            sk_model=tree,
            name="model",
            registered_model_name=args.model_name,
            input_example=X_tr.head(5),
            signature=sig,
            await_registration_for=300,
        )

        v_tree = str(model_info.registered_model_version)
        client.set_model_version_tag(args.model_name, v_tree, "algo", tree_algo)
        client.set_model_version_tag(args.model_name, v_tree, "rmse", f"{rmse_score:.6f}")
        client.set_model_version_tag(args.model_name, v_tree, "r2", f"{r2:.6f}")
        client.set_model_version_tag(args.model_name, v_tree, "features", ",".join(feature_cols))
        client.update_model_version(
            name=args.model_name,
            version=v_tree,
            description=(
                "DecisionTreeRegressor\n"
                f"- rmse: {rmse_score:.6f}\n"
                f"- r2: {r2:.6f}\n"
                f"- run_id: {run.info.run_id}\n"
            ),
        )

    # ----------------------------
    # Model B: LinearSVR (candidate)
    # ----------------------------
    svr_algo = "linear_svr"
    svr = LinearSVR(C=1.0, epsilon=0.2, random_state=args.seed, max_iter=20000)

    with mlflow.start_run(run_name="train_svr") as run:
        svr.fit(X_tr, y_tr)
        pred = svr.predict(X_te)

        rmse_score = rmse(y_te, pred)
        r2 = float(r2_score(y_te, pred))

        sig = infer_signature(X_tr, svr.predict(X_tr))

        mlflow.log_param("algo", svr_algo)
        mlflow.log_metric("rmse", rmse_score)
        mlflow.log_metric("r2", r2)

        model_info = mlflow.sklearn.log_model(
            sk_model=svr,
            name="model",
            registered_model_name=args.model_name,
            input_example=X_tr.head(5),
            signature=sig,
            await_registration_for=300,
        )

        v_svr = str(model_info.registered_model_version)
        client.set_model_version_tag(args.model_name, v_svr, "algo", svr_algo)
        client.set_model_version_tag(args.model_name, v_svr, "rmse", f"{rmse_score:.6f}")
        client.set_model_version_tag(args.model_name, v_svr, "r2", f"{r2:.6f}")
        client.set_model_version_tag(args.model_name, v_svr, "features", ",".join(feature_cols))
        client.update_model_version(
            name=args.model_name,
            version=v_svr,
            description=(
                "LinearSVR\n"
                f"- rmse: {rmse_score:.6f}\n"
                f"- r2: {r2:.6f}\n"
                f"- run_id: {run.info.run_id}\n"
            ),
        )

    # ----------------------------
    # Aliases
    # ----------------------------
    client.set_registered_model_alias(args.model_name, "production", v_tree)
    client.set_registered_model_alias(args.model_name, "candidate", v_svr)

    print("\n=== DONE ===")
    print(f"Model: {args.model_name}")
    print(f"  production -> v{v_tree}  ({tree_algo})")
    print(f"  candidate  -> v{v_svr}  ({svr_algo})")
    print("\nServe production:")
    print(f'  $env:MLFLOW_TRACKING_URI = "{args.tracking_uri}"')
    print(f'  mlflow models serve -m "models:/{args.model_name}@production" -p 5001 --env-manager local')


if __name__ == "__main__":
    main()
