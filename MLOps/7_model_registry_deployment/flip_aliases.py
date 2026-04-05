"""
flip_aliases.py

promote by moving an MLflow Model Registry alias.

What this script does:
- Reads the current `production` and `candidate` aliases for a registered model.
- Moves `previous_production` to the current production version (rollback anchor).
- Promotes `candidate` by re-pointing `production` to the candidate version.

This is the entire "deployment switch" in MLflow: alias reassignment.

Example:
  python flip_aliases.py --tracking-uri http://localhost:5000 --model-name toy_registry_demo_model
"""
from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", type=str, default="http://localhost:5000")
    p.add_argument("--model-name", type=str, default="toy_registry_demo_model")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    prod = client.get_model_version_by_alias(args.model_name, "production")
    cand = client.get_model_version_by_alias(args.model_name, "candidate")

    client.set_registered_model_alias(args.model_name, "previous_production", prod.version)
    client.set_registered_model_alias(args.model_name, "production", cand.version)

    print(f"Flipped: production v{prod.version} -> v{cand.version}")
    print(f"Rollback target: previous_production -> v{prod.version}")


if __name__ == "__main__":
    main()
