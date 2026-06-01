"""Smoke-test the 22971 grading environment."""

from __future__ import annotations

import importlib
import json


MODULES = [
    "numpy",
    "pandas",
    "sklearn",
    "xgboost",
    "pyarrow",
    "mlflow",
    "metaflow",
    "nannyml",
    "optuna",
    "ray",
    "torch",
    "torchvision",
    "matplotlib",
    "seaborn",
]


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    versions = {name: module_version(name) for name in MODULES}

    import mlflow
    import nannyml as nml
    import numpy as np
    import pandas as pd
    import ray
    import torch
    from metaflow import FlowSpec
    from sklearn.linear_model import LinearRegression

    assert hasattr(mlflow, "log_table")
    assert hasattr(nml, "MissingValuesCalculator")
    assert hasattr(nml, "UnseenValuesCalculator")
    assert hasattr(FlowSpec, "next")
    assert torch.distributed.is_available()

    x = pd.DataFrame({"x": [0.0, 1.0, 2.0], "cat": ["a", "b", "a"]})
    y = np.array([0.0, 1.0, 2.0])
    LinearRegression().fit(x[["x"]], y).predict(x[["x"]])

    ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
    try:
        assert ray.data.from_pandas(x).count() == len(x)
    finally:
        ray.shutdown()

    print(json.dumps(versions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
