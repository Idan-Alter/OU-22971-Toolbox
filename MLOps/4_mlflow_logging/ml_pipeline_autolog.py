"""
ml_pipeline_autolog.py

Scripted version of the ML logic originally developed
in an exploratory Jupyter notebook.

This script does not generate or clean data. It consumes files produced by
generate_data.py from a data directory (default: ./data/) and runs:

- train/test split
- sklearn Pipeline: StandardScaler + SVC
- GridSearchCV hyperparameter search
- classification report + confusion matrix
- MLflow logging: sklearn autolog (no manual logging calls)

Expected input CSV format:
- columns: x1, x2, y
- default file: ./data/clean.csv

Example:
    python ml_pipeline_autolog.py --data data1/clean.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Use a non-interactive backend by default.
# This prevents matplotlib from trying to open GUI windows,
plt.switch_backend("Agg")

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)


def train_and_evaluate(df: pd.DataFrame) -> None:
    X = df[["x1", "x2"]].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    pipe = make_pipeline(StandardScaler(), SVC())

    param_grid = {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.1, 0.01, 0.001],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print(report)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.figure_.tight_layout()
    disp.figure_.savefig("confusion_matrix.png")
    plt.close(disp.figure_)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(grid.cv_results_).to_csv("gridsearch_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/clean.csv",
        help="Path to CSV with columns: x1, x2, y (default: data/clean.csv)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Enable interactive plot windows (optional).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="4_mlflow_autologging",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    # Switch back to an interactive backend only if explicitly requested.
    if args.show_plots:
        plt.switch_backend("TkAgg")  # or QtAgg if available

    df = pd.read_csv(args.data)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        train_and_evaluate(df)


if __name__ == "__main__":
    main()
