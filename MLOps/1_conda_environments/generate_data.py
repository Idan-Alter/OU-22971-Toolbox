# generate_data.py
"""
generate_data.py

Scripted version of the data-generation logic originally developed
in an exploratory Jupyter notebook.

Generates synthetic data, injects controlled corruption, applies a
manual cleaning step, and saves datasets and plots as reproducible
artifacts (CSV + PNG). Downstream code should consume the saved files,
not re-run generation.

Outputs (written to ./data/):
- original.csv / original.png
- dirty.csv
- clean.csv / clean.png

Example:
    python generate_data.py --seed 0 --cutoff 10 --outdir data
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_data(seed: int):
    X, y = make_blobs(
        n_samples=600,
        centers=2,
        n_features=2,
        cluster_std=2.0,
        random_state=seed,
    )
    return X, y


def make_dirty(X, y, seed, n_outliers=18, n_nans=10, label_noise=0.05):
    rng = np.random.default_rng(seed)

    X_dirty = X.copy()
    y_dirty = y.copy()

    idx = rng.choice(len(X_dirty), size=n_outliers, replace=False)
    X_dirty[idx] += rng.normal(0, 30, size=X_dirty[idx].shape)

    nan_idx = rng.choice(len(X_dirty), size=n_nans, replace=False)
    X_dirty[nan_idx, 0] = np.nan

    n_flip = int(label_noise * len(y_dirty))
    flip_idx = rng.choice(len(y_dirty), size=n_flip, replace=False)
    y_dirty[flip_idx] = 1 - y_dirty[flip_idx]

    return X_dirty, y_dirty


def manual_clean(X_dirty, y_dirty, cutoff):
    mask = (
        ~np.isnan(X_dirty).any(axis=1)
        & (np.abs(X_dirty[:, 0]) < cutoff)
        & (np.abs(X_dirty[:, 1]) < cutoff)
    )
    return X_dirty[mask], y_dirty[mask]


def to_df(X, y):
    return pd.DataFrame(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "y": y,
        }
    )


def save_scatter(X, y, title, path):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = generate_data(seed=args.seed)
    X_dirty, y_dirty = make_dirty(X, y, seed=args.seed)
    X_clean, y_clean = manual_clean(X_dirty, y_dirty, cutoff=args.cutoff)

    # Save CSVs
    to_df(X, y).to_csv(os.path.join(args.outdir, "original.csv"), index=False)
    to_df(X_dirty, y_dirty).to_csv(os.path.join(args.outdir, "dirty.csv"), index=False)
    to_df(X_clean, y_clean).to_csv(os.path.join(args.outdir, "clean.csv"), index=False)

    # Save plots
    save_scatter(
        X, y,
        title="Original data",
        path=os.path.join(args.outdir, "original.png"),
    )

    save_scatter(
        X_clean, y_clean,
        title=f"Cleaned data (cutoff = +/-{args.cutoff})",
        path=os.path.join(args.outdir, "clean.png"),
    )

    print(f"Saved artifacts to {args.outdir}")
    print("  original.csv, dirty.csv, clean.csv")
    print("  original.png, clean.png")


if __name__ == "__main__":
    main()
