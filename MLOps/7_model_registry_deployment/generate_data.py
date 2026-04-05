"""
generate_data.py

toy regression data generator

This script produces a single CSV file for downstream training scripts.
Output:

- A CSV with columns: x0..x{d-1}, y
- Written to `--out` (default: data/toy_regression.csv)

Example:
  python generate_data.py --seed 0 --n 2500 --d 6 --out data/toy_regression.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_regression


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/toy_regression.csv"))
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--noise", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)


    X, y = make_regression(
        n_samples=args.n,
        n_features=args.d,
        noise=args.noise,
        random_state=args.seed,
    )

    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(args.d)])
    y = pd.Series(y, name="y", dtype="float64")

    df = pd.concat([X, y], axis=1)
    df.to_csv(args.out, index=False)

    print(f"Wrote: {args.out}  shape={df.shape}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
