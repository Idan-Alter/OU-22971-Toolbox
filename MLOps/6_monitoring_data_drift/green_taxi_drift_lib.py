"""
MLOps Unit 6 (Drift): shared utilities.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient


UNIT_DIR = Path(__file__).resolve().parent
RAW_DATETIME_COLS = ["lpep_pickup_datetime", "lpep_dropoff_datetime"]

# Policy objects (assumptions) live at the top.
# Note: we intentionally do NOT require a "month" column; slice identity should live in metadata.
EXPECTED_SCHEMA: Dict[str, str] = {
    "ehail_fee": "object",
    "RatecodeID": "float64",
    "store_and_fwd_flag": "object",
    "trip_type": "float64",
    "payment_type": "float64",
    "passenger_count": "float64",
    "congestion_surcharge": "float64",
    "DOLocationID": "int64",
    "PULocationID": "int64",
    "lpep_pickup_datetime": "datetime64[us]",
    "lpep_dropoff_datetime": "datetime64[us]",
    "VendorID": "int64",
    "extra": "float64",
    "fare_amount": "float64",
    "trip_distance": "float64",
    "tolls_amount": "float64",
    "tip_amount": "float64",
    "mta_tax": "float64",
    "total_amount": "float64",
    "improvement_surcharge": "float64",
}

RANGE_SPECS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("trip_distance", 0.0, 200.0),
    ("fare_amount", 0.0, 500.0),
    ("tip_amount", 0.0, 200.0),
    ("tolls_amount", 0.0, 200.0),
    ("total_amount", 0.0, 1000.0),
    ("passenger_count", 0.0, 10.0),
    # duration_min is derived; if absent, range checks will skip it (datetime sanity covers duration too).
    ("duration_min", 0.0, 360.0),
]

def cast_ints_to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Avoid MLflow evaluate/SHAP issues with pandas nullable Int64 by logging features as float64."""
    X = X.copy()
    int_cols = [c for c in X.columns if pd.api.types.is_integer_dtype(X[c])]
    if int_cols:
        X[int_cols] = X[int_cols].astype("float64")
    return X


def resolve_input_path(path: Path | str, *, anchor_dir: Path = UNIT_DIR) -> Path:
    """
    Resolve a user-supplied path against the cwd first, then this unit folder.

    This keeps example paths like TLC_data/... working whether the script is run
    from the repo root or from inside the unit directory.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()

    cwd_candidate = p.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (anchor_dir / p).resolve()

def load_taxi_table(path: Path) -> pd.DataFrame:
    """Load TLC-like data from a given path (parquet preferred), normalizing datetime columns if present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(
                "Parquet support missing. Install 'pyarrow' (recommended) or 'fastparquet'. "
                "Example: conda install -c conda-forge pyarrow"
            ) from e
    elif suf == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (expected .parquet or .csv)")

    for c in RAW_DATETIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


# -----------------------------
# Feature engineering
# -----------------------------


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap calendar features and a duration_min column derived from pickup/dropoff timestamps."""
    out = df.copy()
    for c in RAW_DATETIME_COLS:
        if c not in out.columns:
            continue
        dt = pd.to_datetime(out[c], errors="coerce")
        prefix = c.replace("_datetime", "")
        out[f"{prefix}_year"] = dt.dt.year.astype("Int64")
        out[f"{prefix}_month"] = dt.dt.month.astype("Int64")
        out[f"{prefix}_weekday"] = dt.dt.dayofweek.astype("Int64")
        out[f"{prefix}_hour"] = dt.dt.hour.astype("Int64")

    if all(c in out.columns for c in RAW_DATETIME_COLS):
        dur = (out["lpep_dropoff_datetime"] - out["lpep_pickup_datetime"]).dt.total_seconds() / 60.0
        out["duration_min"] = dur

    return out


def make_tip_frame(
    df_raw: pd.DataFrame, *, credit_card_only: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return (X, y, feature_cols) for a simple tip regression task using numeric-only features."""
    df = add_datetime_features(df_raw)

    if credit_card_only and "payment_type" in df.columns:
        df = df[df["payment_type"] == 1].copy()

    if "tip_amount" not in df.columns:
        raise ValueError("Expected column 'tip_amount'.")

    # Robust target extraction: do NOT depend on tip_amount being numeric dtype already.
    y = (
        pd.to_numeric(df["tip_amount"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # Numeric-only features; drop obvious post-hoc leakage totals.
    num = df.select_dtypes(include=["number"]).copy()
    X = num.drop(columns=["tip_amount", "total_amount"], errors="ignore")

    feature_cols = list(X.columns)
    return X, y, feature_cols


# -----------------------------
# Integrity checks
# -----------------------------


@dataclass
class CheckResult:
    metrics: Dict[str, float]
    tables: Dict[str, pd.DataFrame]


def _expected_family(exp: str) -> str:
    exp = str(exp).strip().lower()
    if exp.startswith("datetime64"):
        return "datetime"
    if exp in {"object", "string"}:
        return "string"
    if exp.startswith("int") or exp.startswith("float") or exp in {"number", "numeric"}:
        return "numeric"
    if exp in {"bool", "boolean"}:
        return "bool"
    if exp in {"category"}:
        return "category"
    return "exact"


def _family_ok(actual_dtype: Any, expected: str) -> bool:
    t = pd.api.types
    fam = _expected_family(expected)
    if fam == "datetime":
        return t.is_datetime64_any_dtype(actual_dtype)
    if fam == "numeric":
        return t.is_numeric_dtype(actual_dtype)
    if fam == "string":
        # TLC flags often come as object; some engines may yield pandas string dtype or category.
        return t.is_object_dtype(actual_dtype) or t.is_string_dtype(actual_dtype) or t.is_categorical_dtype(actual_dtype)
    if fam == "bool":
        return t.is_bool_dtype(actual_dtype)
    if fam == "category":
        return t.is_categorical_dtype(actual_dtype)
    # fam == "exact"
    return str(actual_dtype) == str(expected)


def run_integrity_checks(
    df_raw: pd.DataFrame,
    *,
    expected_schema: Optional[Dict[str, str]] = None,
    zone_lookup_path: Optional[Path] = None,
) -> CheckResult:
    """Run cheap schema/range/domain/datetime checks and return loggable tables + scalar metrics."""
    df = df_raw.copy()
    metrics: Dict[str, float] = {}
    tables: Dict[str, pd.DataFrame] = {}

    # ---- schema checks (presence + dtype)
    schema = expected_schema or EXPECTED_SCHEMA
    present_cols = set(df.columns)
    expected_cols = set(schema.keys())

    missing = sorted(expected_cols - present_cols)
    extra = sorted(present_cols - expected_cols)

    dtype_rows: List[Dict[str, Any]] = []
    bad_family = 0
    bad_exact = 0

    for col, exp_dtype in schema.items():
        if col not in df.columns:
            continue
        actual_dtype = df[col].dtype
        actual_str = str(actual_dtype)

        family_ok = _family_ok(actual_dtype, exp_dtype)
        exact_ok = (actual_str == str(exp_dtype))

        if not family_ok:
            bad_family += 1
        if not exact_ok:
            bad_exact += 1

        dtype_rows.append(
            {
                "column": col,
                "expected_dtype": str(exp_dtype),
                "actual_dtype": actual_str,
                "family_ok": bool(family_ok),
                "exact_match": bool(exact_ok),
            }
        )

    tables["schema_presence"] = pd.DataFrame({"missing_column": missing})
    tables["schema_extra_columns"] = pd.DataFrame({"extra_column": extra})
    tables["schema_dtypes"] = pd.DataFrame(
        dtype_rows,
        columns=["column", "expected_dtype", "actual_dtype", "family_ok", "exact_match"],
    )

    metrics["schema_missing_cols"] = float(len(missing))
    metrics["schema_extra_cols"] = float(len(extra))
    metrics["schema_bad_family_dtypes"] = float(bad_family)
    metrics["schema_bad_exact_dtypes"] = float(bad_exact)

    # ---- missingness
    if df.shape[1] == 0:
        tables["missingness"] = pd.DataFrame(
            columns=["dtype", "missing_frac", "missing_count", "n_unique"]
        )
        metrics["missing_frac_mean"] = float("nan")
        metrics["missing_frac_max"] = float("nan")
    else:
        miss = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing_frac": df.isna().mean(),
                "missing_count": df.isna().sum(),
                "n_unique": df.nunique(dropna=False),
            }
        ).sort_values("missing_frac", ascending=False)
        tables["missingness"] = miss
        metrics["missing_frac_mean"] = float(np.nanmean(df.isna().mean().to_numpy()))
        metrics["missing_frac_max"] = float(np.nanmax(df.isna().mean().to_numpy()))

    # ---- duplicates
    dup = int(df.duplicated().sum()) if len(df) else 0
    metrics["duplicate_rows"] = float(dup)
    metrics["duplicate_rows_frac"] = float(dup / max(len(df), 1))

    # ---- range checks (soft)
    def bad_frac_num(col: str, lo: Optional[float], hi: Optional[float]) -> Tuple[float, float, float]:
        if col not in df.columns:
            return np.nan, np.nan, np.nan
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.dropna()
        if valid.empty:
            return 1.0, np.nan, np.nan
        bad = pd.Series(False, index=valid.index)
        if lo is not None:
            bad |= valid < lo
        if hi is not None:
            bad |= valid > hi
        return float(bad.mean()), float(valid.min()), float(valid.max())

    rows: List[Dict[str, Any]] = []
    for col, lo, hi in RANGE_SPECS:
        bf, mn, mx = bad_frac_num(col, lo, hi)
        if not np.isnan(bf):
            rows.append(
                {"column": col, "lo": lo, "hi": hi, "bad_frac": bf, "min": mn, "max": mx}
            )

    if rows:
        rng = pd.DataFrame(rows).sort_values("bad_frac", ascending=False)
        tables["range_checks"] = rng
        metrics["range_worst_bad_frac"] = float(rng["bad_frac"].max())
        metrics["range_any_bad_cols"] = float((rng["bad_frac"] > 0).sum())

    # ---- domain checks
    dom_specs: List[Tuple[str, Iterable[Any]]] = [
        ("store_and_fwd_flag", ["Y", "N"]),
        ("payment_type", [1, 2, 3, 4, 5, 6]),
        ("trip_type", [1, 2]),
        ("RatecodeID", [1, 2, 3, 4, 5, 6]),
    ]

    drows: List[Dict[str, Any]] = []
    for col, allowed in dom_specs:
        if col not in df.columns:
            continue
        s = df[col]
        allowed_set = set(allowed)
        bad = ~s.isna() & ~s.isin(allowed_set)
        drows.append(
            {
                "column": col,
                "bad_frac": float(bad.mean()) if len(s) else 0.0,
                "bad_count": int(bad.sum()) if len(s) else 0,
                "n_unique": int(s.nunique(dropna=True)) if len(s) else 0,
            }
        )

    if drows:
        dom = pd.DataFrame(drows).sort_values("bad_frac", ascending=False)
        tables["domain_checks"] = dom
        metrics["domain_worst_bad_frac"] = float(dom["bad_frac"].max())
        metrics["domain_any_bad_cols"] = float((dom["bad_count"] > 0).sum())

    # ---- datetime sanity
    if all(c in df.columns for c in RAW_DATETIME_COLS):
        pickup = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")
        dropoff = pd.to_datetime(df["lpep_dropoff_datetime"], errors="coerce")
        dur = (dropoff - pickup).dt.total_seconds() / 60.0

        metrics["duration_neg_frac"] = float((dur < 0).mean()) if len(dur) else 0.0
        metrics["duration_over_6h_frac"] = float((dur > 360).mean()) if len(dur) else 0.0
        metrics["duration_nan_frac"] = float(dur.isna().mean()) if len(dur) else 0.0

        tables["datetime_checks"] = pd.DataFrame(
            [
                {"check": "duration_negative", "frac": metrics["duration_neg_frac"]},
                {"check": "duration_over_6h", "frac": metrics["duration_over_6h_frac"]},
                {"check": "duration_nan", "frac": metrics["duration_nan_frac"]},
            ]
        )

    # ---- zone validity (optional)
    if zone_lookup_path and Path(zone_lookup_path).exists():
        zones = pd.read_csv(zone_lookup_path)
        if "LocationID" in zones.columns:
            valid_ids = set(
                pd.to_numeric(zones["LocationID"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            for col in ["PULocationID", "DOLocationID"]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
                    bad = ~s.isin(valid_ids)
                    metrics[f"{col}_unknown_frac"] = float(bad.mean()) if len(s) else 0.0

    return CheckResult(metrics=metrics, tables=tables)


# -----------------------------
# Drift metrics
# -----------------------------


def _safe_probs(counts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert histogram bin counts into a smoothed probability vector for KL/PSI-style computations."""
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    p = counts / max(total, 1.0)
    p = np.clip(p, eps, 1.0)
    return p / float(p.sum())


def psi_numeric(ref: np.ndarray, cur: np.ndarray, *, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) for numeric samples using reference-quantile bins."""
    # Robust coercion: treat non-numeric as missing rather than crashing.
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(cur), errors="coerce").to_numpy(dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 50 or cur.size < 50:
        return float("nan")

    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, qs)

    # enforce strictly increasing interior edges (ties in ref quantiles are common)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    # include all current values, even out-of-range vs ref
    edges[0] = -np.inf
    edges[-1] = np.inf

    rc, _ = np.histogram(ref, bins=edges)
    cc, _ = np.histogram(cur, bins=edges)
    p = _safe_probs(rc)
    q = _safe_probs(cc)

    return float(np.sum((p - q) * np.log(p / q)))


def js_divergence_categorical(ref: pd.Series, cur: pd.Series, *, eps: float = 1e-8) -> float:
    """Compute Jensen–Shannon divergence between two categorical distributions."""
    r = pd.Series(ref).astype("object")
    c = pd.Series(cur).astype("object")

    keys = pd.Index(sorted(set(r.dropna().unique()).union(set(c.dropna().unique()))))
    if len(keys) == 0:
        return float("nan")

    pr = (
        r.value_counts(normalize=True)
        .reindex(keys, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    pc = (
        c.value_counts(normalize=True)
        .reindex(keys, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    pr = np.clip(pr, eps, 1.0)
    pc = np.clip(pc, eps, 1.0)
    pr = pr / float(pr.sum())
    pc = pc / float(pc.sum())

    m = 0.5 * (pr + pc)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))

    return float(0.5 * kl(pr, m) + 0.5 * kl(pc, m))


def compute_drift_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    *,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    bins: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Build a per-feature drift table and summary metrics comparing a reference dataframe to a current dataframe."""
    if numeric_cols is None:
        numeric_cols = [
            c for c in ref_df.columns if pd.api.types.is_numeric_dtype(ref_df[c])
        ]
    if categorical_cols is None:
        categorical_cols = [c for c in ref_df.columns if c not in numeric_cols]

    rows: List[Dict[str, Any]] = []

    for c in numeric_cols:
        if c in ref_df.columns and c in cur_df.columns:
            ref_x = ref_df[c]
            cur_x = cur_df[c]
            rows.append(
                {
                    "feature": c,
                    "type": "numeric",
                    "psi": psi_numeric(ref_x.to_numpy(), cur_x.to_numpy(), bins=bins),
                    "ref_mean": float(np.nanmean(pd.to_numeric(ref_x, errors="coerce"))),
                    "cur_mean": float(np.nanmean(pd.to_numeric(cur_x, errors="coerce"))),
                }
            )

    for c in categorical_cols:
        if c in ref_df.columns and c in cur_df.columns:
            rows.append(
                {
                    "feature": c,
                    "type": "categorical",
                    "jsd": js_divergence_categorical(ref_df[c], cur_df[c]),
                    "ref_unique": int(ref_df[c].nunique(dropna=True)),
                    "cur_unique": int(cur_df[c].nunique(dropna=True)),
                }
            )

    drift = pd.DataFrame(rows)
    if not drift.empty:
        drift = drift.sort_values(["type", "feature"])

    metrics: Dict[str, float] = {}
    if not drift.empty:
        num = drift[drift["type"] == "numeric"]
        cat = drift[drift["type"] == "categorical"]

        if not num.empty:
            metrics["psi_mean"] = float(np.nanmean(num["psi"]))
            metrics["psi_max"] = float(np.nanmax(num["psi"]))
            metrics["psi_gt_0_25"] = float((num["psi"] > 0.25).sum())

        if not cat.empty:
            metrics["jsd_mean"] = float(np.nanmean(cat["jsd"]))
            metrics["jsd_max"] = float(np.nanmax(cat["jsd"]))

    return drift, metrics

### Violin plots for selected numeric features (credit card only)
def log_violin_plots_ref_vs_cur(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    *,
    columns=("tip_amount", "fare_amount", "trip_distance"),
    artifact_file="plots/violin_ref_vs_cur_cc_only.png",
    clip_q: float = 0.995,
):
    """
    CC-only violins (ref vs cur) for selected columns.
    - Clips extreme high tail per feature at clip_q.
    - Excludes nonpositive values from the violin (so the shape isn't a spike at 0),
      but logs + annotates the <=0 fraction for each split/feature.
    Assumes it's called inside an active `mlflow.start_run()` context.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _prep(df: pd.DataFrame, split: str):
        if "payment_type" in df.columns:
            df = df[df["payment_type"] == 1].copy()  # CC only

        stats = {}  # feature -> nonpos_frac
        rows = []

        for c in columns:
            if c not in df.columns:
                continue

            x = pd.to_numeric(df[c], errors="coerce")
            x = x[np.isfinite(x)]

            nonpos_frac = float((x <= 0).mean()) if len(x) else float("nan")
            stats[c] = nonpos_frac
            mlflow.log_metric(f"cc_only_{split}_{c}_nonpos_frac", nonpos_frac)

            x = x[x > 0]
            if len(x) == 0:
                continue

            if 0.0 < clip_q < 1.0 and len(x) > 20:
                hi = float(np.quantile(x, clip_q))
                x = x.clip(upper=hi)

            rows.append(pd.DataFrame({"feature": c, "value": x, "split": f"{split} (CC only)"}))

        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return out, stats

    ref_df, ref_stats = _prep(df_ref, "reference")
    cur_df, cur_stats = _prep(df_cur, "current")

    plot_df = pd.concat([ref_df, cur_df], ignore_index=True).dropna()
    if plot_df.empty:
        return

    plt.figure(figsize=(2.2 * len(columns) + 2, 5))
    ax = sns.violinplot(
        data=plot_df,
        x="feature",
        y="value",
        hue="split",
        cut=0,
        inner="quartile",
        scale="width",
    )

    # Safe tick relabel: fix locator first, then set labels.
    cats = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(range(len(cats)))

    labels = []
    for c in cats:
        r = ref_stats.get(c)
        u = cur_stats.get(c)
        if r is None or u is None or (not np.isfinite(r)) or (not np.isfinite(u)):
            labels.append(c)
        else:
            labels.append(f"{c}\n<=0: ref {r:.1%}, cur {u:.1%}")
    ax.set_xticklabels(labels)

    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), artifact_file)
    plt.close()



# -----------------------------
# Issue injection (for demos)
# -----------------------------


def corrupt_current_slice(
    df_raw: pd.DataFrame, *, seed: int = 0, severity: str = "medium"
) -> pd.DataFrame:
    """Inject controlled issues into a slice (missingness/range/domain/schema/datetime) for demo purposes."""
    rng = np.random.default_rng(seed)
    df = df_raw.copy()
    n = len(df)
    if n == 0:
        return df

    def pick(frac: float) -> np.ndarray:
        k = max(1, int(frac * n))
        return rng.choice(n, size=k, replace=False)

    # Inject missingness
    for col in [c for c in ["trip_distance", "fare_amount", "payment_type", "tip_amount"] if c in df.columns]:
        idx = pick(0.01 if severity == "low" else 0.02)
        df.loc[df.index[idx], col] = np.nan

    # Range bugs
    if "trip_distance" in df.columns:
        idx = pick(0.01)
        td = pd.to_numeric(df.loc[df.index[idx], "trip_distance"], errors="coerce")
        td = td.fillna(1.0)
        df.loc[df.index[idx], "trip_distance"] = -np.abs(td)

    # Domain bug
    if severity in {"medium", "high"} and "payment_type" in df.columns:
        idx = pick(0.01)
        df.loc[df.index[idx], "payment_type"] = 99

    # Datetime ordering bug
    if severity in {"medium", "high"} and all(c in df.columns for c in RAW_DATETIME_COLS):
        idx = pick(0.005)
        df.loc[df.index[idx], "lpep_dropoff_datetime"] = (
            pd.to_datetime(df.loc[df.index[idx], "lpep_pickup_datetime"], errors="coerce")
            - pd.to_timedelta(5, unit="m")
        )

    # Schema break (drop a column)
    if severity == "high":
        for col in ["fare_amount", "trip_distance", "PULocationID"]:
            if col in df.columns:
                df = df.drop(columns=[col])
                break

    return df


# -----------------------------
# Feature alignment
# -----------------------------


def load_feature_cols_from_run(
    run_id: str,
    *,
    artifact_path: str = "feature_cols.json",
) -> Optional[List[str]]:
    """Load feature column order from a run artifact, if present."""
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
        )
    except Exception:
        return None

    try:
        with open(local_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    cols = payload.get("feature_cols")
    if not isinstance(cols, list) or not cols:
        return None

    return [str(c) for c in cols]


def align_feature_frame(X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Select and order columns to match training-time feature order."""
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return X.loc[:, feature_cols].copy()


# -----------------------------
# Baseline model lookup (robust w/o registry)
# -----------------------------

def latest_model_uri(client: MlflowClient, experiment_id: str) -> Tuple[str, str]:
    """
    Return (model_uri, run_id) for the latest FINISHED train/retrain run.

      1) query latest FINISHED runs
      2) filter in Python by tags.pipeline_step in {'train','retrain'}
      3) prefer tag 'model_uri', else fall back to runs:/<run_id>/model if artifact exists
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=200,  
    )

    for r in runs:
        step = r.data.tags.get("pipeline_step", "")
        if step not in {"train", "retrain"}:
            continue

        run_id = r.info.run_id

        # 1) Prefer explicit tag if present
        tag_uri = r.data.tags.get("model_uri")
        if tag_uri:
            return tag_uri, run_id

        # 2) Fallback: deterministic artifact path "model" (if it exists)
        try:
            arts = client.list_artifacts(run_id, path="")
            if any(a.path == "model" for a in arts):
                return f"runs:/{run_id}/model", run_id
        except Exception:
            pass

    raise ValueError("No finished train/retrain run found with a logged model artifact.")
