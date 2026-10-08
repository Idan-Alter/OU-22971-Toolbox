"""Focused M1-M3 acceptance fixtures; run in 22971-mlflow."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock
from contextlib import ExitStack

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "MLOps/6_monitoring_data_drift"))
import green_taxi_drift_lib as lib
import check_drift


class RevisionTests(unittest.TestCase):
    def test_positional_labels(self):
        raw = pd.DataFrame({"tip_amount": [0, "2.5", "bad", None, np.inf, 8, -1],
                            "payment_type": [1, 1, 1, 1, 1, 2, 1],
                            "trip_distance": range(7)}, index=[4, 4, 4, 2, 2, 4, 4])
        X, y, cols = lib.make_tip_frame(raw)
        self.assertEqual(X.trip_distance.tolist(), [0, 1, 6])
        np.testing.assert_array_equal(y, [0, 2.5, -1])
        self.assertEqual(X.index.tolist(), [4, 4, 4])
        self.assertEqual(cols, ["payment_type", "trip_distance"])
        self.assertEqual(lib.tip_label_mask(raw)[2], {
            "label_eligible_rows": 6, "label_valid_rows": 3, "label_excluded_rows": 3})
        self.assertEqual(len(raw), 7)
        self.assertGreater(lib.run_integrity_checks(raw).metrics["missing_frac_max"], 0)

    def test_reference_rmse(self):
        for metrics in [{"root_mean_squared_error": 10}, {"new_rmse": 10},
                        {"root_mean_squared_error": 10, "new_rmse": 30, "baseline_rmse": 90}]:
            self.assertEqual(lib.rmse_comparison(12, metrics), (10., 20., "available"))
        for metrics in [{}, {"root_mean_squared_error": 0}, {"root_mean_squared_error": np.inf}]:
            self.assertIsNone(lib.rmse_comparison(12, metrics)[1])

    def test_categorical_drift_once(self):
        ids = ["payment_type", "RatecodeID", "PULocationID", "DOLocationID"]
        raw = pd.DataFrame({**{c: [1, 2, 1, 2] for c in ids}, "fare_amount": [1., 2., 3., 4.]})
        report, _ = lib.compute_drift_report(raw, raw, categorical_cols=ids, numeric_cols=["fare_amount"])
        self.assertEqual(len(report), 5)
        self.assertFalse(report["feature"].duplicated().any())
        self.assertEqual(set(report.loc[report["type"] == "categorical", "feature"]), set(ids))

    def test_monitor_coverage(self):
        for tips, expected in [([None, "bad"], "unavailable_no_valid_labels"), ([0., 2.], "available")]:
            raw = pd.DataFrame({"tip_amount": tips, "payment_type": [1, 1], "trip_distance": [1., 2.]})
            args = MagicMock(ref_parquet=Path("ref.parquet"), cur_parquet=Path("cur.parquet"),
                             tracking_uri="unused", experiment="fixture", model_uri="fixture-model",
                             simulate_issues=False, run_name="fixture", severity="low", drift_bins=10)
            artifacts = {}
            with ExitStack() as stack:
                stack.enter_context(patch.object(check_drift, "parse_args", return_value=args))
                stack.enter_context(patch.object(check_drift, "load_taxi_table", return_value=raw))
                stack.enter_context(patch.object(check_drift, "MlflowClient"))
                stack.enter_context(patch.object(check_drift, "log_violin_plots_ref_vs_cur"))
                for name in ["set_tracking_uri", "set_experiment", "start_run", "set_tags", "set_tag", "log_input", "log_metrics"]:
                    stack.enter_context(patch.object(check_drift.mlflow, name))
                tables = stack.enter_context(patch.object(check_drift.mlflow, "log_table"))
                stack.enter_context(patch.object(check_drift.mlflow.data, "from_pandas"))
                stack.enter_context(patch.object(check_drift.mlflow, "log_dict", side_effect=lambda d, artifact_file: artifacts.update({artifact_file: d})))
                evaluate = stack.enter_context(patch.object(check_drift.mlflow.models, "evaluate", return_value=MagicMock(metrics={"root_mean_squared_error": 1.})))
                check_drift.main()
            summary = artifacts["monitor_summary.json"]
            self.assertEqual(summary["performance_status"], expected)
            self.assertEqual(summary["label_eligible_rows"], 2)
            self.assertTrue(tables.called)
            if expected == "available":
                evaluate.assert_called_once()
                self.assertEqual(summary["cur_rmse"], 1.)
            else:
                evaluate.assert_not_called()
                self.assertIsNone(summary["cur_rmse"])
                self.assertEqual(summary["label_excluded_rows"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
