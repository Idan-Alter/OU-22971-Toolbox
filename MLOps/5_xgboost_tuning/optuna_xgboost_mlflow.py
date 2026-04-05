"""
optuna_xgboost_mlflow.py

1) Load a simple real-world tabular dataset (sklearn breast cancer).
2) Hyperparameter tuning with Optuna (TPE + MedianPruner).
3) Log runs to MLflow (one parent "study" run + one nested run per trial).
4) XGBoost training reports intermediate validation AUC to Optuna's pruner via a pruning callback.

Run:
  python optuna_xgboost_mlflow.py --n-trials 40
"""

import argparse

import numpy as np
import pandas as pd

import optuna


import xgboost as xgb

import mlflow
import mlflow.xgboost

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class OptunaPruningCallback(xgb.callback.TrainingCallback):
    def __init__(self, trial, data_name: str, metric_name: str):
        self.trial = trial
        self.data_name = data_name
        self.metric_name = metric_name

    def after_iteration(self, model, epoch, evals_log):
        value = evals_log[self.data_name][self.metric_name][-1]
        self.trial.report(float(value), step=epoch)

        if self.trial.should_prune():
            # side-effect before control-flow interruption
            mlflow.set_tag("trial_pruned", "TRUE")
            mlflow.log_metric("pruned_at_iteration", epoch)
            raise optuna.TrialPruned()

        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", type=str, default="http://localhost:5000")
    p.add_argument("--experiment", type=str, default="5_optuna_tuning")
    p.add_argument("--study-name", type=str, default="xgb_optuna_auc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--num-boost-round", type=int, default=400)
    p.add_argument("--log-curve", action="store_true")
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # --- Data (fixed split so trials are comparable)
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target.to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=args.seed, stratify=y_trainval
    )
    # train=60%, valid=20%, test=20%

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtrain_full = xgb.DMatrix(
        pd.concat([X_train, X_valid], axis=0),
        label=np.concatenate([y_train, y_valid]),
    )
    dtest = xgb.DMatrix(X_test, label=y_test)

    # --- Optuna study setup
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
    )

    # Parent run = the study container in MLflow.
    with mlflow.start_run(run_name=f"study_{study.study_name}"):
        mlflow.set_tag("optuna_study_name", study.study_name)
        mlflow.log_param("dataset", "sklearn_breast_cancer")
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("timeout_sec", args.timeout if args.timeout is not None else "None")
        mlflow.log_param("num_boost_round", args.num_boost_round)
        mlflow.log_param("sampler", "TPESampler")
        mlflow.log_param("pruner", "MedianPruner")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc", 
                "tree_method": "hist",
                "seed": args.seed,
                "verbosity": 0, #silent 

                # capacity 
                "max_depth": trial.suggest_int("max_depth", 2, 8), # 2 < depth < 8
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),

                # boosting schedule
                "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),

                # randomness (variance reduction)
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

                # L2 / L1 regularization
                "lambda": trial.suggest_float("lambda", 1e-8, 50.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            }

            # One MLflow child run per trial (nested under the parent study run).
            with mlflow.start_run(run_name=f"trial_{trial.number:04d}", nested=True):
                mlflow.set_tag("optuna_trial_number", trial.number)
                mlflow.set_tag("trial_pruned", "unknown")

                mlflow.log_params(params)
                mlflow.log_param("num_boost_round", args.num_boost_round)

                evals_result = {}

                # Reports intermediate valid AUC each boosting round for pruning.
                pruning_cb = OptunaPruningCallback(trial, data_name="valid", metric_name="auc")

                _booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=args.num_boost_round,
                    evals=[(dtrain, "train"), (dvalid, "valid")],
                    evals_result=evals_result,
                    callbacks=[pruning_cb],
                    verbose_eval=False,
                )

                mlflow.set_tag("trial_pruned", "FALSE")
                valid_auc = np.asarray(evals_result["valid"]["auc"], dtype=float)
                best_iter = int(valid_auc.argmax())
                best_auc = float(valid_auc[best_iter])

                mlflow.log_metric("best_val_auc", best_auc)
                mlflow.log_metric("best_iteration", best_iter)
                mlflow.log_metric("last_val_auc", float(valid_auc[-1]))

                if args.log_curve:
                    curve = pd.DataFrame(
                        {
                            "iter": np.arange(len(valid_auc)),
                            "train_auc": np.asarray(evals_result["train"]["auc"], dtype=float),
                            "valid_auc": valid_auc,
                        }
                    )
                    curve.to_csv("learning_curve.csv", index=False)
                    mlflow.log_artifact("learning_curve.csv")

                return best_auc

        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

        # Log best params to the parent run for easy UI scanning
        mlflow.log_metric("study_best_val_auc", float(study.best_value))
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # Final model:
        # - find best boosting iteration on validation
        # - retrain on (train + valid) for that many rounds
        best_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": args.seed,
            "verbosity": 0,
            **study.best_params,
        }

        probe_evals = {}
        _ = xgb.train(
            params=best_params,
            dtrain=dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dvalid, "valid")],
            evals_result=probe_evals,
            verbose_eval=False, #only errors.
        )
        valid_auc = np.asarray(probe_evals["valid"]["auc"], dtype=float)
        best_iter = int(valid_auc.argmax())

        booster_final = xgb.train(
            params=best_params,
            dtrain=dtrain_full,
            num_boost_round=best_iter + 1,
            verbose_eval=False,
        )

        pred_test = booster_final.predict(dtest)
        test_auc = float(roc_auc_score(y_test, pred_test))

        mlflow.log_metric("final_best_iteration", best_iter)
        mlflow.log_metric("final_test_auc", test_auc)
        mlflow.xgboost.log_model(booster_final, name="final_model")

    print("Done.")
    print(f"Best val AUC: {study.best_value:.5f}")
    print("Best params:", study.best_params)




if __name__ == "__main__":
    main()
