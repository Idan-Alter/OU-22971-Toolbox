# MLOps Unit 5 - Hyperparameter Tuning with Optuna

## Setup

1. Activate the conda env:

   ```bash
   conda activate 22971-mlflow
   ```
  
2. Start the tracking server:

   ```powershell
   mlflow server --workers 1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns
   ```
3. Open the UI at http://localhost:5000.

---
## Prerequisites

1. Review the [**XGBoost** crash course notebook](0_xgboost_crash_course.ipynb).
   
2. Review the [breast cancer **data EDA** notebook](1_breast_cancer_eda.ipynb).

---

## Hyperparameter tuning

ML models come with many knobs (max tree depth, learning rate, regularization rate,...).
So far we've picked them either:

- manually ("magic numbers")
- via grid search

There are better ways to do this.

**Hyperparameter tuning** = searching that space deliberately, using an optimization algorithm that avoids wasting trials.

---

## Optuna

**Optuna** is a popular hyperparameter optimization framework that plays nice with MLflow.

### Mental model

- A **study** is one optimization problem:
   > "Find the best hyperparameters for this model + task + objective."

   It owns:
   - all trials
   - optimization direction (`maximize` / `minimize`)
   - sampler 
   - pruner

- A **trial** is one attempted parameter configuration.

   It:
   1. samples hyperparameters
   2. runs training + evaluation
   3. returns **one score**


- The **sampler** **decides** which parameter configuration to try in the next trial. 
  
  The default algorithm (TPE) is adaptive: it learns from past trials and pushes future ones towards promising regions.

- The **pruner** **monitors** a trial **mid-run** and aborts it if it is unlikely to become competitive.

   in XGBoost terms: 
   >"If after X **boosting rounds** the validation loss is not better than Y, `break`."

- The **objective** is the callback Optuna drives:

   ```python
   def objective(trial):
      ... #ML logic
      return score
   ```
   Rules:
   - Wrap all your ML logic (`model.fit` + `model.evaluate`) into it.
   - Return a single scalar.
   - **You define the score** (*Accuracy, validation AUC, RMSE, etc...*).

   Optuna calls `objective` repeatedly with different hyperparameter values and ranks the results by score.

## Demo

1. Examine `optuna_xgboost_mlflow.py`. Of note:
   - **3-way** data split: train, validation and test sets is essential: the validation set is used to optimize the hyperparameters.
   - `TPESampler` is a good default, but there might be a better choice: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html.
   - `MedianPruner`: stop a run if it is currently worse than the median of all completed runs.
   - `trial.suggest_X` calls the sampler to generate fresh parameter values.
     - Search space bounds are **hyper-hyperparameters**: choose them pragmatically based on past experience.
   - `OptunaPruningCallback` reports AUC to the pruner **after each boosting round**.
   - `objective` reports `best_auc` to Optuna: the AUC of the best iteration in this trial.
   - `study.optimize` starts the search.
   - The final model is trained on the train **and validation** splits.
2. run `optuna_xgboost_mlflow.py`.
3. Examine the run in the mlflow UI.
   - compare child runs within one run (Optuna study)
   - look at parallel coordinates plots (all knobs + val_auc)
   - draw conclusions (data set specific):
     - The problem is easy: many parameter configurations get perfect AUC.
     - `lambda`, `alpha` should be small.
     - `gamma` has a weak effect on AUC.
     - `max_depth > 5` is better.
     - ...
   
