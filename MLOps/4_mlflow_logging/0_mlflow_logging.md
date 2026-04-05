# MLOps Unit 4 - Logging

## Setup

1. Activate the conda env:

   ```bash
   conda activate 22971-mlflow
   ```

2. Generate **two distinct datasets**. Run:
   
   ```bash
   python generate_data.py --seed 0 --cutoff 10 --outdir data1
   ```

   Then, with a **different RNG seed**:

   ```bash
   python generate_data.py --seed 1 --cutoff 10 --outdir data2
   ```
    
3. Start the tracking server:

   ```powershell
   mlflow server --workers 1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns
   ```

   Notes:
   - Keep this shell open.
   - The backend and artifact paths are **relative to the current directory**.
   - Starting the server from a different folder creates a *different tracking database* with no history.

4. Open the UI at http://localhost:5000.

---

## Log *everything*

We now patch `ml_pipeline.py` so that **nothing important disappears**.

Our mental model going forward:

- **If it's small** -> log it.  
  *params, metrics, plots, reports*

- **If it's big** -> log its **metadata**.  
  *datasets, model checkpoints (DL)*

- **If it's big and important** -> log it anyway.

Logging is almost always negligible relative to training cost.
It only hurts when you log too often or too much.
When that happens: log less, or log async.


---


## Examine the MLflow logging pipeline `ml_pipeline_logging.py`

In VS Code:

1. Right-click `1_conda_environments\ml_pipeline.py`.
2. Click **"Select for Compare"**.
3. Right-click `4_mlflow_logging\ml_pipeline_logging.py`.
4. Click **"Compare with selected"**.

```diff
+ MLflow server / experiment setup
+ mlflow.start_run() context manager
+ logging boilerplate **after** training run
```
**Main takeaway:** the ML code stays the same.

---


## Run `ml_pipeline_logging.py`

Run a few times with different data sources:
```bash
python ml_pipeline_logging.py --data data1/clean.csv
```
```bash
python ml_pipeline_logging.py --data data2/clean.csv
```

## Investigate warnings

1. `UserWarning: The specified dataset source can be interpreted in multiple ways`
   
  - **Cause:** harmless MLflow bug.
  - **Fix:** Declare data source type explicitly:
    ```diff
    + from mlflow.data.sources import LocalArtifactDatasetSource
    - ds = mlflow.data.from_pandas(df, source=args.data, name="dataset", targets="y")
    + ds = mlflow.data.from_pandas(df, source=LocalArtifactDatasetSource(args.data), name="dataset", targets="y")
    ```

2. `UserWarning: Hint: Inferred schema contains integer column(s).`

  - **Cause:** MLflow automatically infers feature datatypes. `y` is inferred as int.
  - **Fix:** None required. 

## Examine the experiment in the UI

Pay attention to:

- params vs metrics vs artifacts (what goes where)
- dataset lineage (source + digest)
- the stored model artifact


---

## Autolog


`.autolog()` logs (almost) everything with near-zero boilerplate.

Just add this to the original `ml_pipeline.py`:
```python
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    mlflow.sklearn.autolog()
    with mlflow.start_run():
      ...
```
**Try it:**
1. Compare `ml_pipeline_autolog.py` to `ml_pipeline_logging.py`
2.  Run:
```bash
python ml_pipeline_autolog.py --data data1/clean.csv
```
```bash
python ml_pipeline_autolog.py --data data2/clean.csv
``` 
3. Inspect the autologged experiment in the UI and compare to the manually logged experiment.

---

## Next up

- real-world use cases: long training runs, data drift
- advanced features: hyperparameter tuning, deployment


---
