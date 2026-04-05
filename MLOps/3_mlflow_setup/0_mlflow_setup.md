# MLOps Unit 3 - MLflow Setup

## Setup

Activate the environment from **Unit 1**:

```bash
conda activate 22971-mlflow
```

Sanity check:

```bash
python -c "import mlflow; print(mlflow.__version__)"
```

---

## What we're setting up

MLflow gives you:

- a place to log **params / metrics / artifacts**
- a UI to **compare runs**
- a backend that is more robust than "a folder naming convention"

Our `logging_wrapper.py` from **Unit 2** was a toy version of this idea.
Now we do it properly.

---

## 0. MLflow mental model

MLflow is best thought of as a **runs database with a UI**.

A few core pieces:

- **Tracking server**: a small HTTP service that serves the UI and accepts logs
  - think: a queryable version of the `runs/` folder from `2_logging_persistence`

- **Experiment**: a named bucket of runs (project / task)
  - example: all runs produced while working on `2_logging_persistence`

- **Run**: one execution of your training script
  - for us: each time we run `ml_pipeline.py` with some hyperparameters

- **Inputs**: data a run actually consumed
  -  for us: the `data` folder produced by `generate_data.py`

- **Params**: inputs / knobs 
  - example: `svc__C`, RNG seed, etc.

- **Metrics**: numbers you compare across runs
  - example: CV score, test accuracy / F1, loss

- **Artifacts**: files produced by a run
  - example: saved model, plots, logs

---

## 1. Choose a tracking mode

MLflow can store run metadata in three ways:

### A) Database (recommended)
Metadata in a local **SQLite database** (for solo dev), with artifacts on disk.
**This is the quickstart path and the one we'll use.**

### B) File system (legacy)
MLflow creates a local folder and stores everything there.
This backend is deprecated - don't use it.
(Details: https://github.com/mlflow/mlflow/issues/18534)

### C) Remote tracking server
A shared MLflow server (team setup).

---

## 2. Local setup: SQLite + local artifacts

1. Create a dedicated folder inside the project root: `./mlflow_tracking`

2. Start a local tracking server (run in a shell):

   ```powershell
   mlflow server --workers 1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns
   ```

3. Open the UI (in a browser):

   - `http://localhost:5000`

What this means:

- The MLflow server **process** runs in this shell and is **ephemeral**
  (kill the shell -> server stops).

- The **state** is persistent on disk:
  - `mlflow_tracking/mlflow.db` stores run metadata (params, metrics, tags, status)
  - `mlflow_tracking/mlruns/` stores artifacts (plots, models, logs, files)

- Restarting the server with the same paths **restores the full run history**.

**Note:** these paths are relative to wherever you start the server.
If you run the command from a different folder, you'll create a different DB by accident.

---

## 3. Verify the connection

Copy to `startup_test.py` and run the script:

```python
import json
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")  # Which server to use for logging?
mlflow.set_experiment("3_mlflow_setup")           # Which experiment to log into?

with mlflow.start_run():
    mlflow.log_param("smoke", "test")
    mlflow.log_metric("ok", 1.0)

    # create a tiny artifact
    artifact = {
        "status": "ok",
        "note": "smoke test run"
    }

    with open("artifact.json", "w") as f:
        json.dump(artifact, f, indent=2)

    mlflow.log_artifact("artifact.json")

    print("logged")

```

Refresh the UI and you should see a new experiment.

If nothing shows up, check:

- Is the server actually running?
- Are you logging to the right server (check URI)?
- Did you start the server from a different folder (different SQLite file)?

---

## Next

In **Unit 4**, we'll modify `ml_pipeline.py` so it logs:

- params (hyperparameters)
- metrics (CV score, test metrics)
- artifacts (confusion matrix plot)
- the trained model
