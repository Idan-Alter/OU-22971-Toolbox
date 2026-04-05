# MLOps Unit 7 - Model Registry, Promotion, and Deployment

## Tracking server vs model registry

In the **tracking server**, experiments and runs are the center of gravity; logged models are run artifacts.

The **model registry** flips the hierarchy: **models are first-class objects** and runs matter mainly for provenance.

Core objects:

- **Logged model** (tracking): a model artifact produced by a run, addressable as:
  - `runs:/<run_id>/<artifact_path>` (e.g. `runs:/abc123/model`)

- **Registered model** (registry): a *named container* for a family of models across runs.
  - holds **versions**, **aliases**, and **metadata**

- **Model version**: an integer version inside a registered model.
  - points back to a logged model's **source URI** (lineage)

- **Alias**: a *mutable pointer* to a version (this is "promotion").
  - example: `models:/green_taxi_tip_model@production`

---

## The workflow

1) Train -> log model (tracking)
2) Register it -> creates a new **model version** (registry)
3) Attach metadata (tags + description + eval pointers)
4) Promote by moving an alias (`candidate -> production`)
5) Deploy by serving `@production`

---

## Alias scheme

Use aliases to express **deployment intent**, not model quality:

- `candidate` - newest model produced by training/retraining (evaluation target)
- `production` - the model the production deployment should serve
- `previous_production` - last production model (rollback target)
- `shadow` (optional) - model you want to run in parallel (no traffic / or separate endpoint)

Promotion rule of thumb:
- **Always set `previous_production` before moving `production`.**

---

## Register a model version

When logging, specify the registered model name. If you want to attach tags/description *immediately*, use:
- `await_registration_for=...` so a version is available right away
- `model_info.registered_model_version` to get the version number

```python
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

client = MlflowClient()
registered_name = "green_taxi_tip_model"

with mlflow.start_run():
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",  # artifact path inside the run
        registered_model_name=registered_name,
        input_example=X_tr.head(5),
        await_registration_for=300,
    )

    v = str(model_info.registered_model_version)

    client.set_model_version_tag(registered_name, v, "algo", "ridge")
    client.update_model_version(
        name=registered_name,
        version=v,
        description="Initial baseline. Trained on 2020-01.",
    )
```

---

## Metadata

### Tags (cheap, queryable, automatable)

Use tags for anything that behaves like a status/selector:

- `validation_status`: `pending | approved | rejected`
- `data_slice`: `green_tripdata_2020-01`
- `eval_run_id`: `<run_id>` of the evaluation run

```python
from mlflow import MlflowClient

client = MlflowClient()

client.set_registered_model_tag("green_taxi_tip_model", "task", "tip_regression")
client.set_model_version_tag("green_taxi_tip_model", "1", "validation_status", "approved")
```

---

## Deployment: serve the alias, not a version

### Local inference server

1. Set the MLflow tracking URI environment variable:
    ```powershell
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    ```
2. Start the server:
    ```powershell
    mlflow models serve -m "models:/green_taxi_tip_model@production" -p 5001 --env-manager local
    ```

Online inference:

```powershell
curl.exe http://127.0.0.1:5001/invocations -H "Content-Type: application/json"  --data-binary "@payload.json"
```

Note:
- Payload formats depend on flavor; `{"inputs": ...}` is the common baseline.

### Network deployment (go further)

mlflow models serve is great for smoke tests and demos. For a real production endpoint, you should use containers and deploy via a scalable platform.

See the MLflow deployment docs for details: https://mlflow.org/docs/latest/ml/deployment/

---

## Promotion: move aliases

Promotion is just *moving pointers*.

```python
from mlflow import MlflowClient

client = MlflowClient()
model_name = "green_taxi_tip_model"
candidate_version = 7

# 1) read current production version
prod = client.get_model_version_by_alias(model_name, "production")

# 2) preserve rollback target
client.set_registered_model_alias(model_name, "previous_production", prod.version)

# 3) promote candidate into production
client.set_registered_model_alias(model_name, "production", candidate_version)
```
---

## Lookback at Unit 6 (drift + retrain)

Unit 6 currently scans runs to find a "latest model".

Using the registry is cleaner and decouples monitoring from experiment structure:

- baseline = `models:/green_taxi_tip_model@production`
- retrain logs a new version -> set `candidate`
- evaluate `candidate` vs `production` on an eval slice
- if candidate wins -> promote by moving `production` (and preserving `previous_production`)

Baseline retrieval:

```python
import mlflow.pyfunc

baseline = mlflow.pyfunc.load_model("models:/green_taxi_tip_model@production")
yhat = baseline.predict(X_eval)
```
