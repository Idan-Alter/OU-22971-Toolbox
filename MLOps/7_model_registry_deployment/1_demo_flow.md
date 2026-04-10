# MLOps Unit 7 - Registry and Deployment Demo

## 0) Setup

1. Activate the conda env:

```powershell
conda activate 22971-mlflow
```

2. Start the tracking server:

```powershell
mlflow server --workers 1 --port 5000 --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --default-artifact-root mlflow_tracking/mlruns
```

3. Open the UI: http://localhost:5000 -> **Models**.

---

## 1) Generate data

```powershell
python generate_data.py
```

This creates a CSV file with toy regression data.

---

## 2) Train + register two models + set aliases

```powershell
python train_register.py
```

This creates a single registered model family with two versions:
- `production` -> DecisionTreeRegressor
- `candidate` -> LinearSVR

---

## 3) Deploy production (serve the alias)

In a *new terminal* (keep it open):

1. Set URI environment var:

   PowerShell:

   ```powershell
   $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
   ```

   bash:

   ```bash
   export MLFLOW_TRACKING_URI="http://localhost:5000"
   ```

2. **Linux only:** rollback an incompatible dependency:

   bash:

   ```bash
   pip install "starlette<1.0"
   ```

3. Serve the model:

   ```powershell
   mlflow models serve -m "models:/toy_registry_demo_model@production" -p 5001 --env-manager local
   ```

---

## 4) Run online inference

### 4.1 Create `payload.json`

```json
{
  "dataframe_split": {
    "columns": ["x0", "x1", "x2", "x3", "x4", "x5"],
    "data": [
      [0, 0, 0, 0, 0, 0],
      [1, 2, 3, 4, 5, 6]
    ]
  }
}
```

### 4.2 `POST /invocations` (online inference)

In a *new terminal*:

PowerShell:

```powershell
curl.exe http://127.0.0.1:5001/invocations -H "Content-Type: application/json" --data-binary "@payload.json"
```

bash:

```bash
curl http://127.0.0.1:5001/invocations -H "Content-Type: application/json" --data-binary "@payload.json"
```

---

## 5) Flip aliases (promote candidate -> production)

Do this **while the server is still running**:

```powershell
python flip_aliases.py
```

---

## 6) Check deployed model inference again

Run the same commands again:

```powershell
curl.exe http://127.0.0.1:5001/invocations -H "Content-Type: application/json" --data-binary "@payload.json"
```

**Expected result:** same predictions as before.

**Takeaway:** *Alias flips change the registry pointer, not the already-running process - `mlflow models serve` is static until you redeploy.*

---

## 7) Redeploy (restart the serving process)

Stop the server terminal (`Ctrl+C`) and start it again:

```powershell
mlflow models serve -m "models:/toy_registry_demo_model@production" -p 5001 --env-manager local
```

---

## 8) Check inference again

```powershell
curl.exe http://127.0.0.1:5001/invocations -H "Content-Type: application/json" --data-binary "@payload.json"
```

Now you should get predictions from the new production model.
