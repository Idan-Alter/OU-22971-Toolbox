# 22971 grading environment

This folder defines a local Docker image for grading Course 22971 capstone submissions.

The image is based on Miniconda and creates a single conda environment named
`22971-grading`. It is intended to run the MLOps, Ray, and PyTorch Distributed
capstone projects from one consistent CPU-only environment.

## Included stacks

- Core ML/data: NumPy, pandas, scikit-learn, XGBoost, SciPy, PyArrow
- MLOps: MLflow, Metaflow, NannyML, Optuna, FastAPI, Uvicorn
- Distributed computing: Ray with `default`, `data`, `train`, and `tune` extras
- PyTorch Distributed: CPU PyTorch and torchvision
- Notebook/plotting support: JupyterLab, ipykernel, matplotlib, seaborn
- Test helpers: pytest, requests

## Build

Run from the repository root:

```powershell
docker build -t 22971-grading:latest .\repo_maintenance\22971_grading_env
```

## Verify

Run the bundled smoke test:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace 22971-grading:latest python repo_maintenance/22971_grading_env/verify_env.py
```

The smoke test imports the grading libraries, fits a small scikit-learn model,
checks MLflow/Metaflow/NannyML APIs used in the MLOps capstone, verifies
`torch.distributed`, and starts a local Ray Data job.

## Use for grading

Mount a student's repo and run their commands inside the image:

```powershell
docker run --rm -it -v "C:\path\to\student_repo:/workspace" -w /workspace 22971-grading:latest bash
```

Inside the container, `python`, `pip`, and CLI tools resolve to the
`22971-grading` conda environment by default. The image also sets a default
Metaflow user identity (`grader`) so local Metaflow runs work in clean Docker
containers without requiring host user environment variables.

For MLOps projects that hard-code `http://localhost:5000`, start MLflow inside
the same container before running the flow:

```bash
mlflow server \
  --backend-store-uri sqlite:////tmp/mlflow.db \
  --default-artifact-root /tmp/mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000
```
