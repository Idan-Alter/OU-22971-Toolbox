# Metaflow on Windows: setup options

Metaflow is part of the capstone because it gives you explicit workflow steps, resumable runs, and a clean way to structure the monitoring and retraining loop.

If Metaflow gives you installation or runtime problems on Windows, use a Linux-backed environment instead.

Use the first option that works for you:

1. **GitHub Codespaces** - best default if you want a browser-based, reproducible setup.
2. **Conda dev container** - best if you want to work locally in VS Code and already have Docker working.
3. **Colab launcher** - quickest zero-install fallback.

---

## Option 1: GitHub Codespaces

Codespaces gives you a Linux VM in the browser with Conda already available.

### What you need

- A GitHub account
- Your capstone code pushed to a GitHub repo
- An `environment.yml` copied from the appendix below

### How to run

1. Push the project to GitHub.
2. Open the repo in GitHub.
3. Click `Code`.
4. Open the `Codespaces` tab.
5. Click `Create codespace`.


### Pros and tradeoffs

Pros:
- No local installation beyond a browser.
- No Docker or dev container setup.
- Reproducible Linux environment.
- MLflow UI port forwarding works well.

Tradeoffs:
- Depends on GitHub Codespaces availability and account limits; the free allowance is usually enough for this project.

---

## Option 2: dev container

Use this if you want a local VS Code workflow with a reproducible Linux environment.

### Prerequisites

You need:

- Docker Desktop
- VS Code
- VS Code Dev Containers extension
- Your project repo cloned locally

### Files to add to the project repo

Add this structure to the root of your project repo:

```text
.devcontainer/
  devcontainer.json
environment.yml
```

Example `.devcontainer/devcontainer.json`:

```json
{
  "name": "mlops-capstone",
  "image": "mcr.microsoft.com/devcontainers/miniconda:1-3",
  "features": {
    "ghcr.io/devcontainers/features/git:1": {}
  },
  "postCreateCommand": "conda env create -f environment.yml && conda clean -afy",
  "remoteEnv": {
    "CONDA_DEFAULT_ENV": "22971-mlops-capstone",
    "PATH": "/opt/conda/envs/22971-mlops-capstone/bin:${containerEnv:PATH}"
  },
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/opt/conda/envs/22971-mlops-capstone/bin/python"
      },
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter"
      ]
    }
  },
  "forwardPorts": [5000]
}
```

Use the `environment.yml` template in the appendix below. Remove optional packages you do not use.

### How to run

1. Open the project folder in VS Code.
2. Open the Command Palette.
3. Choose `Dev Containers: Reopen in Container`.
4. Wait for the container to build and install dependencies.


### Pros and tradeoffs

Pros:
- Very close to a real Linux development environment.
- Uses the same Conda environment style as the earlier MLOps units.

Tradeoffs:
- Requires Docker Desktop setup.

---

## Option 3: Colab launcher

Use this only if Codespaces and Docker are unavailable.

Create a notebook called `capstone_launcher.ipynb` and paste this into one cell:

```python
# Colab launcher for the MLOps capstone.
# Assumption: your repo includes flow.py and any needed data download scripts.
# Replace these values before running.
REPO_URL = "https://github.com/<your-user>/<your-repo>.git"
BRANCH = "main"
REFERENCE_PATH = "data/reference.parquet"
BATCH_PATH = "data/batch.parquet"

# Clone the project into the Colab VM.
!git clone --branch "{BRANCH}" "{REPO_URL}" capstone
%cd /content/capstone

# Install the project-specific packages from the Conda environment template.
# Colab does not use the dev container, so install into the active runtime.
!python -m pip install --upgrade pip
!pip install mlflow==3.8 "optuna==4.6.*" "metaflow==2.19.*" nannyml 

# Optional: run your repo's data download script if you use one.
# !python scripts/download_data.py

# Run the Metaflow workflow.
!python flow.py run --reference-path "{REFERENCE_PATH}" --batch-path "{BATCH_PATH}"

# If you intentionally fail a step for the required demo, fix the code and then run:
# !python flow.py resume retrain
```

After the run finishes, download `mlruns/` from the Colab file browser and inspect it later locally.

### Pros and tradeoffs

Pros:
- No local Python, Docker, or WSL setup.
- Runs on Linux.
- Useful when you are otherwise blocked.

Tradeoffs:
- Runtime state is temporary.
- Direct MLflow UI inspection in Colab is flaky.

---

## Appendix: environment.yml

Copy this file into the root of your capstone repo as `environment.yml`.

```yaml
name: 22971-mlops-capstone
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy=2.4.*
  - pandas=2.3.*
  - scikit-learn=1.8.*
  - xgboost=3.1.*
  - matplotlib=3.10.*
  - seaborn=0.13.*
  - pyarrow=22.0.*
  - ipykernel
  - pip
  - pip:
      - mlflow==3.8
      - optuna==4.6.*
      - metaflow==2.19.*
      - nannyml
```
