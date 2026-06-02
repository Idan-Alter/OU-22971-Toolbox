# Repository Maintenance

This folder contains repo-wide validation, cleanup, notebook rendering, and grading-environment tooling. Run the PowerShell scripts from the repository root in Windows PowerShell unless a section says otherwise.

## `test_repo.ps1`

Use `test_repo.ps1` as the manual pre-release or pre-teaching smoke test for the whole repo.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\test_repo.ps1
```

The harness writes a timestamped session under `test_logs/<session>/`.

- Start with `test_logs/<session>/summary.md`.
- Use `test_logs/<session>/summary.json` for machine-readable results.
- Per-task stdout/stderr and invocation records are under `test_logs/<session>/dump/` and `test_logs/<session>/sandboxes/`.
- Treat a non-zero exit code as a failed smoke run.

By default, the script reuses existing Conda environments and Docker images, starts MLflow and the Ray lesson cluster as needed, runs curated scripts and notebooks, compile-checks helper modules, then runs `clean_ignored.ps1`.

Environment routing:

- MLOps tasks run with Conda env `22971-mlflow`.
- Ray tasks run with Conda env `22971-ray`.
- Distributed DL tasks run in the Distributed DL devcontainer with Conda env `22971-td`.
- Harness helper Python prefers the base Conda Python instead of the Windows Store Python alias.

Useful flags:

- `-SessionName <name>` writes logs to a stable session folder.
- `-SetupEnvs` refreshes the lesson Conda environments before the run.
- `-BuildDocker` rebuilds the Ray and Distributed DL Docker images before the run.
- `-SkipCleanIgnored` keeps ignored generated artifacts for inspection.
- `-AllowNotebookShell` allows notebook shell lines; by default shell-style notebook lines are skipped.

Full refresh example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\test_repo.ps1 -SessionName maintainer_check -SetupEnvs -BuildDocker
```

## `render_notebooks_html.ps1`

Use `render_notebooks_html.ps1` to regenerate the single-file HTML payload next to each tracked notebook.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\render_notebooks_html.ps1
```

The renderer:

- Finds tracked `*.ipynb` files with Git.
- Uses the base Conda environment by default.
- Pins the GitHub-compatible renderer stack to `nbformat==5.10.4` and `nbconvert==7.17.0`.
- Does not execute notebooks.
- Embeds local Markdown images into the HTML payload.
- Fills missing notebook cell IDs deterministically in memory before rendering so repeated renders do not churn random HTML IDs.

Useful flags:

- `-WhatIf` previews notebook-to-HTML targets without installing packages or writing files.
- `-SkipInstall` fails if the pinned renderer packages are missing instead of installing them.
- `-CondaEnv <name>` renders with a different Conda environment.
- `-CondaExecutable <path>` uses a specific Conda executable.
- `-NotebookRoot <dir>` limits rendering to a subtree.
- `-OutputRoot <dir>` writes HTML into a separate folder while preserving repo-relative paths.
- `-IncludeUntracked` includes untracked notebooks in addition to tracked notebooks.

Subtree preview example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\render_notebooks_html.ps1 -NotebookRoot Ray -WhatIf
```

## `clean_ignored.ps1`

Use `clean_ignored.ps1` to remove ignored generated artifacts without running the full smoke test.

Preview first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\clean_ignored.ps1 -WhatIf
```

Run cleanup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\clean_ignored.ps1
```

The script enumerates ignored paths with Git, refuses to delete anything outside the repo root, and intentionally preserves paths under:

- `TLC_data`
- `runpod_output`
- `colab_output`
- `test_logs`
- `exam_prep`

## `22971_grading_env/`

`22971_grading_env/` defines a CPU-only Docker image for grading Course 22971 capstone submissions in one consistent environment.

The image creates and activates a Conda environment named `22971-grading`. It includes the MLOps, Ray, and PyTorch Distributed stacks used by the capstones, including MLflow, Metaflow, NannyML, Optuna, Ray Data/Train/Tune, CPU PyTorch, JupyterLab, pytest, and common data-science packages.

Build from the repository root:

```powershell
docker build -t 22971-grading:latest .\repo_maintenance\22971_grading_env
```

Verify the image with the bundled smoke test:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace 22971-grading:latest python repo_maintenance/22971_grading_env/verify_env.py
```

Use it for a student submission:

```powershell
docker run --rm -it -v "C:\path\to\student_repo:/workspace" -w /workspace 22971-grading:latest bash
```

Inside the container, `python`, `pip`, and CLI tools resolve to `22971-grading` by default. The image also sets `METAFLOW_USER=grader` so local Metaflow runs work in clean Docker containers.

For MLOps projects that expect MLflow at `http://localhost:5000`, start MLflow inside the same container before running the project:

```bash
mlflow server \
  --backend-store-uri sqlite:////tmp/mlflow.db \
  --default-artifact-root /tmp/mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

## `scripts/`

This subfolder contains Python helpers used by the maintenance scripts.

- `run_notebook_code.py` executes notebook code cells sequentially in a plain Python process for smoke testing.
- `run_logged_subprocess.py` runs a command with file-backed logs and writes a structured JSON result for the harness.
- `render_notebook_html.py` renders one notebook to a single-file HTML payload for `render_notebooks_html.ps1`.
