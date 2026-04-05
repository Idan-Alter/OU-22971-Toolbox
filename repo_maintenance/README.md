# Repository maintenance

This folder contains the repo-wide validation, cleanup, and harness helper scripts.

## `test_repo.ps1`

Use `test_repo.ps1` as the manual pre-release / pre-teaching validation pass for the whole repo.

- Run it from the repo root in Windows PowerShell.
- Make sure `conda` and `docker` are available first.
- Use the default smoke-test command: `powershell -ExecutionPolicy Bypass -File .\repo_maintenance\test_repo.ps1`
- Review `test_logs/<session>/summary.md` first, then the per-task dumps under `test_logs/<session>/dump/`; the machine-readable summary is `test_logs/<session>/summary.json`.
- Treat a non-zero exit code as a failed smoke run.

By default, the script reuses existing envs and Docker images, starts MLflow and the Ray cluster as needed, runs scripts and notebooks, writes log dumps and summaries, and runs `clean_ignored.ps1` at the end by default.

Useful flags:

- `-SessionName <name>` for a stable log folder name
- `-SetupEnvs` to refresh the environments before the smoke run
- `-BuildDocker` to rebuild the Docker images before the smoke run
- `-SkipCleanIgnored` if you want to inspect ignored generated artifacts before cleanup
- Shell-style notebook lines are skipped by default unless `-AllowNotebookShell` is passed

## `clean_ignored.ps1`

`clean_ignored.ps1` is useful on its own when you want to clear generated ignored artifacts without running the full smoke test.

- Preview the cleanup first with: `powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\clean_ignored.ps1 -WhatIf`
- Run the actual cleanup with: `powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\clean_ignored.ps1`
- The script deletes ignored paths inside the repo and intentionally preserves anything under `TLC_data`, `runpod_output`, `colab_output`, and `test_logs`

## `scripts/`

This subfolder contains harness-only Python helpers used by `test_repo.ps1`.

- `run_notebook_code.py` executes notebook code cells sequentially in a plain Python process for smoke testing.
- `run_logged_subprocess.py` runs a command with file-backed logs and writes a structured JSON result for the harness.
