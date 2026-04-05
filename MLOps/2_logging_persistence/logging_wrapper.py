"""
Minimal experiment wrapper.

Creates a timestamped run directory, executes the existing training pipeline
inside it, and captures all side effects (artifacts and logs) in one place.

Example
--------
Run with default data path:
    python logging_wrapper.py
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="../1_conda_environments/data/clean.csv",
        help="Path to input CSV (relative to 2_logging_persistence/ or absolute).",
    )
    args = p.parse_args()

    # Anchor everything to *this file's* location (works no matter where you run it from)
    here = Path(__file__).resolve().parent  # .../2_logging_persistence
    project_root = here.parent              # .../MLOps
    src_dir = project_root / "1_conda_environments"
    pipeline_py = src_dir / "ml_pipeline.py"

    data_path = (here / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    data_path = data_path.resolve()

    if not pipeline_py.exists():
        raise FileNotFoundError(f"Can't find ml_pipeline.py at: {pipeline_py}")
    if not data_path.exists():
        raise FileNotFoundError(f"Can't find data file at: {data_path}")

    runs_dir = here / "runs"
    runs_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir()

    # Minimal provenance (no copying)
    (run_dir / "meta.txt").write_text(
        f"script={pipeline_py}\n"
        f"data={data_path}\n"
        f"run_id={run_id}\n",
        encoding="utf-8",
    )

    log_path = run_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            ["python", str(pipeline_py), "--data", str(data_path)],
            cwd=str(run_dir),              # <-- forces artifacts into the run folder
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print(f"Run stored in: {run_dir}")
    print(f"- {log_path}")
    print(f"- {run_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
