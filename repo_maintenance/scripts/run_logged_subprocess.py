"""Run a command with file-backed logs and a structured result record."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_path", type=Path)
    return parser.parse_args()


def write_result(result_path: Path, payload: dict[str, Any]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec_path.read_text(encoding="utf-8-sig"))

    stdout_path = Path(spec["stdout_path"])
    stderr_path = Path(spec["stderr_path"])
    result_path = Path(spec["result_path"])

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "exit_code": None,
        "timed_out": False,
        "runner_error": None,
    }
    env = os.environ.copy()
    for key, value in (spec.get("env_overrides") or {}).items():
        env[str(key)] = str(value)

    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
            with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
                completed = subprocess.run(
                    spec["command"],
                    cwd=spec["cwd"],
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=spec["timeout_sec"],
                    env=env,
                    check=False,
                )
        result["exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        with stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_file:
            stderr_file.write(f"Timed out after {spec['timeout_sec']} seconds.\n")
    except Exception as exc:  # pragma: no cover - harness failure path
        result["runner_error"] = repr(exc)
        result["exit_code"] = 1
        with stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_file:
            stderr_file.write(f"Harness runner failed: {exc!r}\n")
    finally:
        write_result(result_path, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
