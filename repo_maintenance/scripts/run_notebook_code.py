from __future__ import annotations

import argparse
import json
import os
import pprint
import subprocess
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute notebook code cells sequentially in a plain Python process."
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file to execute.")
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Working directory to enter before execution. Defaults to the notebook's parent.",
    )
    parser.add_argument(
        "--allow-shell",
        action="store_true",
        help="Run lines that start with !. By default they are printed and skipped.",
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=>NEW",
        help="Literal source replacement applied to every code cell before execution.",
    )
    return parser.parse_args()


def parse_replacements(raw_pairs: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_pair in raw_pairs:
        if "=>" not in raw_pair:
            raise SystemExit(f"Invalid --replace value: {raw_pair!r}. Expected OLD=>NEW.")
        old, new = raw_pair.split("=>", 1)
        pairs.append((old, new))
    return pairs


def apply_replacements(source: str, replacements: list[tuple[str, str]]) -> str:
    updated = source
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated


def rewrite_magics(source: str, allow_shell: bool) -> str:
    rewritten: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("!"):
            if allow_shell:
                command = stripped[1:].strip()
                rewritten.append(f"{indent}__run_shell__({command!r})")
            else:
                rewritten.append(
                    f"{indent}print({('[notebook-runner] skipped shell line: ' + stripped)!r})"
                )
            continue
        if stripped.startswith("%"):
            rewritten.append(
                f"{indent}print({('[notebook-runner] skipped magic line: ' + stripped)!r})"
            )
            continue
        rewritten.append(line)
    return "\n".join(rewritten)


def display(*objects: object) -> None:
    for obj in objects:
        if hasattr(obj, "to_string") and callable(getattr(obj, "to_string")):
            try:
                print(obj.to_string())
                continue
            except Exception:
                pass
        pprint.pp(obj)


def run_shell(command: str) -> None:
    print(f"[notebook-runner] shell: {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


class DummyIPython:
    def system(self, command: str) -> None:
        run_shell(command)

    def run_line_magic(self, magic_name: str, line: str) -> None:
        print(f"[notebook-runner] skipped line magic %{magic_name} {line}".rstrip(), flush=True)

    def run_cell_magic(self, magic_name: str, line: str, cell: str) -> None:
        print(f"[notebook-runner] skipped cell magic %%{magic_name} {line}".rstrip(), flush=True)


def build_namespace(notebook_path: Path) -> dict[str, object]:
    return {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "__package__": None,
        "__run_shell__": run_shell,
        "display": display,
        "get_ipython": lambda: DummyIPython(),
    }


def main() -> int:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    if not notebook_path.is_file():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    work_dir = (args.cwd or notebook_path.parent).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    replacements = parse_replacements(args.replace)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    namespace = build_namespace(notebook_path)

    print(f"[notebook-runner] notebook={notebook_path}", flush=True)
    print(f"[notebook-runner] cwd={work_dir}", flush=True)
    print(f"[notebook-runner] code_cells={len(code_cells)}", flush=True)

    for index, cell in enumerate(code_cells, start=1):
        raw_source = "".join(cell.get("source", []))
        if not raw_source.strip():
            print(f"[notebook-runner] cell {index}: skipped empty cell", flush=True)
            continue

        source = apply_replacements(raw_source, replacements)
        source = rewrite_magics(source, allow_shell=args.allow_shell)
        cell_start = time.perf_counter()
        print(f"[notebook-runner] cell {index}: start", flush=True)
        try:
            code = compile(source, f"{notebook_path}#cell-{index}", "exec")
            exec(code, namespace, namespace)
        except Exception:
            duration = time.perf_counter() - cell_start
            print(
                f"[notebook-runner] cell {index}: failed after {duration:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            return 1

        duration = time.perf_counter() - cell_start
        print(f"[notebook-runner] cell {index}: ok in {duration:.2f}s", flush=True)

    print("[notebook-runner] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
