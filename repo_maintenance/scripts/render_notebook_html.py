from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


def stable_cell_id(index: int, seen: set[str]) -> str:
    base = f"cell-{index:04d}"
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render one notebook to a single HTML file with deterministic missing cell IDs."
    )
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    output_dir = args.output_dir.resolve()
    raw_notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    seen: set[str] = set()
    for index, cell in enumerate(raw_notebook.get("cells", []), start=1):
        cell_id = cell.get("id")
        if not cell_id or cell_id in seen:
            cell_id = stable_cell_id(index, seen)
            cell["id"] = cell_id
        seen.add(cell_id)

    notebook = nbformat.reads(json.dumps(raw_notebook), as_version=4)
    exporter = HTMLExporter()
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(
        notebook,
        resources={"metadata": {"path": str(notebook_path.parent), "name": args.output_stem}},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output_stem}.html"
    output_path.write_text(body, encoding="utf-8")
    print(f"[render-notebook-html] Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
