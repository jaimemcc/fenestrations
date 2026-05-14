from __future__ import annotations

import sys
from pathlib import Path

import jupytext

from notebook_workflow_config import collect_source_notebooks
from notebook_workflow_config import load_managed_roots
from notebook_workflow_config import tracked_path_for_source


ROOT = Path(__file__).resolve().parents[2]


def read_source_notebook(source_notebook: Path):
    return jupytext.reads(source_notebook.read_text(encoding="utf-8-sig"), fmt="ipynb")


def main() -> int:
    try:
        managed_roots = load_managed_roots(ROOT)
    except ValueError as exc:
        print(f"Notebook workflow config error: {exc}", file=sys.stderr)
        return 2

    notebooks = collect_source_notebooks(managed_roots)
    if not notebooks:
        print("Notebook sync check passed: no source notebooks found.")
        return 0

    for managed_root, source_notebook in notebooks:
        target_notebook = tracked_path_for_source(source_notebook, managed_root)

        if not target_notebook.exists():
            print(
                f"Notebook sync check failed for {source_notebook.relative_to(ROOT)}. Regenerate it with: pixi run sync",
                file=sys.stderr,
            )
            return 1

        regenerated_text = jupytext.writes(read_source_notebook(source_notebook), fmt="py:percent")
        current_text = target_notebook.read_text(encoding="utf-8")

        if regenerated_text != current_text:
            print(
                f"Notebook sync check failed for {source_notebook.relative_to(ROOT)}. Regenerate it with: pixi run sync",
                file=sys.stderr,
            )
            return 1

    print("Notebook sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
