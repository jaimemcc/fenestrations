from __future__ import annotations

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
        print(f"Notebook workflow config error: {exc}")
        return 2

    notebooks = collect_source_notebooks(managed_roots)
    if not notebooks:
        print("No source notebooks found under configured managed roots.")
        return 0

    for managed_root, source_notebook in notebooks:
        target_notebook = tracked_path_for_source(source_notebook, managed_root)
        target_notebook.parent.mkdir(parents=True, exist_ok=True)

        notebook_object = read_source_notebook(source_notebook)
        jupytext.write(notebook_object, target_notebook, fmt="py:percent")

    print("Notebook sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
