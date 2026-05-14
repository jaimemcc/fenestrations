from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jupytext

from notebook_workflow_config import collect_tracked_notebooks
from notebook_workflow_config import load_managed_roots
from notebook_workflow_config import resolve_tracked_notebook_arg
from notebook_workflow_config import source_path_for_tracked


ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate source .ipynb notebooks from tracked .py files.")
    parser.add_argument(
        "notebook",
        nargs="?",
        help="Specific notebook to regenerate (e.g., 'starter_notebook.py'). If omitted, regenerates all notebooks.",
    )
    args = parser.parse_args(argv)

    try:
        managed_roots = load_managed_roots(ROOT)
    except ValueError as exc:
        print(f"Notebook workflow config error: {exc}", file=sys.stderr)
        return 2

    if args.notebook:
        tracked_notebook = resolve_tracked_notebook_arg(args.notebook, managed_roots, ROOT)
        if tracked_notebook is None:
            print(
                f"Notebook not found or ambiguous: {args.notebook}. "
                "Try a path relative to repository root, such as 'feature1/notebooks/text/example.py'.",
                file=sys.stderr,
            )
            return 1
        notebook_entries = [
            (managed_root, tracked_notebook)
            for managed_root, candidate in collect_tracked_notebooks(managed_roots)
            if candidate == tracked_notebook
        ]
    else:
        notebook_entries = collect_tracked_notebooks(managed_roots)
        if not notebook_entries:
            print("No tracked notebooks found under configured managed roots.")
            return 0

    for managed_root, tracked_notebook in notebook_entries:
        source_notebook = source_path_for_tracked(tracked_notebook, managed_root)
        source_notebook.parent.mkdir(parents=True, exist_ok=True)

        notebook_object = jupytext.read(tracked_notebook, fmt="py:percent")
        jupytext.write(notebook_object, source_notebook, fmt="ipynb")

    print("Notebook regeneration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
