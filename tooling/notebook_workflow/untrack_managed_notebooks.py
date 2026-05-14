from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from notebook_workflow_config import is_managed_source_notebook
from notebook_workflow_config import load_managed_roots


ROOT = Path(__file__).resolve().parents[2]


def tracked_managed_notebooks() -> list[str]:
    managed_roots = load_managed_roots(ROOT)

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    tracked: list[str] = []
    for line in completed.stdout.splitlines():
        path = line.strip()
        if not path:
            continue
        candidate = ROOT / path
        if is_managed_source_notebook(candidate, managed_roots):
            tracked.append(path)
    return sorted(tracked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List or untrack managed .ipynb files currently tracked by git.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run git rm --cached on matched files. Without this flag, only preview changes.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --apply to confirm untracking changes in git index.",
    )
    args = parser.parse_args(argv)

    try:
        tracked = tracked_managed_notebooks()
    except ValueError as exc:
        print(f"Notebook workflow config error: {exc}", file=sys.stderr)
        return 2

    if not tracked:
        print("No tracked managed .ipynb files found.")
        return 0

    print("Managed .ipynb files currently tracked by git:")
    for path in tracked:
        print(f"  - {path}")
    print(f"\nTotal tracked managed notebooks: {len(tracked)}")

    if not args.apply:
        print("\nPreview mode only. Re-run with --apply to untrack these files.")
        return 0

    if not args.yes:
        print("\nRefusing to apply without explicit confirmation.", file=sys.stderr)
        print("Re-run with: --apply --yes", file=sys.stderr)
        return 2

    command = ["git", "rm", "--cached", "--", *tracked]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        print("Failed to untrack one or more files.", file=sys.stderr)
        return completed.returncode

    print("\nUntracked managed .ipynb files from git index.")
    print("Run 'pixi run sync' and commit the updated tracked .py files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
