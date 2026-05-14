from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from notebook_workflow_config import is_managed_source_notebook
from notebook_workflow_config import load_managed_roots


ROOT = Path(__file__).resolve().parents[2]


def git_list_files(*, staged: bool) -> list[str]:
    command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def find_managed_notebook_violations(paths: list[str]) -> list[str]:
    managed_roots = load_managed_roots(ROOT)
    violations: list[str] = []
    for relative_path in paths:
        candidate = ROOT / relative_path
        if is_managed_source_notebook(candidate, managed_roots):
            violations.append(relative_path)
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check repository notebook policy.")
    parser.add_argument("--staged", action="store_true", help="Check staged files instead of tracked files.")
    args = parser.parse_args(argv)

    try:
        violations = find_managed_notebook_violations(git_list_files(staged=args.staged))
    except ValueError as exc:
        print(f"Notebook workflow config error: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("Notebook policy violation: managed .ipynb files should not be tracked in git.", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        print(
            "Fix: keep source notebooks local, then run pixi run sync to refresh tracked .py copies.",
            file=sys.stderr,
        )
        return 1

    print("Notebook policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
