from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from untrack_managed_notebooks import tracked_managed_notebooks


ROOT = Path(__file__).resolve().parents[2]


def run_step(command: list[str], *, step_name: str) -> int:
    print(f"\n[{step_name}] {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        print(f"Step failed: {step_name}", file=sys.stderr)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate existing repositories to the notebook text-first workflow: "
            "preview/untrack tracked managed .ipynb files, then sync and validate."
        ),
    )
    parser.add_argument(
        "--apply-untrack",
        action="store_true",
        help="Apply git rm --cached to tracked managed .ipynb files. Without this flag, preview only.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --apply-untrack to confirm index changes.",
    )
    args = parser.parse_args(argv)

    tracked = tracked_managed_notebooks()
    if tracked:
        print("Managed .ipynb files currently tracked by git:")
        for path in tracked:
            print(f"  - {path}")
        print(f"\nTotal tracked managed notebooks: {len(tracked)}")

        if not args.apply_untrack:
            print("\nPreview only: no git index changes were made.")
            print("Recommended migration flow:")
            print("  1. Review the tracked managed notebooks listed above.")
            print("  2. Run 'pixi run migrate-existing-notebooks' to untrack, sync, and validate in one step.")
            print("  3. Review 'git status', then stage and commit the result.")
            print("Advanced/manual option: re-run this script with --apply-untrack --yes.")
            return 0

        if not args.yes:
            print("\nRefusing to apply without explicit confirmation.", file=sys.stderr)
            print("Re-run with: --apply-untrack --yes", file=sys.stderr)
            return 2

        untrack_rc = run_step(
            [sys.executable, "tooling/notebook_workflow/untrack_managed_notebooks.py", "--apply", "--yes"],
            step_name="untrack",
        )
        if untrack_rc != 0:
            return untrack_rc
    else:
        print("No tracked managed .ipynb files found.")

    sync_rc = run_step([sys.executable, "tooling/notebook_workflow/sync_notebooks.py"], step_name="sync")
    if sync_rc != 0:
        return sync_rc

    check_sync_rc = run_step([sys.executable, "tooling/notebook_workflow/check_notebook_sync.py"], step_name="check-sync")
    if check_sync_rc != 0:
        return check_sync_rc

    check_policy_rc = run_step([sys.executable, "tooling/notebook_workflow/check_notebook_policy.py"], step_name="check-policy")
    if check_policy_rc != 0:
        return check_policy_rc

    print("\nMigration checks passed.")
    print("Next steps:")
    print("  1. Review 'git status' to confirm the notebook removals and tracked .py additions.")
    print("  2. Run 'git add .' to stage the cleaned working tree.")
    print("  3. Commit the migration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
