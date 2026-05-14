#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_WORKFLOW_SOURCE_REPO = "jaimemcc/noisefree-notebooks"
DEFAULT_WORKFLOW_SOURCE_REF = "main"


def _read_workflow_config(root: Path) -> tuple[list[str], str, str, str]:
    config_path = root / "notebook_workflow_config.json"
    if not config_path.exists():
        return ["notebooks"], "text", DEFAULT_WORKFLOW_SOURCE_REPO, DEFAULT_WORKFLOW_SOURCE_REF

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    managed_roots = payload.get("managed_roots")
    if not isinstance(managed_roots, list) or not managed_roots:
        raise ValueError("notebook_workflow_config.json must define managed_roots")

    source_dirs: list[str] = []
    tracked_subdir: str | None = None
    for index, entry in enumerate(managed_roots, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"managed_roots[{index}] must be an object")
        source_dir = entry.get("source_dir")
        if not isinstance(source_dir, str):
            raise ValueError(f"managed_roots[{index}].source_dir must be a string")
        source_dirs.append(source_dir)

        root_tracked_subdir = entry.get("tracked_subdir", "text")
        if not isinstance(root_tracked_subdir, str):
            raise ValueError(f"managed_roots[{index}].tracked_subdir must be a string")
        if tracked_subdir is None:
            tracked_subdir = root_tracked_subdir
        elif tracked_subdir != root_tracked_subdir:
            raise ValueError("Managed roots have different tracked_subdir values; update manually with explicit setup args")

    workflow = payload.get("workflow")
    source_repo = DEFAULT_WORKFLOW_SOURCE_REPO
    source_ref = DEFAULT_WORKFLOW_SOURCE_REF
    if isinstance(workflow, dict):
        repo_candidate = workflow.get("source_repo")
        ref_candidate = workflow.get("source_ref")
        if isinstance(repo_candidate, str) and repo_candidate.strip():
            source_repo = repo_candidate.strip()
        if isinstance(ref_candidate, str) and ref_candidate.strip():
            source_ref = ref_candidate.strip()

    return source_dirs, (tracked_subdir or "text"), source_repo, source_ref


def _read_python_pin(root: Path) -> str | None:
    manifest_path = root / "pixi.toml"
    dependencies_section = "[dependencies]"
    if not manifest_path.exists():
        manifest_path = root / "pyproject.toml"
        dependencies_section = "[tool.pixi.dependencies]"

    if not manifest_path.exists():
        return None

    in_section = False
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == dependencies_section
            continue
        if in_section and line.startswith("python"):
            match = re.match(r'python\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                return match.group(1)
    return None


def _pull_setup_script(root: Path, *, repo: str, ref: str) -> None:
    setup_path = root / "setup_notebook_workflow.py"
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/setup_notebook_workflow.py"
    try:
        with urlopen(url) as response:
            content = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(f"Failed to download setup script from {url}: {exc}") from exc

    previous_content = setup_path.read_text(encoding="utf-8") if setup_path.exists() else None
    setup_path.write_text(content, encoding="utf-8")

    if previous_content == content:
        print(f"Pull setup: no changes (already up to date) from {url}")
    else:
        print(f"Pull setup: updated setup_notebook_workflow.py from {url}")


def main(argv: list[str] | None = None) -> int:
    raw_args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Update notebook workflow files using existing configuration.",
    )
    parser.add_argument("--pull-setup", action="store_true", help="Download setup_notebook_workflow.py from GitHub before updating")
    parser.add_argument("--repo", help="GitHub repo in owner/name format for setup download")
    parser.add_argument("--ref", help="Git ref/tag/sha used when downloading setup script")
    parser.add_argument("--on-existing", choices=["overwrite", "skip", "fail"], default="overwrite")
    parser.add_argument("--skip-pixi", action="store_true", help="Pass through to setup script")
    parser.add_argument("--dry-run", action="store_true", help="Preview update without changing files")
    args = parser.parse_args(argv)

    root = Path.cwd()

    try:
        source_dirs, tracked_subdir, default_repo, default_ref = _read_workflow_config(root)
    except ValueError as exc:
        print(f"Workflow config error: {exc}", file=sys.stderr)
        return 2

    repo = args.repo or default_repo
    ref = args.ref or default_ref

    if args.pull_setup:
        pull_url = f"https://raw.githubusercontent.com/{repo}/{ref}/setup_notebook_workflow.py"
        print(f"Pull setup: enabled ({pull_url})")
    else:
        print(f"Pull setup: disabled (using local {root / 'setup_notebook_workflow.py'})")

    if args.pull_setup:
        try:
            _pull_setup_script(root, repo=repo, ref=ref)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    command = [sys.executable, "setup_notebook_workflow.py", "--on-existing", args.on_existing]
    for source_dir in source_dirs:
        command.extend(["--source-dir", source_dir])
    command.extend(["--tracked-dir", tracked_subdir])

    python_pin = _read_python_pin(root)
    if python_pin:
        command.extend(["--python-version", python_pin])

    if args.skip_pixi:
        command.append("--skip-pixi")
    if args.dry_run:
        command.append("--dry-run")

    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
