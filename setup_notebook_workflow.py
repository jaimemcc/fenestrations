#!/usr/bin/env python
"""
Automated setup script for Pixi + Jupytext notebook workflow.

Usage:
    python setup_notebook_workflow.py                    # Default: notebooks/ and notebooks/text/
    python setup_notebook_workflow.py --notebook-dir notebooks --tracked-dir text
    python setup_notebook_workflow.py --notebook-dir analysis --tracked-dir .tracked
    python setup_notebook_workflow.py --source-dir scripts --source-dir analysis/notebooks --tracked-dir text
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
import re
from pathlib import Path
from textwrap import dedent


DEFAULT_PYTHON_SPEC = "3.11.*"
PYTHON_SPEC_ALLOWED_PATTERN = re.compile(r"^[0-9A-Za-z.*<>=!,|^~+\-]+$")
WORKFLOW_VERSION = "0.1.0"
DEFAULT_WORKFLOW_SOURCE_REPO = "jaimemcc/noisefree-notebooks"
DEFAULT_WORKFLOW_SOURCE_REF = "main"

WORKFLOW_PYPI_DEPENDENCIES = {
    "jupytext": '">=1.16"',
    "pre-commit": '">=3.7"',
}

WORKFLOW_TASKS = {
    "bootstrap": '"pre-commit install"',
    "check-notebook-policy": '"python tooling/notebook_workflow/check_notebook_policy.py"',
    "check-notebook-sync": '"python tooling/notebook_workflow/check_notebook_sync.py"',
    "check-notebooks": '{ depends-on = ["check-notebook-policy", "check-notebook-sync"] }',
    "sync-notebooks": '"python tooling/notebook_workflow/sync_notebooks.py"',
    "regenerate-notebooks": '"python tooling/notebook_workflow/regenerate_notebooks.py"',
    "untrack-managed-notebooks": '"python tooling/notebook_workflow/untrack_managed_notebooks.py --apply --yes"',
    "preview-untrack-managed-notebooks": '"python tooling/notebook_workflow/untrack_managed_notebooks.py"',
    "migrate-existing-notebooks-preview": '"python tooling/notebook_workflow/migrate_existing_notebooks.py"',
    "migrate-existing-notebooks": '"python tooling/notebook_workflow/migrate_existing_notebooks.py --apply-untrack --yes"',
    "update-notebook-workflow": '"python update_notebook_workflow.py --pull-setup --skip-pixi"',
    "sync": '{ depends-on = ["sync-notebooks"] }',
    "regen": '{ depends-on = ["regenerate-notebooks"] }',
    "check": '{ depends-on = ["check-notebooks"] }',
    "update": '{ depends-on = ["update-notebook-workflow"] }',
}

JUPYTEXT_SECTION = [("tool.jupytext", {"formats": '"ipynb,py:percent"'})]

PYPROJECT_WORKFLOW_SECTIONS = [
    ("tool.jupytext", {"formats": '"ipynb,py:percent"'}),
    (
        "tool.pixi.workspace",
        {
            "name": '"notebook-project"',
            "channels": '["conda-forge"]',
            "platforms": '["win-64", "linux-64"]',
        },
    ),
    ("tool.pixi.pypi-dependencies", WORKFLOW_PYPI_DEPENDENCIES),
    ("tool.pixi.tasks", WORKFLOW_TASKS),
]

PIXI_TOML_DEFAULT_SECTIONS = [
    (
        "workspace",
        {
            "name": '"notebook-project"',
            "channels": '["conda-forge"]',
            "platforms": '["win-64", "linux-64"]',
        },
    ),
    ("pypi-dependencies", WORKFLOW_PYPI_DEPENDENCIES),
    ("tasks", WORKFLOW_TASKS),
]


def write_text_file(path: Path, content: str, *, on_existing: str, dry_run: bool) -> str:
    """Write file content with configurable behavior for existing files."""
    existed_before = path.exists()

    if existed_before:
        if on_existing == "skip":
            return "skipped"
        if on_existing == "fail":
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    if dry_run:
        return "would-overwrite" if existed_before else "would-create"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "overwritten" if existed_before else "created"


def report_write(path: Path, status: str) -> None:
    labels = {
        "created": "✓ Created",
        "overwritten": "✓ Updated",
        "skipped": "• Skipped existing",
        "would-create": "• Dry run would create",
        "would-overwrite": "• Dry run would update",
    }
    print(f"{labels.get(status, '•')} {path}")


def detect_pixi_manifest_path(root: Path) -> Path:
    """Return the Pixi manifest file that this repository should use."""
    pixi_toml = root / "pixi.toml"
    if pixi_toml.exists():
        return pixi_toml
    return root / "pyproject.toml"


def _find_section_bounds(lines: list[str], section_name: str) -> tuple[int, int] | None:
    header = f"[{section_name}]"
    start_index: int | None = None

    for index, raw_line in enumerate(lines):
        if raw_line.strip().lstrip("\ufeff") == header:
            start_index = index
            break

    if start_index is None:
        return None

    end_index = len(lines)
    for index in range(start_index + 1, len(lines)):
        stripped = lines[index].strip().lstrip("\ufeff")
        if stripped.startswith("[") and stripped.endswith("]"):
            end_index = index
            break

    return start_index, end_index


def _merge_toml_sections(existing_text: str, sections: list[tuple[str, dict[str, str]]], *, on_existing: str) -> tuple[str, bool]:
    lines = existing_text.splitlines()
    changed = False

    for section_name, entries in sections:
        section_bounds = _find_section_bounds(lines, section_name)
        if section_bounds is None:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[{section_name}]")
            for key, value in entries.items():
                lines.append(f"{key} = {value}")
            changed = True
            continue

        section_start, section_end = section_bounds
        for key, value in entries.items():
            desired_line = f"{key} = {value}"
            existing_index: int | None = None
            for index in range(section_start + 1, section_end):
                if re.match(rf"\s*{re.escape(key)}\s*=", lines[index]):
                    existing_index = index
                    break

            if existing_index is None:
                lines.insert(section_end, desired_line)
                section_end += 1
                changed = True
                continue

            if lines[existing_index].strip() == desired_line:
                continue

            if on_existing == "skip":
                continue
            if on_existing == "fail":
                raise FileExistsError(f"Refusing to overwrite existing key '{key}' in [{section_name}]")

            lines[existing_index] = desired_line
            changed = True

    return "\n".join(lines) + "\n", changed


def write_toml_sections(
    path: Path,
    sections: list[tuple[str, dict[str, str]]],
    *,
    on_existing: str,
    dry_run: bool,
) -> str:
    """Create or merge TOML sections into a manifest-like file."""
    existed_before = path.exists()

    if not existed_before:
        if dry_run:
            return "would-create"

        content_lines: list[str] = []
        for index, (section_name, entries) in enumerate(sections):
            if index:
                content_lines.append("")
            content_lines.append(f"[{section_name}]")
            for key, value in entries.items():
                content_lines.append(f"{key} = {value}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
        return "created"

    existing_text = path.read_text(encoding="utf-8").lstrip("\ufeff")
    updated_text, changed = _merge_toml_sections(existing_text, sections, on_existing=on_existing)
    if not changed:
        return "skipped"

    if dry_run:
        return "would-overwrite"

    path.write_text(updated_text, encoding="utf-8")
    return "overwritten"


def active_manifest_defines_task(root: Path, task_name: str) -> bool:
    """Return True when the active Pixi manifest already defines the given task."""
    manifest_path = detect_pixi_manifest_path(root)
    if not manifest_path.exists():
        return False

    section_name = "[tasks]"
    if manifest_path.name == "pyproject.toml":
        section_name = "[tool.pixi.tasks]"

    in_tasks_section = False
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_tasks_section = line == section_name
            continue
        if in_tasks_section and re.match(rf"{re.escape(task_name)}\s*=", line):
            return True

    return False


def bootstrap_pre_commit_hooks(root: Path) -> int:
    """Install pre-commit hooks, falling back to a direct executable call."""
    print("\n🚀 Bootstrapping pre-commit hooks...")

    if active_manifest_defines_task(root, "bootstrap"):
        result = subprocess.run(["pixi", "run", "bootstrap"], cwd=root)
        if result.returncode == 0:
            return 0

        print("⚠️  bootstrap failed. Check the Pixi/pre-commit output above.", file=sys.stderr)
        return result.returncode

    print(
        "   bootstrap task is not defined in the active Pixi manifest; trying pre-commit directly through Pixi...",
        file=sys.stderr,
    )
    result = subprocess.run(["pixi", "run", "--executable", "pre-commit", "install"], cwd=root)
    if result.returncode == 0:
        return 0

    print(
        "⚠️  pre-commit bootstrap failed. Run 'pixi run --executable pre-commit install' manually.",
        file=sys.stderr,
    )
    return result.returncode


def count_existing_source_notebooks(root: Path, source_dirs: list[str]) -> int:
    """Count source notebooks already present under managed roots."""
    total = 0
    for source_dir in source_dirs:
        total += sum(1 for _ in (root / source_dir).rglob("*.ipynb"))
    return total


def run_initial_notebook_sync(root: Path, existing_notebook_count: int) -> int:
    """Populate tracked text notebooks for repos that already contain source notebooks."""
    if existing_notebook_count == 0:
        return 0

    print(f"\n📝 Syncing {existing_notebook_count} existing notebook(s) into tracked text files...")
    result = subprocess.run(["pixi", "run", "sync-notebooks"], cwd=root)
    if result.returncode != 0:
        print("⚠️  initial notebook sync failed. Run 'pixi run sync' manually.", file=sys.stderr)
        return result.returncode
    return 0


def normalize_relative_repo_path(path_text: str, *, field_name: str) -> str:
    candidate = Path(path_text)
    if candidate.is_absolute():
        raise ValueError(f"{field_name} must be repository-relative, got absolute path: {path_text}")

    cleaned = Path(*[part for part in candidate.parts if part not in ("", ".")])
    if not cleaned.parts:
        raise ValueError(f"{field_name} cannot be empty")
    if ".." in cleaned.parts:
        raise ValueError(f"{field_name} cannot contain '..': {path_text}")
    return str(cleaned).replace("\\", "/")


def resolve_source_dirs(requested_source_dirs: list[str] | None, notebook_dir: str) -> list[str]:
    source_values = requested_source_dirs if requested_source_dirs else [notebook_dir]

    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(source_values, start=1):
        cleaned = normalize_relative_repo_path(raw, field_name=f"source_dir[{index}]")
        if cleaned in seen:
            raise ValueError(f"Duplicate source directory specified: {cleaned}")
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def load_existing_managed_roots(root: Path) -> tuple[list[str], str] | None:
    """Read source dirs + shared tracked_subdir from existing config when available."""
    config_path = root / "notebook_workflow_config.json"
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"notebook_workflow_config.json is not valid JSON: {exc}") from exc

    managed_roots = payload.get("managed_roots")
    if not isinstance(managed_roots, list) or not managed_roots:
        raise ValueError("notebook_workflow_config.json must define a non-empty managed_roots array")

    source_dirs: list[str] = []
    tracked_subdir: str | None = None
    seen_sources: set[str] = set()
    for index, entry in enumerate(managed_roots, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"managed_roots[{index}] must be an object")

        source_dir_raw = entry.get("source_dir")
        if not isinstance(source_dir_raw, str):
            raise ValueError(f"managed_roots[{index}].source_dir must be a string")
        source_dir = normalize_relative_repo_path(source_dir_raw, field_name=f"managed_roots[{index}].source_dir")
        if source_dir in seen_sources:
            raise ValueError(f"Duplicate source_dir in notebook_workflow_config.json: {source_dir}")
        seen_sources.add(source_dir)
        source_dirs.append(source_dir)

        tracked_subdir_raw = entry.get("tracked_subdir")
        if tracked_subdir_raw is None:
            # Support config entries that use tracked_dir by deriving subdir when nested under source_dir.
            tracked_dir_raw = entry.get("tracked_dir")
            if isinstance(tracked_dir_raw, str):
                tracked_dir = normalize_relative_repo_path(
                    tracked_dir_raw,
                    field_name=f"managed_roots[{index}].tracked_dir",
                )
                prefix = f"{source_dir}/"
                if tracked_dir.startswith(prefix):
                    tracked_subdir_candidate = tracked_dir[len(prefix):]
                else:
                    tracked_subdir_candidate = "text"
            else:
                tracked_subdir_candidate = "text"
        else:
            if not isinstance(tracked_subdir_raw, str):
                raise ValueError(f"managed_roots[{index}].tracked_subdir must be a string")
            tracked_subdir_candidate = normalize_relative_repo_path(
                tracked_subdir_raw,
                field_name=f"managed_roots[{index}].tracked_subdir",
            )

        if tracked_subdir is None:
            tracked_subdir = tracked_subdir_candidate
        elif tracked_subdir != tracked_subdir_candidate:
            raise ValueError(
                "Existing managed roots use different tracked subdirectories; "
                "rerun setup with explicit --source-dir/--tracked-dir arguments."
            )

    return source_dirs, (tracked_subdir or "text")


def infer_existing_pixi_python(root: Path) -> str | None:
    """Read the current Pixi python pin from the active manifest if present."""
    manifest_path = detect_pixi_manifest_path(root)
    if not manifest_path.exists():
        return None

    dependencies_section = "[dependencies]"
    if manifest_path.name == "pyproject.toml":
        dependencies_section = "[tool.pixi.dependencies]"

    in_pixi_dependencies = False
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            in_pixi_dependencies = line == dependencies_section
            continue

        if in_pixi_dependencies and line.startswith("python"):
            match = re.match(r'python\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                return match.group(1)
    return None


def validate_python_spec(python_spec: str) -> str:
    """Validate a Pixi-compatible Python version spec and normalize whitespace."""
    normalized = python_spec.strip()
    if not normalized:
        raise ValueError(
            "Invalid Python version spec: value is empty. Use examples like '3.11.*', '3.12.*', or '>=3.11,<3.13'."
        )

    if any(ch.isspace() for ch in normalized):
        raise ValueError(
            "Invalid Python version spec: whitespace is not allowed. Use examples like '3.11.*' or '>=3.11,<3.13'."
        )

    if not PYTHON_SPEC_ALLOWED_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Invalid Python version spec: contains unsupported characters. "
            "Use examples like '3.11.*', '3.12.*', or '>=3.11,<3.13'."
        )

    if not any(ch.isdigit() for ch in normalized):
        raise ValueError(
            "Invalid Python version spec: must contain at least one digit. "
            "Use examples like '3.11.*' or '>=3.11,<3.13'."
        )

    return normalized


def resolve_python_spec(root: Path, requested_python_spec: str | None) -> tuple[str, str]:
    """Choose python spec from CLI, existing pyproject, or default."""
    if requested_python_spec:
        return validate_python_spec(requested_python_spec), "--python-version"

    existing_python_spec = infer_existing_pixi_python(root)
    if existing_python_spec:
        return validate_python_spec(existing_python_spec), f"existing {detect_pixi_manifest_path(root).name}"

    return validate_python_spec(DEFAULT_PYTHON_SPEC), "default"


def create_pixi_manifests(
    root: Path,
    notebook_dir: str,
    tracked_dir: str,
    python_spec: str,
    *,
    on_existing: str,
    dry_run: bool,
) -> None:
    """Generate or merge Pixi/Jupytext manifests using the repo's active Pixi file."""
    manifest_path = detect_pixi_manifest_path(root)

    if manifest_path.name == "pixi.toml":
        pyproject_path = root / "pyproject.toml"
        pyproject_status = write_toml_sections(
            pyproject_path,
            JUPYTEXT_SECTION,
            on_existing=on_existing,
            dry_run=dry_run,
        )
        report_write(pyproject_path, pyproject_status)

        manifest_sections = list(PIXI_TOML_DEFAULT_SECTIONS)
        manifest_sections.append(("dependencies", {"python": f'"{python_spec}"'}))
        manifest_status = write_toml_sections(
            manifest_path,
            manifest_sections,
            on_existing=on_existing,
            dry_run=dry_run,
        )
        report_write(manifest_path, manifest_status)
        return

    manifest_sections = list(PYPROJECT_WORKFLOW_SECTIONS)
    manifest_sections.append(("tool.pixi.dependencies", {"python": f'"{python_spec}"'}))
    manifest_status = write_toml_sections(
        manifest_path,
        manifest_sections,
        on_existing=on_existing,
        dry_run=dry_run,
    )
    report_write(manifest_path, manifest_status)


def create_gitignore(root: Path, source_dirs: list[str], *, on_existing: str, dry_run: bool) -> None:
    """Generate .gitignore that ignores .ipynb but tracks .py."""
    notebook_lines = [f"{source_dir}/**/*.ipynb" for source_dir in source_dirs]
    content_lines = [
        "# Notebook binaries (track .py via Jupytext instead)",
        *notebook_lines,
        "",
        "# Python",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.so",
        "",
        "# Jupyter",
        ".ipynb_checkpoints/",
        "",
        "# Build and distribution",
        "build/",
        "dist/",
        "*.egg-info/",
        "*.egg",
        "",
        "# Testing",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "",
        "# Linting and formatting caches",
        ".ruff_cache/",
        "",
        "# Pixi environments",
        ".pixi/*",
        "!.pixi/config.toml",
        "",
        "# Virtual environments (backup, pixi is primary)",
        "venv/",
        "env/",
        "",
        "# Environment files",
        ".env",
        ".env.local",
        "",
        "# OS files",
        ".DS_Store",
        "Thumbs.db",
    ]
    content = "\n".join(content_lines) + "\n"
    target = root / ".gitignore"

    if target.exists():
        existing_lines = target.read_text(encoding="utf-8").splitlines()
        required_lines = [
            "# Notebook binaries (track .py via Jupytext instead)",
            *notebook_lines,
            ".ipynb_checkpoints/",
        ]
        missing_lines = [line for line in required_lines if line not in existing_lines]

        if not missing_lines:
            status = "skipped"
        elif on_existing == "fail":
            raise FileExistsError(f"Refusing to update existing file: {target}")
        elif dry_run:
            status = "would-overwrite"
        else:
            appended_lines = [""]
            if "# Notebook binaries (track .py via Jupytext instead)" in missing_lines:
                appended_lines.append("# Notebook binaries (track .py via Jupytext instead)")
            for notebook_line in notebook_lines:
                if notebook_line in missing_lines:
                    appended_lines.append(notebook_line)
            if ".ipynb_checkpoints/" in missing_lines:
                appended_lines.extend(["", "# Jupyter", ".ipynb_checkpoints/"])

            target.write_text("\n".join(existing_lines + appended_lines).rstrip() + "\n", encoding="utf-8")
            status = "overwritten"
    else:
        status = write_text_file(target, content, on_existing=on_existing, dry_run=dry_run)

    report_write(target, status)


def create_notebook_workflow_config(
    root: Path,
    source_dirs: list[str],
    tracked_subdir: str,
    *,
    on_existing: str,
    dry_run: bool,
) -> None:
    content_object = {
        "workflow": {
            "version": WORKFLOW_VERSION,
            "source_repo": DEFAULT_WORKFLOW_SOURCE_REPO,
            "source_ref": DEFAULT_WORKFLOW_SOURCE_REF,
        },
        "managed_roots": [
            {
                "source_dir": source_dir,
                "tracked_subdir": tracked_subdir,
            }
            for source_dir in source_dirs
        ]
    }
    content = json.dumps(content_object, indent=2) + "\n"
    target = root / "notebook_workflow_config.json"
    status = write_text_file(target, content, on_existing=on_existing, dry_run=dry_run)
    report_write(target, status)


def create_update_script(root: Path, *, on_existing: str, dry_run: bool) -> None:
    content = f'''\
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


DEFAULT_WORKFLOW_SOURCE_REPO = "{DEFAULT_WORKFLOW_SOURCE_REPO}"
DEFAULT_WORKFLOW_SOURCE_REF = "{DEFAULT_WORKFLOW_SOURCE_REF}"


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
            raise ValueError(f"managed_roots[{{index}}] must be an object")
        source_dir = entry.get("source_dir")
        if not isinstance(source_dir, str):
            raise ValueError(f"managed_roots[{{index}}].source_dir must be a string")
        source_dirs.append(source_dir)

        root_tracked_subdir = entry.get("tracked_subdir", "text")
        if not isinstance(root_tracked_subdir, str):
            raise ValueError(f"managed_roots[{{index}}].tracked_subdir must be a string")
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
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == dependencies_section
            continue
        if in_section and line.startswith("python"):
            match = re.match(r'python\\s*=\\s*["\\']([^"\\']+)["\\']', line)
            if match:
                return match.group(1)
    return None


def _pull_setup_script(root: Path, *, repo: str, ref: str) -> None:
    setup_path = root / "setup_notebook_workflow.py"
    url = f"https://raw.githubusercontent.com/{{repo}}/{{ref}}/setup_notebook_workflow.py"
    try:
        with urlopen(url) as response:
            content = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(f"Failed to download setup script from {{url}}: {{exc}}") from exc

    previous_content = setup_path.read_text(encoding="utf-8") if setup_path.exists() else None
    setup_path.write_text(content, encoding="utf-8")

    if previous_content == content:
        print(f"Pull setup: no changes (already up to date) from {{url}}")
    else:
        print(f"Pull setup: updated setup_notebook_workflow.py from {{url}}")


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
        print(f"Workflow config error: {{exc}}", file=sys.stderr)
        return 2

    repo = args.repo or default_repo
    ref = args.ref or default_ref

    if args.pull_setup:
        pull_url = f"https://raw.githubusercontent.com/{{repo}}/{{ref}}/setup_notebook_workflow.py"
        print(f"Pull setup: enabled ({{pull_url}})")
    else:
        print(f"Pull setup: disabled (using local {{root / 'setup_notebook_workflow.py'}})")

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
'''

    target = root / "update_notebook_workflow.py"
    status = write_text_file(target, content, on_existing=on_existing, dry_run=dry_run)
    report_write(target, status)


def create_precommit_config(root: Path, *, on_existing: str, dry_run: bool) -> None:
    """Generate .pre-commit-config.yaml with Pixi-routed hooks."""
    content = '''\
repos:
  - repo: local
    hooks:
      - id: notebook-policy
        name: notebook-policy
        entry: pixi run python tooling/notebook_workflow/check_notebook_policy.py --staged
        language: system
        pass_filenames: false
      - id: notebook-sync
        name: notebook-sync
        entry: pixi run python tooling/notebook_workflow/check_notebook_sync.py
        language: system
        pass_filenames: false
'''
    target = root / ".pre-commit-config.yaml"
    status = write_text_file(target, content, on_existing=on_existing, dry_run=dry_run)
    report_write(target, status)


def create_github_workflow(root: Path, *, on_existing: str, dry_run: bool) -> None:
    """Generate GitHub Actions workflow for CI."""
    workflow_dir = root / ".github" / "workflows"
    if not dry_run:
        workflow_dir.mkdir(parents=True, exist_ok=True)

    content = '''\
name: notebook-policy

on:
  pull_request:
  push:
    branches:
      - main
      - master

jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: prefix-dev/setup-pixi@v0
        with:
          cache: true

      - name: Install notebook tooling
        run: pixi install --locked

      - name: Check notebook workflow
        run: pixi run check-notebooks
'''
    target = workflow_dir / "notebook-policy.yml"
    status = write_text_file(target, content, on_existing=on_existing, dry_run=dry_run)
    report_write(target, status)


def create_scripts(root: Path, *, on_existing: str, dry_run: bool) -> None:
    """Generate workflow scripts in tooling/notebook_workflow/."""
    scripts_dir = root / "tooling" / "notebook_workflow"
    if not dry_run:
        scripts_dir.mkdir(parents=True, exist_ok=True)

    script_map: dict[str, str] = {
        "__init__.py": '"""Notebook workflow infrastructure utilities."""\n',
        "notebook_workflow_config.py": '''\
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_FILENAME = "notebook_workflow_config.json"
DEFAULT_SOURCE_DIR = "notebooks"
DEFAULT_TRACKED_SUBDIR = "text"


@dataclass(frozen=True)
class ManagedRoot:
    source_dir: Path
    tracked_dir: Path
    source_label: str
    tracked_label: str


def _normalize_relative_path(path_text: str, *, field_name: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        raise ValueError(f"{field_name} must be repository-relative, got absolute path: {path_text}")

    cleaned = Path(*[part for part in candidate.parts if part not in ("", ".")])
    if not cleaned.parts:
        raise ValueError(f"{field_name} cannot be empty")
    if ".." in cleaned.parts:
        raise ValueError(f"{field_name} cannot contain '..': {path_text}")
    return cleaned


def _default_managed_roots(root: Path) -> list[ManagedRoot]:
    source_rel = Path(DEFAULT_SOURCE_DIR)
    tracked_rel = source_rel / DEFAULT_TRACKED_SUBDIR
    return [
        ManagedRoot(
            source_dir=root / source_rel,
            tracked_dir=root / tracked_rel,
            source_label=str(source_rel).replace("\\\\", "/"),
            tracked_label=str(tracked_rel).replace("\\\\", "/"),
        )
    ]


def load_managed_roots(root: Path) -> list[ManagedRoot]:
    config_path = root / CONFIG_FILENAME
    if not config_path.exists():
        return _default_managed_roots(root)

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{CONFIG_FILENAME} is not valid JSON: {exc}") from exc

    managed_roots = raw.get("managed_roots")
    if not isinstance(managed_roots, list) or not managed_roots:
        raise ValueError(f"{CONFIG_FILENAME} must define a non-empty 'managed_roots' array")

    parsed: list[ManagedRoot] = []
    seen_sources: set[Path] = set()

    for index, entry in enumerate(managed_roots, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"managed_roots[{index}] must be an object")

        source_dir_text = entry.get("source_dir")
        if not isinstance(source_dir_text, str):
            raise ValueError(f"managed_roots[{index}].source_dir must be a string")
        source_rel = _normalize_relative_path(source_dir_text, field_name=f"managed_roots[{index}].source_dir")

        tracked_dir_text = entry.get("tracked_dir")
        tracked_subdir_text = entry.get("tracked_subdir")

        if tracked_dir_text is not None and tracked_subdir_text is not None:
            raise ValueError(
                f"managed_roots[{index}] cannot set both tracked_dir and tracked_subdir; use one"
            )

        if tracked_dir_text is not None:
            if not isinstance(tracked_dir_text, str):
                raise ValueError(f"managed_roots[{index}].tracked_dir must be a string")
            tracked_rel = _normalize_relative_path(
                tracked_dir_text,
                field_name=f"managed_roots[{index}].tracked_dir",
            )
        else:
            if tracked_subdir_text is None:
                tracked_subdir_text = DEFAULT_TRACKED_SUBDIR
            if not isinstance(tracked_subdir_text, str):
                raise ValueError(f"managed_roots[{index}].tracked_subdir must be a string")
            tracked_subdir_rel = _normalize_relative_path(
                tracked_subdir_text,
                field_name=f"managed_roots[{index}].tracked_subdir",
            )
            tracked_rel = source_rel / tracked_subdir_rel

        if source_rel in seen_sources:
            raise ValueError(f"Duplicate source_dir in {CONFIG_FILENAME}: {source_rel}")
        seen_sources.add(source_rel)

        parsed.append(
            ManagedRoot(
                source_dir=root / source_rel,
                tracked_dir=root / tracked_rel,
                source_label=str(source_rel).replace("\\\\", "/"),
                tracked_label=str(tracked_rel).replace("\\\\", "/"),
            )
        )

    return parsed


def _is_within(path: Path, parent: Path) -> bool:
    return parent == path or parent in path.parents


def collect_source_notebooks(managed_roots: list[ManagedRoot]) -> list[tuple[ManagedRoot, Path]]:
    notebooks: list[tuple[ManagedRoot, Path]] = []
    for managed_root in managed_roots:
        if not managed_root.source_dir.exists():
            continue

        for path in managed_root.source_dir.rglob("*.ipynb"):
            if not path.is_file():
                continue
            if _is_within(path, managed_root.tracked_dir):
                continue
            notebooks.append((managed_root, path))

    notebooks.sort(key=lambda item: str(item[1]))
    return notebooks


def collect_tracked_notebooks(managed_roots: list[ManagedRoot]) -> list[tuple[ManagedRoot, Path]]:
    notebooks: list[tuple[ManagedRoot, Path]] = []
    for managed_root in managed_roots:
        if not managed_root.tracked_dir.exists():
            continue
        for path in managed_root.tracked_dir.rglob("*.py"):
            if path.is_file():
                notebooks.append((managed_root, path))

    notebooks.sort(key=lambda item: str(item[1]))
    return notebooks


def tracked_path_for_source(source_notebook: Path, managed_root: ManagedRoot) -> Path:
    relative_path = source_notebook.relative_to(managed_root.source_dir).with_suffix(".py")
    return managed_root.tracked_dir / relative_path


def source_path_for_tracked(tracked_notebook: Path, managed_root: ManagedRoot) -> Path:
    relative_path = tracked_notebook.relative_to(managed_root.tracked_dir).with_suffix(".ipynb")
    return managed_root.source_dir / relative_path


def is_managed_source_notebook(path: Path, managed_roots: list[ManagedRoot]) -> bool:
    if path.suffix.lower() != ".ipynb":
        return False

    for managed_root in managed_roots:
        if _is_within(path, managed_root.source_dir) and not _is_within(path, managed_root.tracked_dir):
            return True
    return False


def resolve_tracked_notebook_arg(
    notebook_arg: str,
    managed_roots: list[ManagedRoot],
    root: Path,
) -> Path | None:
    candidate_root_relative = root / notebook_arg
    if candidate_root_relative.exists() and candidate_root_relative.is_file():
        for managed_root in managed_roots:
            if _is_within(candidate_root_relative, managed_root.tracked_dir):
                return candidate_root_relative

    if any(sep in notebook_arg for sep in ("/", "\\\\")):
        for managed_root in managed_roots:
            candidate = managed_root.tracked_dir / notebook_arg
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    matches: list[Path] = []
    for _, tracked_notebook in collect_tracked_notebooks(managed_roots):
        if tracked_notebook.name == notebook_arg:
            matches.append(tracked_notebook)

    if len(matches) == 1:
        return matches[0]
    return None
''',
        "sync_notebooks.py": '''\
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
''',
        "check_notebook_sync.py": '''\
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
''',
        "regenerate_notebooks.py": '''\
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
''',
        "check_notebook_policy.py": '''\
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
''',
        "untrack_managed_notebooks.py": '''\
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
    print(f"\\nTotal tracked managed notebooks: {len(tracked)}")

    if not args.apply:
        print("\\nPreview mode only. Re-run with --apply to untrack these files.")
        return 0

    if not args.yes:
        print("\\nRefusing to apply without explicit confirmation.", file=sys.stderr)
        print("Re-run with: --apply --yes", file=sys.stderr)
        return 2

    command = ["git", "rm", "--cached", "--", *tracked]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        print("Failed to untrack one or more files.", file=sys.stderr)
        return completed.returncode

    print("\\nUntracked managed .ipynb files from git index.")
    print("Run 'pixi run sync' and commit the updated tracked .py files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        "migrate_existing_notebooks.py": '''\
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from untrack_managed_notebooks import tracked_managed_notebooks


ROOT = Path(__file__).resolve().parents[2]


def run_step(command: list[str], *, step_name: str) -> int:
    print(f"\\n[{step_name}] {' '.join(command)}")
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
        print(f"\\nTotal tracked managed notebooks: {len(tracked)}")

        if not args.apply_untrack:
            print("\nPreview only: no git index changes were made.")
            print("Recommended migration flow:")
            print("  1. Review the tracked managed notebooks listed above.")
            print("  2. Run 'pixi run migrate-existing-notebooks' to untrack, sync, and validate in one step.")
            print("  3. Review 'git status', then stage and commit the result.")
            print("Advanced/manual option: re-run this script with --apply-untrack --yes.")
            return 0

        if not args.yes:
            print("\\nRefusing to apply without explicit confirmation.", file=sys.stderr)
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
''',
    }

    for filename, content in script_map.items():
        target = scripts_dir / filename
        status = write_text_file(target, content, on_existing=on_existing, dry_run=dry_run)
        report_write(target, status)

    print("✓ Script generation completed in tooling/notebook_workflow/")


def create_directories(root: Path, source_dirs: list[str], tracked_subdir: str, *, dry_run: bool) -> None:
    """Create source and tracked notebook directories for all managed roots."""
    for source_dir in source_dirs:
        source_path = root / source_dir
        tracked_path = source_path / tracked_subdir

        if dry_run:
            print(f"• Dry run would ensure directory: {source_path}")
            print(f"• Dry run would ensure directory: {tracked_path}")
            continue

        tracked_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory structure: {source_dir}/ and {source_dir}/{tracked_subdir}/")


def main(argv: list[str] | None = None) -> int:
    raw_args = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Set up Pixi + Jupytext notebook workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              python setup_notebook_workflow.py
              python setup_notebook_workflow.py --notebook-dir analysis --tracked-dir .tracked
                            python setup_notebook_workflow.py --source-dir scripts --source-dir analysis/notebooks --tracked-dir text
              python setup_notebook_workflow.py -n analysis -t .tracked -p 3.12.* -o overwrite
        """),
    )
    parser.add_argument(
        "-n",
        "--notebook-dir",
        default="notebooks",
        help="Directory for source .ipynb files (default: notebooks)",
    )
    parser.add_argument(
        "-t",
        "--tracked-dir",
        default="text",
        help="Subdirectory within notebook-dir for tracked .py files (default: text)",
    )
    parser.add_argument(
        "-r",
        "--source-dir",
        dest="source_dirs",
        action="append",
        help=(
            "Managed source notebook directory. Repeat to configure multiple roots. "
            "When provided, --notebook-dir is used only as fallback default."
        ),
    )
    parser.add_argument(
        "-s",
        "--skip-pixi",
        action="store_true",
        help="Skip pixi install and bootstrap (useful for testing)",
    )
    parser.add_argument(
        "-p",
        "--python-version",
        help=(
            "Python version/spec for [tool.pixi.dependencies].python "
            f"(example: 3.11.*). Defaults to existing pyproject pin when present, otherwise {DEFAULT_PYTHON_SPEC}."
        ),
    )
    parser.add_argument(
        "-o",
        "--on-existing",
        choices=["skip", "overwrite", "fail"],
        default="skip",
        help="How to handle existing generated files (default: skip)",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files or running pixi commands",
    )
    args = parser.parse_args(raw_args)

    root = Path.cwd()
    try:
        explicit_source_dirs = args.source_dirs is not None
        explicit_notebook_dir = "--notebook-dir" in raw_args or "-n" in raw_args
        explicit_tracked_dir = "--tracked-dir" in raw_args or "-t" in raw_args

        existing_roots = None
        if not explicit_source_dirs and not explicit_notebook_dir and not explicit_tracked_dir:
            existing_roots = load_existing_managed_roots(root)

        if existing_roots is not None:
            source_dirs, tracked_subdir = existing_roots
            source_resolution = "existing notebook_workflow_config.json"
        else:
            source_dirs = resolve_source_dirs(args.source_dirs, args.notebook_dir)
            tracked_subdir = normalize_relative_repo_path(args.tracked_dir, field_name="tracked_dir")
            source_resolution = "CLI/default arguments"
    except ValueError as exc:
        print(f"⚠️  {exc}", file=sys.stderr)
        return 2

    try:
        python_spec, python_spec_source = resolve_python_spec(root, args.python_version)
    except ValueError as exc:
        print(f"⚠️  {exc}", file=sys.stderr)
        return 2

    print("\n📋 Setting up notebook workflow...")
    print("   Managed source directories:")
    for source_dir in source_dirs:
        print(f"     - {source_dir}/")
    print(f"   Tracked subdirectory (per source): {tracked_subdir}/\n")
    print(f"   Managed roots source: {source_resolution}")
    print(f"   Pixi Python: {python_spec} (from {python_spec_source})\n")

    existing_notebook_count = count_existing_source_notebooks(root, source_dirs)

    # Create configuration files
    create_directories(root, source_dirs, tracked_subdir, dry_run=args.dry_run)

    try:
        create_pixi_manifests(
            root,
            source_dirs[0],
            tracked_subdir,
            python_spec,
            on_existing=args.on_existing,
            dry_run=args.dry_run,
        )
        create_gitignore(root, source_dirs, on_existing=args.on_existing, dry_run=args.dry_run)
        create_notebook_workflow_config(
            root,
            source_dirs,
            tracked_subdir,
            on_existing=args.on_existing,
            dry_run=args.dry_run,
        )
        create_precommit_config(root, on_existing=args.on_existing, dry_run=args.dry_run)
        create_github_workflow(root, on_existing=args.on_existing, dry_run=args.dry_run)
        create_scripts(root, on_existing=args.on_existing, dry_run=args.dry_run)
        create_update_script(root, on_existing=args.on_existing, dry_run=args.dry_run)
    except FileExistsError as exc:
        print(f"⚠️  {exc}", file=sys.stderr)
        print("Use --on-existing overwrite to replace managed files, or --on-existing skip to keep them.", file=sys.stderr)
        return 1

    print("\n✅ Setup complete!\n")

    # Install and bootstrap
    if args.dry_run:
        print("Dry run complete. No files were changed and no commands were executed.")
    elif not args.skip_pixi:
        print("🔧 Installing Pixi environment...")
        try:
            result = subprocess.run(["pixi", "install"], cwd=root)
        except FileNotFoundError:
            print("⚠️  Pixi executable was not found on PATH.", file=sys.stderr)
            print("Install Pixi from https://pixi.sh, then run 'pixi install' and 'pixi run bootstrap'.", file=sys.stderr)
            return 1

        if result.returncode != 0:
            print("⚠️  pixi install failed. Install Pixi from https://pixi.sh and try again.", file=sys.stderr)
            return 1

        if run_initial_notebook_sync(root, existing_notebook_count) != 0:
            return 1

        if bootstrap_pre_commit_hooks(root) != 0:
            return 1

    elif existing_notebook_count > 0:
        print(
            f"Note: {existing_notebook_count} existing notebook(s) were detected. Run 'pixi run sync' before your first commit.",
            file=sys.stderr,
        )

    print("\n" + "=" * 60)
    print("🎉 Notebook workflow is ready!\n")
    manifest_paths = [path.name for path in [root / "pixi.toml", root / "pyproject.toml"] if path.exists()]
    manifest_args = " ".join(manifest_paths)
    print("Next steps:")
    print(f"  0. Commit generated workflow files to establish clean baseline:")
    print(f"     git add .")
    print(f"     git commit -m 'Set up notebook workflow with Pixi + Jupytext'")
    print(f"  1. Create your first notebook in {source_dirs[0]}/<name>.ipynb")
    print(f"  2. Run: pixi run sync")
    print(f"  3. If migrating existing repos: pixi run preview-untrack-managed-notebooks")
    print(f"  4. Or run full migration: pixi run migrate-existing-notebooks")
    print(f"  5. Commit tracked text notebooks under each managed root's {tracked_subdir}/")
    print("  6. For more info, see README.md")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
