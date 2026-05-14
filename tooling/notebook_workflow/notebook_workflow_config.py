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
            source_label=str(source_rel).replace("\\", "/"),
            tracked_label=str(tracked_rel).replace("\\", "/"),
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
                source_label=str(source_rel).replace("\\", "/"),
                tracked_label=str(tracked_rel).replace("\\", "/"),
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

    if any(sep in notebook_arg for sep in ("/", "\\")):
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
