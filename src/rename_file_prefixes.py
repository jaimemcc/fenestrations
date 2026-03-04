"""
Normalize file prefixes in a folder to the format FAS<number>_... or PR<number>_...

Accepted output examples:
    FAS1_sample.tif
    PR23_image.npy
    FAS10_results.csv

This script fixes names like:
    FAS_1_sample.tif  -> FAS1_sample.tif
    PR_23_image.npy   -> PR23_image.npy
    FAS_10_results.csv -> FAS10_results.csv

The part of the filename after the underscore following the number is preserved exactly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


VALID_RANGE = range(1, 25)
NAME_PATTERN = re.compile(r"^(FAS|PR)_?(\d{1,2})_(.*)$")


def _iter_files(folder: Path, recursive: bool) -> list[Path]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in paths if path.is_file())


def _normalized_name(name: str) -> str | None:
    match = NAME_PATTERN.match(name)
    if not match:
        return None

    code, number_text, suffix = match.groups()
    number = int(number_text)
    if number not in VALID_RANGE:
        return None

    new_name = f"{code}{number}_{suffix}"
    if new_name == name:
        return None
    return new_name


def rename_folder(folder: Path, recursive: bool = False, dry_run: bool = False) -> tuple[int, int]:
    candidates: list[tuple[Path, Path]] = []
    skipped = 0

    for src in _iter_files(folder, recursive=recursive):
        new_name = _normalized_name(src.name)
        if new_name is None:
            skipped += 1
            continue
        dst = src.with_name(new_name)
        candidates.append((src, dst))

    if not candidates:
        print("No files needed renaming.")
        return 0, skipped

    dst_map: dict[Path, Path] = {}
    conflicts: list[tuple[Path, Path, str]] = []
    for src, dst in candidates:
        if dst in dst_map and dst_map[dst] != src:
            conflicts.append((src, dst, "multiple files map to the same target"))
            continue
        if dst.exists() and dst != src:
            conflicts.append((src, dst, "target already exists"))
            continue
        dst_map[dst] = src

    if conflicts:
        print("Conflicts detected. Resolve these before renaming:")
        for src, dst, reason in conflicts:
            print(f"  {src.name} -> {dst.name} ({reason})")
        print("No files were renamed.")
        return 0, skipped

    renamed = 0
    for src, dst in candidates:
        if dry_run:
            print(f"[DRY RUN] {src.name} -> {dst.name}")
            renamed += 1
            continue
        src.rename(dst)
        print(f"Renamed: {src.name} -> {dst.name}")
        renamed += 1

    return renamed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename files to the format FAS<number>_... or PR<number>_... "
            "(number must be 1-24)."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder to process (default: current directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process files in subfolders as well.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming files.",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    renamed, skipped = rename_folder(folder, recursive=args.recursive, dry_run=args.dry_run)
    mode = "Dry run" if args.dry_run else "Done"
    print(f"{mode}: renamed={renamed}, skipped={skipped}")


if __name__ == "__main__":
    main()
