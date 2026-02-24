from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
    ".npy",
}


def _unique_destination_path(destination_dir: Path, filename: str) -> Path:
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        new_candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not new_candidate.exists():
            return new_candidate
        counter += 1


def flatten_images(input_dir: Path, output_dir: Path, move_files: bool = False) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    operation = "Move" if move_files else "Copy"

    image_paths = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        print(f"No supported files found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} supported file(s) in {input_dir}")
    print(f"{operation}ing files to: {output_dir}")

    copied = 0
    for index, source_path in enumerate(sorted(image_paths), start=1):
        destination_path = _unique_destination_path(output_dir, source_path.name)
        if move_files:
            shutil.move(str(source_path), str(destination_path))
        else:
            shutil.copy2(source_path, destination_path)

        copied += 1
        print(f"[{index}/{len(image_paths)}] {operation}d: {source_path} -> {destination_path.name}")

    print(f"Done. {operation}d {copied} file(s).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flatten image and .npy files from nested folders into one folder."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source folder to scan recursively.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\\TestData\\fenestrations\\Input_images_normalized"),
        help="Destination folder for flattened files.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    args = parser.parse_args()

    flatten_images(args.input, args.output, move_files=args.move)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
