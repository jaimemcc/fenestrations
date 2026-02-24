from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import tifffile


def _stub_from_tif_path(tif_path: Path, root_dir: Path) -> str:
    rel = tif_path.relative_to(root_dir)
    stub = str(rel.parent / rel.stem).replace("\\", "/")
    if stub.startswith("./"):
        stub = stub[2:]
    return stub


def _parse_pixel_size_from_sem_metadata(metadata: dict[str, Any], tif_path: Path) -> float:
    if "ap_image_pixel_size" not in metadata:
        raise KeyError(f"Missing ap_image_pixel_size in sem_metadata for: {tif_path}")

    value = metadata["ap_image_pixel_size"]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[1])
    return float(value)


def _extract_pixel_size(tif_path: Path) -> float:
    with tifffile.TiffFile(tif_path) as tif:
        metadata = tif.sem_metadata
        if not metadata:
            raise KeyError(f"Missing sem_metadata for: {tif_path}")
        return _parse_pixel_size_from_sem_metadata(metadata, tif_path)


def extract_pixel_sizes(
    input_dir: Path,
    output_csv: Path,
    continue_on_error: bool = False,
) -> int:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a folder: {input_dir}")

    tif_paths = sorted([*input_dir.rglob("*.tif"), *input_dir.rglob("*.tiff")])
    if not tif_paths:
        raise ValueError(f"No .tif or .tiff files found in {input_dir}")

    rows: list[dict[str, Any]] = []
    failed: list[str] = []

    print(f"Found {len(tif_paths)} tif file(s) in {input_dir}")
    for index, tif_path in enumerate(tif_paths, start=1):
        stub = _stub_from_tif_path(tif_path, input_dir)
        print(f"[{index}/{len(tif_paths)}] Reading: {stub}")
        try:
            pixel_size = _extract_pixel_size(tif_path)
            rows.append(
                {
                    "stub": stub,
                    "pixel_size": pixel_size,
                    "tif_path": str(tif_path),
                }
            )
            print(f"[{index}/{len(tif_paths)}] OK: {stub} (pixel_size={pixel_size})")
        except Exception as exc:
            print(f"[{index}/{len(tif_paths)}] Error: {stub}")
            print(f"Error details: {exc}")
            failed.append(stub)
            if not continue_on_error:
                raise

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("stub").drop_duplicates(subset=["stub"], keep="last")
    else:
        df = pd.DataFrame(columns=["stub", "pixel_size", "tif_path"])

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} row(s) to {output_csv}")
    if failed:
        print(f"Failed files: {len(failed)}")
        for stub in failed:
            print(f" - {stub}")

    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively extract TIFF pixel sizes and save stub/pixel_size to CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Root folder to scan recursively.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pixel_sizes.csv"),
        help="Output CSV path (default: data/pixel_sizes.csv).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other TIFF files if one fails.",
    )
    args = parser.parse_args()

    extract_pixel_sizes(
        input_dir=args.input,
        output_csv=args.output,
        continue_on_error=args.continue_on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
