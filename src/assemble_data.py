"""
Assemble ROI Data from Segmentation Files

This script processes segmentation files (.npy) and their corresponding TIF images
to extract ROI (Region of Interest) data and generate summary statistics.

Usage:
    # Basic usage - process all *_seg.npy files in the data directory
    python assemble_data.py

    # Specify custom input directory
    python assemble_data.py --input /path/to/segmentation/files

    # Specify separate directory for TIF files
    python assemble_data.py --input /path/to/npy --tif-input /path/to/tif

    # Append to existing output instead of overwriting
    python assemble_data.py --append --output data/roi_data.pickle

    # Custom output path
    python assemble_data.py --output /path/to/output.pickle

Arguments:
    --input         Directory containing *_seg.npy segmentation files (default: data/)
    --tif-input     Directory containing .tif image files (default: same as --input)
    --append        Append results to existing output pickle file instead of overwriting
    --output        Output pickle file path (default: data/roi_data.pickle)

Output:
    A pickle file containing a dictionary with two DataFrames:
    - 'rois': Detailed ROI measurements for each region
    - 'summary': Summary statistics for each processed image

The script automatically adds an 'id' column derived from the stub filenames and
handles deduplication when appending to existing data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

from roi_analysis import get_stub, run_batch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    return _repo_root() / "data"


def _collect_stubs(npy_dir: Path, tif_dir: Path) -> list[str]:
    seg_paths = list(npy_dir.glob("*_seg.npy"))
    use_recursive = False
    if not seg_paths:
        seg_paths = list(npy_dir.rglob("*_seg.npy"))
        use_recursive = True
        if seg_paths:
            print(f"No .npy files found in {npy_dir}, searching recursively")

    stubs: list[str] = []
    for seg_path in sorted(seg_paths):
        # If files are in subdirectories, preserve relative path
        if use_recursive:
            # Get path relative to npy_dir
            try:
                rel_path = seg_path.relative_to(npy_dir)
                # Remove _seg.npy suffix to get stub with relative path
                stub = str(rel_path.parent / rel_path.stem)
                if stub.endswith("_seg"):
                    stub = stub[:-4]
                # Normalize path separators to forward slashes
                stub = stub.replace("\\", "/")
                if stub.startswith("./"):
                    stub = stub[2:]
            except ValueError:
                # Fallback if relative path fails
                stub = get_stub(seg_path)
                if stub.endswith("_seg"):
                    stub = stub[:-4]
        else:
            stub = get_stub(seg_path)
            if stub.endswith("_seg"):
                stub = stub[:-4]
        
        # Check for matching tif file
        tif_path = tif_dir / f"{stub}.tif"
        if not tif_path.exists():
            # Try recursive search
            filename_only = Path(stub).name
            recursive_paths = list(tif_dir.rglob(f"{filename_only}.tif"))
            if recursive_paths:
                tif_path = recursive_paths[0]
                print(f"Found {filename_only}.tif recursively at {tif_path}")
            else:
                print(f"Warning: missing tif for {stub}, skipping")
                continue
        stubs.append(stub)
    return stubs


def _add_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df.insert(0, "id", pd.Series(dtype=str))
        return df

    if "stub" not in df.columns:
        df = df.copy()
        df.insert(0, "id", pd.Series(dtype=str, index=df.index))
        return df

    ids = df["stub"].astype(str).str.split("_").str[:2].str.join("_")
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.insert(0, "id", ids)
    return df


def _load_existing(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    payload = pd.read_pickle(path)
    if isinstance(payload, dict):
        rois = payload.get("rois", pd.DataFrame())
        summary = payload.get("summary", pd.DataFrame())
        return rois, summary
    if isinstance(payload, pd.DataFrame):
        return payload, pd.DataFrame()
    raise ValueError(f"Unexpected pickle format in {path}")


def _merge_dedup(existing: pd.DataFrame, new: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new.copy()
    else:
        merged = pd.concat([existing, new], ignore_index=True)

    if merged.empty:
        return merged

    subset_cols = [col for col in subset if col in merged.columns]
    if subset_cols:
        return merged.drop_duplicates(subset=subset_cols, keep="last")

    return merged.drop_duplicates(keep="last")


def main() -> int:
    default_input = _default_data_dir()
    default_tif = None
    default_output = default_input / "roi_data.pickle"

    parser = argparse.ArgumentParser(description="Assemble ROI data from segmentation files.")
    parser.add_argument("--input", type=Path, default=default_input, help="Folder with *_seg.npy files.")
    parser.add_argument("--tif-input", type=Path, default=default_tif, help="Folder with .tif files.")
    parser.add_argument("--append", action="store_true", help="Append to existing output pickle.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output pickle path.")
    args = parser.parse_args()

    input_dir = args.input
    tif_dir = args.tif_input if args.tif_input is not None else input_dir
    stubs = _collect_stubs(input_dir, tif_dir)
    if not stubs:
        raise ValueError(f"No stubs found in {input_dir}")

    rois_df, summary_df = run_batch(input_dir, stubs, tif_dir=tif_dir)
    rois_df = _add_id_column(rois_df)
    summary_df = _add_id_column(summary_df)

    if args.append and args.output.exists():
        existing_rois, existing_summary = _load_existing(args.output)
        existing_rois = _add_id_column(existing_rois)
        existing_summary = _add_id_column(existing_summary)

        rois_df = _merge_dedup(existing_rois, rois_df, subset=["stub", "roi_id"])
        summary_df = _merge_dedup(existing_summary, summary_df, subset=["stub"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"rois": rois_df, "summary": summary_df}, args.output)

    print(f"Saved ROIs: {len(rois_df)} rows")
    print(f"Saved summary: {len(summary_df)} rows")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
