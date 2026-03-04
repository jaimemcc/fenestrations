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

    # Continue processing remaining stubs even if some fail
    python assemble_data.py --continue-on-error

    # Remove robust-tail outliers before computing summary stats
    python assemble_data.py --remove-outliers

    # Reuse cached raw ROI extraction (skip re-reading all source files)
    python assemble_data.py --use-raw-cache

    # Set cluster-related fields to NaN for invalid clusters
    python assemble_data.py --nan-invalid-clusters

    # Custom output path
    python assemble_data.py --output /path/to/output.pickle

Arguments:
    --input         Directory containing *_seg.npy segmentation files (default: data/)
    --tif-input     Directory containing .tif image files (default: same as --input)
    --append        Append results to existing output pickle file instead of overwriting
    --continue-on-error Continue processing other stubs if one fails
    --remove-outliers Remove ROI outliers using robust z-score on log(area)
    --outlier-z-thresh Robust z-score threshold used when --remove-outliers is enabled
    --max-k         Maximum nearest neighbors considered during ROI analysis
    --use-raw-cache Reuse cached raw ROI extraction when inputs match
    --refresh-raw-cache Force rebuild of raw ROI cache
    --raw-cache-path Path to raw ROI cache pickle
    --metadata-path Path to metadata file (.xlsx or .csv) with experiment/condition columns
    --metadata-key Merge key to use for metadata: auto, id, or stub
    --id-token-count Number of underscore-separated stub tokens used to build id
    --nan-invalid-clusters Set cluster-related columns to NaN when cluster_is_valid is False
    --output        Output pickle file path (default: data/roi_data.pickle)

Output:
    A pickle file containing a dictionary with DataFrames:
    - 'rois': Detailed ROI measurements for each region
    - 'summary': Summary statistics for each processed image
    - 'outlier_rois_df': ROIs removed by robust-tail outlier filtering

The script automatically adds an 'id' column derived from the stub filenames and
handles deduplication when appending to existing data.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Tuple

import numpy as np
import pandas as pd

from roi_analysis import compute_summary_df, get_stub, normalize_stub_path, run_batch


# -----------------------------
# Config defaults (editable)
# -----------------------------
DEFAULT_APPEND = False
DEFAULT_CONTINUE_ON_ERROR = False
DEFAULT_REMOVE_OUTLIERS = False
DEFAULT_OUTLIER_Z_THRESH = 5.0
DEFAULT_MAX_K = 10
DEFAULT_USE_RAW_CACHE = True
DEFAULT_REFRESH_RAW_CACHE = False
DEFAULT_METADATA_PATH = None
DEFAULT_METADATA_KEY = "auto"
DEFAULT_ID_TOKEN_COUNT = 3
DEFAULT_NAN_INVALID_CLUSTERS = False


def _default_raw_cache_path() -> Path:
    return _default_data_dir() / "roi_raw_cache.pickle"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    return _repo_root() / "data"


def _default_log_file_path(output_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return output_path.parent / f"assemble_data_{timestamp}.log"


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _collect_stubs(npy_dir: Path, tif_dir: Path) -> list[str]:
    if not npy_dir.exists() or not npy_dir.is_dir():
        print(f"Warning: input folder not found, skipping scan: {npy_dir}")
        return []

    tif_dir_exists = tif_dir.exists() and tif_dir.is_dir()
    if not tif_dir_exists:
        print(f"Warning: tif folder not found, continuing with stub discovery: {tif_dir}")

    seg_paths = list(npy_dir.glob("*_seg.npy"))
    use_recursive = False
    if not seg_paths:
        seg_paths = list(npy_dir.rglob("*_seg.npy"))
        use_recursive = True
        if seg_paths:
            print(f"No .npy files found in {npy_dir}, searching recursively")

    stubs: list[str] = []
    for seg_path in sorted(seg_paths):
        if use_recursive:
            try:
                rel_path = seg_path.relative_to(npy_dir)
                stub = str(rel_path.parent / rel_path.stem)
                if stub.endswith("_seg"):
                    stub = stub[:-4]
                stub = normalize_stub_path(stub)
            except ValueError:
                stub = get_stub(seg_path)
                if stub.endswith("_seg"):
                    stub = stub[:-4]
                stub = normalize_stub_path(stub)
        else:
            stub = get_stub(seg_path)
            if stub.endswith("_seg"):
                stub = stub[:-4]
            stub = normalize_stub_path(stub)

        if tif_dir_exists:
            tif_path = tif_dir / f"{stub}.tif"
            tiff_path = tif_dir / f"{stub}.tiff"
            if not tif_path.exists() and not tiff_path.exists():
                filename_only = Path(stub).name
                recursive_paths = [*tif_dir.rglob(f"{filename_only}.tif"), *tif_dir.rglob(f"{filename_only}.tiff")]
                if recursive_paths:
                    print(f"Found {filename_only} TIFF recursively at {recursive_paths[0]}")
                else:
                    print(f"Warning: missing tif/tiff for {stub}; will attempt processing anyway")
        stubs.append(stub)
    return stubs


def _add_id_column(df: pd.DataFrame, id_token_count: int = DEFAULT_ID_TOKEN_COUNT) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df.insert(0, "id", pd.Series(dtype=str))
        return df

    if "stub" not in df.columns:
        df = df.copy()
        df.insert(0, "id", pd.Series(dtype=str, index=df.index))
        return df

    stubs = df["stub"].astype(str).map(normalize_stub_path)
    ids = stubs.str.split("_").str[:id_token_count].str.join("_")
    df = df.copy()
    df["stub"] = stubs
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.insert(0, "id", ids)
    return df


def _load_existing(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payload = pd.read_pickle(path)
    if isinstance(payload, dict):
        rois = payload.get("rois", pd.DataFrame())
        summary = payload.get("summary", pd.DataFrame())
        outlier_rois_df = payload.get("outlier_rois_df", pd.DataFrame())
        return rois, summary, outlier_rois_df
    if isinstance(payload, pd.DataFrame):
        return payload, pd.DataFrame(), pd.DataFrame()
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


def _run_batch_with_progress(
    input_dir: Path,
    stubs: list[str],
    tif_dir: Path,
    max_k: int,
    continue_on_error: bool = False,
) -> Tuple[pd.DataFrame, list[str], list[str]]:
    rois_frames: list[pd.DataFrame] = []
    failed_stubs: list[str] = []
    successful_stubs: list[str] = []

    total = len(stubs)
    for index, stub in enumerate(stubs, start=1):
        print(f"[{index}/{total}] Processing stub: {stub}")
        try:
            rois_part, _ = run_batch(
                input_dir,
                [stub],
                max_k=max_k,
                tif_dir=tif_dir,
                compute_summary=False,
            )
            rois_frames.append(rois_part)
            successful_stubs.append(stub)
            print(f"[{index}/{total}] Completed stub: {stub} (rois={len(rois_part)})")
        except Exception as exc:
            print(f"[{index}/{total}] Error processing stub: {stub}")
            print(f"Error details: {exc}")
            if continue_on_error:
                failed_stubs.append(stub)
                print(f"[{index}/{total}] Continuing after error")
                continue
            raise

    rois_df = pd.concat(rois_frames, ignore_index=True) if rois_frames else pd.DataFrame()
    return rois_df, failed_stubs, successful_stubs


def _build_cache_meta(input_dir: Path, tif_dir: Path, stubs: list[str], max_k: int) -> dict:
    return {
        "input_dir": str(input_dir.resolve()),
        "tif_dir": str(tif_dir.resolve()),
        "max_k": int(max_k),
        "stubs": list(stubs),
    }


def _cache_meta_matches_run(cache_meta: dict, input_dir: Path, tif_dir: Path, max_k: int) -> bool:
    return (
        cache_meta.get("input_dir") == str(input_dir.resolve())
        and cache_meta.get("tif_dir") == str(tif_dir.resolve())
        and int(cache_meta.get("max_k", -1)) == int(max_k)
    )


def _stubs_in_rois(rois_df: pd.DataFrame) -> set[str]:
    if rois_df.empty or "stub" not in rois_df.columns:
        return set()
    return set(rois_df["stub"].astype(str).map(normalize_stub_path))


def _load_raw_cache(path: Path) -> tuple[pd.DataFrame, list[str], dict] | None:
    if not path.exists():
        return None

    payload = pd.read_pickle(path)
    if not isinstance(payload, dict):
        return None

    rois_df = payload.get("raw_rois_df")
    failed_stubs = payload.get("failed_stubs", [])
    cache_meta = payload.get("cache_meta", {})

    if not isinstance(rois_df, pd.DataFrame):
        return None
    if not isinstance(failed_stubs, list):
        failed_stubs = []
    if not isinstance(cache_meta, dict):
        cache_meta = {}

    return rois_df, failed_stubs, cache_meta


def _save_raw_cache(path: Path, rois_df: pd.DataFrame, failed_stubs: list[str], cache_meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {
            "raw_rois_df": rois_df,
            "failed_stubs": failed_stubs,
            "cache_meta": cache_meta,
        },
        path,
    )


def _load_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def _normalize_metadata_columns(metadata_df: pd.DataFrame) -> pd.DataFrame:
    normalized = metadata_df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    return normalized


def _normalize_merge_key_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold()


def _key_conflict_count(metadata_df: pd.DataFrame, key: str) -> int:
    subset = metadata_df[[key, "experiment", "condition"]].dropna(subset=[key]).copy()
    if subset.empty:
        return 0
    subset[key] = _normalize_merge_key_series(subset[key])
    unique_pairs = subset.drop_duplicates(subset=[key, "experiment", "condition"])
    conflicts = unique_pairs.groupby(key).size()
    return int((conflicts > 1).sum())


def _choose_metadata_key(
    metadata_df: pd.DataFrame,
    rois_df: pd.DataFrame,
    requested_key: str,
) -> tuple[str | None, dict]:
    requested_key = str(requested_key).lower().strip()
    candidate_keys = [key for key in ["stub", "id"] if key in metadata_df.columns and key in rois_df.columns]

    if not candidate_keys:
        return None, {"metadata_key_selected": None, "metadata_key_reason": "no-common-key"}

    if requested_key in {"id", "stub"}:
        if requested_key in candidate_keys:
            return requested_key, {
                "metadata_key_selected": requested_key,
                "metadata_key_reason": "user-requested",
            }
        return None, {
            "metadata_key_selected": None,
            "metadata_key_reason": f"requested-key-{requested_key}-missing",
        }

    scores: list[tuple[str, int, int, int]] = []
    for key in candidate_keys:
        metadata_keys = set(_normalize_merge_key_series(metadata_df[key].dropna()))
        rois_keys = set(_normalize_merge_key_series(rois_df[key].dropna()))
        overlap = len(metadata_keys.intersection(rois_keys))
        matched = int(rois_df[key].dropna().shape[0]) if overlap > 0 else 0
        conflicts = _key_conflict_count(metadata_df, key)
        scores.append((key, overlap, matched, conflicts))

    best_key, best_overlap, best_matched, best_conflicts = sorted(
        scores,
        key=lambda item: (-item[1], -item[2], item[3], item[0]),
    )[0]
    if best_overlap == 0:
        return None, {
            "metadata_key_selected": None,
            "metadata_key_reason": "auto-no-overlap",
            "metadata_key_overlap": 0,
            "metadata_key_matched": 0,
            "metadata_key_conflicts": int(best_conflicts),
        }
    return best_key, {
        "metadata_key_selected": best_key,
        "metadata_key_reason": "auto",
        "metadata_key_overlap": int(best_overlap),
        "metadata_key_matched": int(best_matched),
        "metadata_key_conflicts": int(best_conflicts),
    }


def _build_metadata_lookup(metadata_df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, int]:
    subset = metadata_df[[key, "experiment", "condition"]].dropna(subset=[key]).copy()
    if subset.empty:
        return subset, 0

    subset[key] = _normalize_merge_key_series(subset[key])
    subset = subset.drop_duplicates(subset=[key, "experiment", "condition"])

    pair_counts = (
        subset.groupby([key, "experiment", "condition"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([key, "count"], ascending=[True, False])
    )

    conflicts = int((pair_counts.groupby(key).size() > 1).sum())
    lookup = pair_counts.drop_duplicates(subset=[key], keep="first")[[key, "experiment", "condition"]]
    return lookup, conflicts


def _merge_metadata_into_df(df: pd.DataFrame, metadata_lookup: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    merged = df.drop(columns=["experiment", "condition"], errors="ignore").copy()
    merge_key_col = "__merge_key"
    merged[merge_key_col] = _normalize_merge_key_series(merged[key])
    lookup = metadata_lookup.rename(columns={key: merge_key_col})
    merged = merged.merge(lookup, on=merge_key_col, how="left")
    return merged.drop(columns=[merge_key_col])


def _apply_metadata_merge(
    rois_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outlier_rois_df: pd.DataFrame,
    metadata_path: Path | None,
    metadata_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    info: dict = {
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "metadata_merge_applied": False,
    }

    if metadata_path is None:
        info["metadata_merge_reason"] = "no-metadata-path"
        return rois_df, summary_df, outlier_rois_df, info

    if not metadata_path.exists():
        info["metadata_merge_reason"] = "metadata-file-missing"
        return rois_df, summary_df, outlier_rois_df, info

    metadata_df = _normalize_metadata_columns(_load_metadata(metadata_path))
    required = {"experiment", "condition"}
    if not required.issubset(set(metadata_df.columns)):
        info["metadata_merge_reason"] = "missing-required-columns"
        info["metadata_columns"] = list(metadata_df.columns)
        return rois_df, summary_df, outlier_rois_df, info

    selected_key, key_info = _choose_metadata_key(metadata_df, rois_df, metadata_key)
    info.update(key_info)
    if selected_key is None:
        info["metadata_merge_reason"] = "no-valid-merge-key"
        return rois_df, summary_df, outlier_rois_df, info

    lookup_df, conflicts = _build_metadata_lookup(metadata_df, selected_key)
    info["metadata_lookup_rows"] = int(len(lookup_df))
    info["metadata_key_conflicts_resolved"] = int(conflicts)

    rois_merged = _merge_metadata_into_df(rois_df, lookup_df, selected_key)
    summary_merged = _merge_metadata_into_df(summary_df, lookup_df, selected_key)
    outlier_merged = _merge_metadata_into_df(outlier_rois_df, lookup_df, selected_key)

    info["metadata_merge_applied"] = True
    info["metadata_merge_reason"] = "ok"
    info["metadata_rois_missing_condition"] = int(rois_merged["condition"].isna().sum()) if "condition" in rois_merged.columns else int(len(rois_merged))
    return rois_merged, summary_merged, outlier_merged, info


def _get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = result.stdout.strip()
        return commit_hash if commit_hash else None
    except Exception:
        return None


def _build_run_parameters(
    args: argparse.Namespace,
    input_dir: Path,
    tif_dir: Path,
    stubs: list[str],
    cache_hit: bool,
    rois_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outlier_rois_df: pd.DataFrame,
    failed_stubs: list[str],
    metadata_info: dict,
) -> dict:
    cleaning_applied = bool(args.remove_outliers)
    outliers_removed_count = int(len(outlier_rois_df))
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    git_commit_hash = _get_git_commit_hash()

    params = {
        "run_timestamp_utc": run_timestamp_utc,
        "git_commit_hash": git_commit_hash,
        "cleaning_applied": cleaning_applied,
        "remove_outliers": bool(args.remove_outliers),
        "outlier_z_thresh": float(args.outlier_z_thresh),
        "max_k": int(args.max_k),
        "append": bool(args.append),
        "continue_on_error": bool(args.continue_on_error),
        "use_raw_cache": bool(args.use_raw_cache),
        "refresh_raw_cache": bool(args.refresh_raw_cache),
        "raw_cache_path": str(Path(args.raw_cache_path)),
        "raw_cache_hit": bool(cache_hit),
        "input_dir": str(input_dir),
        "tif_dir": str(tif_dir),
        "output_path": str(Path(args.output)),
        "n_stubs": int(len(stubs)),
        "failed_stubs": list(failed_stubs),
        "n_failed_stubs": int(len(failed_stubs)),
        "n_rois": int(len(rois_df)),
        "n_summary": int(len(summary_df)),
        "n_outlier_rois": outliers_removed_count,
    }
    params.update(metadata_info)
    return params


def _nan_invalid_cluster_fields(rois_df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    if rois_df.empty:
        return rois_df.copy(), 0, []

    if "cluster_is_valid" not in rois_df.columns:
        return rois_df.copy(), 0, []

    cluster_cols = [
        col for col in rois_df.columns if col.startswith("cluster_") and col != "cluster_is_valid"
    ]
    if not cluster_cols:
        return rois_df.copy(), 0, []

    invalid_mask = rois_df["cluster_is_valid"].eq(False)
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return rois_df.copy(), 0, cluster_cols

    updated = rois_df.copy()
    updated.loc[invalid_mask, cluster_cols] = np.nan
    return updated, invalid_count, cluster_cols


def _robust_tail_filter_log_area(rois_df: pd.DataFrame, z_thresh: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rois_df.empty:
        return rois_df.copy(), rois_df.copy()

    if "area" not in rois_df.columns:
        raise ValueError("Cannot remove outliers: 'area' column not found in ROI dataframe")

    area_vals = rois_df["area"].to_numpy(dtype=float)
    finite_positive = np.isfinite(area_vals) & (area_vals > 0)

    keep = np.ones(len(rois_df), dtype=bool)
    keep[~finite_positive] = False

    if finite_positive.any():
        log_area = np.log(area_vals[finite_positive])
        median_log = np.median(log_area)
        mad_log = np.median(np.abs(log_area - median_log))

        if mad_log > 0:
            robust_sigma = 1.4826 * mad_log
            robust_z = (log_area - median_log) / robust_sigma
            keep_finite = np.abs(robust_z) <= z_thresh
            keep[finite_positive] = keep_finite

    filtered_df = rois_df.loc[keep].copy()
    outlier_df = rois_df.loc[~keep].copy()
    return filtered_df, outlier_df


def _mean_valid(series: pd.Series) -> float:
    arr = series.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(arr.mean()) if arr.size else np.nan


def _mean_profile(series: pd.Series) -> np.ndarray | None:
    profiles = [profile for profile in series if profile is not None]
    if not profiles:
        return None
    return np.mean(np.vstack(profiles), axis=0)


def _recompute_summary_from_rois(rois_df: pd.DataFrame) -> pd.DataFrame:
    if rois_df.empty:
        return pd.DataFrame()

    if "stub" not in rois_df.columns:
        raise ValueError("Cannot recompute summary: 'stub' column not found in ROI dataframe")

    summary_frames: list[pd.DataFrame] = []

    for stub, rois_stub in rois_df.groupby("stub", sort=False):
        if rois_stub.empty:
            continue

        mean_profile_major = _mean_profile(rois_stub["profile_major"]) if "profile_major" in rois_stub.columns else None
        mean_profile_minor = _mean_profile(rois_stub["profile_minor"]) if "profile_minor" in rois_stub.columns else None
        mean_four_axis = _mean_profile(rois_stub["four_axis_mean"]) if "four_axis_mean" in rois_stub.columns else None

        mean_step_major_px = _mean_valid(rois_stub["step_major"]) if "step_major" in rois_stub.columns else np.nan
        mean_step_minor_px = _mean_valid(rois_stub["step_minor"]) if "step_minor" in rois_stub.columns else np.nan
        mean_step_diag45_px = _mean_valid(rois_stub["step_diag45"]) if "step_diag45" in rois_stub.columns else np.nan
        mean_step_diag135_px = _mean_valid(rois_stub["step_diag135"]) if "step_diag135" in rois_stub.columns else np.nan

        step_candidates = np.array(
            [mean_step_major_px, mean_step_minor_px, mean_step_diag45_px, mean_step_diag135_px],
            dtype=float,
        )
        step_candidates = step_candidates[np.isfinite(step_candidates) & (step_candidates > 0)]
        mean_step_four_axis_px = float(step_candidates.mean()) if step_candidates.size else np.nan

        summary_part = compute_summary_df(
            rois_stub,
            str(stub),
            mean_profile_major=mean_profile_major,
            mean_profile_minor=mean_profile_minor,
            mean_four_axis=mean_four_axis,
            mean_step_major_px=mean_step_major_px,
            mean_step_minor_px=mean_step_minor_px,
            mean_step_four_axis_px=mean_step_four_axis_px,
        )
        summary_part = summary_part.assign(
            mean_profile_major_smpls=[mean_profile_major],
            mean_profile_minor_smpls=[mean_profile_minor],
            mean_four_axis_smpls=[mean_four_axis],
        )
        summary_frames.append(summary_part)

    return pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()


def _run_pipeline(args: argparse.Namespace) -> int:
    input_dir = args.input
    tif_dir = args.tif_input if args.tif_input is not None else input_dir
    print(f"Scanning segmentation files in: {input_dir}")
    print(f"Using TIF directory: {tif_dir}")

    rois_df: pd.DataFrame
    failed_stubs: list[str]
    cache_hit = False
    discovered_stubs: list[str] = []
    stubs: list[str] = []

    discovered_stubs = _collect_stubs(input_dir, tif_dir)

    if discovered_stubs:
        print(f"Found {len(discovered_stubs)} stub(s) on disk")
    else:
        print("No stubs discovered on disk")

    new_stubs: list[str] = list(discovered_stubs)

    if args.use_raw_cache and not args.refresh_raw_cache:
        cached = _load_raw_cache(args.raw_cache_path)
        if cached is not None:
            cached_rois_df, cached_failed_stubs, cached_meta = cached
            if _cache_meta_matches_run(cached_meta, input_dir, tif_dir, args.max_k):
                rois_df = cached_rois_df.copy()
                failed_stubs = [normalize_stub_path(stub) for stub in cached_failed_stubs]
                cache_hit = True
                print(f"Using raw ROI cache: {args.raw_cache_path}")

                processed_stubs = _stubs_in_rois(rois_df)
                new_stubs = [stub for stub in discovered_stubs if stub not in processed_stubs]
                if new_stubs:
                    print(f"Discovered {len(new_stubs)} new stub(s) not in cache")
                else:
                    print("No new stubs to process beyond cache")
            else:
                print("Raw cache metadata does not match current run settings; rebuilding cache")

    if not cache_hit:
        if not discovered_stubs:
            raise ValueError(f"No stubs found in {input_dir}")
        stubs = list(discovered_stubs)
        rois_df, failed_stubs, _ = _run_batch_with_progress(
            input_dir,
            stubs,
            tif_dir,
            max_k=args.max_k,
            continue_on_error=args.continue_on_error,
        )
    else:
        if new_stubs:
            new_rois_df, new_failed_stubs, new_successful_stubs = _run_batch_with_progress(
                input_dir,
                new_stubs,
                tif_dir,
                max_k=args.max_k,
                continue_on_error=True,
            )
            rois_df = _merge_dedup(rois_df, new_rois_df, subset=["stub", "roi_id"])
            recovered_stubs = set(new_successful_stubs)
            failed_stubs = [stub for stub in failed_stubs if stub not in recovered_stubs]
            failed_stubs = list(dict.fromkeys([*failed_stubs, *new_failed_stubs]))
        stubs = list(dict.fromkeys([*discovered_stubs, *list(_stubs_in_rois(rois_df)), *failed_stubs]))

    cache_meta = _build_cache_meta(input_dir, tif_dir, stubs, args.max_k)
    if args.use_raw_cache:
        _save_raw_cache(args.raw_cache_path, rois_df, failed_stubs, cache_meta)
        print(f"Saved raw ROI cache: {args.raw_cache_path}")

    invalid_cluster_rows = 0
    nan_cluster_columns: list[str] = []
    if args.nan_invalid_clusters:
        rois_df, invalid_cluster_rows, nan_cluster_columns = _nan_invalid_cluster_fields(rois_df)
        print(
            "Set cluster-related fields to NaN for invalid clusters: "
            f"rows={invalid_cluster_rows}, columns={len(nan_cluster_columns)}"
        )

    if args.remove_outliers:
        rois_df, outlier_rois_df = _robust_tail_filter_log_area(rois_df, z_thresh=args.outlier_z_thresh)
        print(f"Removed outlier ROIs: {len(outlier_rois_df)}")
    else:
        outlier_rois_df = pd.DataFrame(columns=rois_df.columns)

    summary_df = _recompute_summary_from_rois(rois_df)

    rois_df = _add_id_column(rois_df, id_token_count=args.id_token_count)
    summary_df = _add_id_column(summary_df, id_token_count=args.id_token_count)
    outlier_rois_df = _add_id_column(outlier_rois_df, id_token_count=args.id_token_count)

    if args.append and args.output.exists():
        existing_rois, existing_summary, existing_outlier_rois = _load_existing(args.output)
        existing_rois = _add_id_column(existing_rois, id_token_count=args.id_token_count)
        existing_summary = _add_id_column(existing_summary, id_token_count=args.id_token_count)
        existing_outlier_rois = _add_id_column(existing_outlier_rois, id_token_count=args.id_token_count)

        rois_df = _merge_dedup(existing_rois, rois_df, subset=["stub", "roi_id"])
        summary_df = _merge_dedup(existing_summary, summary_df, subset=["stub"])
        outlier_rois_df = _merge_dedup(existing_outlier_rois, outlier_rois_df, subset=["stub", "roi_id"])

    rois_df, summary_df, outlier_rois_df, metadata_info = _apply_metadata_merge(
        rois_df,
        summary_df,
        outlier_rois_df,
        metadata_path=args.metadata_path,
        metadata_key=args.metadata_key,
    )

    if metadata_info.get("metadata_merge_applied"):
        print(
            "Metadata merged using key: "
            f"{metadata_info.get('metadata_key_selected')} "
            f"(missing conditions in rois: {metadata_info.get('metadata_rois_missing_condition')})"
        )
    else:
        print(f"Metadata merge skipped: {metadata_info.get('metadata_merge_reason')}")

    parameters = _build_run_parameters(
        args=args,
        input_dir=input_dir,
        tif_dir=tif_dir,
        stubs=stubs,
        cache_hit=cache_hit,
        rois_df=rois_df,
        summary_df=summary_df,
        outlier_rois_df=outlier_rois_df,
        failed_stubs=failed_stubs,
        metadata_info=metadata_info,
    )
    parameters["nan_invalid_clusters"] = bool(args.nan_invalid_clusters)
    parameters["nan_invalid_cluster_rows"] = int(invalid_cluster_rows)
    parameters["nan_invalid_cluster_columns"] = list(nan_cluster_columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {
            "rois": rois_df,
            "summary": summary_df,
            "outlier_rois_df": outlier_rois_df,
            "parameters": parameters,
        },
        args.output,
    )

    print(f"Saved ROIs: {len(rois_df)} rows")
    print(f"Saved summary: {len(summary_df)} rows")
    print(f"Saved outlier_rois_df: {len(outlier_rois_df)} rows")
    print(f"Saved parameters: {len(parameters)} fields")
    print(f"Output: {args.output}")
    if failed_stubs:
        print(f"Failed stubs: {len(failed_stubs)}")
        for stub in failed_stubs:
            print(f" - {stub}")
    return 0


def main() -> int:
    default_input = _default_data_dir()
    default_tif = None
    default_output = default_input / "roi_data.pickle"
    default_raw_cache = _default_raw_cache_path()
    default_metadata = default_input / "fenestrations_metafile.xlsx"

    parser = argparse.ArgumentParser(description="Assemble ROI data from segmentation files.")
    parser.add_argument("--input", type=Path, default=default_input, help="Folder with *_seg.npy files.")
    parser.add_argument("--tif-input", type=Path, default=default_tif, help="Folder with .tif files.")
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_APPEND,
        help="Append to existing output pickle.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONTINUE_ON_ERROR,
        help="Continue processing remaining stubs if one fails.",
    )
    parser.add_argument(
        "--remove-outliers",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REMOVE_OUTLIERS,
        help="Remove ROI outliers using robust z-score on log(area) before summary calculation.",
    )
    parser.add_argument(
        "--outlier-z-thresh",
        type=float,
        default=DEFAULT_OUTLIER_Z_THRESH,
        help="Robust z-score threshold for outlier removal on log(area).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=DEFAULT_MAX_K,
        help="Maximum nearest neighbors considered during ROI neighbor analysis.",
    )
    parser.add_argument(
        "--use-raw-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_RAW_CACHE,
        help="Reuse cached raw ROI extraction when input/tif/stubs/max-k match.",
    )
    parser.add_argument(
        "--refresh-raw-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REFRESH_RAW_CACHE,
        help="Force rebuild of raw ROI cache before analysis.",
    )
    parser.add_argument(
        "--raw-cache-path",
        type=Path,
        default=default_raw_cache,
        help="Path to pickle storing cached raw ROI extraction.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=default_metadata if DEFAULT_METADATA_PATH is None else DEFAULT_METADATA_PATH,
        help="Path to metadata file (.xlsx/.xls/.csv) with experiment and condition columns.",
    )
    parser.add_argument(
        "--metadata-key",
        type=str,
        choices=["auto", "id", "stub"],
        default=DEFAULT_METADATA_KEY,
        help="Key used to merge metadata into outputs.",
    )
    parser.add_argument(
        "--id-token-count",
        type=int,
        default=DEFAULT_ID_TOKEN_COUNT,
        help="Number of underscore-separated tokens from stub used to build id.",
    )
    parser.add_argument(
        "--nan-invalid-clusters",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_NAN_INVALID_CLUSTERS,
        help="Set cluster-related output columns to NaN where cluster_is_valid is False.",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output pickle path.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to logfile capturing terminal output (default: output folder with timestamped name).",
    )
    args = parser.parse_args()

    log_file = args.log_file if args.log_file is not None else _default_log_file_path(args.output)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as log_handle:
        tee_stdout = _TeeStream(sys.stdout, log_handle)
        tee_stderr = _TeeStream(sys.stderr, log_handle)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"Logging terminal output to: {log_file}")
            print(f"Run started (UTC): {datetime.now(timezone.utc).isoformat()}")
            try:
                exit_code = _run_pipeline(args)
            except Exception:
                print("Run failed with exception:")
                traceback.print_exc()
                return 1
            print(f"Run finished (UTC): {datetime.now(timezone.utc).isoformat()}")
            return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
