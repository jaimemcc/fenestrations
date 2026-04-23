from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath

from src.roi_analysis import compute_polygon_cluster_stats, get_stub


def _parse_neighbor_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(value, (list, tuple)):
        return []

    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def build_union_mask_for_stub(stub_df: pd.DataFrame, image_h: int, image_w: int) -> np.ndarray:
    """Build a boolean union mask for one stub from valid polygon clusters."""
    id_to_xy = {
        int(row.roi_id): (float(row.centroid_x), float(row.centroid_y))
        for _, row in stub_df.iterrows()
    }

    union_mask = np.zeros((image_h, image_w), dtype=bool)
    valid_rows = stub_df[stub_df["cluster_is_valid"].fillna(False)]

    for _, row in valid_rows.iterrows():
        ids_val = _parse_neighbor_ids(row["cluster_neighbor_ids"])
        if len(ids_val) < 3:
            continue

        poly = np.array([id_to_xy[i] for i in ids_val if i in id_to_xy], dtype=float)
        if poly.shape[0] < 3:
            continue

        min_x = max(int(np.floor(np.min(poly[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(poly[:, 0]))), image_w - 1)
        min_y = max(int(np.floor(np.min(poly[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(poly[:, 1]))), image_h - 1)
        if max_x < min_x or max_y < min_y:
            continue

        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        xx, yy = np.meshgrid(xs, ys)
        points = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
        inside = MplPath(poly).contains_points(points, radius=1e-9).reshape(yy.shape)
        union_mask[min_y : max_y + 1, min_x : max_x + 1] |= inside

    return union_mask


def build_and_save_union_masks(
    rois_df: pd.DataFrame,
    images_path: str | Path,
    output_dir: str | Path,
    compute_cluster_stats_if_missing: bool = True,
    max_neighbors: int = 10,
) -> pd.DataFrame:
    """Build and save union masks for all stubs and write union_mask_summary.csv."""
    source_df = rois_df.copy()
    if source_df.empty:
        raise ValueError("No ROI rows available.")

    required_cols = {"stub", "roi_id", "centroid_x", "centroid_y", "pixel_size"}
    missing = [c for c in sorted(required_cols) if c not in source_df.columns]
    if missing:
        raise KeyError(f"rois_df is missing required columns: {missing}")

    has_cluster_cols = {"cluster_is_valid", "cluster_neighbor_ids"}.issubset(source_df.columns)
    if not has_cluster_cols:
        if not compute_cluster_stats_if_missing:
            raise KeyError(
                "rois_df is missing cluster columns ['cluster_is_valid', 'cluster_neighbor_ids'] "
                "and compute_cluster_stats_if_missing is False."
            )

        out_parts = []
        for _, stub_df in source_df.groupby("stub", sort=True):
            out_parts.append(compute_polygon_cluster_stats(stub_df.copy(), max_neighbors=max_neighbors))
        source_df = pd.concat(out_parts, ignore_index=True)

    images_path = Path(images_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stub_to_tif = {get_stub(p.name): p for p in sorted(images_path.glob("*.tif"))}

    summary_rows: list[dict[str, Any]] = []
    for stub, stub_df in source_df.groupby("stub", sort=True):
        tif_path = stub_to_tif.get(str(stub))
        if tif_path is None:
            print(f"Skipping {stub}: TIFF not found in {images_path}")
            continue

        raw = tifffile.imread(tif_path)
        image = raw[0] if raw.ndim > 2 else raw
        image_h, image_w = image.shape[:2]

        union_mask = build_union_mask_for_stub(stub_df, image_h=image_h, image_w=image_w)

        covered_px = int(union_mask.sum())
        total_px = int(union_mask.size)
        coverage_fraction = covered_px / total_px if total_px else np.nan
        pixel_size_nm_per_px = float(stub_df["pixel_size"].iloc[0]) if not stub_df.empty else np.nan

        covered_area_nm2 = covered_px * (pixel_size_nm_per_px ** 2) if np.isfinite(pixel_size_nm_per_px) else np.nan
        covered_area_um2 = covered_area_nm2 / 1_000_000.0 if np.isfinite(covered_area_nm2) else np.nan

        mask_path = output_dir / f"{stub}_union_mask.npy"
        np.save(mask_path, union_mask)

        summary_rows.append(
            {
                "stub": stub,
                "mask_path": str(mask_path),
                "pixel_size_nm_per_px": pixel_size_nm_per_px,
                "covered_px": covered_px,
                "total_px": total_px,
                "coverage_fraction": coverage_fraction,
                "coverage_percent": 100.0 * coverage_fraction if np.isfinite(coverage_fraction) else np.nan,
                "covered_area_nm2": covered_area_nm2,
                "covered_area_um2": covered_area_um2,
                "covered_area_phys2": covered_area_nm2,
            }
        )

    union_summary_df = pd.DataFrame(summary_rows)
    if not union_summary_df.empty:
        union_summary_df = union_summary_df.sort_values("coverage_fraction", ascending=False).reset_index(drop=True)

    union_summary_path = output_dir / "union_mask_summary.csv"
    union_summary_df.to_csv(union_summary_path, index=False)

    return union_summary_df
