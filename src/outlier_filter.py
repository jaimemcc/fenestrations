from __future__ import annotations

import numpy as np
import pandas as pd


def robust_tail_filter_log_area(
    rois_df: pd.DataFrame,
    z_thresh: float = 4.5,
    add_score_column: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter ROI rows using a robust z-score on log(area).

    Args:
        rois_df: ROI dataframe containing an ``area`` column.
        z_thresh: Keep rows with ``abs(z) <= z_thresh``.
        add_score_column: Add ``robust_z_log_area`` to returned dataframes.

    Returns:
        Tuple of ``(kept_rows, outlier_rows)``.
    """
    if rois_df.empty:
        return rois_df.copy(), rois_df.copy()

    if "area" not in rois_df.columns:
        raise ValueError("Cannot remove outliers: 'area' column not found in ROI dataframe")

    area_vals = rois_df["area"].to_numpy(dtype=float)
    finite_positive = np.isfinite(area_vals) & (area_vals > 0)

    keep = np.ones(len(rois_df), dtype=bool)
    keep[~finite_positive] = False

    robust_z_all = np.full(len(rois_df), np.nan, dtype=float)

    if finite_positive.any():
        log_area = np.log(area_vals[finite_positive])
        median_log = np.median(log_area)
        mad_log = np.median(np.abs(log_area - median_log))

        if mad_log > 0:
            robust_sigma = 1.4826 * mad_log
            robust_z = (log_area - median_log) / robust_sigma
            robust_z_all[finite_positive] = robust_z
            keep[finite_positive] = np.abs(robust_z) <= z_thresh

    kept_df = rois_df.loc[keep].copy()
    outlier_df = rois_df.loc[~keep].copy()

    if add_score_column:
        kept_df["robust_z_log_area"] = robust_z_all[keep]
        outlier_df["robust_z_log_area"] = robust_z_all[~keep]

    return kept_df, outlier_df