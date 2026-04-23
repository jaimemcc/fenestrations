from __future__ import annotations

import numpy as np
import pandas as pd


def add_lattice_area_columns(
    df: pd.DataFrame,
    spacing_col: str = "mean_cluster_neighbor_distance_nm",
    neighbor_count_col: str = "mean_neighbor_count",
    square_col: str = "cell_area_square_nm2",
    hex_col: str = "cell_area_hex_nm2",
    blend_weight_col: str = "blend_weight",
    blend_col: str = "cell_area_blend_nm2",
) -> pd.DataFrame:
    """Add square/hex/blend lattice cell-area columns to a DataFrame.

    The blend weight follows the existing project convention:
    blend_weight = clip((mean_neighbor_count - 4) / 2, 0, 1)
    """
    out = df.copy()
    out[square_col] = out[spacing_col] ** 2
    out[hex_col] = (np.sqrt(3) / 2.0) * (out[spacing_col] ** 2)
    out[blend_weight_col] = np.clip((out[neighbor_count_col] - 4.0) / 2.0, 0.0, 1.0)
    out[blend_col] = out[blend_weight_col] * out[hex_col] + (1.0 - out[blend_weight_col]) * out[square_col]
    return out


def add_lattice_blend_metrics_from_area(
    df: pd.DataFrame,
    area_col: str,
    blend_area_col: str = "cell_area_blend_nm2",
    porosity_col: str = "porosity_lattice_from_area",
    porosity_pct_col: str = "porosity_lattice_from_area_pct",
    density_nm2_col: str = "density_lattice_from_area_per_nm2",
    density_um2_col: str = "density_lattice_from_area_per_um2",
) -> pd.DataFrame:
    """Compute lattice-blend porosity and density columns from an ROI-equivalent area column."""
    out = df.copy()
    out[porosity_col] = out[area_col] / out[blend_area_col]
    out[porosity_pct_col] = out[porosity_col] * 100.0
    out[density_nm2_col] = out[porosity_col] / out[area_col]
    out[density_um2_col] = out[density_nm2_col] * 1_000_000.0
    return out
