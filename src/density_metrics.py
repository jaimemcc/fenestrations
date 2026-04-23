from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_key(value) -> str | None:
    """Normalise a stub label to an alphanumeric key for safe joins."""
    if pd.isna(value):
        return None
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


# ---------------------------------------------------------------------------
# Lattice density
# ---------------------------------------------------------------------------

def add_lattice_density_columns(
    summary_df: pd.DataFrame,
    spacing_col: str = "mean_cluster_neighbor_distance_nm",
    neighbor_count_col: str = "mean_neighbor_count",
) -> pd.DataFrame:
    """Add lattice-based density columns (fenestrations per µm²) to *summary_df*.

    Three geometry assumptions are provided:
    - ``density_lattice_square``  – square lattice
    - ``density_lattice_hex``     – hexagonal lattice
    - ``density_lattice_blend``   – weighted blend based on neighbour count

    Intermediate cell-area columns (nm²) are added if not already present.
    """
    from src.lattice_porosity import add_lattice_area_columns

    if not {"cell_area_square_nm2", "cell_area_hex_nm2", "cell_area_blend_nm2"}.issubset(
        summary_df.columns
    ):
        summary_df = add_lattice_area_columns(
            summary_df,
            spacing_col=spacing_col,
            neighbor_count_col=neighbor_count_col,
        )

    summary_df = summary_df.copy()
    summary_df["density_lattice_square"] = 1_000_000.0 / summary_df["cell_area_square_nm2"]
    summary_df["density_lattice_hex"] = 1_000_000.0 / summary_df["cell_area_hex_nm2"]
    summary_df["density_lattice_blend"] = 1_000_000.0 / summary_df["cell_area_blend_nm2"]
    return summary_df


# ---------------------------------------------------------------------------
# Union-mask area
# ---------------------------------------------------------------------------

def add_union_mask_area(
    summary_df: pd.DataFrame,
    rois_df: pd.DataFrame,
    images_path: Path,
    union_dir: Path,
    area_col: str = "union_mask_area_um2",
) -> pd.DataFrame:
    """Map union-mask covered area (µm²) onto every row of *summary_df*.

    If *images_path* exists the union masks are (re)built from scratch via
    :func:`src.union_masks.build_and_save_union_masks`.  Otherwise the
    pre-existing ``union_mask_summary.csv`` inside *union_dir* is loaded.

    The result is a copy of *summary_df* with an additional column *area_col*
    (``union_mask_area_um2`` by default).
    """
    from src.union_masks import build_and_save_union_masks

    union_summary_path = Path(union_dir) / "union_mask_summary.csv"
    images_path = Path(images_path)

    if images_path.exists():
        union_summary_df = build_and_save_union_masks(
            rois_df=rois_df,
            images_path=images_path,
            output_dir=union_dir,
            compute_cluster_stats_if_missing=True,
            max_neighbors=10,
        )
    elif union_summary_path.exists():
        union_summary_df = pd.read_csv(union_summary_path)
    else:
        raise FileNotFoundError(
            f"Need either IMAGESFOLDER images to build union masks, or an existing "
            f"summary file at {union_summary_path}"
        )

    if "covered_area_um2" not in union_summary_df.columns:
        if "covered_area_nm2" in union_summary_df.columns:
            union_summary_df = union_summary_df.copy()
            union_summary_df["covered_area_um2"] = (
                union_summary_df["covered_area_nm2"] / 1_000_000.0
            )
        else:
            raise KeyError(
                "union summary is missing 'covered_area_um2' "
                "(and 'covered_area_nm2' for fallback conversion)"
            )

    union_summary_df["_stub_key"] = union_summary_df["stub"].map(_stub_key)
    stub_to_area = (
        union_summary_df.drop_duplicates(subset=["_stub_key"], keep="first")
        .set_index("_stub_key")["covered_area_um2"]
        .to_dict()
    )

    summary_df = summary_df.copy()
    summary_df[area_col] = summary_df["stub"].map(_stub_key).map(stub_to_area)
    return summary_df


# ---------------------------------------------------------------------------
# Union-based density
# ---------------------------------------------------------------------------

def add_union_density(
    summary_df: pd.DataFrame,
    rois_df: pd.DataFrame | None = None,
    area_col: str = "union_mask_area_um2",
    density_col: str = "density_union",
) -> pd.DataFrame:
    """Add a union-mask-based fenestration density column to *summary_df*.

    Requires *area_col* (``union_mask_area_um2``) to be present; call
    :func:`add_union_mask_area` first if it is missing.

    If ``roi_count`` is absent from *summary_df* and *rois_df* is provided,
    the per-stub ROI count is computed from *rois_df* and merged in.
    """
    if area_col not in summary_df.columns:
        raise KeyError(
            f"summary_df is missing '{area_col}'. "
            "Call add_union_mask_area() first."
        )

    summary_df = summary_df.copy()

    if "roi_count" not in summary_df.columns:
        if rois_df is None:
            raise ValueError(
                "'roi_count' column is absent from summary_df and no rois_df was provided."
            )
        roi_counts = (
            rois_df[["stub"]]
            .dropna(subset=["stub"])
            .assign(_stub_key=lambda d: d["stub"].map(_stub_key))
            .groupby("_stub_key", as_index=False)
            .size()
            .rename(columns={"size": "roi_count"})
        )
        summary_df["_stub_key"] = summary_df["stub"].map(_stub_key)
        summary_df = summary_df.merge(roi_counts, on="_stub_key", how="left")
        summary_df = summary_df.drop(columns=["_stub_key"])

    summary_df[density_col] = summary_df["roi_count"] / summary_df[area_col]
    return summary_df
