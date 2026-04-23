from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.density_metrics import (
    add_lattice_density_columns,
    add_porosity_from_diameters,
    add_union_density,
    add_union_mask_area,
)
from src.diameter_calcs import _compute_dip_width, _compute_dip_width_derivative
from src.roi_analysis import estimate_diameter_from_profile


def _compute_axis_profile_diams(
    profile: object,
    step_nm: float,
    pixel_size_nm: float,
) -> tuple[float, float, float]:
    if profile is None or not np.isfinite(step_nm) or not np.isfinite(pixel_size_nm):
        return np.nan, np.nan, np.nan

    profile_arr = np.asarray(profile, dtype=float)
    if profile_arr.size < 5:
        return np.nan, np.nan, np.nan

    d_p2p = estimate_diameter_from_profile(
        profile_arr,
        pixel_size=float(pixel_size_nm),
        sample_step_px=float(step_nm) / float(pixel_size_nm),
    )
    d_fwhm = _compute_dip_width(profile_arr, float(step_nm))
    d_deriv = _compute_dip_width_derivative(profile_arr, float(step_nm))
    return d_p2p, d_fwhm, d_deriv


def _ensure_roi_base_columns(rois_df: pd.DataFrame) -> pd.DataFrame:
    out = rois_df.copy()

    if "circularity" not in out.columns and {
        "minor_axis_length_px",
        "major_axis_length_px",
    }.issubset(out.columns):
        out["circularity"] = out["minor_axis_length_px"] / out["major_axis_length_px"]

    if "step_major_nm" not in out.columns and {"step_major", "pixel_size"}.issubset(out.columns):
        out["step_major_nm"] = out["step_major"] * out["pixel_size"]

    if "step_minor_nm" not in out.columns and {"step_minor", "pixel_size"}.issubset(out.columns):
        out["step_minor_nm"] = out["step_minor"] * out["pixel_size"]

    if "cluster_neighbor_distance_nm" not in out.columns and {
        "cluster_neighbor_distance",
        "pixel_size",
    }.issubset(out.columns):
        out["cluster_neighbor_distance_nm"] = out["cluster_neighbor_distance"] * out["pixel_size"]

    if "diameter_elliptical_nm" not in out.columns and {
        "major_axis_extent",
        "minor_axis_extent",
        "pixel_size",
    }.issubset(out.columns):
        major_nm = 2.0 * out["major_axis_extent"] * out["pixel_size"]
        minor_nm = 2.0 * out["minor_axis_extent"] * out["pixel_size"]
        out["diameter_elliptical_nm"] = np.sqrt(major_nm * minor_nm)

    return out


def _add_roi_profile_diameter_columns(rois_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "profile_major",
        "profile_minor",
        "step_major_nm",
        "step_minor_nm",
        "pixel_size",
    }
    if not required.issubset(rois_df.columns):
        return rois_df.copy()

    out = rois_df.copy()

    for axis in ("major", "minor"):
        p2p_vals: list[float] = []
        fwhm_vals: list[float] = []
        deriv_vals: list[float] = []
        profile_col = f"profile_{axis}"
        step_col = f"step_{axis}_nm"

        for profile, step_nm, pixel_size_nm in zip(
            out[profile_col],
            out[step_col],
            out["pixel_size"],
            strict=False,
        ):
            d_p2p, d_fwhm, d_deriv = _compute_axis_profile_diams(profile, step_nm, pixel_size_nm)
            p2p_vals.append(d_p2p)
            fwhm_vals.append(d_fwhm)
            deriv_vals.append(d_deriv)

        out[f"diameter_p2p_{axis}_nm"] = p2p_vals
        out[f"diameter_fwhm_{axis}_nm"] = fwhm_vals
        out[f"diameter_deriv_{axis}_nm"] = deriv_vals

    for method in ("p2p", "fwhm", "deriv"):
        major_col = f"diameter_{method}_major_nm"
        minor_col = f"diameter_{method}_minor_nm"
        out[f"diameter_{method}_nm"] = np.where(
            np.isfinite(out[major_col]) & np.isfinite(out[minor_col]),
            np.sqrt(out[major_col] * out[minor_col]),
            np.nan,
        )

    return out


def _add_summary_profile_means(rois_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if "stub" not in rois_df.columns or "stub" not in summary_df.columns:
        return summary_df.copy()

    profile_cols = [
        "diameter_p2p_major_nm",
        "diameter_p2p_minor_nm",
        "diameter_p2p_nm",
        "diameter_fwhm_major_nm",
        "diameter_fwhm_minor_nm",
        "diameter_fwhm_nm",
        "diameter_deriv_major_nm",
        "diameter_deriv_minor_nm",
        "diameter_deriv_nm",
    ]
    available = [col for col in profile_cols if col in rois_df.columns]
    if not available:
        return summary_df.copy()

    means_df = (
        rois_df.groupby("stub", as_index=False)[available]
        .mean(numeric_only=True)
        .rename(columns={col: f"mean_{col}" for col in available})
    )
    return summary_df.drop(columns=list(means_df.columns.difference(["stub"])), errors="ignore").merge(
        means_df,
        on="stub",
        how="left",
    )


def add_extra_columns(
    rois_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    union_dir: str | Path = "union_masks",
    images_path: str | Path | None = None,
    add_union_area: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add notebook-ready derived columns to ROI and summary DataFrames in one pass."""
    rois_out = _ensure_roi_base_columns(rois_df)
    rois_out = _add_roi_profile_diameter_columns(rois_out)

    summary_out = summary_df.copy()
    summary_out = _add_summary_profile_means(rois_out, summary_out)
    summary_out = add_lattice_density_columns(summary_out)

    if add_union_area:
        # If images_path is omitted, fall back to union_mask_summary.csv when available.
        image_root = Path(images_path) if images_path is not None else Path("__missing_images_path__")
        try:
            summary_out = add_union_mask_area(
                summary_df=summary_out,
                rois_df=rois_out,
                images_path=image_root,
                union_dir=Path(union_dir),
                area_col="union_mask_area_um2",
            )
            summary_out = add_union_density(summary_out, rois_df=rois_out)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.warn(
                f"Skipping union-area enrichment: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    summary_out = add_porosity_from_diameters(summary_out)

    profile_diameter_cols = {
        "mean_diameter_p2p_nm": "p2p",
        "mean_diameter_fwhm_nm": "fwhm",
        "mean_diameter_deriv_nm": "deriv",
    }
    if set(profile_diameter_cols).issubset(summary_out.columns):
        summary_out = add_porosity_from_diameters(
            summary_out,
            diameter_cols=profile_diameter_cols,
            output_prefix="porosity_blend_profile",
        )

    return rois_out, summary_out


def enrich_roi_data_pickle(
    input_path: str | Path,
    output_path: str | Path | None = None,
    union_dir: str | Path = "union_masks",
    images_path: str | Path | None = None,
    add_union_area: bool = True,
) -> Path:
    input_file = Path(input_path)
    if output_path is None:
        output_file = input_file
    else:
        output_file = Path(output_path)

    payload = pd.read_pickle(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected pickle payload to be a dict, got: {type(payload)}")

    rois_df = payload.get("rois", pd.DataFrame())
    summary_df = payload.get("summary", pd.DataFrame())
    outlier_rois_df = payload.get("outlier_rois_df", pd.DataFrame())

    rois_df, summary_df = add_extra_columns(
        rois_df=rois_df,
        summary_df=summary_df,
        union_dir=union_dir,
        images_path=images_path,
        add_union_area=add_union_area,
    )

    pd.to_pickle(
        {
            "rois": rois_df,
            "summary": summary_df,
            "outlier_rois_df": outlier_rois_df,
        },
        output_file,
    )
    return output_file


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich roi_data pickle with extra derived columns for notebook plotting.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/roi_data.pickle"),
        help="Input roi_data pickle path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path (defaults to overwriting --input).",
    )
    parser.add_argument(
        "--union-dir",
        type=Path,
        default=Path("union_masks"),
        help="Directory containing union masks and union_mask_summary.csv.",
    )
    parser.add_argument(
        "--images-path",
        type=Path,
        default=None,
        help="Optional TIFF folder used to rebuild union masks if needed.",
    )
    parser.add_argument(
        "--no-union-area",
        action="store_true",
        help="Skip union-mask area and density columns.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_file = enrich_roi_data_pickle(
        input_path=args.input,
        output_path=args.output,
        union_dir=args.union_dir,
        images_path=args.images_path,
        add_union_area=not args.no_union_area,
    )
    print(f"Saved enriched dataset: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
