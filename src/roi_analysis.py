from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter


def get_stub(file: str | Path) -> str:
    return Path(file).stem.split(".")[0]


def load_segmentation(datafolder: str | Path, stub: str) -> Tuple[object, np.ndarray]:
    path = Path(datafolder) / f"{stub}_seg.npy"
    arr = np.load(path, allow_pickle=True)
    payload = arr.item() if arr.ndim == 0 and arr.dtype == object else arr
    if isinstance(payload, dict) and "masks" in payload:
        masks = payload["masks"]
    else:
        masks = payload
    return payload, masks


def get_pixel_size(datafolder: str | Path, stub: str) -> float:
    file = Path(datafolder) / f"{stub}.tif"
    with tifffile.TiffFile(file) as tif:
        metadata = tif.sem_metadata
        if not metadata or "ap_image_pixel_size" not in metadata:
            raise KeyError(f"Missing ap_image_pixel_size in sem_metadata for: {file}")
        return metadata["ap_image_pixel_size"][1]


def compute_centroids_df(masks: np.ndarray, pixel_size: float, stub: str | None = None) -> pd.DataFrame:
    labels = masks
    ys, xs = np.indices(labels.shape)

    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels)

    sum_x = np.bincount(flat_labels, weights=xs.ravel())
    sum_y = np.bincount(flat_labels, weights=ys.ravel())

    roi_ids = np.nonzero(counts)[0]
    roi_ids = roi_ids[roi_ids != 0]

    if roi_ids.size == 0:
        columns = ["roi_id", "centroid_x", "centroid_y", "pixel_count", "pixel_size", "area"]
        if stub is not None:
            columns = ["stub"] + columns
        return pd.DataFrame(columns=columns)

    centroid_x = sum_x[roi_ids] / counts[roi_ids]
    centroid_y = sum_y[roi_ids] / counts[roi_ids]

    area = counts[roi_ids] * (pixel_size ** 2)

    data = {
        "roi_id": roi_ids,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "pixel_count": counts[roi_ids],
        "pixel_size": pixel_size,
        "area": area,
    }
    if stub is not None:
        data = {"stub": np.repeat(stub, roi_ids.size), **data}

    return pd.DataFrame(data)


def load_image(datafolder: str | Path, stub: str) -> np.ndarray:
    raw = tifffile.imread(Path(datafolder) / f"{stub}.tif")
    return raw[0] if raw.ndim > 2 else raw


def _compute_dip_width(
    profile: np.ndarray,
    step_px: float,
    baseline_n: int = 10,
    level_frac: float = 0.5,
) -> float:
    if profile.size == 0:
        return np.nan

    n = profile.size
    n_base = max(1, min(baseline_n, n // 2))
    baseline = 0.5 * (profile[:n_base].mean() + profile[-n_base:].mean())
    min_idx = int(np.argmin(profile))
    dip = baseline - profile[min_idx]
    if dip <= 0:
        return np.nan

    level = baseline - dip * level_frac
    below = profile <= level
    if not np.any(below):
        return np.nan

    left = min_idx
    while left > 0 and below[left]:
        left -= 1
    right = min_idx
    while right < n - 1 and below[right]:
        right += 1

    width_samples = max(1, right - left - 1)
    return width_samples * step_px


def _compute_dip_width_derivative(
    profile: np.ndarray,
    step_px: float,
    window: int = 11,
    polyorder: int = 2,
) -> float:
    if profile.size == 0:
        return np.nan

    n = profile.size
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 5:
        return np.nan

    smoothed = savgol_filter(profile, window_length=win, polyorder=polyorder, mode="nearest")
    deriv = np.gradient(smoothed)
    min_idx = int(np.argmin(smoothed))

    if min_idx <= 1 or min_idx >= n - 2:
        return np.nan

    left = np.argmin(deriv[:min_idx])
    right = min_idx + np.argmax(deriv[min_idx:])
    if right <= left:
        return np.nan

    return (right - left) * step_px


def _compute_baseline_dip_width(
    profile: np.ndarray,
    step_px: float,
    window_range: Tuple[int, int] = (10, 30),
    min_run: int = 5,
) -> float:
    if profile.size == 0:
        return np.nan

    n = profile.size
    widths = []
    w_start, w_end = window_range

    for w in range(w_start, w_end + 1):
        if w * 2 >= n:
            break

        left_mean = profile[:w].mean()
        right_mean = profile[-w:].mean()
        baseline = np.linspace(left_mean, right_mean, n)

        below = profile < baseline
        below_int = below.astype(int)
        run = np.convolve(below_int, np.ones(min_run, dtype=int), mode="valid")
        valid = run >= min_run
        if not np.any(valid):
            continue

        start = int(np.argmax(valid))
        end = int(len(valid) - 1 - np.argmax(valid[::-1]) + min_run - 1)
        if end <= start:
            continue

        widths.append((end - start + 1) * step_px)

    if not widths:
        return np.nan

    return float(max(widths))


def compute_profiles_df(
    masks: np.ndarray,
    image: np.ndarray,
    pixel_size: float,
    margin_pixels: int = 10,
    num_samples: int = 100,
    baseline_n: int = 10,
) -> pd.DataFrame:
    rows = []
    max_label = int(masks.max()) if masks.size else 0

    for label in range(1, max_label + 1):
        single_mask = masks == label
        if not np.any(single_mask):
            continue

        ys, xs = np.nonzero(single_mask)
        cy = float(ys.mean())
        cx = float(xs.mean())

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        mask_width = x_max - x_min
        mask_height = y_max - y_min

        profiles = {}
        fwhm_values = []

        # Horizontal profile
        length = mask_width + 2 * margin_pixels
        x_coords = np.linspace(cx - length / 2, cx + length / 2, num_samples)
        y_coords = np.full_like(x_coords, cy)
        h_profile = map_coordinates(image, [y_coords, x_coords], order=1, mode="nearest")
        step_px = length / (num_samples - 1)
        profiles["profile_h"] = h_profile
        profiles["profile_length_h"] = length
        fwhm_values.append(_compute_dip_width(h_profile, step_px, baseline_n=baseline_n))

        # Vertical profile
        length = mask_height + 2 * margin_pixels
        y_coords = np.linspace(cy - length / 2, cy + length / 2, num_samples)
        x_coords = np.full_like(y_coords, cx)
        v_profile = map_coordinates(image, [y_coords, x_coords], order=1, mode="nearest")
        step_px = length / (num_samples - 1)
        profiles["profile_v"] = v_profile
        profiles["profile_length_v"] = length
        fwhm_values.append(_compute_dip_width(v_profile, step_px, baseline_n=baseline_n))

        # Diagonal 45
        diag_length = np.sqrt(mask_width**2 + mask_height**2)
        length = diag_length + 2 * margin_pixels
        x_coords = np.linspace(cx - length / 2, cx + length / 2, num_samples)
        y_coords = np.linspace(cy - length / 2, cy + length / 2, num_samples)
        d45_profile = map_coordinates(image, [y_coords, x_coords], order=1, mode="nearest")
        step_px = length / (num_samples - 1)
        profiles["profile_d45"] = d45_profile
        profiles["profile_length_d"] = length
        fwhm_values.append(_compute_dip_width(d45_profile, step_px, baseline_n=baseline_n))

        # Diagonal 135
        x_coords = np.linspace(cx - length / 2, cx + length / 2, num_samples)
        y_coords = np.linspace(cy + length / 2, cy - length / 2, num_samples)
        d135_profile = map_coordinates(image, [y_coords, x_coords], order=1, mode="nearest")
        step_px = length / (num_samples - 1)
        profiles["profile_d135"] = d135_profile
        fwhm_values.append(_compute_dip_width(d135_profile, step_px, baseline_n=baseline_n))

        profile_list = [
            p
            for p in (
                profiles["profile_h"],
                profiles["profile_v"],
                profiles["profile_d45"],
                profiles["profile_d135"],
            )
            if p is not None
        ]
        if profile_list:
            combined_profile = np.mean(np.vstack(profile_list), axis=0)
        else:
            combined_profile = None

        diameter_fwhm_px = np.nanmean(fwhm_values) if fwhm_values else np.nan
        diameter_fwhm = diameter_fwhm_px * pixel_size if np.isfinite(diameter_fwhm_px) else np.nan

        diameter_deriv = np.nan
        if combined_profile is not None:
            width_px = _compute_dip_width_derivative(combined_profile, step_px, window=11, polyorder=2)
            diameter_deriv = width_px * pixel_size if np.isfinite(width_px) else np.nan

        rows.append(
            {
                "roi_id": label,
                "profile_h": profiles["profile_h"],
                "profile_v": profiles["profile_v"],
                "profile_d45": profiles["profile_d45"],
                "profile_d135": profiles["profile_d135"],
                "profile_mean": combined_profile,
                "profile_length_h": profiles.get("profile_length_h"),
                "profile_length_v": profiles.get("profile_length_v"),
                "profile_length_d": profiles.get("profile_length_d"),
                "diameter_fwhm": diameter_fwhm,
                "diameter_deriv": diameter_deriv,
            }
        )

    profiles_df = pd.DataFrame(rows)
    if profiles_df.empty:
        return profiles_df

    mean_profiles = [p for p in profiles_df["profile_mean"] if p is not None]
    if mean_profiles:
        global_mean = np.mean(np.vstack(mean_profiles), axis=0)
        profile_mad = []
        for profile in profiles_df["profile_mean"]:
            if profile is None:
                profile_mad.append(np.nan)
            else:
                profile_mad.append(np.mean(np.abs(profile - global_mean)))
        profiles_df["profile_mad"] = profile_mad
    else:
        profiles_df["profile_mad"] = np.nan

    return profiles_df


def compute_neighbor_stats(centroids_df: pd.DataFrame, max_k: int = 10) -> Tuple[pd.DataFrame, np.ndarray]:
    coords = centroids_df[["centroid_x", "centroid_y"]].to_numpy()
    n_rois = coords.shape[0]

    if n_rois == 0:
        centroids_df = centroids_df.assign(neighbor_count=[], mean_neighbor_distance=[])
        return centroids_df, np.empty((0, 0))

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))

    nearest = np.sort(dists, axis=1)[:, 1 : max_k + 1]

    if nearest.shape[1] == 0:
        neighbor_count = np.zeros(n_rois, dtype=int)
    else:
        jumps = np.diff(nearest, axis=1)
        if jumps.shape[1] == 0:
            neighbor_count = np.ones(n_rois, dtype=int)
        else:
            neighbor_count = np.argmax(jumps, axis=1) + 1

    mean_neighbor_distance = np.array(
        [
            nearest[i, : neighbor_count[i]].mean() if neighbor_count[i] > 0 else np.nan
            for i in range(n_rois)
        ]
    )

    centroids_df = centroids_df.assign(
        neighbor_count=neighbor_count,
        mean_neighbor_distance=mean_neighbor_distance,
    )

    return centroids_df, dists


def compute_summary_df(centroids_df: pd.DataFrame, stub: str) -> pd.DataFrame:
    mean_neighbor_count = centroids_df["neighbor_count"].mean()
    mean_neighbor_distance = centroids_df["mean_neighbor_distance"].mean()
    mean_roi_area = centroids_df["area"].mean()
    mean_pixel_size = centroids_df["pixel_size"].mean()
    mean_diameter_area = centroids_df["diameter_area"].mean()
    mean_diameter_fwhm = centroids_df["diameter_fwhm"].mean()
    mean_diameter_deriv = centroids_df["diameter_deriv"].mean()
    mean_profile_mad = centroids_df["profile_mad"].mean()

    porosity_square = mean_roi_area / (mean_neighbor_distance ** 2)
    porosity_hex = mean_roi_area / ((np.sqrt(3) / 2) * (mean_neighbor_distance ** 2))

    w = np.clip((mean_neighbor_count - 4) / 2, 0, 1)
    cell_area_blend = w * ((np.sqrt(3) / 2) * (mean_neighbor_distance ** 2)) + (1 - w) * (
        mean_neighbor_distance ** 2
    )
    porosity_blend = mean_roi_area / cell_area_blend

    return pd.DataFrame(
        {
            "stub": [stub],
            "mean_neighbor_count": [mean_neighbor_count],
            "mean_neighbor_distance": [mean_neighbor_distance],
            "mean_roi_area": [mean_roi_area],
            "mean_pixel_size": [mean_pixel_size],
            "mean_diameter_area": [mean_diameter_area],
            "mean_diameter_fwhm": [mean_diameter_fwhm],
            "mean_diameter_deriv": [mean_diameter_deriv],
            "mean_profile_mad": [mean_profile_mad],
            "porosity_square": [porosity_square],
            "porosity_hex": [porosity_hex],
            "porosity_blend": [porosity_blend],
            "blend_weight": [w],
        }
    )


def run_stub(datafolder: str | Path, stub: str, max_k: int = 10, tif_dir: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tif_dir is None:
        tif_dir = datafolder
    _, masks = load_segmentation(datafolder, stub)
    pixel_size = get_pixel_size(tif_dir, stub)
    image = load_image(tif_dir, stub)

    centroids_df = compute_centroids_df(masks, pixel_size, stub=stub)
    centroids_df, _ = compute_neighbor_stats(centroids_df, max_k=max_k)
    profiles_df = compute_profiles_df(masks, image, pixel_size)

    mean_profile = None
    mean_profile_diameter = np.nan
    mean_profile_diameter_deriv = np.nan
    mean_profile_diameter_baseline = np.nan
    mean_profile_step = np.nan

    if not profiles_df.empty:
        centroids_df = centroids_df.merge(profiles_df, on="roi_id", how="left")
        mean_profiles = [p for p in profiles_df["profile_mean"] if p is not None]
        if mean_profiles:
            mean_profile = np.mean(np.vstack(mean_profiles), axis=0)

            lengths_h = profiles_df["profile_length_h"].dropna().to_numpy()
            lengths_v = profiles_df["profile_length_v"].dropna().to_numpy()
            lengths_d = profiles_df["profile_length_d"].dropna().to_numpy()
            lengths_all = np.concatenate([lengths_h, lengths_v, lengths_d])
            if lengths_all.size > 0:
                mean_length = lengths_all.mean()
                mean_profile_step = mean_length / (mean_profile.size - 1)
                mean_profile_diameter = (
                    _compute_dip_width(mean_profile, mean_profile_step, level_frac=0.1) * pixel_size
                )
                mean_profile_diameter_deriv = (
                    _compute_dip_width_derivative(mean_profile, mean_profile_step) * pixel_size
                )
                mean_profile_diameter_baseline = (
                    _compute_baseline_dip_width(mean_profile, mean_profile_step, window_range=(10, 30), min_run=5)
                    * pixel_size
                )
    else:
        centroids_df = centroids_df.assign(
            profile_h=None,
            profile_v=None,
            profile_d45=None,
            profile_d135=None,
            profile_mean=None,
            profile_length_h=np.nan,
            profile_length_v=np.nan,
            profile_length_d=np.nan,
            profile_mad=np.nan,
            diameter_fwhm=np.nan,
            diameter_deriv=np.nan,
            diameter_baseline=np.nan,
        )

    centroids_df["diameter_area"] = 2 * np.sqrt(centroids_df["area"] / np.pi)
    summary_df = compute_summary_df(centroids_df, stub)
    summary_df = summary_df.assign(
        mean_profile=[mean_profile],
        mean_profile_diameter=[mean_profile_diameter],
        mean_profile_diameter_deriv=[mean_profile_diameter_deriv],
        mean_profile_diameter_baseline=[mean_profile_diameter_baseline],
        mean_profile_step=[mean_profile_step],
    )

    return centroids_df, summary_df


def run_batch(
    datafolder: str | Path, stubs: Iterable[str], max_k: int = 10, tif_dir: str | Path | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rois_list = []
    summary_list = []

    for stub in stubs:
        rois_df, summary_df = run_stub(datafolder, stub, max_k=max_k, tif_dir=tif_dir)
        rois_list.append(rois_df)
        summary_list.append(summary_df)

    all_rois_df = pd.concat(rois_list, ignore_index=True) if rois_list else pd.DataFrame()
    all_summary_df = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()

    return all_rois_df, all_summary_df
