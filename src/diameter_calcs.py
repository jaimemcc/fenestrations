from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter


def _compute_dip_width(
    profile: np.ndarray,
    step_px: float,
    baseline_n: int = 10,
    level_frac: float = 0.5,
) -> float:
    """
    Compute width of a dip in a profile using a Full Width Half Maximum-style threshold.
    """
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
    """
    Compute width of a dip using slopes from a smoothed derivative profile.
    """
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
    window_range: tuple[int, int] = (10, 30),
    min_run: int = 5,
) -> float:
    """
    Compute dip width from the longest contiguous run below an edge-derived baseline.
    """
    if profile.size == 0:
        return np.nan

    n = profile.size
    widths: list[float] = []
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
    """
    Compute directional profiles and ROI diameters using multiple profile-based methods.
    """
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

        profiles: dict[str, np.ndarray | float] = {}
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
        combined_profile = np.mean(np.vstack(profile_list), axis=0) if profile_list else None

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


__all__ = [
    "compute_profiles_df",
    "_compute_dip_width",
    "_compute_dip_width_derivative",
    "_compute_baseline_dip_width",
]
