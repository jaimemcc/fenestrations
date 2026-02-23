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


def _polygon_internal_angles_deg(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([], dtype=float)

    prev_pts = np.roll(points, shift=1, axis=0)
    next_pts = np.roll(points, shift=-1, axis=0)

    v1 = prev_pts - points
    v2 = next_pts - points

    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    denom = norm1 * norm2

    angles = np.full(points.shape[0], np.nan, dtype=float)
    valid = denom > 0
    if np.any(valid):
        cosang = np.einsum("ij,ij->i", v1[valid], v2[valid]) / denom[valid]
        cosang = np.clip(cosang, -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(cosang))

    return angles


def _is_convex_polygon(points: np.ndarray, eps: float = 1e-12) -> bool:
    n = points.shape[0]
    if n < 3:
        return False

    edges = np.roll(points, shift=-1, axis=0) - points
    next_edges = np.roll(edges, shift=-1, axis=0)
    cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]

    nonzero = np.abs(cross) > eps
    if not np.any(nonzero):
        return False

    signs = np.sign(cross[nonzero])
    return np.all(signs > 0) or np.all(signs < 0)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray, eps: float = 1e-12) -> bool:
    x, y = point
    inside = False
    n = polygon.shape[0]

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
        if abs(cross) <= eps and dot <= eps:
            return True

        intersects = (y1 > y) != (y2 > y)
        if intersects:
            xinters = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xinters:
                inside = not inside

    return inside


def compute_polygon_cluster_stats(
    centroids_df: pd.DataFrame,
    max_neighbors: int = 8,
    max_internal_angle_deg: float = 175.0,
    max_angular_gap_deg: float = 180.0,
) -> pd.DataFrame:
    coords = centroids_df[["centroid_x", "centroid_y"]].to_numpy()
    roi_ids = centroids_df["roi_id"].to_numpy()
    n_rois = coords.shape[0]

    if n_rois == 0:
        return centroids_df.assign(
            cluster_neighbor_count=[],
            cluster_neighbor_ids=[],
            cluster_max_internal_angle_deg=[],
            cluster_max_angular_gap_deg=[],
            cluster_is_convex=[],
            cluster_center_inside_polygon=[],
            cluster_is_valid=[],
        )

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))

    cluster_neighbor_count = np.zeros(n_rois, dtype=int)
    cluster_neighbor_ids: list[list[int]] = [[] for _ in range(n_rois)]
    cluster_max_internal_angle = np.full(n_rois, np.nan, dtype=float)
    cluster_max_angular_gap = np.full(n_rois, np.nan, dtype=float)
    cluster_is_convex = np.zeros(n_rois, dtype=bool)
    cluster_center_inside = np.zeros(n_rois, dtype=bool)
    cluster_is_valid = np.zeros(n_rois, dtype=bool)
    cluster_neighbor_distance = np.full(n_rois, np.nan, dtype=float)

    for i in range(n_rois):
        center = coords[i]
        order_by_dist = np.argsort(dists[i])
        max_neighbors_i = min(max_neighbors, n_rois - 1)
        neighbor_idx_full = order_by_dist[1 : max_neighbors_i + 1]
        if neighbor_idx_full.size < 3:
            continue

        fallback_result = None
        chosen_result = None

        for k in range(neighbor_idx_full.size, 2, -1):
            neighbor_idx = neighbor_idx_full[:k]
            neighbor_pts = coords[neighbor_idx]
            rel = neighbor_pts - center
            theta = np.arctan2(rel[:, 1], rel[:, 0])
            sort_idx = np.argsort(theta)

            polygon = neighbor_pts[sort_idx]
            polygon_ids = roi_ids[neighbor_idx][sort_idx]
            theta_sorted = theta[sort_idx]

            internal_angles = _polygon_internal_angles_deg(polygon)
            max_angle = np.nanmax(internal_angles)

            gaps = np.diff(np.r_[theta_sorted, theta_sorted[0] + 2 * np.pi])
            max_gap = float(np.degrees(np.max(gaps)))

            is_convex = _is_convex_polygon(polygon)
            center_inside = _point_in_polygon(center, polygon)

            is_valid = (
                is_convex
                and center_inside
                and np.isfinite(max_angle)
                and (max_angle < max_internal_angle_deg)
                and (max_gap <= max_angular_gap_deg)
            )

            result = {
                "count": polygon.shape[0],
                "ids": [int(v) for v in polygon_ids],
                "max_angle": max_angle,
                "max_gap": max_gap,
                "is_convex": is_convex,
                "center_inside": center_inside,
                "is_valid": is_valid,
                "neighbor_distance": float(np.linalg.norm(neighbor_pts - center, axis=1).mean()),
            }

            if fallback_result is None:
                fallback_result = result
            if is_valid:
                chosen_result = result
                break

        final_result = chosen_result if chosen_result is not None else fallback_result
        if final_result is None:
            continue

        cluster_neighbor_count[i] = final_result["count"]
        cluster_neighbor_ids[i] = final_result["ids"]
        cluster_max_internal_angle[i] = final_result["max_angle"]
        cluster_max_angular_gap[i] = final_result["max_gap"]
        cluster_is_convex[i] = final_result["is_convex"]
        cluster_center_inside[i] = final_result["center_inside"]
        cluster_is_valid[i] = final_result["is_valid"]
        if final_result["is_valid"]:
            cluster_neighbor_distance[i] = final_result["neighbor_distance"]

    return centroids_df.assign(
        cluster_neighbor_count=cluster_neighbor_count,
        cluster_neighbor_ids=cluster_neighbor_ids,
        cluster_max_internal_angle_deg=cluster_max_internal_angle,
        cluster_max_angular_gap_deg=cluster_max_angular_gap,
        cluster_is_convex=cluster_is_convex,
        cluster_center_inside_polygon=cluster_center_inside,
        cluster_is_valid=cluster_is_valid,
        cluster_neighbor_distance=cluster_neighbor_distance,
    )


def compute_profiles_from_principal_axes(
    roi_mask: np.ndarray,
    image: np.ndarray,
    centroid_x: float,
    centroid_y: float,
    pixel_size: float,
    num_samples: int = 256,
    extension_factor: float = 0.25,
) -> dict:
    """
    Extract major and minor axis profiles for an ROI based on principal axes.
    
    Parameters
    ----------
    roi_mask : np.ndarray
        2D boolean or label mask for the ROI
    image : np.ndarray
        2D original image to sample intensity from
    centroid_x, centroid_y : float
        ROI centroid coordinates in pixels
    pixel_size : float
        Physical size of each pixel (µm/pixel)
    num_samples : int
        Number of samples along each profile (default 256)
    extension_factor : float
        How much to extend beyond ROI (0.25 = 25% on each side)
    
    Returns
    -------
    dict
        Keys: 'major_angle', 'major_extent', 'minor_extent', 
              'major_length_px', 'minor_length_px', 'step_major', 'step_minor',
              'profile_major', 'profile_minor', 'success'
    """
    
    # Get ROI pixel coordinates
    ys, xs = np.nonzero(roi_mask)
    if len(xs) < 3:
        return {
            "success": False,
            "profile_major": None,
            "profile_minor": None,
            "major_angle": np.nan,
            "major_extent": np.nan,
            "minor_extent": np.nan,
            "major_length_px": np.nan,
            "minor_length_px": np.nan,
            "step_major": np.nan,
            "step_minor": np.nan,
        }
    
    # Find major axis by searching for angle with max extent
    max_extent = 0
    best_angle = 0
    
    for angle_deg in np.linspace(0, 180, 181):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Project pixels onto this angle
        proj = (xs - centroid_x) * cos_a + (ys - centroid_y) * sin_a
        extent = proj.max() - proj.min()
        
        if extent > max_extent:
            max_extent = extent
            best_angle = angle_deg
    
    major_angle = best_angle
    minor_angle = (major_angle + 90) % 180
    
    # Get extents along both axes
    major_rad = np.radians(major_angle)
    minor_rad = np.radians(minor_angle)
    
    cos_maj = np.cos(major_rad)
    sin_maj = np.sin(major_rad)
    cos_min = np.cos(minor_rad)
    sin_min = np.sin(minor_rad)
    
    proj_major = (xs - centroid_x) * cos_maj + (ys - centroid_y) * sin_maj
    proj_minor = (xs - centroid_x) * cos_min + (ys - centroid_y) * sin_min
    
    major_extent = (proj_major.max() - proj_major.min()) / 2.0  # half-extent from center
    minor_extent = (proj_minor.max() - proj_minor.min()) / 2.0
    
    # Extend by extension_factor (25%) on each side
    major_extent_total = major_extent * (1 + 2 * extension_factor)
    minor_extent_total = minor_extent * (1 + 2 * extension_factor)
    
    # Total path lengths in pixels
    major_length_px = 2 * major_extent_total
    minor_length_px = 2 * minor_extent_total
    
    # Sample step in pixels
    step_major = major_length_px / (num_samples - 1) if num_samples > 1 else 1.0
    step_minor = minor_length_px / (num_samples - 1) if num_samples > 1 else 1.0
    
    # Sample along major axis
    sample_dists_major = np.linspace(-major_extent_total, major_extent_total, num_samples)
    coords_major_x = centroid_x + sample_dists_major * cos_maj
    coords_major_y = centroid_y + sample_dists_major * sin_maj
    profile_major = map_coordinates(image, [coords_major_y, coords_major_x], order=1, cval=np.nan)
    
    # Sample along minor axis
    sample_dists_minor = np.linspace(-minor_extent_total, minor_extent_total, num_samples)
    coords_minor_x = centroid_x + sample_dists_minor * cos_min
    coords_minor_y = centroid_y + sample_dists_minor * sin_min
    profile_minor = map_coordinates(image, [coords_minor_y, coords_minor_x], order=1, cval=np.nan)
    
    return {
        "success": True,
        "profile_major": profile_major,
        "profile_minor": profile_minor,
        "major_angle": major_angle,
        "major_extent": major_extent,
        "minor_extent": minor_extent,
        "major_length_px": major_length_px,
        "minor_length_px": minor_length_px,
        "step_major": step_major,
        "step_minor": step_minor,
    }


def plot_cluster_centroids(
    centroids_df: pd.DataFrame,
    valid_col: str = "cluster_is_valid",
    figsize: tuple[float, float] = (6.0, 6.0),
    ax=None,
):
    import matplotlib.pyplot as plt

    required = {"centroid_x", "centroid_y", valid_col}
    missing = required - set(centroids_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for cluster plot: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    is_valid = centroids_df[valid_col].fillna(False).to_numpy(dtype=bool)
    x = centroids_df["centroid_x"].to_numpy()
    y = centroids_df["centroid_y"].to_numpy()

    ax.scatter(x[~is_valid], y[~is_valid], s=18, alpha=0.65, label="not cluster", color="tab:gray")
    ax.scatter(x[is_valid], y[is_valid], s=28, alpha=0.9, label="cluster", color="tab:blue")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("centroid_x")
    ax.set_ylabel("centroid_y")
    ax.set_title("ROI centroid clusters")
    ax.invert_yaxis()
    ax.legend(frameon=False)

    return fig, ax


def plot_cluster_polygons(
    centroids_df: pd.DataFrame,
    valid_col: str = "cluster_is_valid",
    neighbor_ids_col: str = "cluster_neighbor_ids",
    roi_id_col: str = "roi_id",
    figsize: tuple[float, float] = (6.0, 6.0),
    ax=None,
):
    import ast
    import matplotlib.pyplot as plt

    required = {"centroid_x", "centroid_y", roi_id_col, valid_col, neighbor_ids_col}
    missing = required - set(centroids_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for polygon plot: {sorted(missing)}")

    id_to_xy = {
        int(row[roi_id_col]): (float(row["centroid_x"]), float(row["centroid_y"]))
        for _, row in centroids_df.iterrows()
    }

    fig, ax = plot_cluster_centroids(centroids_df, valid_col=valid_col, figsize=figsize, ax=ax)

    valid_rows = centroids_df[centroids_df[valid_col].fillna(False)]
    for _, row in valid_rows.iterrows():
        ids_value = row[neighbor_ids_col]
        if isinstance(ids_value, str):
            try:
                ids_value = ast.literal_eval(ids_value)
            except (ValueError, SyntaxError):
                continue

        if not isinstance(ids_value, (list, tuple)) or len(ids_value) < 3:
            continue

        polygon_xy = []
        for rid in ids_value:
            rid_int = int(rid)
            if rid_int in id_to_xy:
                polygon_xy.append(id_to_xy[rid_int])

        if len(polygon_xy) < 3:
            continue

        poly = np.asarray(polygon_xy, dtype=float)
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="tab:orange", alpha=0.45, linewidth=1.2)

    ax.set_title("ROI centroid clusters with polygons")
    return fig, ax


def plot_roi_neighbor_comparison(
    image: np.ndarray,
    masks: np.ndarray,
    centroids_df: pd.DataFrame,
    roi_id: int | None = None,
    max_neighbors: int = 10,
    figsize: tuple[float, float] = (12.0, 6.0),
    random_seed: int | None = None,
):
    import ast
    import matplotlib.pyplot as plt
    import time

    required = {"roi_id", "centroid_x", "centroid_y"}
    missing = required - set(centroids_df.columns)
    if missing:
        raise KeyError(f"Missing required columns for ROI comparison plot: {sorted(missing)}")

    plot_df = centroids_df.copy()

    if "cluster_neighbor_ids" not in plot_df.columns:
        plot_df = compute_polygon_cluster_stats(plot_df, max_neighbors=max_neighbors)

    roi_ids_all = plot_df["roi_id"].to_numpy(dtype=int)
    if roi_ids_all.size == 0:
        raise ValueError("No ROIs available in centroids_df")

    if roi_id is None:
        seed = int(time.time_ns()) if random_seed is None else int(random_seed)
        rng = np.random.default_rng(seed)
        roi_id = int(rng.choice(roi_ids_all))

    roi_index = np.where(roi_ids_all == int(roi_id))[0]
    if roi_index.size == 0:
        raise ValueError(f"ROI id {roi_id} not found in centroids_df")
    roi_index = int(roi_index[0])

    cluster_ids_value = plot_df.iloc[roi_index]["cluster_neighbor_ids"]
    if isinstance(cluster_ids_value, str):
        try:
            cluster_ids_value = ast.literal_eval(cluster_ids_value)
        except (ValueError, SyntaxError):
            cluster_ids_value = []

    if isinstance(cluster_ids_value, (list, tuple, np.ndarray)):
        cluster_neighbor_ids = np.asarray([int(v) for v in cluster_ids_value], dtype=int)
    else:
        cluster_neighbor_ids = np.asarray([], dtype=int)

    is_valid = plot_df.iloc[roi_index].get("cluster_is_valid", False)
    valid_str = "valid" if is_valid else "invalid"

    mask_main = masks == int(roi_id)
    mask_cluster = np.isin(masks, cluster_neighbor_ids)

    main_masked = np.ma.masked_where(~mask_main, mask_main)
    cluster_masked = np.ma.masked_where(~mask_cluster, mask_cluster)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(image, cmap="gray")
    ax.imshow(cluster_masked, cmap="Blues", alpha=0.85)
    ax.imshow(main_masked, cmap="Reds", alpha=0.6)
    ax.set_title(f"ROI {roi_id} cluster neighbors ({valid_str}, n={len(cluster_neighbor_ids)})")
    ax.axis("off")

    plt.tight_layout()

    return fig, ax



def compute_summary_df(centroids_df: pd.DataFrame, stub: str) -> pd.DataFrame:
    if "cluster_is_valid" in centroids_df.columns:
        valid_clusters = centroids_df[centroids_df["cluster_is_valid"] == True]
        if not valid_clusters.empty:
            mean_neighbor_count = valid_clusters["cluster_neighbor_count"].mean()
        else:
            mean_neighbor_count = centroids_df["cluster_neighbor_count"].mean()
    else:
        mean_neighbor_count = np.nan

    if "cluster_neighbor_count" in centroids_df.columns:
        valid_mask = centroids_df.get("cluster_is_valid", pd.Series(True, index=centroids_df.index)).fillna(False)
        coords = centroids_df.loc[valid_mask, ["centroid_x", "centroid_y"]].to_numpy()
        if coords.shape[0] > 0:
            diffs = coords[:, None, :] - coords[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            mean_neighbor_distance = float(np.nanmean(np.min(dists, axis=1))) if np.isfinite(dists).any() else np.nan
        else:
            mean_neighbor_distance = np.nan
    else:
        mean_neighbor_distance = np.nan

    mean_roi_area = centroids_df["area"].mean()
    mean_pixel_size = centroids_df["pixel_size"].mean()
    mean_diameter_area = centroids_df["diameter_area"].mean()
    mean_diameter_fwhm = centroids_df["diameter_fwhm"].mean()
    mean_diameter_deriv = centroids_df["diameter_deriv"].mean()
    mean_profile_mad = centroids_df["profile_mad"].mean()

    if "cluster_neighbor_distance" in centroids_df.columns:
        mean_cluster_neighbor_distance = centroids_df["cluster_neighbor_distance"].mean()
    else:
        mean_cluster_neighbor_distance = np.nan

    # Convert pixel distances to physical units
    if np.isfinite(mean_neighbor_distance) and np.isfinite(mean_pixel_size):
        mean_neighbor_distance_phys = mean_neighbor_distance * mean_pixel_size
    else:
        mean_neighbor_distance_phys = np.nan

    if np.isfinite(mean_cluster_neighbor_distance) and np.isfinite(mean_pixel_size):
        mean_cluster_neighbor_distance_phys = mean_cluster_neighbor_distance * mean_pixel_size
    else:
        mean_cluster_neighbor_distance_phys = np.nan

    # Recalculate porosity with physical units (dimensionless)
    porosity_square = mean_roi_area / (mean_neighbor_distance_phys ** 2) if np.isfinite(mean_neighbor_distance_phys) else np.nan
    porosity_hex = mean_roi_area / ((np.sqrt(3) / 2) * (mean_neighbor_distance_phys ** 2)) if np.isfinite(mean_neighbor_distance_phys) else np.nan

    w = np.clip((mean_neighbor_count - 4) / 2, 0, 1) if np.isfinite(mean_neighbor_count) else np.nan
    if np.isfinite(mean_neighbor_distance_phys):
        cell_area_blend = w * ((np.sqrt(3) / 2) * (mean_neighbor_distance_phys ** 2)) + (1 - w) * (
            mean_neighbor_distance_phys ** 2
        )
        porosity_blend = mean_roi_area / cell_area_blend
    else:
        porosity_blend = np.nan

    return pd.DataFrame(
        {
            "stub": [stub],
            "mean_neighbor_count": [mean_neighbor_count],
            "mean_neighbor_distance": [mean_neighbor_distance],
            "mean_neighbor_distance_phys": [mean_neighbor_distance_phys],
            "mean_cluster_neighbor_distance": [mean_cluster_neighbor_distance],
            "mean_cluster_neighbor_distance_phys": [mean_cluster_neighbor_distance_phys],
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
    centroids_df = compute_polygon_cluster_stats(centroids_df, max_neighbors=max_k)
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

    # Compute principal axes profiles for each ROI
    profile_major_list = []
    profile_minor_list = []
    major_angle_list = []
    major_extent_list = []
    minor_extent_list = []
    major_length_px_list = []
    minor_length_px_list = []
    step_major_list = []
    step_minor_list = []

    for _, row in centroids_df.iterrows():
        roi_id = int(row["roi_id"])
        roi_mask = masks == roi_id
        centroid_x = float(row["centroid_x"])
        centroid_y = float(row["centroid_y"])

        result = compute_profiles_from_principal_axes(
            roi_mask=roi_mask,
            image=image,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            pixel_size=pixel_size,
            num_samples=256,
            extension_factor=0.50,
        )

        profile_major_list.append(result["profile_major"] if result["success"] else None)
        profile_minor_list.append(result["profile_minor"] if result["success"] else None)
        major_angle_list.append(result["major_angle"] if result["success"] else np.nan)
        major_extent_list.append(result["major_extent"] if result["success"] else np.nan)
        minor_extent_list.append(result["minor_extent"] if result["success"] else np.nan)
        major_length_px_list.append(result["major_length_px"] if result["success"] else np.nan)
        minor_length_px_list.append(result["minor_length_px"] if result["success"] else np.nan)
        step_major_list.append(result["step_major"] if result["success"] else np.nan)
        step_minor_list.append(result["step_minor"] if result["success"] else np.nan)

    centroids_df = centroids_df.assign(
        profile_major=profile_major_list,
        profile_minor=profile_minor_list,
        major_axis_angle=major_angle_list,
        major_axis_extent=major_extent_list,
        minor_axis_extent=minor_extent_list,
        major_axis_length_px=major_length_px_list,
        minor_axis_length_px=minor_length_px_list,
        step_major=step_major_list,
        step_minor=step_minor_list,
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
