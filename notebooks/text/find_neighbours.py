# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python (fenestrations)
#     language: python
#     name: fenestrations-venv
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import tifffile

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roi_analysis import (
    compute_centroids_df,
    compute_neighbor_stats,
    get_pixel_size,
    get_stub,
    load_segmentation,
    run_batch,
)


# %%
# Load ROI data
from pathlib import Path as FSPath

data_path = FSPath("C:/Users/sthui4072/Github/fenestrations/roi_data.pickle")
images_path = FSPath("C:/Users/sthui4072/Github/fenestrations/images")
metafile_path = FSPath("C:/Users/sthui4072/Github/fenestrations/fenestrations_metafile.xlsx")

# data_path = FSPath("../data/test.pickle")
data = pd.read_pickle(data_path)

# Extract the DataFrames
all_rois_df = data['rois']
all_summary_df = data['summary']

# load metafile
metadata = pd.read_excel(metafile_path)


# %%
# Optional batch re-run from segmentation outputs (if available).
DATAFOLDER = FSPath("C:/Users/sthui4072/Github/fenestrations/flatten_npy")  # expected location for *_seg.npy files
TIF_DIR = images_path

files = sorted(images_path.glob("*.tif"))
stubs = [get_stub(file.name) for file in files]
OUTFILE = ROOT / "roi_neighbor_results.pkl"

print(f"Found {len(stubs)} image stubs in {images_path}")
print(f"Using segmentation folder: {DATAFOLDER}")

# %%
# Recompute from raw segmentation only if *_seg.npy files are available.
seg_files = list(DATAFOLDER.glob("*_seg.npy"))
seg_stubs = {p.name[: -len("_seg.npy")] for p in seg_files}
stubs_to_run = [s for s in stubs if s in seg_stubs]

if not stubs_to_run:
    print("No segmentation files found; using data loaded from roi_data.pickle")
    all_rois_df = rois_df.copy()
    all_summary_df = summary_df.copy()
else:
    print(f"Running batch on {len(stubs_to_run)} stubs")
    all_rois_df, all_summary_df = run_batch(DATAFOLDER, stubs_to_run, max_k=10, tif_dir=TIF_DIR)

all_rois_df.head()
# all_summary_df

# %%
all_rois_df.columns

# %%
all_summary_df

# %%
with open(OUTFILE, "wb") as f:
    pickle.dump({"rois": all_rois_df, "summary": all_summary_df}, f)


OUTFILE

# %%
all_summary_df

# %%
# Visualize mean profile and diameter estimates per stub
rows = all_summary_df.dropna(subset=["mean_profile"]).copy()
if rows.empty:
    raise ValueError("No mean profiles available in summary")

nrows = len(rows)
fig, axes = plt.subplots(nrows, 1, figsize=(8, 3 * nrows), sharex=True)
if nrows == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, rows.iterrows()):
    profile = row["mean_profile"]
    if profile is None:
        continue

    x = np.arange(len(profile))
    center = (len(profile) - 1) / 2

    ax.plot(profile, color="k", linewidth=1, label="Mean profile")

    step_px = row.get("mean_profile_step")
    px_size = row.get("mean_pixel_size")

    if pd.notna(step_px) and pd.notna(px_size):
        if pd.notna(row.get("mean_diameter_area")):
            half = (row["mean_diameter_area"] / (step_px * px_size)) / 2
            ax.axvline(center - half, color="g", linestyle="--", linewidth=1, label="mean_diam_area")
            ax.axvline(center + half, color="g", linestyle="--", linewidth=1)

        if pd.notna(row.get("mean_diameter_fwhm")):
            half = (row["mean_diameter_fwhm"] / (step_px * px_size)) / 2
            ax.axvline(center - half, color="b", linestyle=":", linewidth=1, label="mean_diam_fwhm")
            ax.axvline(center + half, color="b", linestyle=":", linewidth=1)

        if pd.notna(row.get("mean_profile_diameter")):
            half = (row["mean_profile_diameter"] / (step_px * px_size)) / 2
            ax.axvline(center - half, color="m", linestyle="-.", linewidth=1, label="mean_profile_diam")
            ax.axvline(center + half, color="m", linestyle="-.", linewidth=1)

        if pd.notna(row.get("mean_profile_diameter_deriv")):
            half = (row["mean_profile_diameter_deriv"] / (step_px * px_size)) / 2
            ax.axvline(center - half, color="c", linestyle="-", linewidth=1, label="mean_profile_deriv")
            ax.axvline(center + half, color="c", linestyle="-", linewidth=1)

        if pd.notna(row.get("mean_profile_diameter_baseline")):
            half = (row["mean_profile_diameter_baseline"] / (step_px * px_size)) / 2
            ax.axvline(center - half, color="y", linestyle="-", linewidth=1, label="mean_profile_baseline")
            ax.axvline(center + half, color="y", linestyle="-", linewidth=1)

    ax.set_title(f"Stub {row['stub']}")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best")
 
axes[-1].set_xlabel("Sample index")
plt.tight_layout()
plt.show()

# %%
# Visualize profiles for a selected ROI (with global mean for the stub)
profile_stub = next((stub for stub in stubs if not all_rois_df[all_rois_df["stub"] == stub].empty), None)
if profile_stub is None:
    raise ValueError("No stub from the image list exists in all_rois_df")

rois_stub = all_rois_df[all_rois_df["stub"] == profile_stub].copy()
if rois_stub.empty:
    raise ValueError(f"No ROIs found for stub {profile_stub}")

roi_index_choice = min(301, len(rois_stub) - 1)
roi_id = int(rois_stub["roi_id"].iloc[roi_index_choice])  # change this to any ROI id
roi_row = rois_stub[rois_stub["roi_id"] == roi_id].iloc[0]

import re

def resolve_stub_file(base_path, stub, suffix):
    candidates = [base_path / f"{stub}{suffix}"]
    match = re.match(r"^(FAS|PR)(\d+)(_.*)$", stub)
    if match:
        candidates.append(base_path / f"{match.group(1)}_{match.group(2)}{match.group(3)}{suffix}")
    match = re.match(r"^(FAS|PR)_(\d+)(_.*)$", stub)
    if match:
        candidates.append(base_path / f"{match.group(1)}{match.group(2)}{match.group(3)}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve file for stub {stub} in {base_path}")

seg_path = resolve_stub_file(DATAFOLDER, profile_stub, "_seg.npy")
payload = np.load(seg_path, allow_pickle=True)
payload = payload.item() if payload.ndim == 0 and payload.dtype == object else payload
masks = payload["masks"] if isinstance(payload, dict) and "masks" in payload else payload

tif_path = resolve_stub_file(images_path, profile_stub, ".tif")
raw = tifffile.imread(tif_path)
image = raw[0] if raw.ndim > 2 else raw

def pick_profile(row, *names):
    for name in names:
        if name in row.index and row[name] is not None:
            return row[name]
    return None

profiles = {
    "Horizontal": pick_profile(roi_row, "profile_h", "profile_major"),
    "Vertical": pick_profile(roi_row, "profile_v", "profile_minor"),
    "Diagonal 45": pick_profile(roi_row, "profile_d45", "profile_diag45"),
    "Diagonal 135": pick_profile(roi_row, "profile_d135", "profile_diag135"),
}

mean_profiles = [p for p in (rois_stub["profile_mean"] if "profile_mean" in rois_stub.columns else rois_stub["four_axis_mean"]) if p is not None]
if mean_profiles:
    global_mean = np.mean(np.vstack(mean_profiles), axis=0)
    global_std = np.std(np.vstack(mean_profiles), axis=0)
else:
    global_mean = None
    global_std = None

single_mask = masks == roi_id
ys, xs = np.nonzero(single_mask)
if ys.size == 0 or xs.size == 0:
    raise ValueError(f"ROI id {roi_id} has no pixels")

x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()
mask_width = x_max - x_min
mask_height = y_max - y_min

margin_pixels = 10
num_samples = len(next(p for p in profiles.values() if p is not None))

length_h = mask_width + 2 * margin_pixels
length_v = mask_height + 2 * margin_pixels
length_d = np.sqrt(mask_width ** 2 + mask_height ** 2) + 2 * margin_pixels

step_h = length_h / (num_samples - 1)
step_v = length_v / (num_samples - 1)
step_d = length_d / (num_samples - 1)

pixel_size = float(roi_row["pixel_size"])
diam_area_px = float(roi_row["diameter_area"]) / pixel_size

diam_fwhm = roi_row.get("diameter_fwhm")
diam_fwhm_px = float(diam_fwhm) / pixel_size if pd.notna(diam_fwhm) else np.nan

diam_deriv = roi_row.get("diameter_deriv")
diam_deriv_px = float(diam_deriv) / pixel_size if pd.notna(diam_deriv) else np.nan

diam_baseline = roi_row.get("diameter_baseline")
diam_baseline_px = float(diam_baseline) / pixel_size if pd.notna(diam_baseline) else np.nan

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
axes = axes.ravel()

for ax, (name, profile) in zip(axes, profiles.items()):
    if profile is None:
        ax.set_title(f"{name} (missing)")
        ax.axis("off")
        continue

    if name == "Horizontal":
        step = step_h
    elif name == "Vertical":
        step = step_v
    else:
        step = step_d

    x = np.arange(len(profile))
    center = (len(profile) - 1) / 2

    ax.plot(profile, color="k", alpha=0.6, label="ROI profile")
    if global_mean is not None:
        ax.plot(global_mean, color="r", linewidth=2, label="Stub mean")
        ax.fill_between(x, global_mean - global_std, global_mean + global_std, color="r", alpha=0.2)

    if np.isfinite(diam_area_px):
        half = (diam_area_px / 2) / step
        ax.axvline(center - half, color="g", linestyle="--", linewidth=1, label="diam_area")
        ax.axvline(center + half, color="g", linestyle="--", linewidth=1)

    if np.isfinite(diam_fwhm_px):
        half = (diam_fwhm_px / 2) / step
        ax.axvline(center - half, color="b", linestyle=":", linewidth=1, label="diam_fwhm")
        ax.axvline(center + half, color="b", linestyle=":", linewidth=1)

    if np.isfinite(diam_deriv_px):
        half = (diam_deriv_px / 2) / step
        ax.axvline(center - half, color="c", linestyle="-", linewidth=1, label="diam_deriv")
        ax.axvline(center + half, color="c", linestyle="-", linewidth=1)

    if np.isfinite(diam_baseline_px):
        half = (diam_baseline_px / 2) / step
        ax.axvline(center - half, color="y", linestyle="-", linewidth=1, label="diam_baseline")
        ax.axvline(center + half, color="y", linestyle="-", linewidth=1)

    ax.set_title(name)

fig.suptitle(f"Profiles for ROI {roi_id} (stub {profile_stub})", y=0.98)
for ax in axes:
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Intensity")

axes[0].legend(loc="best")
plt.tight_layout()
plt.show()

# %%
# Profiles using principal axes (major + minor at 90 degrees, 50% extension) with ROI image panel
from src.roi_analysis import compute_profiles_from_principal_axes, estimate_diameter_from_profile
import re


def resolve_stub_file(base_path, stub, suffix):
    candidates = [base_path / f"{stub}{suffix}"]
    match = re.match(r"^(FAS|PR)(\d+)(_.*)$", stub)
    if match:
        candidates.append(base_path / f"{match.group(1)}_{match.group(2)}{match.group(3)}{suffix}")
    match = re.match(r"^(FAS|PR)_(\d+)(_.*)$", stub)
    if match:
        candidates.append(base_path / f"{match.group(1)}{match.group(2)}{match.group(3)}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve file for stub {stub} in {base_path}")


def has_required_files(stub):
    try:
        _ = resolve_stub_file(DATAFOLDER, stub, "_seg.npy")
        _ = resolve_stub_file(images_path, stub, ".tif")
        return True
    except FileNotFoundError:
        return False


def estimate_fwhm_from_profile(profile, pixel_size, sample_step_px):
    if profile is None:
        return np.nan

    y = np.asarray(profile, dtype=float).ravel()
    if y.size < 5 or not np.isfinite(pixel_size) or not np.isfinite(sample_step_px) or sample_step_px <= 0:
        return np.nan

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.nan

    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return np.nan

    half = y_min + 0.5 * (y_max - y_min)
    above = y >= half
    idx = np.where(above)[0]
    if idx.size < 2:
        return np.nan

    left = int(idx[0])
    right = int(idx[-1])

    left_cross = float(left)
    if left > 0 and np.isfinite(y[left - 1]) and np.isfinite(y[left]) and y[left] != y[left - 1]:
        left_cross = (left - 1) + (half - y[left - 1]) / (y[left] - y[left - 1])

    right_cross = float(right)
    if right < y.size - 1 and np.isfinite(y[right]) and np.isfinite(y[right + 1]) and y[right + 1] != y[right]:
        right_cross = right + (half - y[right]) / (y[right + 1] - y[right])

    width_samples = right_cross - left_cross
    if width_samples <= 0 or not np.isfinite(width_samples):
        return np.nan

    return float(width_samples) * float(sample_step_px) * float(pixel_size)


def fmt_nm(value):
    return f"{value:.1f}" if np.isfinite(value) else "nan"


# Randomly sample ROIs each run from stubs that have both seg + image files.
target_n_rois = 8
roi_table = all_rois_df[["stub", "roi_id"]].dropna().copy()
roi_table["roi_id"] = roi_table["roi_id"].astype(int)
roi_table = roi_table.drop_duplicates().reset_index(drop=True)

valid_stubs = [s for s in roi_table["stub"].unique() if has_required_files(s)]
roi_table = roi_table[roi_table["stub"].isin(valid_stubs)].reset_index(drop=True)

if roi_table.empty:
    print("No ROIs available with matching segmentation and TIFF files")
else:
    rng = np.random.default_rng()  # unseeded: new random selection each run
    n_rois = min(target_n_rois, len(roi_table))
    sel_idx = rng.choice(len(roi_table), size=n_rois, replace=False)
    selected_rois = roi_table.iloc[sel_idx].reset_index(drop=True)
    cache = {}

    fig, axes = plt.subplots(
        n_rois,
        3,
        figsize=(18, 3.6 * n_rois),
        gridspec_kw={"width_ratios": [1.1, 1.5, 1.5]},
    )
    if n_rois == 1:
        axes = np.array([axes])

    for row_idx, (_, roi_row_sel) in enumerate(selected_rois.iterrows()):
        stub = roi_row_sel["stub"]
        roi_id = int(roi_row_sel["roi_id"])

        try:
            if stub not in cache:
                seg_path = resolve_stub_file(DATAFOLDER, stub, "_seg.npy")
                payload = np.load(seg_path, allow_pickle=True)
                payload = payload.item() if payload.ndim == 0 and payload.dtype == object else payload
                masks = payload["masks"] if isinstance(payload, dict) and "masks" in payload else payload

                tif_path = resolve_stub_file(images_path, stub, ".tif")
                with tifffile.TiffFile(tif_path) as tif:
                    metadata = tif.sem_metadata
                    if not metadata or "ap_image_pixel_size" not in metadata:
                        raise KeyError(f"Missing ap_image_pixel_size in sem_metadata for: {tif_path}")
                    pixel_size = float(metadata["ap_image_pixel_size"][1])

                raw = tifffile.imread(tif_path)
                image = raw[0] if raw.ndim > 2 else raw
                cache[stub] = {"masks": masks, "image": image, "pixel_size": pixel_size}

            masks = cache[stub]["masks"]
            image = cache[stub]["image"]
            pixel_size = cache[stub]["pixel_size"]

            roi_index = (all_rois_df["stub"] == stub) & (all_rois_df["roi_id"] == roi_id)
            if not roi_index.any():
                for col in range(3):
                    axes[row_idx, col].text(0.5, 0.5, f"ROI {roi_id} not found", ha="center", va="center")
                    axes[row_idx, col].axis("off")
                continue

            roi_row = all_rois_df[roi_index].iloc[0]
            roi_mask = masks == roi_id
            centroid_x = roi_row["centroid_x"]
            centroid_y = roi_row["centroid_y"]

            # ROI image panel
            ax_img = axes[row_idx, 0]
            ys, xs = np.nonzero(roi_mask)
            if ys.size > 0:
                margin = 12
                x0 = max(0, xs.min() - margin)
                x1 = min(image.shape[1], xs.max() + margin + 1)
                y0 = max(0, ys.min() - margin)
                y1 = min(image.shape[0], ys.max() + margin + 1)

                img_crop = image[y0:y1, x0:x1]
                mask_crop = roi_mask[y0:y1, x0:x1]

                ax_img.imshow(img_crop, cmap="gray")
                ax_img.imshow(np.ma.masked_where(~mask_crop, mask_crop), cmap="autumn", alpha=0.45)
                ax_img.set_title(f"ROI {roi_id} ({stub})", fontsize=9)
                ax_img.axis("off")
            else:
                ax_img.text(0.5, 0.5, f"ROI {roi_id} has no pixels", ha="center", va="center", fontsize=8)
                ax_img.axis("off")

            # Compute principal axis profiles
            num_samples = 256
            result = compute_profiles_from_principal_axes(
                roi_mask=roi_mask,
                image=image,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                pixel_size=pixel_size,
                num_samples=num_samples,
                extension_factor=0.50,
            )

            if result["success"]:
                profile_major = result["profile_major"]
                profile_minor = result["profile_minor"]
                major_angle = result["major_angle"]
                step_major = result["step_major"]
                step_minor = result["step_minor"]

                major_diameter_nm = estimate_diameter_from_profile(profile_major, pixel_size, sample_step_px=step_major)
                minor_diameter_nm = estimate_diameter_from_profile(profile_minor, pixel_size, sample_step_px=step_minor)
                major_fwhm_nm = estimate_fwhm_from_profile(profile_major, pixel_size, sample_step_px=step_major)
                minor_fwhm_nm = estimate_fwhm_from_profile(profile_minor, pixel_size, sample_step_px=step_minor)

                # Major axis intensity plot
                sample_index_major = np.arange(num_samples)
                center_major = sample_index_major[len(sample_index_major) // 2]
                x_major = (sample_index_major - center_major) * step_major * pixel_size

                ax_major = axes[row_idx, 1]
                ax_major.plot(x_major, profile_major, color="k", linewidth=1.5)
                ax_major.axvline(-major_diameter_nm / 2, color="g", linestyle=":", linewidth=2, alpha=0.7)
                ax_major.axvline(major_diameter_nm / 2, color="g", linestyle=":", linewidth=2, alpha=0.7)
                if np.isfinite(major_fwhm_nm):
                    ax_major.axvline(-major_fwhm_nm / 2, color="darkorange", linestyle="--", linewidth=1.8, alpha=0.9)
                    ax_major.axvline(major_fwhm_nm / 2, color="darkorange", linestyle="--", linewidth=1.8, alpha=0.9)
                ax_major.set_ylabel("Intensity", fontsize=9)
                ax_major.set_title(
                    f"Major (θ={major_angle:.0f}°, p2p={fmt_nm(major_diameter_nm)} nm, fwhm={fmt_nm(major_fwhm_nm)} nm)",
                    fontsize=8,
                )
                ax_major.grid(True, alpha=0.3)

                # Minor axis intensity plot
                sample_index_minor = np.arange(num_samples)
                center_minor = sample_index_minor[len(sample_index_minor) // 2]
                x_minor = (sample_index_minor - center_minor) * step_minor * pixel_size

                ax_minor = axes[row_idx, 2]
                ax_minor.plot(x_minor, profile_minor, color="b", linewidth=1.5)
                ax_minor.axvline(-minor_diameter_nm / 2, color="g", linestyle=":", linewidth=2, alpha=0.7)
                ax_minor.axvline(minor_diameter_nm / 2, color="g", linestyle=":", linewidth=2, alpha=0.7)
                if np.isfinite(minor_fwhm_nm):
                    ax_minor.axvline(-minor_fwhm_nm / 2, color="darkorange", linestyle="--", linewidth=1.8, alpha=0.9)
                    ax_minor.axvline(minor_fwhm_nm / 2, color="darkorange", linestyle="--", linewidth=1.8, alpha=0.9)
                ax_minor.set_ylabel("Intensity", fontsize=9)
                ax_minor.set_title(
                    f"Minor (θ={(major_angle + 90) % 180:.0f}°, p2p={fmt_nm(minor_diameter_nm)} nm, fwhm={fmt_nm(minor_fwhm_nm)} nm)",
                    fontsize=8,
                )
                ax_minor.grid(True, alpha=0.3)
            else:
                axes[row_idx, 1].text(0.5, 0.5, "Could not compute\nmajor profile", ha="center", va="center")
                axes[row_idx, 2].text(0.5, 0.5, "Could not compute\nminor profile", ha="center", va="center")

        except Exception as e:
            axes[row_idx, 0].text(0.5, 0.5, f"Error loading ROI\n{roi_id}", ha="center", va="center", fontsize=8)
            axes[row_idx, 1].text(0.5, 0.5, f"Error: {str(e)[:40]}", ha="center", va="center", fontsize=8)
            axes[row_idx, 2].text(0.5, 0.5, "Error loading\nminor plot", ha="center", va="center", fontsize=8)
            axes[row_idx, 0].axis("off")

    axes[-1, 1].set_xlabel("Distance (nm)", fontsize=9)
    axes[-1, 2].set_xlabel("Distance (nm)", fontsize=9)

    fig.suptitle(
        f"ROI image + principal-axis profiles for {n_rois} random ROIs",
        y=0.995,
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()


# %%
# Per-stub normalized major/minor axis profile averages + peak-to-peak and FWHM in nm
from src.roi_analysis import compute_centroids_df, compute_profiles_from_principal_axes
import re


def resolve_stub_file(base_path, stub, suffix):
    candidates = [base_path / f"{stub}{suffix}"]
    match = re.match(r"^(FAS|PR)(\d+)(_.*)$", stub)
    if match:
        candidates.append(base_path / f"{match.group(1)}_{match.group(2)}{match.group(3)}{suffix}")
    match = re.match(r"^(FAS|PR)_(\d+)(_.*)$", stub)
    if match:
        candidates.append(base_path / f"{match.group(1)}{match.group(2)}{match.group(3)}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve file for stub {stub} in {base_path}")


def load_stub_image_and_masks(stub):
    seg_path = resolve_stub_file(DATAFOLDER, stub, "_seg.npy")
    tif_path = resolve_stub_file(images_path, stub, ".tif")

    payload = np.load(seg_path, allow_pickle=True)
    payload = payload.item() if payload.ndim == 0 and payload.dtype == object else payload
    masks = payload["masks"] if isinstance(payload, dict) and "masks" in payload else payload

    with tifffile.TiffFile(tif_path) as tif:
        metadata = tif.sem_metadata
        if not metadata or "ap_image_pixel_size" not in metadata:
            raise KeyError(f"Missing ap_image_pixel_size in sem_metadata for: {tif_path}")
        pixel_size = float(metadata["ap_image_pixel_size"][1])

    raw = tifffile.imread(tif_path)
    image = raw[0] if raw.ndim > 2 else raw
    return image, masks, pixel_size


def estimate_peak_to_peak_samples(profile):
    if profile is None:
        return np.nan

    y = np.asarray(profile, dtype=float).ravel()
    if y.size < 5:
        return np.nan

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.nan

    mid = y.size // 2
    if mid < 1 or mid >= y.size:
        return np.nan

    left = y[:mid]
    right = y[mid:]
    if left.size == 0 or right.size == 0:
        return np.nan
    if np.all(np.isnan(left)) or np.all(np.isnan(right)):
        return np.nan

    left_idx = int(np.nanargmax(left))
    right_idx = int(np.nanargmax(right)) + mid
    width_samples = right_idx - left_idx
    return float(width_samples) if width_samples > 0 else np.nan


def estimate_fwhm_samples(profile):
    if profile is None:
        return np.nan

    y = np.asarray(profile, dtype=float).ravel()
    if y.size < 5:
        return np.nan

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.nan

    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return np.nan

    half = y_min + 0.5 * (y_max - y_min)
    above = y >= half
    idx = np.where(above)[0]
    if idx.size < 2:
        return np.nan

    left = int(idx[0])
    right = int(idx[-1])

    left_cross = float(left)
    if left > 0 and np.isfinite(y[left - 1]) and np.isfinite(y[left]) and y[left] != y[left - 1]:
        left_cross = (left - 1) + (half - y[left - 1]) / (y[left] - y[left - 1])

    right_cross = float(right)
    if right < y.size - 1 and np.isfinite(y[right]) and np.isfinite(y[right + 1]) and y[right + 1] != y[right]:
        right_cross = right + (half - y[right]) / (y[right + 1] - y[right])

    width_samples = right_cross - left_cross
    return float(width_samples) if width_samples > 0 and np.isfinite(width_samples) else np.nan


def collect_normalized_axis_profiles(stub, num_samples=256, extension_factor=0.5):
    image, masks, pixel_size = load_stub_image_and_masks(stub)
    centroids_df = compute_centroids_df(masks, pixel_size, stub=stub)

    if centroids_df.empty:
        return None

    major_profiles = []
    minor_profiles = []
    roi_ids_used = []
    major_step_nm_values = []
    minor_step_nm_values = []
    major_p2p_nm_values = []
    major_fwhm_nm_values = []
    minor_p2p_nm_values = []
    minor_fwhm_nm_values = []

    for _, roi_row in centroids_df.iterrows():
        roi_id = int(roi_row["roi_id"])
        roi_mask = masks == roi_id

        result = compute_profiles_from_principal_axes(
            roi_mask=roi_mask,
            image=image,
            centroid_x=float(roi_row["centroid_x"]),
            centroid_y=float(roi_row["centroid_y"]),
            pixel_size=pixel_size,
            num_samples=num_samples,
            extension_factor=extension_factor,
        )

        if not result["success"]:
            continue

        major_profile = np.asarray(result["profile_major"], dtype=float).ravel()
        minor_profile = np.asarray(result["profile_minor"], dtype=float).ravel()

        if major_profile.size != num_samples or minor_profile.size != num_samples:
            continue
        if not np.any(np.isfinite(major_profile)) or not np.any(np.isfinite(minor_profile)):
            continue

        major_step_nm = float(result["step_major"]) * float(pixel_size)
        minor_step_nm = float(result["step_minor"]) * float(pixel_size)
        if not np.isfinite(major_step_nm) or major_step_nm <= 0 or not np.isfinite(minor_step_nm) or minor_step_nm <= 0:
            continue

        major_p2p_samples = estimate_peak_to_peak_samples(major_profile)
        major_fwhm_samples = estimate_fwhm_samples(major_profile)
        minor_p2p_samples = estimate_peak_to_peak_samples(minor_profile)
        minor_fwhm_samples = estimate_fwhm_samples(minor_profile)

        major_profiles.append(major_profile)
        minor_profiles.append(minor_profile)
        roi_ids_used.append(roi_id)
        major_step_nm_values.append(major_step_nm)
        minor_step_nm_values.append(minor_step_nm)
        major_p2p_nm_values.append(major_p2p_samples * major_step_nm if np.isfinite(major_p2p_samples) else np.nan)
        major_fwhm_nm_values.append(major_fwhm_samples * major_step_nm if np.isfinite(major_fwhm_samples) else np.nan)
        minor_p2p_nm_values.append(minor_p2p_samples * minor_step_nm if np.isfinite(minor_p2p_samples) else np.nan)
        minor_fwhm_nm_values.append(minor_fwhm_samples * minor_step_nm if np.isfinite(minor_fwhm_samples) else np.nan)

    if not major_profiles:
        return None

    major_stack = np.vstack(major_profiles)
    minor_stack = np.vstack(minor_profiles)
    normalized_axis = np.linspace(-1.0, 1.0, major_stack.shape[1])

    major_mean = np.nanmean(major_stack, axis=0)
    major_var = np.nanvar(major_stack, axis=0)
    major_std = np.sqrt(major_var)

    minor_mean = np.nanmean(minor_stack, axis=0)
    minor_var = np.nanvar(minor_stack, axis=0)
    minor_std = np.sqrt(minor_var)

    major_step_nm_mean = float(np.nanmean(major_step_nm_values))
    minor_step_nm_mean = float(np.nanmean(minor_step_nm_values))

    major_p2p_nm_mean = float(np.nanmean(major_p2p_nm_values))
    major_p2p_nm_var = float(np.nanvar(major_p2p_nm_values))
    major_fwhm_nm_mean = float(np.nanmean(major_fwhm_nm_values))
    major_fwhm_nm_var = float(np.nanvar(major_fwhm_nm_values))

    minor_p2p_nm_mean = float(np.nanmean(minor_p2p_nm_values))
    minor_p2p_nm_var = float(np.nanvar(minor_p2p_nm_values))
    minor_fwhm_nm_mean = float(np.nanmean(minor_fwhm_nm_values))
    minor_fwhm_nm_var = float(np.nanvar(minor_fwhm_nm_values))

    return {
        "stub": stub,
        "pixel_size": pixel_size,
        "n_rois_total": int(len(centroids_df)),
        "n_rois_used": int(len(roi_ids_used)),
        "roi_ids_used": roi_ids_used,
        "normalized_axis": normalized_axis,
        "major_mean": major_mean,
        "major_var": major_var,
        "major_std": major_std,
        "major_stack": major_stack,
        "minor_mean": minor_mean,
        "minor_var": minor_var,
        "minor_std": minor_std,
        "minor_stack": minor_stack,
        "major_step_nm_mean": major_step_nm_mean,
        "minor_step_nm_mean": minor_step_nm_mean,
        "major_p2p_nm_mean": major_p2p_nm_mean,
        "major_p2p_nm_var": major_p2p_nm_var,
        "major_fwhm_nm_mean": major_fwhm_nm_mean,
        "major_fwhm_nm_var": major_fwhm_nm_var,
        "minor_p2p_nm_mean": minor_p2p_nm_mean,
        "minor_p2p_nm_var": minor_p2p_nm_var,
        "minor_fwhm_nm_mean": minor_fwhm_nm_mean,
        "minor_fwhm_nm_var": minor_fwhm_nm_var,
    }


stub_profile_results = {}
stub_profile_rows = []

for stub in sorted(all_rois_df["stub"].dropna().unique()):
    try:
        result = collect_normalized_axis_profiles(stub, num_samples=256, extension_factor=0.5)
    except Exception as exc:
        print(f"Skipping {stub}: {exc}")
        continue

    if result is None:
        print(f"Skipping {stub}: no valid ROI profiles")
        continue

    stub_profile_results[stub] = result
    stub_profile_rows.append(
        {
            "stub": stub,
            "pixel_size": result["pixel_size"],
            "n_rois_total": result["n_rois_total"],
            "n_rois_used": result["n_rois_used"],
            "major_mean_profile": result["major_mean"],
            "major_variance_profile": result["major_var"],
            "minor_mean_profile": result["minor_mean"],
            "minor_variance_profile": result["minor_var"],
            "major_step_nm_mean": result["major_step_nm_mean"],
            "minor_step_nm_mean": result["minor_step_nm_mean"],
            "major_p2p_nm_mean": result["major_p2p_nm_mean"],
            "major_p2p_nm_var": result["major_p2p_nm_var"],
            "major_fwhm_nm_mean": result["major_fwhm_nm_mean"],
            "major_fwhm_nm_var": result["major_fwhm_nm_var"],
            "minor_p2p_nm_mean": result["minor_p2p_nm_mean"],
            "minor_p2p_nm_var": result["minor_p2p_nm_var"],
            "minor_fwhm_nm_mean": result["minor_fwhm_nm_mean"],
            "minor_fwhm_nm_var": result["minor_fwhm_nm_var"],
        }
    )

stub_profile_summary_df = pd.DataFrame(stub_profile_rows).sort_values("stub").reset_index(drop=True)

print(f"Built normalized profile averages for {len(stub_profile_summary_df)} stubs")
display_cols = [
    "stub",
    "n_rois_total",
    "n_rois_used",
    "major_p2p_nm_mean",
    "major_fwhm_nm_mean",
    "minor_p2p_nm_mean",
    "minor_fwhm_nm_mean",
]
stub_profile_summary_df[display_cols].head(10)

plot_stubs = list(stub_profile_results.keys())[:min(4, len(stub_profile_results))]
if plot_stubs:
    fig, axes = plt.subplots(len(plot_stubs), 2, figsize=(14, 3.8 * len(plot_stubs)), sharex=True)
    if len(plot_stubs) == 1:
        axes = np.array([axes])

    for row_idx, stub in enumerate(plot_stubs):
        result = stub_profile_results[stub]
        x = result["normalized_axis"]
        major_mean = result["major_mean"]
        major_std = result["major_std"]
        minor_mean = result["minor_mean"]
        minor_std = result["minor_std"]

        ax_major = axes[row_idx, 0]
        ax_minor = axes[row_idx, 1]

        ax_major.plot(x, major_mean, color="tab:blue", linewidth=2, label="mean")
        ax_major.fill_between(x, major_mean - major_std, major_mean + major_std, color="tab:blue", alpha=0.2, label="mean +/- 1 std")
        ax_major.set_title(
            f"{stub} major axis | p2p={result['major_p2p_nm_mean']:.1f} nm | FWHM={result['major_fwhm_nm_mean']:.1f} nm",
            fontsize=9,
        )
        ax_major.set_ylabel("Intensity")
        ax_major.grid(True, alpha=0.25)

        ax_minor.plot(x, minor_mean, color="tab:green", linewidth=2, label="mean")
        ax_minor.fill_between(x, minor_mean - minor_std, minor_mean + minor_std, color="tab:green", alpha=0.2, label="mean +/- 1 std")
        ax_minor.set_title(
            f"{stub} minor axis | p2p={result['minor_p2p_nm_mean']:.1f} nm | FWHM={result['minor_fwhm_nm_mean']:.1f} nm",
            fontsize=9,
        )
        ax_minor.set_ylabel("Intensity")
        ax_minor.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("Normalized axis position")
    axes[-1, 1].set_xlabel("Normalized axis position")
    axes[0, 0].legend(frameon=False, loc="best")
    axes[0, 1].legend(frameon=False, loc="best")
    fig.suptitle("Per-stub normalized major/minor profiles with peak-to-peak and FWHM metrics", y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print("No stubs produced valid normalized profiles.")

# %%
# Replot normalized profiles with visual p2p, FWHM, and robust steepest-boundary marker lines
if "stub_profile_results" not in globals() or not stub_profile_results:
    raise ValueError("Run the normalized profile metrics cell first.")


def width_nm_to_half_norm(width_nm, step_nm, n_samples):
    if not np.isfinite(width_nm) or not np.isfinite(step_nm) or step_nm <= 0 or n_samples < 2:
        return np.nan
    width_samples = width_nm / step_nm
    return width_samples / (n_samples - 1)


def moving_average_1d(y, window=9):
    if window <= 1:
        return y.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_pad, kernel, mode="valid")


def _crossing_indices(y, half_level, center_idx):
    left_cross = np.nan
    right_cross = np.nan

    if center_idx > 0:
        for i in range(center_idx - 1, -1, -1):
            y0 = y[i]
            y1 = y[i + 1]
            if (y0 - half_level) * (y1 - half_level) <= 0 and y1 != y0:
                left_cross = i + (half_level - y0) / (y1 - y0)
                break

    if center_idx < y.size - 1:
        for i in range(center_idx, y.size - 1):
            y0 = y[i]
            y1 = y[i + 1]
            if (y0 - half_level) * (y1 - half_level) <= 0 and y1 != y0:
                right_cross = i + (half_level - y0) / (y1 - y0)
                break

    return left_cross, right_cross


def estimate_steepest_boundary_samples(profile):
    if profile is None:
        return np.nan, np.nan, np.nan

    y = np.asarray(profile, dtype=float).ravel()
    if y.size < 9:
        return np.nan, np.nan, np.nan

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.nan, np.nan, np.nan

    y_filled = y.copy()
    if not np.all(finite):
        idx = np.arange(y.size)
        y_filled[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    # Smooth first, then use derivative extrema as steepest edges.
    y_smooth = moving_average_1d(y_filled, window=11)
    grad = np.gradient(y_smooth)

    center_float = 0.5 * (y_smooth.size - 1)
    n = y_smooth.size
    edge_pad = max(2, int(0.04 * (n - 1)))
    center_pad = max(4, int(0.10 * (n - 1)))

    left_start = edge_pad
    left_stop = int(np.floor(center_float - center_pad)) + 1
    right_start = int(np.ceil(center_float + center_pad))
    right_stop = n - edge_pad

    if left_stop <= left_start or right_stop <= right_start:
        return np.nan, np.nan, np.nan

    left_candidates = np.arange(left_start, left_stop)
    right_candidates = np.arange(right_start, right_stop)
    if left_candidates.size == 0 or right_candidates.size == 0:
        return np.nan, np.nan, np.nan

    left_local = grad[left_candidates]
    right_local = grad[right_candidates]

    # Derivative-based steepest edges: most negative on left, most positive on right.
    left_idx = int(left_candidates[int(np.argmin(left_local))])
    right_idx = int(right_candidates[int(np.argmax(right_local))])

    width_samples = right_idx - left_idx
    if width_samples <= 0:
        return np.nan, np.nan, np.nan

    left_offset = left_idx - center_float
    right_offset = right_idx - center_float

    return float(width_samples), float(left_offset), float(right_offset)


all_plot_stubs = list(stub_profile_results.keys())
n_plot_stubs = min(4, len(all_plot_stubs))
rng_plot = np.random.default_rng()
plot_stubs = list(rng_plot.choice(all_plot_stubs, size=n_plot_stubs, replace=False)) if n_plot_stubs > 0 else []
fig, axes = plt.subplots(len(plot_stubs), 2, figsize=(14, 4.0 * len(plot_stubs)), sharex=True)
if len(plot_stubs) == 1:
    axes = np.array([axes])

for row_idx, stub in enumerate(plot_stubs):
    result = stub_profile_results[stub]
    x = result["normalized_axis"]
    n_samples = len(x)

    major_half_p2p = width_nm_to_half_norm(result["major_p2p_nm_mean"], result["major_step_nm_mean"], n_samples)
    major_half_fwhm = width_nm_to_half_norm(result["major_fwhm_nm_mean"], result["major_step_nm_mean"], n_samples)

    minor_half_p2p = width_nm_to_half_norm(result["minor_p2p_nm_mean"], result["minor_step_nm_mean"], n_samples)
    minor_half_fwhm = width_nm_to_half_norm(result["minor_fwhm_nm_mean"], result["minor_step_nm_mean"], n_samples)

    major_steep_width_samples, major_left_off, major_right_off = estimate_steepest_boundary_samples(result["major_mean"])
    minor_steep_width_samples, minor_left_off, minor_right_off = estimate_steepest_boundary_samples(result["minor_mean"])

    major_steep_nm = major_steep_width_samples * result["major_step_nm_mean"] if np.isfinite(major_steep_width_samples) else np.nan
    minor_steep_nm = minor_steep_width_samples * result["minor_step_nm_mean"] if np.isfinite(minor_steep_width_samples) else np.nan

    major_left_norm = (2.0 * major_left_off) / (n_samples - 1) if np.isfinite(major_left_off) else np.nan
    major_right_norm = (2.0 * major_right_off) / (n_samples - 1) if np.isfinite(major_right_off) else np.nan
    minor_left_norm = (2.0 * minor_left_off) / (n_samples - 1) if np.isfinite(minor_left_off) else np.nan
    minor_right_norm = (2.0 * minor_right_off) / (n_samples - 1) if np.isfinite(minor_right_off) else np.nan

    ax_major = axes[row_idx, 0]
    ax_minor = axes[row_idx, 1]

    # Major axis plot
    ax_major.plot(x, result["major_mean"], color="tab:blue", linewidth=2, label="mean")
    ax_major.fill_between(
        x,
        result["major_mean"] - result["major_std"],
        result["major_mean"] + result["major_std"],
        color="tab:blue",
        alpha=0.2,
        label="mean +/- 1 std",
    )

    if np.isfinite(major_half_p2p):
        ax_major.axvline(-major_half_p2p, color="forestgreen", linestyle=":", linewidth=2, alpha=0.9, label="p2p")
        ax_major.axvline(major_half_p2p, color="forestgreen", linestyle=":", linewidth=2, alpha=0.9)
    if np.isfinite(major_half_fwhm):
        ax_major.axvline(-major_half_fwhm, color="darkorange", linestyle="--", linewidth=2, alpha=0.9, label="FWHM")
        ax_major.axvline(major_half_fwhm, color="darkorange", linestyle="--", linewidth=2, alpha=0.9)
    if np.isfinite(major_left_norm) and np.isfinite(major_right_norm):
        ax_major.axvline(major_left_norm, color="crimson", linestyle="-.", linewidth=2, alpha=0.95, label="derivative")
        ax_major.axvline(major_right_norm, color="crimson", linestyle="-.", linewidth=2, alpha=0.95)

    ax_major.set_title(
        (
            f"{stub} major | p2p={result['major_p2p_nm_mean']:.1f} nm | "
            f"FWHM={result['major_fwhm_nm_mean']:.1f} nm | derivative={major_steep_nm:.1f} nm"
        ),
        fontsize=9,
    )
    ax_major.set_ylabel("Intensity")
    ax_major.grid(True, alpha=0.25)

    # Minor axis plot
    ax_minor.plot(x, result["minor_mean"], color="tab:green", linewidth=2, label="mean")
    ax_minor.fill_between(
        x,
        result["minor_mean"] - result["minor_std"],
        result["minor_mean"] + result["minor_std"],
        color="tab:green",
        alpha=0.2,
        label="mean +/- 1 std",
    )

    if np.isfinite(minor_half_p2p):
        ax_minor.axvline(-minor_half_p2p, color="forestgreen", linestyle=":", linewidth=2, alpha=0.9, label="p2p")
        ax_minor.axvline(minor_half_p2p, color="forestgreen", linestyle=":", linewidth=2, alpha=0.9)
    if np.isfinite(minor_half_fwhm):
        ax_minor.axvline(-minor_half_fwhm, color="darkorange", linestyle="--", linewidth=2, alpha=0.9, label="FWHM")
        ax_minor.axvline(minor_half_fwhm, color="darkorange", linestyle="--", linewidth=2, alpha=0.9)
    if np.isfinite(minor_left_norm) and np.isfinite(minor_right_norm):
        ax_minor.axvline(minor_left_norm, color="crimson", linestyle="-.", linewidth=2, alpha=0.95, label="derivative")
        ax_minor.axvline(minor_right_norm, color="crimson", linestyle="-.", linewidth=2, alpha=0.95)

    ax_minor.set_title(
        (
            f"{stub} minor | p2p={result['minor_p2p_nm_mean']:.1f} nm | "
            f"FWHM={result['minor_fwhm_nm_mean']:.1f} nm | derivative={minor_steep_nm:.1f} nm"
        ),
        fontsize=9,
    )
    ax_minor.set_ylabel("Intensity")
    ax_minor.grid(True, alpha=0.25)

axes[-1, 0].set_xlabel("Normalized axis position")
axes[-1, 1].set_xlabel("Normalized axis position")
axes[0, 0].legend(frameon=False, loc="best")
axes[0, 1].legend(frameon=False, loc="best")
fig.suptitle("Normalized profiles with p2p (green), FWHM (orange), and steepest-edge (red) markers", y=0.995)
plt.tight_layout()
plt.show()

# %%
# Save per-stub major/minor p2p, FWHM, and derivative sizes to Excel (with ID)
if "stub_profile_results" not in globals() or not stub_profile_results:
    raise ValueError("Run Cell 12 first so stub_profile_results is available.")

def _moving_average_1d(y, window=11):
    if window <= 1:
        return y.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y_pad, kernel, mode="valid")

def _derivative_width_nm(profile, step_nm):
    if profile is None or not np.isfinite(step_nm) or step_nm <= 0:
        return np.nan
    y = np.asarray(profile, dtype=float).ravel()
    if y.size < 9:
        return np.nan

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.nan

    y_filled = y.copy()
    if not np.all(finite):
        idx = np.arange(y.size)
        y_filled[~finite] = np.interp(idx[~finite], idx[finite], y[finite])

    y_smooth = _moving_average_1d(y_filled, window=11)
    grad = np.gradient(y_smooth)

    n = y_smooth.size
    center_float = 0.5 * (n - 1)
    edge_pad = max(2, int(0.04 * (n - 1)))
    center_pad = max(4, int(0.10 * (n - 1)))

    left_start = edge_pad
    left_stop = int(np.floor(center_float - center_pad)) + 1
    right_start = int(np.ceil(center_float + center_pad))
    right_stop = n - edge_pad

    if left_stop <= left_start or right_stop <= right_start:
        return np.nan

    left_candidates = np.arange(left_start, left_stop)
    right_candidates = np.arange(right_start, right_stop)
    if left_candidates.size == 0 or right_candidates.size == 0:
        return np.nan

    left_idx = int(left_candidates[int(np.argmin(grad[left_candidates]))])
    right_idx = int(right_candidates[int(np.argmax(grad[right_candidates]))])

    width_samples = right_idx - left_idx
    if width_samples <= 0:
        return np.nan

    return float(width_samples) * float(step_nm)

def _build_stub_id_lookup():
    # Prefer summary tables, then fall back to ROI table if needed.
    candidates = []
    if "all_summary_df" in globals() and isinstance(all_summary_df, pd.DataFrame):
        candidates.append(all_summary_df)
    if "summary_df" in globals() and isinstance(summary_df, pd.DataFrame):
        candidates.append(summary_df)
    if "all_rois_df" in globals() and isinstance(all_rois_df, pd.DataFrame):
        candidates.append(all_rois_df)
    if "rois_df" in globals() and isinstance(rois_df, pd.DataFrame):
        candidates.append(rois_df)

    for df in candidates:
        if {"stub", "id"}.issubset(df.columns):
            lookup = df[["stub", "id"]].dropna(subset=["stub"]).drop_duplicates(subset=["stub"])
            if not lookup.empty:
                return dict(zip(lookup["stub"], lookup["id"]))

    return {}

stub_to_id = _build_stub_id_lookup()

rows = []
for stub, result in sorted(stub_profile_results.items()):
    major_derivative_nm = _derivative_width_nm(result.get("major_mean"), result.get("major_step_nm_mean", np.nan))
    minor_derivative_nm = _derivative_width_nm(result.get("minor_mean"), result.get("minor_step_nm_mean", np.nan))

    rows.append(
        {
            "id": stub_to_id.get(stub, np.nan),
            "stub": stub,
            "major_p2p_nm": result.get("major_p2p_nm_mean", np.nan),
            "major_fwhm_nm": result.get("major_fwhm_nm_mean", np.nan),
            "major_derivative_nm": major_derivative_nm,
            "minor_p2p_nm": result.get("minor_p2p_nm_mean", np.nan),
            "minor_fwhm_nm": result.get("minor_fwhm_nm_mean", np.nan),
            "minor_derivative_nm": minor_derivative_nm,
        }
    )

stub_axis_metrics_df = pd.DataFrame(rows).sort_values(["id", "stub"], na_position="last").reset_index(drop=True)
excel_path = ROOT / "stub_profile_metrics_major_minor.xlsx"
stub_axis_metrics_df.to_excel(excel_path, index=False)

print(f"Saved Excel file: {excel_path}")
stub_axis_metrics_df.head(10)

# %%
# Debug: Check which ROIs are being selected and if their files exist
available_stubs_debug = all_rois_df["stub"].dropna().unique()[:3]
print(f"First 3 available stubs: {available_stubs_debug}")

selected_rois_debug = []
for stub in available_stubs_debug:
    stub_rois = all_rois_df[all_rois_df["stub"] == stub][["stub", "roi_id"]].dropna().copy()
    stub_rois["roi_id"] = stub_rois["roi_id"].astype(int)
    stub_rois = stub_rois.drop_duplicates()
    n_to_take = min(4, len(stub_rois))
    print(f"Stub {stub}: {len(stub_rois)} ROIs available, taking {n_to_take}")
    selected_rois_debug.append(stub_rois.iloc[:n_to_take])

selected_rois_debug = pd.concat(selected_rois_debug, ignore_index=True)[:10]
print(f"\nSelected {len(selected_rois_debug)} ROIs:")
print(selected_rois_debug)

# Check which files exist
print("\nFile existence check:")
for _, roi_row in selected_rois_debug.iterrows():
    stub = roi_row["stub"]
    seg_file = DATAFOLDER / f"{stub}_seg.npy"
    seg_file_alt = DATAFOLDER / f"{stub[:-6]}_{stub[-5:].replace('_', '')}_seg.npy"
    exists = seg_file.exists() or seg_file_alt.exists()
    print(f"Stub {stub}: {exists}")


# %%
# Check what files actually exist
seg_files = sorted(DATAFOLDER.glob("*_seg.npy"))[:15]
print("First 15 segmentation files in flatten_npy:")
for f in seg_files:
    print(f.name)


# %%
# Test the resolve function
test_stub = "FAS15_ME_04"
try:
    result = resolve_stub_file(DATAFOLDER, test_stub, "_seg.npy")
    print(f"Successfully resolved {test_stub} to {result.name}")
except FileNotFoundError as e:
    print(f"Failed to resolve: {e}")


# %%
# Visualize one ROI and its nearest neighbors over the original image
DATAFOLDER = FSPath("C:/Users/sthui4072/Github/fenestrations/flatten_npy")
viz_stub = "FAS_15_ME_04"  # change this to any stub

seg_path = DATAFOLDER / f"{viz_stub}_seg.npy"
if not seg_path.exists():
    raise FileNotFoundError(f"Missing segmentation file: {seg_path}")

payload = np.load(seg_path, allow_pickle=True)
payload = payload.item() if payload.ndim == 0 and payload.dtype == object else payload
masks = payload["masks"] if isinstance(payload, dict) and "masks" in payload else payload

tif_path = images_path / f"{viz_stub}.tif"
if not tif_path.exists():
    raise FileNotFoundError(f"Missing image file: {tif_path}")
with tifffile.TiffFile(tif_path) as tif:
    metadata = tif.sem_metadata
    if not metadata or "ap_image_pixel_size" not in metadata:
        raise KeyError(f"Missing ap_image_pixel_size in sem_metadata for: {tif_path}")
    pixel_size = float(metadata["ap_image_pixel_size"][1])
raw = tifffile.imread(tif_path)
image = raw[0] if raw.ndim > 2 else raw

centroids_df = compute_centroids_df(masks, pixel_size, stub=viz_stub)
centroids_df, dists = compute_neighbor_stats(centroids_df, max_k=10)

if centroids_df.empty:
    raise ValueError(f"No ROIs found for stub {viz_stub}")
roi_index_choice = min(350, len(centroids_df) - 1)
roi_id = int(centroids_df["roi_id"].iloc[roi_index_choice])  # change this to any ROI id

coords = centroids_df[["centroid_x", "centroid_y"]].to_numpy()
roi_ids_all = centroids_df["roi_id"].to_numpy()

roi_index = np.where(roi_ids_all == roi_id)[0]
if roi_index.size == 0:
    raise ValueError(f"ROI id {roi_id} not found")
roi_index = roi_index[0]

dist_row = dists[roi_index]

# Get the nearest neighbors for this ROI
order = np.argsort(dist_row)
order = order[order != roi_index]

n = int(centroids_df.loc[roi_index, "neighbor_count"])
neighbor_indices = order[:n]

# Build a label mask for display
mask_main = (masks == roi_id)
mask_neighbors = np.isin(masks, roi_ids_all[neighbor_indices])

# Mask background so only ROI pixels are drawn
neighbors_masked = np.ma.masked_where(~mask_neighbors, mask_neighbors)
main_masked = np.ma.masked_where(~mask_main, mask_main)

plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")
plt.imshow(neighbors_masked, cmap="Blues", alpha=0.9)
plt.imshow(main_masked, cmap="Reds", alpha=0.6)
plt.title(f"ROI {roi_id} with {n} neighbors")
plt.axis("off")
plt.show()

# %%
# Load ROI data
data_path = Path("C:/Users/sthui4072/Github/fenestrations/roi_data.pickle")
images_path = Path("C:/Users/sthui4072/Github/fenestrations/images")
metafile_path = Path("C:/Users/sthui4072/Github/fenestrations/fenestrations_metafile.xlsx")

# data_path = Path("../data/test.pickle")
data = pd.read_pickle(data_path)

# Extract the DataFrames
rois_df = data['rois']
summary_df = data['summary']

# load metafile
metadata = pd.read_excel(metafile_path)

# %%
rois_df.head(  )


# %%
# Quick QC: side-by-side cluster views for one stub
from src.roi_analysis import compute_polygon_cluster_stats, plot_cluster_centroids, plot_cluster_polygons

qc_stub = "FAS24_ME_11"  # change stub as needed
qc_df = rois_df[rois_df["stub"] == qc_stub].copy()
print(len(qc_df), "ROIs found for stub", qc_stub)

if qc_df.empty:
    raise ValueError(f"No rows found for stub {qc_stub}")

if "cluster_is_valid" not in qc_df.columns:
    qc_df = compute_polygon_cluster_stats(qc_df, max_neighbors=10)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
plot_cluster_centroids(qc_df, ax=axes[0])
plot_cluster_polygons(qc_df, ax=axes[1])
axes[0].set_title(f"{qc_stub}: centroid clusters")
axes[1].set_title(f"{qc_stub}: centroid clusters + polygons")
ymin, ymax = axes[0].get_ylim()
for axis in axes:
    axis.set_ylim([ymax, ymin])
plt.tight_layout()
plt.show()

# %%
# Total area of any image without legend
(0.690*3.692) * (1.080*3.692)

# %%
summary_df.head()

# %%
import ast
import numpy as np
from matplotlib.path import Path as MplPath

# Build a union mask of all valid cluster polygons (no double counting).
# raw = tifffile.imread(images_path / f"{qc_stub}.tif")
# image = raw[0] if raw.ndim > 2 else raw
# image_h, image_w = image.shape[:2]

image_h, image_w = 768, 1024

id_to_xy = {
    int(row.roi_id): (float(row.centroid_x), float(row.centroid_y))
    for _, row in qc_df.iterrows()
}

union_mask = np.zeros((image_h, image_w), dtype=bool)
valid_rows = qc_df[qc_df["cluster_is_valid"].fillna(False)]

for _, row in valid_rows.iterrows():
    ids_val = row["cluster_neighbor_ids"]
    if isinstance(ids_val, str):
        try:
            ids_val = ast.literal_eval(ids_val)
        except (ValueError, SyntaxError):
            continue

    if not isinstance(ids_val, (list, tuple)) or len(ids_val) < 3:
        continue

    poly = np.array([id_to_xy[int(i)] for i in ids_val if int(i) in id_to_xy], dtype=float)
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

covered_px = int(union_mask.sum())
total_px = int(union_mask.size)
coverage_fraction = covered_px / total_px if total_px else np.nan
coverage_percent = 100.0 * coverage_fraction

pixel_size = float(qc_df["pixel_size"].iloc[0]) if not qc_df.empty else np.nan
covered_area_phys2 = covered_px * (pixel_size ** 2) if np.isfinite(pixel_size) else np.nan

print(f"Stub: {qc_stub}")
print(f"Covered pixels (union): {covered_px}")
print(f"Total pixels: {total_px}")
print(f"Coverage fraction: {coverage_fraction:.6f}")
print(f"Coverage percent: {coverage_percent:.3f}%")
print(f"Covered area (physical^2): {covered_area_phys2:.6f}")

# %%
# Build union masks for all stubs and save to disk.
import ast
from matplotlib.path import Path as MplPath

masks_outdir = ROOT / "union_masks"
masks_outdir.mkdir(parents=True, exist_ok=True)

source_df = all_rois_df.copy() if "all_rois_df" in globals() and not all_rois_df.empty else rois_df.copy()
if source_df.empty:
    raise ValueError("No ROI rows available. Run data loading first.")

if "cluster_is_valid" not in source_df.columns or "cluster_neighbor_ids" not in source_df.columns:
    from src.roi_analysis import compute_polygon_cluster_stats

    out_parts = []
    for stub, stub_df in source_df.groupby("stub", sort=True):
        out_parts.append(compute_polygon_cluster_stats(stub_df.copy(), max_neighbors=10))
    source_df = pd.concat(out_parts, ignore_index=True)

stub_to_tif = {get_stub(p.name): p for p in sorted(images_path.glob("*.tif"))}

summary_rows = []
for stub, stub_df in source_df.groupby("stub", sort=True):
    tif_path = stub_to_tif.get(stub)
    if tif_path is None:
        print(f"Skipping {stub}: TIFF not found in {images_path}")
        continue

    raw = tifffile.imread(tif_path)
    image = raw[0] if raw.ndim > 2 else raw
    image_h, image_w = image.shape[:2]

    id_to_xy = {
        int(row.roi_id): (float(row.centroid_x), float(row.centroid_y))
        for _, row in stub_df.iterrows()
    }

    union_mask = np.zeros((image_h, image_w), dtype=bool)
    valid_rows = stub_df[stub_df["cluster_is_valid"].fillna(False)]

    for _, row in valid_rows.iterrows():
        ids_val = row["cluster_neighbor_ids"]
        if isinstance(ids_val, str):
            try:
                ids_val = ast.literal_eval(ids_val)
            except (ValueError, SyntaxError):
                continue

        if not isinstance(ids_val, (list, tuple)) or len(ids_val) < 3:
            continue

        poly = np.array([id_to_xy[int(i)] for i in ids_val if int(i) in id_to_xy], dtype=float)
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

    covered_px = int(union_mask.sum())
    total_px = int(union_mask.size)
    coverage_fraction = covered_px / total_px if total_px else np.nan
    pixel_size_nm_per_px = float(stub_df["pixel_size"].iloc[0]) if not stub_df.empty else np.nan

    covered_area_nm2 = covered_px * (pixel_size_nm_per_px ** 2) if np.isfinite(pixel_size_nm_per_px) else np.nan
    covered_area_um2 = covered_area_nm2 / 1_000_000.0 if np.isfinite(covered_area_nm2) else np.nan

    np.save(masks_outdir / f"{stub}_union_mask.npy", union_mask)

    summary_rows.append(
        {
            "stub": stub,
            "mask_path": str(masks_outdir / f"{stub}_union_mask.npy"),
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

union_summary_df = pd.DataFrame(summary_rows).sort_values("coverage_fraction", ascending=False)
union_summary_path = masks_outdir / "union_mask_summary.csv"
union_summary_df.to_csv(union_summary_path, index=False)

print(f"Saved {len(union_summary_df)} union masks to: {masks_outdir}")
print(f"Summary CSV: {union_summary_path}")
union_summary_df.head(10)

# %%
# ROI density per stub: n_rois / covered_area_nm2 (and per µm²)
roi_counts_df = (
    all_rois_df.groupby("stub", as_index=False)
    .agg(n_rois=("roi_id", "nunique"))
)

roi_density_df = roi_counts_df.merge(
    union_summary_df[["stub", "covered_area_nm2", "covered_area_um2", "coverage_fraction", "coverage_percent"]],
    on="stub",
    how="inner",
)

roi_density_df["rois_per_nm2"] = roi_density_df["n_rois"] / roi_density_df["covered_area_nm2"]
roi_density_df["rois_per_um2"] = roi_density_df["n_rois"] / roi_density_df["covered_area_um2"]

roi_density_df = roi_density_df.sort_values("rois_per_um2", ascending=False)

density_outfile = masks_outdir / "roi_density_per_stub.csv"
roi_density_df.to_csv(density_outfile, index=False)

print(f"Saved ROI density table to: {density_outfile}")
roi_density_df.head(20)

# %%
# Total fenestration area per stub using mean_roi_area_nm2 * n_rois
source_area_df = all_rois_df.copy() if "all_rois_df" in globals() and not all_rois_df.empty else rois_df.copy()

fenestration_area_df = (
    source_area_df.groupby("stub", as_index=False)
    .agg(
        n_rois=("roi_id", "nunique"),
        mean_roi_area_nm2=("area", "mean"),
        summed_roi_area_nm2=("area", "sum"),
    )
)

fenestration_area_df["total_fenestration_area_nm2"] = (
    fenestration_area_df["mean_roi_area_nm2"] * fenestration_area_df["n_rois"]
)
fenestration_area_df["total_fenestration_area_um2"] = (
    fenestration_area_df["total_fenestration_area_nm2"] / 1_000_000.0
)
fenestration_area_df["summed_roi_area_um2"] = fenestration_area_df["summed_roi_area_nm2"] / 1_000_000.0

fenestration_area_df = fenestration_area_df.sort_values("total_fenestration_area_nm2", ascending=False)

fenestration_area_outfile = masks_outdir / "fenestration_area_per_stub.csv"
fenestration_area_df.to_csv(fenestration_area_outfile, index=False)

print(f"Saved fenestration area table to: {fenestration_area_outfile}")
fenestration_area_df.head(20)

# %%
# Porosity per stub: total fenestration area / covered area (unitless)
from datetime import datetime

porosity_df = fenestration_area_df[["stub", "total_fenestration_area_nm2", "total_fenestration_area_um2"]].merge(
    union_summary_df[["stub", "covered_area_nm2", "covered_area_um2"]],
    on="stub",
    how="inner",
)

porosity_df["porosity_fraction"] = porosity_df["total_fenestration_area_nm2"] / porosity_df["covered_area_nm2"]
porosity_df["porosity_percent"] = 100.0 * porosity_df["porosity_fraction"]

porosity_df = porosity_df.sort_values("porosity_fraction", ascending=False)

porosity_outfile = masks_outdir / "porosity_per_stub.csv"
try:
    porosity_df.to_csv(porosity_outfile, index=False)
except PermissionError:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    porosity_outfile = masks_outdir / f"porosity_per_stub_{ts}.csv"
    porosity_df.to_csv(porosity_outfile, index=False)

print(f"Saved porosity table to: {porosity_outfile}")
porosity_df.head(20)

# %%
# Consolidated per-stub metrics table
stub_metrics_df = (
    union_summary_df[[
        "stub",
        "pixel_size_nm_per_px",
        "covered_px",
        "total_px",
        "coverage_fraction",
        "coverage_percent",
        "covered_area_nm2",
        "covered_area_um2",
    ]]
    .merge(
        roi_density_df[["stub", "n_rois", "rois_per_nm2", "rois_per_um2"]],
        on="stub",
        how="inner",
    )
    .merge(
        fenestration_area_df[[
            "stub",
            "mean_roi_area_nm2",
            "summed_roi_area_nm2",
            "summed_roi_area_um2",
            "total_fenestration_area_nm2",
            "total_fenestration_area_um2",
        ]],
        on="stub",
        how="inner",
    )
    .merge(
        porosity_df[["stub", "porosity_fraction", "porosity_percent"]],
        on="stub",
        how="inner",
    )
    .sort_values("porosity_fraction", ascending=False)
)

stub_metrics_outfile = masks_outdir / "stub_metrics_all_in_one.csv"
stub_metrics_df.to_csv(stub_metrics_outfile, index=False)

print(f"Saved consolidated metrics to: {stub_metrics_outfile}")
print(f"Rows: {len(stub_metrics_df)}")
stub_metrics_df.head(20)

# %%
# Add consolidated stub metrics as columns in summary_df (preserve summary_df row order)
if "stub" not in summary_df.columns:
    raise KeyError("summary_df must contain a 'stub' column")

summary_df = summary_df.copy()
metrics_lookup = stub_metrics_df.set_index("stub")

# Add all metric columns except the key.
metric_cols = [c for c in stub_metrics_df.columns if c != "stub"]
for col in metric_cols:
    summary_df[col] = summary_df["stub"].map(metrics_lookup[col])

summary_with_all_metrics_outfile = masks_outdir / "summary_df_with_all_stub_metrics.csv"
summary_df.to_csv(summary_with_all_metrics_outfile, index=False)

print(f"Updated summary_df with {len(metric_cols)} metric columns")
print(f"Saved to: {summary_with_all_metrics_outfile}")
summary_df[["stub", "rois_per_um2", "porosity_percent", "covered_area_um2"]].head(20)

# %%
# Unit sanity checks
# 1) covered_area_phys2 should equal covered_area_nm2 (legacy alias)
max_rel_diff_cover = np.nanmax(
    np.abs(union_summary_df["covered_area_phys2"] - union_summary_df["covered_area_nm2"])
    / np.maximum(np.abs(union_summary_df["covered_area_nm2"]), 1e-12)
)

# 2) mean*n should match summed ROI area (same unit: nm^2)
fenestration_area_df["rel_diff_mean_vs_sum"] = np.abs(
    fenestration_area_df["total_fenestration_area_nm2"] - fenestration_area_df["summed_roi_area_nm2"]
) / np.maximum(np.abs(fenestration_area_df["summed_roi_area_nm2"]), 1e-12)

print(f"Max relative diff covered_area_phys2 vs covered_area_nm2: {max_rel_diff_cover:.3e}")
print(
    "Max relative diff total_fenestration_area_nm2 (mean*n) vs summed_roi_area_nm2: "
    f"{fenestration_area_df['rel_diff_mean_vs_sum'].max():.3e}"
)

fenestration_area_df[[
    "stub",
    "n_rois",
    "mean_roi_area_nm2",
    "summed_roi_area_nm2",
    "total_fenestration_area_nm2",
    "rel_diff_mean_vs_sum",
]].head(10)

# %%
summary_df.columns


# %%
summary_control = summary_df.query("condition == 'control'")
summary_fasted = summary_df.query("condition == 'fasted'")
summary_restricted = summary_df.query("condition == 'restricted'")

# %%
# Average metrics per animal
summary_per_animal = (
    summary_df.groupby(["id", "condition"], as_index=False)
    .agg(
        porosity_percent=("porosity_percent", "mean"),
        rois_per_um2=("rois_per_um2", "mean"),
    )
)

summary_per_animal

# %%
# Density and porosity control 
summary_control = summary_per_animal.query("condition == 'control'")
summary_control_mean = summary_control[["porosity_percent", "rois_per_um2"]].mean()
summary_control_mean

# %%
control_rows_summary_per_animal = int((summary_per_animal["condition"] == "control").sum())
control_rows_summary_df = int((summary_df["condition"] == "control").sum())
print("control rows in summary_per_animal:", control_rows_summary_per_animal)
print("control rows in summary_df:", control_rows_summary_df)

# %%
# QC plot: union mask overlay + ROI centroids
fig, ax = plt.subplots(figsize=(7, 7))

h, w = union_mask.shape
mask_overlay = np.ma.masked_where(~union_mask, union_mask)
ax.imshow(
    mask_overlay,
    cmap="autumn",
    alpha=0.45,
    interpolation="nearest",
    origin="upper",
    extent=(0, w, h, 0),
)

is_valid = qc_df["cluster_is_valid"].fillna(False).to_numpy(dtype=bool)
xs = qc_df["centroid_x"].to_numpy()
ys = qc_df["centroid_y"].to_numpy()

ax.scatter(xs[~is_valid], ys[~is_valid], s=16, c="white", edgecolors="black", linewidths=0.4, alpha=0.8, label="non-cluster")
ax.scatter(xs[is_valid], ys[is_valid], s=22, c="cyan", edgecolors="black", linewidths=0.5, alpha=0.95, label="cluster")

# Lock axes to image-style coordinates (x right, y downward).
ax.set_xlim(0, w)
ax.set_ylim(h, 0)

ax.set_title(f"{qc_stub}: polygon union mask over ROI centroids")
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.show()

# %%
print('qc_df columns:', list(qc_df.columns))
print('metadata columns:', list(metadata.columns))
print('summary_df columns:', list(summary_df.columns))
print('stub exists in metadata:', qc_stub in metadata.astype(str).to_string())

# %%
from src.roi_analysis import plot_roi_neighbor_comparison

compare_stub = stubs[1]
compare_roi_id = None  # set int roi_id to force a specific ROI
compare_random_seed = 1  # set int for reproducible random ROI; None uses time

payload, masks = load_segmentation(DATAFOLDER, compare_stub)
raw = tifffile.imread(DATAFOLDER / f"{compare_stub}.tif")
image = raw[0] if raw.ndim > 2 else raw

compare_df = all_rois_df[all_rois_df["stub"] == compare_stub].copy()
if compare_df.empty:
    raise ValueError(f"No ROIs found for stub {compare_stub}")

fig, ax = plot_roi_neighbor_comparison(
    image=image,
    masks=masks,
    centroids_df=compare_df,
    roi_id=compare_roi_id,
    max_neighbors=10,
    figsize=(8, 8),
    random_seed=compare_random_seed,
)
plt.show()

# %%
all_rois_df.columns

# %%
all_rois_df.query("cluster_is_valid == True").cluster_neighbor_count.plot(kind="hist", bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# %%

# %%
# Display all three original TIFF images and their masks side by side
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col, stub in enumerate(stubs):
    # Top row: original TIFF images
    raw = tifffile.imread(DATAFOLDER / f"{stub}.tif")
    image = raw[0] if raw.ndim > 2 else raw
    axes[0, col].imshow(image, cmap="gray")
    axes[0, col].set_title(f"{stub} (original)")
    axes[0, col].axis("off")
    
    # Bottom row: ROI masks
    payload, masks = load_segmentation(DATAFOLDER, stub)
    axes[1, col].imshow(masks, cmap="tab20", interpolation="nearest")
    axes[1, col].set_title(f"{stub} (masks)")
    axes[1, col].axis("off")

plt.tight_layout()
plt.show()

# %%

# %%
# Random ROI montage across all files: 10x10 grid with mask overlay + major axis
rng_seed = 42  # set to None for non-reproducible sampling
n_rows, n_cols = 10, 10
n_show = n_rows * n_cols
margin = 8

if "all_rois_df" not in globals() or all_rois_df is None or all_rois_df.empty:
    raise ValueError("Run the batch cell first so all_rois_df is available")

required_cols = {"stub", "roi_id"}
missing_cols = required_cols - set(all_rois_df.columns)
if missing_cols:
    raise ValueError(f"all_rois_df is missing required columns: {missing_cols}")

roi_table = all_rois_df[["stub", "roi_id"]].dropna().copy()
roi_table["roi_id"] = roi_table["roi_id"].astype(int)
roi_table = roi_table.drop_duplicates().reset_index(drop=True)

if roi_table.empty:
    raise ValueError("No ROIs available to plot")

n_pick = min(n_show, len(roi_table))
rng = np.random.default_rng(rng_seed)
sel_idx = rng.choice(len(roi_table), size=n_pick, replace=False)
selected = roi_table.iloc[sel_idx].reset_index(drop=True)

# Cache image + masks per stub to avoid repeated I/O
cache = {}
for stub in selected["stub"].unique():
    payload, masks = load_segmentation(DATAFOLDER, stub)
    raw = tifffile.imread(DATAFOLDER / f"{stub}.tif")
    image = raw[0] if raw.ndim > 2 else raw
    cache[stub] = {"image": image, "masks": masks}

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
axes = axes.ravel()

for i, ax in enumerate(axes):
    if i >= n_pick:
        ax.axis("off")
        continue

    stub = selected.loc[i, "stub"]
    roi_id = int(selected.loc[i, "roi_id"])

    image = cache[stub]["image"]
    masks = cache[stub]["masks"]
    roi_mask = masks == roi_id

    ys, xs = np.nonzero(roi_mask)
    if ys.size < 2:
        ax.axis("off")
        continue

    # Crop around ROI for visibility
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x0 = max(0, x_min - margin)
    x1 = min(image.shape[1], x_max + margin + 1)
    y0 = max(0, y_min - margin)
    y1 = min(image.shape[0], y_max + margin + 1)

    img_crop = image[y0:y1, x0:x1]
    mask_crop = roi_mask[y0:y1, x0:x1]

    # Major axis from PCA of ROI pixel coordinates
    coords = np.column_stack((xs, ys)).astype(float)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]

    # Endpoints using min/max projection along major axis
    proj = centered @ major_vec
    p0 = center + proj.min() * major_vec
    p1 = center + proj.max() * major_vec

    # Convert to crop coordinates
    p0x, p0y = p0[0] - x0, p0[1] - y0
    p1x, p1y = p1[0] - x0, p1[1] - y0

    ax.imshow(img_crop, cmap="gray")
    ax.imshow(np.ma.masked_where(~mask_crop, mask_crop), cmap="autumn", alpha=0.5)
    ax.plot([p0x, p1x], [p0y, p1y], color="cyan", linewidth=1.5)

    ax.set_title(f"{stub}\nROI {roi_id}", fontsize=7)
    ax.axis("off")

fig.suptitle(f"Random ROI montage ({n_pick} sampled across all stubs)", fontsize=16)
plt.tight_layout()
plt.show()

# %%

# %%
# Compare center-line vs ±1 px averaged profiles on major/minor axes for 10 random ROIs
import builtins

random_seed = 123
n_rois = 10
num_samples = 256
extension_factor = 1.00  # extend line beyond ROI half-length by 100% (50% more than before)
offset_pixels = 1.0       # offsets: [-1, 0, +1] pixels

if "all_rois_df" not in globals() or all_rois_df is None or all_rois_df.empty:
    raise ValueError("Run the batch cell first so all_rois_df is available")

roi_table = all_rois_df[["stub", "roi_id"]].dropna().copy()
roi_table["roi_id"] = roi_table["roi_id"].astype(int)
roi_table = roi_table.drop_duplicates().reset_index(drop=True)
if roi_table.empty:
    raise ValueError("No ROIs found in all_rois_df")


def bilinear_sample(image_2d, x, y):
    h, w = image_2d.shape

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    out = np.full_like(x, np.nan, dtype=float)
    if not np.any(valid):
        return out

    xv = x[valid]
    yv = y[valid]
    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    wx = xv - x0v
    wy = yv - y0v

    v00 = image_2d[y0v, x0v]
    v10 = image_2d[y0v, x1v]
    v01 = image_2d[y1v, x0v]
    v11 = image_2d[y1v, x1v]

    out_valid = (
        (1 - wx) * (1 - wy) * v00
        + wx * (1 - wy) * v10
        + (1 - wx) * wy * v01
        + wx * wy * v11
    )
    out[valid] = out_valid
    return out


def axis_from_mask(roi_mask):
    ys, xs = np.nonzero(roi_mask)
    if xs.size < 5:
        return None

    coords = np.column_stack((xs, ys)).astype(float)
    center = coords.mean(axis=0)
    centered = coords - center

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    major_vec = eigvecs[:, order[0]]
    major_vec = major_vec / np.linalg.norm(major_vec)
    minor_vec = np.array([-major_vec[1], major_vec[0]])

    proj_major = centered @ major_vec
    proj_minor = centered @ minor_vec

    major_half = builtins.max(abs(proj_major.min()), abs(proj_major.max()))
    minor_half = builtins.max(abs(proj_minor.min()), abs(proj_minor.max()))

    if major_half <= 0 or minor_half <= 0:
        return None

    return center, major_vec, minor_vec, major_half, minor_half


def sample_axis_family(
    image_2d,
    center_xy,
    axis_vec,
    perp_vec,
    half_extent,
    nsamples,
    ext_factor,
    offsets,
    pixel_size_nm,
):
    half_len = half_extent * (1 + ext_factor)
    t = np.linspace(-half_len, half_len, nsamples)  # pixel units
    x_nm = t * pixel_size_nm  # already in nm

    profiles = []
    lines = []
    for off in offsets:
        line_center = center_xy + off * perp_vec
        x = line_center[0] + t * axis_vec[0]
        y = line_center[1] + t * axis_vec[1]
        p = bilinear_sample(image_2d, x, y)
        profiles.append(p)
        lines.append((x, y))

    profiles = np.vstack(profiles)
    center_profile = profiles[len(offsets) // 2]
    mean_profile = np.nanmean(profiles, axis=0)
    return center_profile, mean_profile, profiles, lines, x_nm


rng = np.random.default_rng(random_seed)
n_pick = builtins.min(n_rois, len(roi_table))
sel_idx = rng.choice(len(roi_table), size=n_pick, replace=False)
selected = roi_table.iloc[sel_idx].reset_index(drop=True)

cache = {}
for stub in selected["stub"].unique():
    payload, masks = load_segmentation(DATAFOLDER, stub)
    raw = tifffile.imread(DATAFOLDER / f"{stub}.tif")
    image = raw[0] if raw.ndim > 2 else raw
    pixel_size_nm = get_pixel_size(DATAFOLDER, stub)
    cache[stub] = {"image": image, "masks": masks, "pixel_size_nm": pixel_size_nm}

offsets = np.array([-offset_pixels, 0.0, offset_pixels], dtype=float)

fig, axes = plt.subplots(n_pick, 3, figsize=(14, 3.0 * n_pick))
if n_pick == 1:
    axes = np.array([axes])

for row_idx, (_, row) in enumerate(selected.iterrows()):
    stub = row["stub"]
    roi_id = int(row["roi_id"])

    image = cache[stub]["image"]
    masks = cache[stub]["masks"]
    pixel_size_nm = cache[stub]["pixel_size_nm"]
    roi_mask = masks == roi_id

    out = axis_from_mask(roi_mask)
    if out is None:
        for c in range(3):
            axes[row_idx, c].axis("off")
        continue

    center_xy, major_vec, minor_vec, major_half, minor_half = out

    major_center, major_mean, major_stack, major_lines, x_major_nm = sample_axis_family(
        image_2d=image,
        center_xy=center_xy,
        axis_vec=major_vec,
        perp_vec=minor_vec,
        half_extent=major_half,
        nsamples=num_samples,
        ext_factor=extension_factor,
        offsets=offsets,
        pixel_size_nm=pixel_size_nm,
    )

    minor_center, minor_mean, minor_stack, minor_lines, x_minor_nm = sample_axis_family(
        image_2d=image,
        center_xy=center_xy,
        axis_vec=minor_vec,
        perp_vec=major_vec,
        half_extent=minor_half,
        nsamples=num_samples,
        ext_factor=extension_factor,
        offsets=offsets,
        pixel_size_nm=pixel_size_nm,
    )

    ys, xs = np.nonzero(roi_mask)
    margin = 12
    x0 = builtins.max(0, xs.min() - margin)
    x1 = builtins.min(image.shape[1], xs.max() + margin + 1)
    y0 = builtins.max(0, ys.min() - margin)
    y1 = builtins.min(image.shape[0], ys.max() + margin + 1)

    img_crop = image[y0:y1, x0:x1]
    mask_crop = roi_mask[y0:y1, x0:x1]

    ax_img = axes[row_idx, 0]
    ax_img.imshow(img_crop, cmap="gray")
    ax_img.imshow(np.ma.masked_where(~mask_crop, mask_crop), cmap="autumn", alpha=0.45)

    for line_idx, (x, y) in enumerate(major_lines):
        color = "cyan" if line_idx == 1 else "deepskyblue"
        ax_img.plot(x - x0, y - y0, color=color, linewidth=1.2, alpha=0.9)

    for line_idx, (x, y) in enumerate(minor_lines):
        color = "lime" if line_idx == 1 else "springgreen"
        ax_img.plot(x - x0, y - y0, color=color, linewidth=1.0, alpha=0.9)

    ax_img.set_title(f"{stub} | ROI {roi_id}", fontsize=9)
    ax_img.axis("off")

    ax_maj = axes[row_idx, 1]
    for p in major_stack:
        ax_maj.plot(x_major_nm, p, color="0.75", linewidth=0.8)
    ax_maj.plot(x_major_nm, major_center, color="tab:blue", linewidth=1.2, label="center")
    ax_maj.plot(x_major_nm, major_mean, color="tab:red", linewidth=1.8, label="avg(-1,0,+1 px)")
    ax_maj.set_title("Major axis profile", fontsize=9)
    ax_maj.set_xlabel("Distance (nm)")
    ax_maj.set_ylabel("Intensity")
    if row_idx == 0:
        ax_maj.legend(loc="best", fontsize=8)

    ax_min = axes[row_idx, 2]
    for p in minor_stack:
        ax_min.plot(x_minor_nm, p, color="0.75", linewidth=0.8)
    ax_min.plot(x_minor_nm, minor_center, color="tab:green", linewidth=1.2, label="center")
    ax_min.plot(x_minor_nm, minor_mean, color="tab:red", linewidth=1.8, label="avg(-1,0,+1 px)")
    ax_min.set_title("Minor axis profile", fontsize=9)
    ax_min.set_xlabel("Distance (nm)")
    ax_min.set_ylabel("Intensity")

fig.suptitle(
    "10 random ROIs: center-line profiles vs ±1 px averaged profiles (major/minor axes)",
    fontsize=13,
    y=1.002,
)
plt.tight_layout()
plt.show()

# %%

# %%
f, ax = plt.subplots(ncols=2, figsize=(12, 5))

for idx, row in all_summary_df.iterrows():
    stub = row['stub']
    pixel_size_nm = row.get('mean_pixel_size_nm_per_px', np.nan)
    if np.isnan(pixel_size_nm):
        # Fallback: try to get from any rois_df row with this stub
        stub_rows = all_rois_df[all_rois_df['stub'] == stub]
        if not stub_rows.empty:
            pixel_size_nm = stub_rows.iloc[0]['pixel_size']
    
    major = row['mean_profile_major_smpls']
    minor = row['mean_profile_minor_smpls']

    major_diam_nm = row.get('mean_diameter_major_nm', np.nan)
    minor_diam_nm = row.get('mean_diameter_minor_nm', np.nan)

    # Estimate mean sampling step (nm) per stub using ROI-level step columns
    stub_rows = all_rois_df[all_rois_df['stub'] == stub]
    step_major_nm = np.nan
    step_minor_nm = np.nan
    if not stub_rows.empty:
        if 'step_major' in stub_rows.columns:
            step_major_nm = float(np.nanmean(stub_rows['step_major'].to_numpy(dtype=float))) * pixel_size_nm
        if 'step_minor' in stub_rows.columns:
            step_minor_nm = float(np.nanmean(stub_rows['step_minor'].to_numpy(dtype=float))) * pixel_size_nm

    # Fallback to 1 pixel step in nm if unavailable
    if not np.isfinite(step_major_nm):
        step_major_nm = pixel_size_nm
    if not np.isfinite(step_minor_nm):
        step_minor_nm = pixel_size_nm

    # Center x-axis at 0 nm
    x_major_nm = (np.arange(len(major)) - (len(major) - 1) / 2.0) * step_major_nm
    x_minor_nm = (np.arange(len(minor)) - (len(minor) - 1) / 2.0) * step_minor_nm
    
    # Plot major axis
    ax[0].plot(x_major_nm, major, label=f"{stub}", alpha=0.7)
    if np.isfinite(major_diam_nm):
        ax[0].text(0.02, 0.95 - idx*0.08, f"{stub} major: {major_diam_nm:.1f} nm", 
                   transform=ax[0].transAxes, fontsize=8, verticalalignment='top')
    
    # Plot minor axis
    ax[1].plot(x_minor_nm, minor, label=f"{stub}", alpha=0.7)
    if np.isfinite(minor_diam_nm):
        ax[1].text(0.02, 0.95 - idx*0.08, f"{stub} minor: {minor_diam_nm:.1f} nm", 
                   transform=ax[1].transAxes, fontsize=8, verticalalignment='top')

ax[0].set_title("Mean Major Axis Profiles")
ax[0].set_ylabel("Intensity")
ax[0].set_xlabel("Distance (nm)")
ax[0].grid(True, alpha=0.3)

ax[1].set_title("Mean Minor Axis Profiles")
ax[1].set_ylabel("Intensity")
ax[1].set_xlabel("Distance (nm)")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%

# %%
# Plot mean 4-axis profiles from batch results
if "all_summary_df" not in globals() or all_summary_df is None or all_summary_df.empty:
    raise ValueError("Run the batch cell first so all_summary_df is available")

# Filter rows that have 4-axis profiles
rows_with_profiles = all_summary_df[
    all_summary_df["mean_four_axis_smpls"].notna() & 
    (all_summary_df["mean_four_axis_smpls"].apply(lambda x: x is not None and len(x) > 0 if isinstance(x, (list, np.ndarray)) else False))
].copy()

if rows_with_profiles.empty:
    print("No 4-axis profiles available in summary")
else:
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for idx, (_, row) in enumerate(rows_with_profiles.iterrows()):
        stub = row['stub']
        profile = row['mean_four_axis_smpls']
        
        if profile is None or len(profile) == 0:
            continue

        pixel_size_nm = row.get('mean_pixel_size_nm_per_px', np.nan)
        stub_rows = all_rois_df[all_rois_df['stub'] == stub]

        # Approximate 4-axis step using ROI-level major/minor step means
        step_nm = np.nan
        if not stub_rows.empty and np.isfinite(pixel_size_nm):
            step_vals = []
            if 'step_major' in stub_rows.columns:
                step_vals.append(float(np.nanmean(stub_rows['step_major'].to_numpy(dtype=float))))
            if 'step_minor' in stub_rows.columns:
                step_vals.append(float(np.nanmean(stub_rows['step_minor'].to_numpy(dtype=float))))
            if step_vals:
                step_nm = float(np.nanmean(step_vals)) * pixel_size_nm

        if not np.isfinite(step_nm):
            step_nm = pixel_size_nm if np.isfinite(pixel_size_nm) else 1.0

        four_axis_diam_nm = row.get('mean_diameter_four_axis_nm', np.nan)
        label = f"{stub} (4-axis mean)"
        if np.isfinite(four_axis_diam_nm):
            label = f"{label}, d={four_axis_diam_nm:.1f} nm"

        x_nm = (np.arange(len(profile)) - (len(profile) - 1) / 2.0) * step_nm
        ax.plot(x_nm, profile, linewidth=2, label=label, marker='o', markersize=3, alpha=0.8)
    
    ax.set_title("Mean 4-Axis Composite Profiles per Stub (from batch processing)", fontsize=12)
    ax.set_xlabel("Distance (nm)")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
