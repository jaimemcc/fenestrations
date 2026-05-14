# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import sys
# Add src to path for importing local modules
sys.path.insert(0, str(Path("../src").resolve()))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from trompy import save_figure_atomic

from src.outlier_filter import robust_tail_filter_log_area
from src.add_extra_columns import add_extra_columns

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["font.size"] = 8
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams['savefig.transparent'] = True

SAVEFIGS = True

# %%
DATAFOLDER = Path("./data")
METAFILE = Path("./data/fenestrations_metafile.xlsx")
FIGSFOLDER = Path("./figs/paper1/panels")

# Ensure output folder exists for direct saves.
FIGSFOLDER.mkdir(parents=True, exist_ok=True)

# save_figure_atomic stages files under _tmp/<relative output path>;
# ensure that staging path exists when using relative figure folders.
(Path.cwd() / "_tmp" / FIGSFOLDER).mkdir(parents=True, exist_ok=True)

data = pd.read_pickle(DATAFOLDER / "roi_data.pickle")

# Extract the DataFrames
rois_df = data['rois']
summary_df = data['summary']

# load metafile
metadata = pd.read_excel(METAFILE)

print("ROI Data Statistics")
print("=" * 50)
print(f"\nROIs DataFrame:")
print(f"  Rows: {len(rois_df)}")
print(f"  Columns: {len(rois_df.columns)}")
print(f"  Column names: {list(rois_df.columns)}")
print(f"\n  Memory usage: {rois_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n\nSummary DataFrame:")
print(f"  Rows: {len(summary_df)}")
print(f"  Columns: {len(summary_df.columns)}")
print(f"  Column names: {list(summary_df.columns)}")
print(f"\n  Memory usage: {summary_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 50)
print("\nFirst few rows of ROIs DataFrame:")
print(rois_df.head())

print("\n" + "=" * 50)
print("\nFirst few rows of Summary DataFrame:")
print(summary_df.head())

print("\n" + "=" * 50)
print("\nData types:")
print("\nROIs DataFrame:")
print(rois_df.dtypes)
print("\nSummary DataFrame:")
print(summary_df.dtypes)


# %%
def get_filtered_control_data(rois_df, summary_df):
    
    rois_df, _ = robust_tail_filter_log_area(rois_df)
    rois_df = rois_df.query("condition == 'control'").reset_index(drop=True)
    summary_df = summary_df.query("condition == 'control'").reset_index(drop=True)
    
    return rois_df, summary_df

rois_df, summary_df = get_filtered_control_data(rois_df, summary_df)

# Add extra derived columns once so downstream plotting cells can rely on them.
rois_df, summary_df = add_extra_columns(
    rois_df,
    summary_df,
    union_dir=Path("./union_masks"),
    images_path=None,
    add_union_area=True,
)

# %%
rois_df.info()


# %%
# setting plotting conventions

def init_diameter_panel():
    f, ax = plt.subplots(figsize=(1.5, 1.2),
                         gridspec_kw={"left": 0.25, "right": 0.95, "top": 0.95, "bottom": 0.37})
    
    return f, ax

circular_color      = "#E9B07A"  # muted apricot
elliptical_color    = "#87BFAE"  # soft seafoam
profile_deriv_color = "#6F9DD9"  # muted blue
profile_fwhm_color  = "#8B86CF"  # muted violet
profile_p2p_color   = "#5FA8C7"  # muted cyan

circularity_color = "#8E97A1"

major_color = "#2F6DB3"  # balanced cobalt blue
minor_color = "#C96B28"  # muted amber-orange
neighbor_color = "goldenrod"


# circularity_diam_color = "coral"
# elliptical_color = "lightcoral"
# diameter_color = "steelblue"
# circularity_color = "coral"


# %%
## Panel A: Diameter distribution (circular)

xlims = (0, 110)

f, ax = init_diameter_panel()
sns.kdeplot(data=rois_df.diameter_area,
            fill=True, ax=ax, color=circular_color, alpha=0.7,
            cut=0)

ax.set_xlim(xlims)
ax.set_xlabel("Diameter (nm)")

ax.set_yticks([])

ax.text(80, 0.04, f"Mean:\n{rois_df.diameter_area.mean():.1f} nm", ha="center", color=circular_color)

sns.despine(offset=5)

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "diameter_circular")

# %%
## Panel B: Circularity distribution

rois_df["circularity"] = rois_df.minor_axis_length_px / rois_df.major_axis_length_px


f, ax = init_diameter_panel()
sns.kdeplot(data=rois_df.circularity,
            fill=True, ax=ax, color=circularity_color, alpha=0.7,
            cut=0)
ax.set_xlabel("Circularity")
ax.set_xlim(0, 1.1)
ax.set_yticks([])

ax.text(0.3, 0.94, f"Mean:\n{rois_df.circularity.mean():.3f}", ha="center", color=circularity_color)

sns.despine(offset=5)

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "circularity")

# %%
## Panel C: Diameter distribution (elliptical)

# Figure for diameter based on major/minor axis
# Calculate equivalent diameter in px
rois_df['elliptical_diameter'] = np.sqrt(rois_df.major_axis_length_px * rois_df.minor_axis_length_px) 

f, ax = init_diameter_panel()
sns.kdeplot(data=rois_df.elliptical_diameter,
            fill=True, ax=ax, color=elliptical_color, alpha=0.7,
            cut=0)

ax.set_xlabel("Diameter (nm)")
ax.set_xlim(xlims)
ax.set_yticks([])

ax.text(80, 0.04, f"Mean:\n{rois_df.elliptical_diameter.mean():.1f} nm", ha="center", color=elliptical_color)

sns.despine(offset=5)

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "diameter_elliptical")


# %%
def get_spacing_factor(rois_df, row, axis):
    
    if axis == "major":
        return (rois_df
                .query("stub == @row.stub")
                .step_major_nm
                .mean()
                )
    elif axis == "minor":
        return (rois_df
                .query("stub == @row.stub")
                .step_minor_nm
                .mean()
                )
    else:
        raise ValueError("Axis must be 'major' or 'minor'")

# per-ROI nm per sample
rois_df["step_major_nm"] = rois_df["step_major"] * rois_df["pixel_size"]
rois_df["step_minor_nm"] = rois_df["step_minor"] * rois_df["pixel_size"]

# %%
rois_df.columns

# %%
# Profile-based diameters are computed by add_extra_columns (applied to
# the per-stub averaged profiles) and live directly in summary_df.
print(
    summary_df[
        [
            "mean_diameter_p2p_major_nm", "mean_diameter_p2p_minor_nm", "mean_diameter_p2p_nm",
            "mean_diameter_fwhm_major_nm", "mean_diameter_fwhm_minor_nm", "mean_diameter_fwhm_nm",
            "mean_diameter_deriv_major_nm", "mean_diameter_deriv_minor_nm", "mean_diameter_deriv_nm",
        ]
    ].describe().round(1)
)

# %%
# prep data for profiles plot
row = summary_df.iloc[0]

f, ax = plt.subplots(figsize=(2.5, 1.2), ncols=2, sharey=True,
                         gridspec_kw={"left": 0.25, "right": 0.95, "top": 0.95, "bottom": 0.35,
                                      "wspace": 0.5})

major_profile = row.mean_profile_major_smpls
n = len(major_profile)
x_nm = (np.arange(n) - (n - 1) / 2) * get_spacing_factor(rois_df, row, "major")

ax[0].plot(x_nm, major_profile, color=major_color)
ax[0].set_ylabel("Intensity")
ax[0].set_yticks([])

sns.despine(ax=ax[0], offset=5)

minor_profile = row.mean_profile_minor_smpls
n = len(minor_profile)
x_nm = (np.arange(n) - (n - 1) / 2) * get_spacing_factor(rois_df, row, "minor")

ax[1].plot(x_nm, minor_profile, color=minor_color)

for axis in ax:
    axis.set_xlim(-110, 110)
    
sns.despine(ax=ax[1], offset=5, left=True)

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "profiles")

# %%
[c for c in summary_df.columns if "diameter" in c]

# %%
f, ax = plt.subplots(
    3, 1,
    figsize=(1.8, 1.2),
    # sharex=True,
    gridspec_kw={"left": 0.28, "right": 0.95, "top": 0.95, "bottom": 0.4, "hspace": 0.3}
)

kde_specs = [
    ("mean_diameter_p2p_nm", "P2P", profile_p2p_color),
    ("mean_diameter_fwhm_nm", "FWHM", profile_fwhm_color),
    ("mean_diameter_deriv_nm", "Derivative", profile_deriv_color),
]

for i, (col, label, color) in enumerate(kde_specs):
    vals = summary_df[col].dropna()
    if len(vals) > 1:
        sns.kdeplot(data=vals, fill=True, ax=ax[i], color=color, alpha=0.5, cut=0)

    ax[i].set_xlim(xlims)
    ax[i].set_yticks([])
    # ax[i].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=12)
    ax[i].set_ylabel("")
    
    if i < 2:
        ax[i].set_xlabel("")

    sns.despine(ax=ax[i], offset=5)

xlims = (0, 170)

for axis in ax[:2]:
    sns.despine(ax=axis, bottom=True)
    axis.set_xticks([])
    axis.set_xlim(xlims)

ax[-1].set_xlabel("Diameter (nm)")
ax[-1].set_xticks([0, 50, 100, 150])
ax[-1].set_xlim(xlims)

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "diameter_profiles")

# %%
summary_df[["mean_diameter_p2p_nm", "mean_diameter_fwhm_nm", "mean_diameter_deriv_nm"]]

# %%
methods = {
    "mean_diameter_p2p_nm":  "Peak-to-peak",
    "mean_diameter_fwhm_nm": "FWHM",
    "mean_diameter_deriv_nm": "Derivative",
}

fig, axes = plt.subplots(
    1, 3,
    figsize=(5.5, 1.5),
    sharey=True,
    gridspec_kw={"left": 0.08, "right": 0.97, "top": 0.88,
                 "bottom": 0.3, "wspace": 0.3},
)

xlims = (0, 160)

for ax, (col, label) in zip(axes, methods.items()):
    vals = summary_df[col].dropna()
    if len(vals) > 1:
        sns.kdeplot(vals, ax=ax, fill=True, color="steelblue", alpha=0.7, cut=0)
    ax.axvline(vals.median(), color="steelblue", lw=1, ls="--")
    ax.set_xlim(xlims)
    ax.set_yticks([])
    ax.set_title(label, fontsize=7)
    ax.set_xlabel("Diameter (nm)")
    sns.despine(ax=ax, offset=4)

if SAVEFIGS:
    save_figure_atomic(fig, FIGSFOLDER / "diameter_profile_methods")

# %%
rois_df.columns

# %%

# %%
# Mean diameter comparing five measurement methods

diameter_boxplot_df = pd.concat(
    [
        pd.DataFrame({
            "method": "Circular",
            "diameter_nm": rois_df["diameter_area"],
            "color": circular_color,
        }),
        pd.DataFrame({
            "method": "Elliptical",
            "diameter_nm": rois_df["diameter_elliptical_nm"],
            "color": elliptical_color,
        }),
        pd.DataFrame({
            "method": "Peak-to-peak",
            "diameter_nm": summary_df["mean_diameter_p2p_nm"],
            "color": profile_p2p_color,
        }),
        pd.DataFrame({
            "method": "FWHM",
            "diameter_nm": summary_df["mean_diameter_fwhm_nm"],
            "color": profile_fwhm_color,
        }),
        pd.DataFrame({
            "method": "Derivative",
            "diameter_nm": summary_df["mean_diameter_deriv_nm"],
            "color": profile_deriv_color,
        }),
    ],
    ignore_index=True,
).dropna(subset=["diameter_nm"])

method_order = [
    "Derivative",
    "FWHM",
    "Peak-to-peak",
    "Elliptical",
    "Circular",
]

# Compute mean for each method
mean_data = []
for method in method_order:
    data = diameter_boxplot_df[diameter_boxplot_df["method"] == method]["diameter_nm"].values
    mean = np.mean(data)
    color = diameter_boxplot_df[diameter_boxplot_df["method"] == method]["color"].iloc[0]
    mean_data.append({"method": method, "mean": mean, "color": color})

mean_df = pd.DataFrame(mean_data)

f, ax = plt.subplots(
    figsize=(1.8, 1.7),
    gridspec_kw={"left": 0.5, "right": 0.95, "top": 0.95, "bottom": 0.3}
)

# Plot mean markers only
y_pos = np.arange(len(method_order))
ax.scatter(
    mean_df["mean"],
    y_pos,
    color=mean_df["color"],
    s=50,
    alpha=0.7,
    zorder=3,
)

ax.set_yticks(y_pos)
ax.set_yticklabels(method_order)
ax.set_xlabel("Diameter (nm)")
ax.set_xlim(0, 160)
ax.set_xticks([0, 50, 100, 150])

sns.despine(ax=ax, offset=5)

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "diameter_boxplot_5methods")


# %%
# supplemental - showing different diameter from profile measurements

def init_profile_fig():
    f, ax = plt.subplots(nrows=3, figsize=(1.5, 3),
                         gridspec_kw={"left": 0.10, "right": 0.95, "top": 0.95, "bottom": 0.15})
    
    return f, ax

rows_to_use = [0, 1, 2]

rows_from_df = summary_df.iloc[rows_to_use, :]

diameter_columns = ["mean_diameter_deriv_major_nm", "mean_diameter_fwhm_major_nm", "mean_diameter_p2p_major_nm"]
diameter_colors = [profile_deriv_color, profile_fwhm_color, profile_p2p_color]

def make_diameter_profile_plot(row):
    f, ax = init_profile_fig()

    for diam_col, diam_color, axis in zip(diameter_columns, diameter_colors, ax):
        major_profile = row.mean_profile_major_smpls
        n = len(major_profile)
        x_nm = (np.arange(n) - (n - 1) / 2) * get_spacing_factor(rois_df, row, "major")

        axis.plot(x_nm, major_profile, color="k")
        # ax[0].set_ylabel("Intensity")
        axis.set_yticks([])
        axis.set_xticks([])
        axis.set_ylim(-2, 1.05 * major_profile.max())
        
        # set up fill_betweens
        axis.axvspan(-row[diam_col]/2, row[diam_col]/2, color=diam_color, alpha=0.3, edgecolor="none", linewidth=0)
        
        sns.despine(ax=axis, left=True, bottom=True, offset=5)

    sns.despine(ax=ax[-1], left=True, bottom=False, offset=5)
    ax[-1].set_xlim(-110, 110)
    ax[-1].set_xticks([-100, 0, 100])
    ax[-1].set_xlabel("Distance (nm)")
    
    return f, ax

rows_for_examples = {"1": 7, "2": 12, "3": 24}

for key, val in rows_for_examples.items():

    row = summary_df.iloc[val]
    f, ax = make_diameter_profile_plot(row)

    if SAVEFIGS:
        save_figure_atomic(f, FIGSFOLDER / f"diameter_profile_methods_example_{key}")


# %%
f, ax = init_profile_fig()

for diam_col, diam_color, axis in zip(diameter_columns, diameter_colors, ax):
    
    sns.kdeplot(
        data=summary_df[diam_col].dropna(),
        fill=True,
        ax=axis,
        color=diam_color,
        alpha=0.5,
        cut=0
    )
    
    axis.set_yticks([])
    axis.set_xticks([])
    axis.set_ylabel("")
    axis.set_xlabel("")
    axis.set_xlim(0, 160)
    
    sns.despine(ax=axis, left=True, bottom=True, offset=5)
    
sns.despine(ax=ax[-1], left=True, bottom=False, offset=5)
ax[-1].set_xlim(0, 160)
ax[-1].set_xticks([0, 50, 100, 150])
ax[-1].set_xlabel("Diameter (nm)")

if SAVEFIGS:
    save_figure_atomic(f, FIGSFOLDER / "diameter_profile_methods_kde")
    
    

# %%
# ax[0].axvspan?

# %%
summary_df.columns

# %%
