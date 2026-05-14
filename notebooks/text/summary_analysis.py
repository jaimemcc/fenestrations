# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: fenestrations (.venv) Python 3.13
#     language: python
#     name: fenestrations-313
# ---

# %%
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from src.outlier_filter import robust_tail_filter_log_area

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
# Plot control-only equivalent diameters from major/minor axis pairs and report mean/std
metrics_file = Path("C:/Users/sthui4072/Github/fenestrations/stub_profile_metrics_major_minor.xlsx")
if not metrics_file.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

metrics_df = pd.read_excel(metrics_file)
if "condition" not in metrics_df.columns:
    meta_conditions = metadata[["stub", "condition"]].dropna(subset=["stub"]).drop_duplicates(subset=["stub"], keep="first")
    metrics_df = metrics_df.merge(meta_conditions, on="stub", how="left")

control_df = metrics_df.query("condition == 'control'").copy()
if control_df.empty:
    raise ValueError("No control rows found in the metrics file.")

method_pairs = [
    ("major_p2p_nm", "minor_p2p_nm", "p2p"),
    ("major_fwhm_nm", "minor_fwhm_nm", "FWHM"),
    ("major_derivative_nm", "minor_derivative_nm", "derivative"),
]

rows = []
method_values = []
for major_col, minor_col, title in method_pairs:
    pair_df = control_df[["stub", major_col, minor_col]].dropna().copy()
    pair_df = pair_df[(pair_df[major_col] > 0) & (pair_df[minor_col] > 0)]
    if pair_df.empty:
        method_values.append((title, np.array([])))
        continue
    pair_df["equivalent_diameter_nm"] = np.sqrt(pair_df[major_col] * pair_df[minor_col])
    method_values.append((title, pair_df["equivalent_diameter_nm"].to_numpy()))

all_values = np.concatenate([values for _, values in method_values if values.size > 0])
if all_values.size == 0:
    raise ValueError("No valid equivalent diameter values found for control rows.")

x_min = 25.0
x_max = 125
shared_bins = np.linspace(x_min, x_max, 75) if np.unique(all_values).size > 1 else 10

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

for ax, (title, values), (major_col, minor_col, _) in zip(axes, method_values, method_pairs):
    if values.size == 0:
        rows.append({
            "method": title,
            "n": 0,
            "equivalent_diameter_mean_nm": np.nan,
            "equivalent_diameter_std_nm": np.nan,
        })
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        continue

    mean_value = float(np.nanmean(values))
    std_value = float(np.nanstd(values, ddof=1)) if values.size > 1 else np.nan

    rows.append({
        "method": title,
        "n": int(len(values)),
        "equivalent_diameter_mean_nm": mean_value,
        "equivalent_diameter_std_nm": std_value,
    })

    weights = np.ones_like(values) / len(values) * 100
    ax.hist(values, bins=shared_bins, weights=weights, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.set_title(f"{title} (n={len(values)})")
    ax.set_xlabel("Diameter (nm)")
    ax.set_ylabel("Occurrence (%)")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(x_min, x_max)

equivalent_diameter_metrics_df = pd.DataFrame(rows)

fig.suptitle("Control-only equivalent diameters from major/minor axis pairs", y=1.02)
plt.tight_layout()
plt.show()

display(equivalent_diameter_metrics_df)

# %%
# Per-stub lattice-based porosity and density from equivalent diameters (p2p, FWHM, derivative)
# Lattice cell areas use mean neighbor spacing per stub (control only).

metrics_file = Path("C:/Users/sthui4072/Github/fenestrations/stub_profile_metrics_major_minor.xlsx")
if not metrics_file.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

metrics_df = pd.read_excel(metrics_file)
if "condition" not in metrics_df.columns:
    meta_conditions = metadata[["stub", "condition"]].dropna(subset=["stub"]).drop_duplicates(subset=["stub"], keep="first")
    metrics_df = metrics_df.merge(meta_conditions, on="stub", how="left")
metrics_df = metrics_df.query("condition == 'control'").copy()

required_summary_cols = ["stub", "id", "condition", "mean_cluster_neighbor_distance_nm", "mean_neighbor_count"]
missing_summary = [c for c in required_summary_cols if c not in summary_df.columns]
if missing_summary:
    raise KeyError(f"summary_df is missing required columns: {missing_summary}")

summary_stub = (
    summary_df[required_summary_cols]
    .dropna(subset=["stub"])
    .query("condition == 'control'")
    .drop_duplicates(subset=["stub"], keep="first")
    .copy()
)
if summary_stub.empty:
    raise ValueError("No control rows found in summary_df.")

# Build lattice cell areas directly from neighbor spacing.
summary_stub["cell_area_square_nm2"] = summary_stub["mean_cluster_neighbor_distance_nm"] ** 2
summary_stub["cell_area_hex_nm2"] = (np.sqrt(3) / 2.0) * (summary_stub["mean_cluster_neighbor_distance_nm"] ** 2)
summary_stub["blend_weight"] = np.clip((summary_stub["mean_neighbor_count"] - 4.0) / 2.0, 0.0, 1.0)
summary_stub["cell_area_blend_nm2"] = (
    summary_stub["blend_weight"] * summary_stub["cell_area_hex_nm2"]
    + (1.0 - summary_stub["blend_weight"]) * summary_stub["cell_area_square_nm2"]
)

method_specs = [
    ("p2p", "major_p2p_nm", "minor_p2p_nm"),
    ("fwhm", "major_fwhm_nm", "minor_fwhm_nm"),
    ("derivative", "major_derivative_nm", "minor_derivative_nm"),
]

stub_density_lattice_df = summary_stub.copy()

for method_name, major_col, minor_col in method_specs:
    if major_col not in metrics_df.columns or minor_col not in metrics_df.columns:
        raise KeyError(f"Missing columns in metrics file: {major_col}, {minor_col}")

    method_df = metrics_df[["stub", major_col, minor_col]].dropna().copy()
    method_df = method_df[(method_df[major_col] > 0) & (method_df[minor_col] > 0)]

    method_df[f"{method_name}_equivalent_diameter_nm"] = np.sqrt(method_df[major_col] * method_df[minor_col])
    method_df[f"{method_name}_equivalent_area_nm2"] = np.pi * (method_df[f"{method_name}_equivalent_diameter_nm"] / 2.0) ** 2

    # Keep one row per stub in case duplicates exist in the metrics file.
    method_df = method_df.drop_duplicates(subset=["stub"], keep="first")

    stub_density_lattice_df = stub_density_lattice_df.merge(
        method_df[["stub", f"{method_name}_equivalent_diameter_nm", f"{method_name}_equivalent_area_nm2"]],
        on="stub",
        how="left",
    )

    # Porosity from equivalent diameter area and lattice cell area.
    for lattice_name, cell_area_col in [
        ("square", "cell_area_square_nm2"),
        ("hex", "cell_area_hex_nm2"),
        ("blend", "cell_area_blend_nm2"),
    ]:
        porosity_col = f"porosity_{method_name}_{lattice_name}"
        porosity_pct_col = f"porosity_{method_name}_{lattice_name}_pct"
        density_nm2_col = f"density_{method_name}_{lattice_name}_per_nm2"
        density_um2_col = f"density_{method_name}_{lattice_name}_per_um2"

        stub_density_lattice_df[porosity_col] = (
            stub_density_lattice_df[f"{method_name}_equivalent_area_nm2"] / stub_density_lattice_df[cell_area_col]
        )
        stub_density_lattice_df[porosity_pct_col] = stub_density_lattice_df[porosity_col] * 100.0
        stub_density_lattice_df[density_nm2_col] = (
            stub_density_lattice_df[porosity_col] / stub_density_lattice_df[f"{method_name}_equivalent_area_nm2"]
        )
        stub_density_lattice_df[density_um2_col] = stub_density_lattice_df[density_nm2_col] * 1_000_000.0

preview_cols = [
    "stub",
    "condition",
    "p2p_equivalent_diameter_nm",
    "porosity_p2p_blend_pct",
    "density_p2p_blend_per_um2",
    "fwhm_equivalent_diameter_nm",
    "porosity_fwhm_blend_pct",
    "density_fwhm_blend_per_um2",
    "derivative_equivalent_diameter_nm",
    "porosity_derivative_blend_pct",
    "density_derivative_blend_per_um2",
]

display(stub_density_lattice_df[preview_cols].sort_values(["condition", "stub"]).reset_index(drop=True))

# %%
# Per-stub porosity and density from equivalent diameters (p2p, FWHM, derivative),
# using union-mask total area per stub instead of lattice-model cell area (control only).

metrics_file = Path("C:/Users/sthui4072/Github/fenestrations/stub_profile_metrics_major_minor.xlsx")
union_summary_file = Path("C:/Users/sthui4072/Github/fenestrations/union_masks/union_mask_summary.csv")

if not metrics_file.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
if not union_summary_file.exists():
    raise FileNotFoundError(f"Union-mask summary file not found: {union_summary_file}")

metrics_df = pd.read_excel(metrics_file)
union_area_df = pd.read_csv(union_summary_file)

if "condition" not in metrics_df.columns:
    meta_conditions = metadata[["stub", "condition"]].dropna(subset=["stub"]).drop_duplicates(subset=["stub"], keep="first")
    metrics_df = metrics_df.merge(meta_conditions, on="stub", how="left")
metrics_df = metrics_df.query("condition == 'control'").copy()

required_union_cols = ["stub", "covered_area_nm2", "covered_area_um2"]
missing_union = [c for c in required_union_cols if c not in union_area_df.columns]
if missing_union:
    raise KeyError(f"union_mask_summary.csv is missing required columns: {missing_union}")

required_summary_cols = ["stub", "id", "condition"]
missing_summary = [c for c in required_summary_cols if c not in summary_df.columns]
if missing_summary:
    raise KeyError(f"summary_df is missing required columns: {missing_summary}")

def _stub_key(value):
    if pd.isna(value):
        return None
    return "".join(ch for ch in str(value).upper() if ch.isalnum())

summary_stub = (
    summary_df[required_summary_cols]
    .dropna(subset=["stub"])
    .query("condition == 'control'")
    .drop_duplicates(subset=["stub"], keep="first")
    .copy()
)
if summary_stub.empty:
    raise ValueError("No control rows found in summary_df.")
summary_stub["_stub_key"] = summary_stub["stub"].map(_stub_key)

# Count ROIs per control stub so union porosity uses total (count-scaled) equivalent area.
roi_counts = (
    rois_df.query("condition == 'control'")[["stub"]]
    .dropna(subset=["stub"])
    .assign(_stub_key=lambda d: d["stub"].map(_stub_key))
    .groupby("_stub_key", as_index=False)
    .size()
    .rename(columns={"size": "roi_count"})
)

union_area_df = union_area_df.copy()
union_area_df["_stub_key"] = union_area_df["stub"].map(_stub_key)
union_area_df = union_area_df.drop_duplicates(subset=["_stub_key"], keep="first")

stub_density_union_df = summary_stub.merge(
    union_area_df[["_stub_key", "covered_area_nm2", "covered_area_um2", "coverage_percent"]],
    on="_stub_key",
    how="left",
)
stub_density_union_df = stub_density_union_df.merge(roi_counts, on="_stub_key", how="left")

method_specs = [
    ("p2p", "major_p2p_nm", "minor_p2p_nm"),
    ("fwhm", "major_fwhm_nm", "minor_fwhm_nm"),
    ("derivative", "major_derivative_nm", "minor_derivative_nm"),
]

for method_name, major_col, minor_col in method_specs:
    if major_col not in metrics_df.columns or minor_col not in metrics_df.columns:
        raise KeyError(f"Missing columns in metrics file: {major_col}, {minor_col}")

    method_df = metrics_df[["stub", major_col, minor_col]].dropna().copy()
    method_df = method_df[(method_df[major_col] > 0) & (method_df[minor_col] > 0)]
    method_df["_stub_key"] = method_df["stub"].map(_stub_key)

    method_df[f"{method_name}_equivalent_diameter_nm"] = np.sqrt(method_df[major_col] * method_df[minor_col])
    method_df[f"{method_name}_equivalent_area_nm2"] = np.pi * (method_df[f"{method_name}_equivalent_diameter_nm"] / 2.0) ** 2

    method_df = method_df.drop_duplicates(subset=["_stub_key"], keep="first")

    stub_density_union_df = stub_density_union_df.merge(
        method_df[["_stub_key", f"{method_name}_equivalent_diameter_nm", f"{method_name}_equivalent_area_nm2"]],
        on="_stub_key",
        how="left",
    )

    total_area_col = f"total_{method_name}_equivalent_area_nm2"
    porosity_col = f"porosity_{method_name}_union"
    porosity_pct_col = f"porosity_{method_name}_union_pct"
    density_nm2_col = f"density_{method_name}_union_per_nm2"
    density_um2_col = f"density_{method_name}_union_per_um2"

    stub_density_union_df[total_area_col] = (
        stub_density_union_df[f"{method_name}_equivalent_area_nm2"] * stub_density_union_df["roi_count"]
    )
    stub_density_union_df[porosity_col] = (
        stub_density_union_df[total_area_col] / stub_density_union_df["covered_area_nm2"]
    )
    stub_density_union_df[porosity_pct_col] = stub_density_union_df[porosity_col] * 100.0
    stub_density_union_df[density_nm2_col] = (
        stub_density_union_df["roi_count"] / stub_density_union_df["covered_area_nm2"]
    )
    stub_density_union_df[density_um2_col] = stub_density_union_df[density_nm2_col] * 1_000_000.0

stub_density_union_df = stub_density_union_df.drop(columns=["_stub_key"])

preview_cols_union = [
    "stub",
    "condition",
    "roi_count",
    "covered_area_um2",
    "coverage_percent",
    "p2p_equivalent_diameter_nm",
    "porosity_p2p_union_pct",
    "density_p2p_union_per_um2",
    "fwhm_equivalent_diameter_nm",
    "porosity_fwhm_union_pct",
    "density_fwhm_union_per_um2",
    "derivative_equivalent_diameter_nm",
    "porosity_derivative_union_pct",
    "density_derivative_union_per_um2",
]

display(stub_density_union_df[preview_cols_union].sort_values(["condition", "stub"]).reset_index(drop=True))

# %%
# Plot porosity and density for method-based and ROI-derived models (control-only stubs),
# and include mean +- std values in each panel.

required_lattice_cols = [
    "porosity_p2p_blend_pct", "porosity_fwhm_blend_pct", "porosity_derivative_blend_pct",
    "density_p2p_blend_per_um2", "density_fwhm_blend_per_um2", "density_derivative_blend_per_um2",
]
required_union_cols = [
    "porosity_p2p_union_pct", "porosity_fwhm_union_pct", "porosity_derivative_union_pct",
    "density_p2p_union_per_um2", "density_fwhm_union_per_um2", "density_derivative_union_per_um2",
]
required_area_cols = [
    "porosity_lattice_from_area_pct", "density_lattice_from_area_per_um2",
    "porosity_union_from_area_pct", "density_union_from_area_per_um2",
]

missing_lattice = [c for c in required_lattice_cols if c not in stub_density_lattice_df.columns]
missing_union = [c for c in required_union_cols if c not in stub_density_union_df.columns]
missing_area = [c for c in required_area_cols if c not in stub_area_model_df.columns]
if missing_lattice:
    raise KeyError(f"Missing lattice columns. Run Cell 4 first. Missing: {missing_lattice}")
if missing_union:
    raise KeyError(f"Missing union columns. Run Cell 5 first. Missing: {missing_union}")
if missing_area:
    raise KeyError(f"Missing ROI-derived columns. Run Cell 7 first. Missing: {missing_area}")

lattice_plot_df = pd.concat([
    stub_density_lattice_df[["stub", "porosity_p2p_blend_pct", "density_p2p_blend_per_um2"]].rename(
        columns={"porosity_p2p_blend_pct": "porosity_pct", "density_p2p_blend_per_um2": "density_per_um2"}
    ).assign(method="p2p"),
    stub_density_lattice_df[["stub", "porosity_fwhm_blend_pct", "density_fwhm_blend_per_um2"]].rename(
        columns={"porosity_fwhm_blend_pct": "porosity_pct", "density_fwhm_blend_per_um2": "density_per_um2"}
    ).assign(method="fwhm"),
    stub_density_lattice_df[["stub", "porosity_derivative_blend_pct", "density_derivative_blend_per_um2"]].rename(
        columns={"porosity_derivative_blend_pct": "porosity_pct", "density_derivative_blend_per_um2": "density_per_um2"}
    ).assign(method="derivative"),
] , ignore_index=True)

union_plot_df = pd.concat([
    stub_density_union_df[["stub", "porosity_p2p_union_pct", "density_p2p_union_per_um2"]].rename(
        columns={"porosity_p2p_union_pct": "porosity_pct", "density_p2p_union_per_um2": "density_per_um2"}
    ).assign(method="p2p"),
    stub_density_union_df[["stub", "porosity_fwhm_union_pct", "density_fwhm_union_per_um2"]].rename(
        columns={"porosity_fwhm_union_pct": "porosity_pct", "density_fwhm_union_per_um2": "density_per_um2"}
    ).assign(method="fwhm"),
    stub_density_union_df[["stub", "porosity_derivative_union_pct", "density_derivative_union_per_um2"]].rename(
        columns={"porosity_derivative_union_pct": "porosity_pct", "density_derivative_union_per_um2": "density_per_um2"}
    ).assign(method="derivative"),
] , ignore_index=True)

area_plot_df = pd.concat([
    stub_area_model_df[["stub", "porosity_lattice_from_area_pct", "density_lattice_from_area_per_um2"]].rename(
        columns={
            "porosity_lattice_from_area_pct": "porosity_pct",
            "density_lattice_from_area_per_um2": "density_per_um2",
        }
    ).assign(model="lattice_from_area"),
    stub_area_model_df[["stub", "porosity_union_from_area_pct", "density_union_from_area_per_um2"]].rename(
        columns={
            "porosity_union_from_area_pct": "porosity_pct",
            "density_union_from_area_per_um2": "density_per_um2",
        }
    ).assign(model="union_from_area"),
] , ignore_index=True)

method_order = ["p2p", "fwhm", "derivative"]
model_order = ["lattice_from_area", "union_from_area"]
palette_method = {"p2p": "#1f77b4", "fwhm": "#2ca02c", "derivative": "#ff7f0e"}
palette_model = {
    "lattice_from_area": "#6a4c93",
    "union_from_area": "#1982c4",
}

def _stats_text(df: pd.DataFrame, group_col: str, value_col: str, unit_label: str, order: list[str]) -> str:
    lines = []
    for group in order:
        values = df.loc[df[group_col] == group, value_col].dropna()
        if values.empty:
            lines.append(f"{group}: n/a")
        else:
            lines.append(f"{group}: {values.mean():.2f} +- {values.std(ddof=1):.2f} {unit_label}")
    return "\n".join(lines)

fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=False)

sns.boxplot(
    data=lattice_plot_df, x="method", y="porosity_pct", hue="method",
    order=method_order, hue_order=method_order, palette=palette_method, dodge=False, legend=False, ax=axes[0, 0],
 )
sns.stripplot(data=lattice_plot_df, x="method", y="porosity_pct", order=method_order, color="black", alpha=0.25, size=2.5, jitter=0.25, ax=axes[0, 0])
axes[0, 0].set_title("Lattice (blend): Porosity")
axes[0, 0].set_xlabel("Method")
axes[0, 0].set_ylabel("Porosity (%)")
axes[0, 0].text(
    1.02, 0.98, _stats_text(lattice_plot_df, "method", "porosity_pct", "%", method_order),
    transform=axes[0, 0].transAxes, va="top", ha="left", fontsize=9,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

sns.boxplot(
    data=lattice_plot_df, x="method", y="density_per_um2", hue="method",
    order=method_order, hue_order=method_order, palette=palette_method, dodge=False, legend=False, ax=axes[0, 1],
 )
sns.stripplot(data=lattice_plot_df, x="method", y="density_per_um2", order=method_order, color="black", alpha=0.25, size=2.5, jitter=0.25, ax=axes[0, 1])
axes[0, 1].set_title("Lattice (blend): Density")
axes[0, 1].set_xlabel("Method")
axes[0, 1].set_ylabel("Density (per um^2)")
axes[0, 1].text(
    1.02, 0.98, _stats_text(lattice_plot_df, "method", "density_per_um2", "per um^2", method_order),
    transform=axes[0, 1].transAxes, va="top", ha="left", fontsize=9,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

sns.boxplot(
    data=union_plot_df, x="method", y="porosity_pct", hue="method",
    order=method_order, hue_order=method_order, palette=palette_method, dodge=False, legend=False, ax=axes[1, 0],
 )
sns.stripplot(data=union_plot_df, x="method", y="porosity_pct", order=method_order, color="black", alpha=0.25, size=2.5, jitter=0.25, ax=axes[1, 0])
axes[1, 0].set_title("Union mask: Porosity")
axes[1, 0].set_xlabel("Method")
axes[1, 0].set_ylabel("Porosity (%)")
axes[1, 0].text(
    1.02, 0.98, _stats_text(union_plot_df, "method", "porosity_pct", "%", method_order),
    transform=axes[1, 0].transAxes, va="top", ha="left", fontsize=9,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

sns.boxplot(
    data=union_plot_df, x="method", y="density_per_um2", hue="method",
    order=method_order, hue_order=method_order, palette=palette_method, dodge=False, legend=False, ax=axes[1, 1],
 )
sns.stripplot(data=union_plot_df, x="method", y="density_per_um2", order=method_order, color="black", alpha=0.25, size=2.5, jitter=0.25, ax=axes[1, 1])
axes[1, 1].set_title("Union mask: Density")
axes[1, 1].set_xlabel("Method")
axes[1, 1].set_ylabel("Density (per um^2)")
axes[1, 1].text(
    1.02, 0.98, _stats_text(union_plot_df, "method", "density_per_um2", "per um^2", method_order),
    transform=axes[1, 1].transAxes, va="top", ha="left", fontsize=9,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

sns.boxplot(
    data=area_plot_df, x="model", y="porosity_pct", hue="model",
    order=model_order, hue_order=model_order, palette=palette_model, dodge=False, legend=False, ax=axes[2, 0],
 )
sns.stripplot(data=area_plot_df, x="model", y="porosity_pct", order=model_order, color="black", alpha=0.25, size=2.5, jitter=0.25, ax=axes[2, 0])
axes[2, 0].set_title("ROI-derived models: Porosity")
axes[2, 0].set_xlabel("Model")
axes[2, 0].set_ylabel("Porosity (%)")
axes[2, 0].tick_params(axis="x", rotation=15)
axes[2, 0].text(
    1.02, 0.98, _stats_text(area_plot_df, "model", "porosity_pct", "%", model_order),
    transform=axes[2, 0].transAxes, va="top", ha="left", fontsize=9,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

sns.boxplot(
    data=area_plot_df, x="model", y="density_per_um2", hue="model",
    order=model_order, hue_order=model_order, palette=palette_model, dodge=False, legend=False, ax=axes[2, 1],
 )
sns.stripplot(data=area_plot_df, x="model", y="density_per_um2", order=model_order, color="black", alpha=0.25, size=2.5, jitter=0.25, ax=axes[2, 1])
axes[2, 1].set_title("ROI-derived models: Density")
axes[2, 1].set_xlabel("Model")
axes[2, 1].set_ylabel("Density (per um^2)")
axes[2, 1].tick_params(axis="x", rotation=15)
axes[2, 1].text(
    1.02, 0.98, _stats_text(area_plot_df, "model", "density_per_um2", "per um^2", model_order),
    transform=axes[2, 1].transAxes, va="top", ha="left", fontsize=9,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

for ax in axes.ravel():
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# %%
# Per-stub density and porosity from ROI-derived area-based diameter, for lattice-blend and union-mask models (control only).

from src.lattice_porosity import add_lattice_area_columns, add_lattice_blend_metrics_from_area

if "diameter_area" not in rois_df.columns:
    raise KeyError("rois_df is missing 'diameter_area'.")
if "condition" not in rois_df.columns or "stub" not in rois_df.columns:
    raise KeyError("rois_df must contain 'condition' and 'stub'.")

required_summary_cols = ["stub", "id", "condition", "mean_cluster_neighbor_distance_nm", "mean_neighbor_count"]
missing_summary = [c for c in required_summary_cols if c not in summary_df.columns]
if missing_summary:
    raise KeyError(f"summary_df is missing required columns: {missing_summary}")

union_summary_file = Path("C:/Users/sthui4072/Github/fenestrations/union_masks/union_mask_summary.csv")
if not union_summary_file.exists():
    raise FileNotFoundError(f"Union-mask summary file not found: {union_summary_file}")
union_area_df = pd.read_csv(union_summary_file)

required_union_cols = ["stub", "covered_area_nm2", "covered_area_um2", "coverage_percent"]
missing_union = [c for c in required_union_cols if c not in union_area_df.columns]
if missing_union:
    raise KeyError(f"union_mask_summary.csv is missing required columns: {missing_union}")

def _stub_key(value):
    if pd.isna(value):
        return None
    return "".join(ch for ch in str(value).upper() if ch.isalnum())

summary_control = (
    summary_df[required_summary_cols]
    .dropna(subset=["stub"])
    .query("condition == 'control'")
    .drop_duplicates(subset=["stub"], keep="first")
    .copy()
)
if summary_control.empty:
    raise ValueError("No control rows found in summary_df.")
summary_control["_stub_key"] = summary_control["stub"].map(_stub_key)

rois_control = rois_df.query("condition == 'control'").copy()
rois_control = rois_control.dropna(subset=["stub", "diameter_area"])
rois_control = rois_control[rois_control["diameter_area"] > 0].copy()
if rois_control.empty:
    raise ValueError("No valid control rows with positive diameter_area in rois_df.")

rois_control["_stub_key"] = rois_control["stub"].map(_stub_key)
rois_control["area_from_diameter_nm2"] = np.pi * (rois_control["diameter_area"] / 2.0) ** 2

area_based_stub = (
    rois_control.groupby("_stub_key", as_index=False)
    .agg(
        roi_count=("diameter_area", "size"),
        mean_diameter_area_nm=("diameter_area", "mean"),
        mean_area_from_diameter_nm2=("area_from_diameter_nm2", "mean"),
        total_area_from_diameter_nm2=("area_from_diameter_nm2", "sum"),
    )
)

union_area_df = union_area_df.copy()
union_area_df["_stub_key"] = union_area_df["stub"].map(_stub_key)
union_area_df = union_area_df.drop_duplicates(subset=["_stub_key"], keep="first")

stub_area_model_df = summary_control.merge(area_based_stub, on="_stub_key", how="left")
stub_area_model_df = stub_area_model_df.merge(
    union_area_df[["_stub_key", "covered_area_nm2", "covered_area_um2", "coverage_percent"]],
    on="_stub_key",
    how="left",
)

stub_area_model_df = add_lattice_area_columns(
    stub_area_model_df,
    spacing_col="mean_cluster_neighbor_distance_nm",
    neighbor_count_col="mean_neighbor_count",
)
stub_area_model_df = add_lattice_blend_metrics_from_area(
    stub_area_model_df,
    area_col="mean_area_from_diameter_nm2",
    blend_area_col="cell_area_blend_nm2",
    porosity_col="porosity_lattice_from_area",
    porosity_pct_col="porosity_lattice_from_area_pct",
    density_nm2_col="density_lattice_from_area_per_nm2",
    density_um2_col="density_lattice_from_area_per_um2",
)

stub_area_model_df["porosity_union_from_area"] = (
    stub_area_model_df["total_area_from_diameter_nm2"] / stub_area_model_df["covered_area_nm2"]
)
stub_area_model_df["porosity_union_from_area_pct"] = stub_area_model_df["porosity_union_from_area"] * 100.0
stub_area_model_df["density_union_from_area_per_nm2"] = (
    stub_area_model_df["roi_count"] / stub_area_model_df["covered_area_nm2"]
)
stub_area_model_df["density_union_from_area_per_um2"] = stub_area_model_df["density_union_from_area_per_nm2"] * 1_000_000.0

stub_area_model_df = stub_area_model_df.drop(columns=["_stub_key"])

preview_cols_area_based = [
    "stub",
    "condition",
    "roi_count",
    "mean_diameter_area_nm",
    "porosity_lattice_from_area_pct",
    "density_lattice_from_area_per_um2",
    "covered_area_um2",
    "porosity_union_from_area_pct",
    "density_union_from_area_per_um2",
]

display(stub_area_model_df[preview_cols_area_based].sort_values(["condition", "stub"]).reset_index(drop=True))

# %%
# Retry writing condition-augmented metrics into the original Excel file
metrics_file = Path("C:/Users/sthui4072/Github/fenestrations/stub_profile_metrics_major_minor.xlsx")
if not metrics_file.exists():
    raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

metrics_df = pd.read_excel(metrics_file)
meta_df = pd.read_excel(metafile_path)

if "condition" not in meta_df.columns:
    raise KeyError("Metafile is missing the 'condition' column")
if "stub" not in metrics_df.columns:
    raise KeyError("Metrics file is missing 'stub'")

def _stub_key(value):
    if pd.isna(value):
        return None
    text = str(value).upper()
    return "".join(ch for ch in text if ch.isalnum())

metrics_df["_stub_key"] = metrics_df["stub"].map(_stub_key)
meta_df["_stub_key"] = meta_df["stub"].map(_stub_key) if "stub" in meta_df.columns else None

stub_to_condition = {}
if "stub" in meta_df.columns:
    lookup = meta_df[["_stub_key", "condition"]].dropna(subset=["_stub_key"]).drop_duplicates(subset=["_stub_key"], keep="first")
    stub_to_condition = dict(zip(lookup["_stub_key"], lookup["condition"]))

metrics_df["condition"] = metrics_df["_stub_key"].map(stub_to_condition)

if "id" in metrics_df.columns and "id" in meta_df.columns:
    id_lookup = (
        meta_df[["id", "condition"]]
        .dropna(subset=["id"])
        .drop_duplicates(subset=["id"], keep="first")
        .set_index("id")["condition"]
        .to_dict()
    )
    metrics_df["condition"] = metrics_df["condition"].fillna(metrics_df["id"].map(id_lookup))

metrics_df = metrics_df.drop(columns=["_stub_key"])
if "condition" in metrics_df.columns:
    cols = metrics_df.columns.tolist()
    cols.insert(2, cols.pop(cols.index("condition")))
    metrics_df = metrics_df[cols]

metrics_df.to_excel(metrics_file, index=False)
print(f"Updated original file: {metrics_file}")
print(f"Rows with missing condition: {int(metrics_df['condition'].isna().sum())} / {len(metrics_df)}")
metrics_df.head(10)

# %%
summary_df.head()


# %%
# Remove outliers using robust z-score on log(area)
rois_df_no_outliers, rois_df_outliers = robust_tail_filter_log_area(rois_df, z_thresh=4.5)

print(f"Kept rows: {len(rois_df_no_outliers)}")
print(f"Excluded outliers: {len(rois_df_outliers)}")
print(f"Kept fraction: {len(rois_df_no_outliers)/len(rois_df):.2%}")

# %%

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Diameter area
rois_df["diameter_area"].dropna().plot.hist(bins=np.linspace(0, 110, 51), alpha=0.7, ax=axes[0], color="steelblue")
axes[0].set_title("Diameter from Area")
axes[0].set_xlabel("Diameter (nm)")
axes[0].set_ylabel("Count")

# Neighbor count
rois_df["cluster_neighbor_count"].dropna().plot.hist(bins=np.arange(0, rois_df["cluster_neighbor_count"].max() + 2, 1), alpha=0.7, ax=axes[1], color="coral")
axes[1].set_title("Number of Neighbors")
axes[1].set_xlabel("Neighbor Count")
axes[1].set_ylabel("Count")
axes[1].set_xlim(2, 10)

# Mean neighbor distance
rois_df["cluster_neighbor_distance"].dropna().plot.hist(bins=np.linspace(0, 100, 51), alpha=0.7, ax=axes[2], color="seagreen")
axes[2].set_title("Mean Neighbor Distance")
axes[2].set_xlabel("Distance (nm)")
axes[2].set_ylabel("Count")

for ax in axes:
    sns.despine(ax=ax, offset=5)

plt.tight_layout()
plt.show()

fig.savefig("roi_feature_distributions.png", dpi=300)

# %%
# to split based on condition

rois_control = rois_df_no_outliers.query("condition == 'control'")
rois_fasted = rois_df_no_outliers.query("condition == 'fasted'")
rois_restricted = rois_df_no_outliers.query("condition == 'restricted'")

rois_control.describe()

# %%
rois_control_diameter_area_stats = rois_control["diameter_area"].agg(["mean", "std"])
rois_control_diameter_area_stats.rename(index={"mean": "mean_diameter_area", "std": "std_diameter_area"})

# %%
rois_control.columns

# %%
rois_control_axis_lengths = (
    rois_control.assign(
        major_axis_length_raw_px=rois_control["major_axis_extent"],
        minor_axis_length_raw_px=rois_control["minor_axis_extent"],
        major_axis_length_raw_nm=rois_control["major_axis_extent"] * rois_control["pixel_size"],
        minor_axis_length_raw_nm=rois_control["minor_axis_extent"] * rois_control["pixel_size"],
    )[[
        "id",
        "stub",
        "roi_id",
        "major_axis_length_raw_px",
        "minor_axis_length_raw_px",
        "major_axis_length_raw_nm",
        "minor_axis_length_raw_nm",
    ]]
)

rois_control_axis_lengths

# %%
rois_control_elliptical = rois_control_axis_lengths.assign(
    elliptical_diameter_nm=np.sqrt(
        rois_control_axis_lengths["major_axis_length_raw_nm"]
        * rois_control_axis_lengths["minor_axis_length_raw_nm"]
    )
)

rois_control_elliptical_stats = rois_control_elliptical["elliptical_diameter_nm"].agg(["mean", "std"])
rois_control_elliptical_stats.rename(index={"mean": "mean_elliptical_diameter_nm", "std": "std_elliptical_diameter_nm"})

# %%
summary_per_stub = (
    rois_control.groupby(["id", "stub"], as_index=False)
    .agg(mean_diameter_area_nm=("diameter_area", "mean"))
)
summary_per_stub

# %%
summary_per_id = (
    summary_per_stub.groupby("id", as_index=False)
    .agg(mean_diameter_area_nm=("mean_diameter_area_nm", "mean"))
)
summary_per_id

# %%
#rois for control condition
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Diameter area
rois_control["diameter_area"].dropna().plot.hist(bins=np.linspace(0, 110, 51), alpha=0.7, ax=axes[0], color="steelblue")
axes[0].set_title("Diameter from Area")
axes[0].set_xlabel("Diameter (nm)")
axes[0].set_ylabel("Count")

# Neighbor count
rois_control["cluster_neighbor_count"].dropna().plot.hist(bins=np.arange(0, rois_control["cluster_neighbor_count"].max() + 2, 1), alpha=0.7, ax=axes[1], color="coral")
axes[1].set_title("Number of Neighbors")
axes[1].set_xlabel("Neighbor Count")
axes[1].set_ylabel("Count")
axes[1].set_xlim(2, 10)

# Mean neighbor distance
rois_control["cluster_neighbor_distance"].dropna().plot.hist(bins=np.linspace(0, 100, 51), alpha=0.7, ax=axes[2], color="seagreen")
axes[2].set_title("Mean Neighbor Distance")
axes[2].set_xlabel("Distance (nm)")
axes[2].set_ylabel("Count")

for ax in axes:
    sns.despine(ax=ax, offset=5)

plt.tight_layout()
plt.show()

fig.savefig("roi_feature_distributions.png", dpi=300)

# %%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Common bins
bins_diam = np.linspace(0, 110, 56)
bins_dist = np.linspace(0, 100, 56)
max_neighbors = int(rois_df["cluster_neighbor_count"].max())
bins_neighbors = np.arange(0, max_neighbors + 2, 1)

# --- Diameter (% occurrence per condition) ---
sns.histplot(
    data=rois_control,
    x="diameter_area",
    hue="condition",
    bins=bins_diam,
    multiple="dodge",
    shrink=0.9,
    stat="density",       
    ax=axes[0],
    common_norm=False,   # normalize each group separately
    palette={"control": "steelblue", "fasted": "seagreen", "restricted": "coral"}
)
axes[0].set_title("Diameter")
axes[0].set_xlabel("Diameter (nm)")
axes[0].set_ylabel("Occurrence (%)")

# --- Neighbor Count (raw counts) ---
sns.histplot(
    data=rois_control,
    x="cluster_neighbor_count",
    hue="condition",
    bins=bins_neighbors,
    multiple="dodge",
    shrink=0.8,
    discrete=True,
    stat="percent",
    ax=axes[1],
    common_norm=False,   # normalize each group separately
    palette={"control": "steelblue", "fasted": "seagreen", "restricted": "coral"}
)
axes[1].set_title("Number of Neighbors")
axes[1].set_xlabel("Neighbor Count")
axes[1].set_ylabel("Occurrence (%)")

# --- Mean Neighbor Distance (raw counts) ---
sns.histplot(
    data=rois_control,
    x="cluster_neighbor_distance",
    hue="condition",
    bins=bins_dist,
    multiple="dodge",
    shrink=0.9,
    stat="percent",
    ax=axes[2],
    common_norm=False,   # normalize each group separately
    palette={"control": "steelblue", "fasted": "seagreen", "restricted": "coral"}
)
  
axes[2].set_title("Mean Neighbor Distance")
axes[2].set_xlabel("Distance (nm)")
axes[2].set_ylabel("Occurrence (%)")

plt.tight_layout()
fig.savefig("roi_feature_distributions_grouped_percent.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
# Figure for diameter distribution per condition based on area
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Bins
bins_diam = np.linspace(0, 110, 56)

fig, ax = plt.subplots(figsize=(6,4))

sns.histplot(
    data=rois_df,
    x="diameter_area",
    hue="condition",
    bins=bins_diam,
    multiple="dodge",
    shrink=0.9,
    stat="percent",
    common_norm=False,
    palette={"control": "steelblue",
              "fasted": "seagreen", 
              "restricted": "coral"
            },
    ax=ax
)

#ax.set_xlim(25, 80)
ax.set_xlabel("Diameter (nm)")
ax.set_ylabel("Occurrence (%)")
ax.set_title("Diameter Distribution per Condition")




plt.tight_layout()
fig.savefig(r"D:\figures\diameter_distribution_control.png", bbox_inches="tight", dpi=300)
plt.show()




# %%
# Check normalization of the bins (should add to 100% per group)
bins = np.linspace(0, 110, 56)

# Create a table with counts and percentages per bin per condition
results = []
for cond, group in summary_df.groupby("condition"):
    counts, edges = np.histogram(group["mean_diameter_area_nm"], bins=bins)
    percent = counts / counts.sum() * 100  # percent per group
    centers = (edges[:-1] + edges[1:]) / 2

    df = pd.DataFrame({
        "condition": cond,
        "bin_center": centers,
        "count": counts,
        "percent": percent
    })
    results.append(df)

# Combine all groups
hist_table = pd.concat(results)

# Check percentages sum to 100 per group
print(hist_table.groupby("condition")["percent"].sum())


# %%
summary_df['mean_diameter_major_nm']

# %%
rois_df.columns

# %%
# Figure for diameter based on major/minor axis
# Calculate equivalent diameter in px
rois_df['equivalent_diameter'] = np.sqrt(rois_df['major_axis_length_px'] * rois_df['minor_axis_length_px']) 

# Figure for diameter distribution per condition based on equivalent diameter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Bins
bins_diam = np.linspace(0, 110, 56)

fig, ax = plt.subplots(figsize=(6,4))

sns.histplot(
    data=rois_df,
    x="equivalent_diameter",
    #hue="condition",
    bins=bins_diam,
    multiple="dodge",
    shrink=0.9,
    stat="percent",
    common_norm=False,
    palette={"control": "steelblue",
              "fasted": "seagreen", 
              "restricted": "coral"
            },
    ax=ax
)

#ax.set_xlim(25, 200)
ax.set_xlabel("Diameter (nm)")
ax.set_ylabel("Occurrence (%)")
ax.set_title("Diameter Distribution per Condition")




plt.tight_layout()
#fig.savefig(r"/media/cellpose/T7/figures/diameter_distribution_control_equivalent_diameter.png", bbox_inches="tight", dpi=300)
plt.show()

# %%
rois_control = rois_df.query("condition == 'control'")
rois_fasted = rois_df.query("condition == 'fasted'")
rois_restricted = rois_df.query("condition == 'restricted'")

# %%



# %%
f, ax = plt.subplots(figsize=(6, 4))

sns.kdeplot(rois_control["diameter_area"].dropna(), label="Control",
            ax=ax, 
            # fill=True, alpha=0.5,
            cumulative=True
            )

sns.kdeplot(rois_fasted["diameter_area"].dropna(), label="Fasted",
            ax=ax,
            # fill=True, alpha=0.5,
            cumulative=True
            )

sns.kdeplot(rois_restricted["diameter_area"].dropna(), label="Restricted",
            ax=ax,
            # fill=True, alpha=0.5,
            cumulative=True
            )

ax.legend()

# %%
# Average diameter per animal
summary_per_animal = summary_df.groupby(["id", "condition"], as_index=False)["mean_diameter_area_nm"].mean()

summary_per_animal


# %%
# New DataFrame excluding robust-z outliers (|z| > 3)
z_cutoff = 5

if "robust_z_log_area" not in rois_df.columns:
    log_area = np.log(rois_df["area"].to_numpy())
    median_log = np.median(log_area)
    mad_log = np.median(np.abs(log_area - median_log))
    if mad_log == 0:
        raise ValueError("MAD is zero; robust z-score cannot be computed.")
    robust_sigma = 1.4826 * mad_log
    rois_df["robust_z_log_area" ] = (log_area - median_log) / robust_sigma

rois_df_no_outliers = rois_df[np.abs(rois_df["robust_z_log_area"]) <= z_cutoff].copy()
rois_df_outliers = rois_df[np.abs(rois_df["robust_z_log_area"]) > z_cutoff].copy()

print(f"z cutoff: {z_cutoff}")
print(f"Kept rows: {len(rois_df_no_outliers)}")
print(f"Excluded outliers: {len(rois_df_outliers)}")
print(f"Kept fraction: {len(rois_df_no_outliers)/len(rois_df):.2%}")

# %%
rois_df_no_outliers.diameter_area.describe()

# %%
import matplotlib.pyplot as plt
import tifffile


def _load_masks(seg_path: Path):
    arr = np.load(seg_path, allow_pickle=True)
    payload = arr.item() if arr.ndim == 0 and arr.dtype == object else arr
    if isinstance(payload, dict) and "masks" in payload:
        return payload["masks"]
    return payload


def _resolve_file(root: Path, stub: str, kind: str) -> Path | None:
    stub_path = Path(stub)
    stem = stub_path.stem
    basename = stub_path.name

    if kind == "tif":
        direct_candidates = [
            root / f"{stub}.tif",
            root / f"{stub}.tiff",
            root / f"{basename}.tif",
            root / f"{basename}.tiff",
            root / f"{stem}.tif",
            root / f"{stem}.tiff",
        ]
        patterns = [f"{basename}.tif", f"{basename}.tiff", f"{stem}.tif", f"{stem}.tiff"]
    elif kind == "seg":
        direct_candidates = [
            root / f"{stub}_seg.npy",
            root / f"{basename}_seg.npy",
            root / f"{stem}_seg.npy",
        ]
        patterns = [f"{basename}_seg.npy", f"{stem}_seg.npy"]
    else:
        return None

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]

    return None


def _roi_view_window(roi_mask: np.ndarray, shape: tuple[int, int], pad: int = 80, min_size: int = 180):
    h, w = shape
    ys, xs = np.where(roi_mask)

    if ys.size == 0 or xs.size == 0:
        cx, cy = w // 2, h // 2
        half = max(min_size // 2, 1)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        return x0, x1, y0, y1

    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)

    if (x1 - x0) < min_size:
        cx = (x0 + x1) // 2
        half = min_size // 2
        x0, x1 = max(0, cx - half), min(w, cx + half)
    if (y1 - y0) < min_size:
        cy = (y0 + y1) // 2
        half = min_size // 2
        y0, y1 = max(0, cy - half), min(h, cy + half)

    return x0, x1, y0, y1


n_samples = min(15, len(rois_df_outliers))
if n_samples == 0:
    print("No outliers found in rois_df_outliers.")
else:
    sample_df = rois_df_outliers.sample(n=n_samples, random_state=42).reset_index(drop=True)

    image_root = Path(r"D:\TestData\fenestrations\images")
    seg_root = Path(r"D:\TestData\fenestrations\masks")

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    if not seg_root.exists():
        raise FileNotFoundError(f"Segmentation directory does not exist: {seg_root}")

    print(f"Image root: {image_root}")
    print(f"Segmentation root: {seg_root}")

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()

    missing_tif = 0
    missing_seg = 0
    missing_roi = 0
    plotted = 0

    for i, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes[i]
        stub = str(row["stub"])
        roi_id = int(row["roi_id"])

        tif_path = _resolve_file(image_root, stub, "tif")
        seg_path = _resolve_file(seg_root, stub, "seg")

        if tif_path is None:
            missing_tif += 1
        if seg_path is None:
            missing_seg += 1

        if tif_path is None or seg_path is None:
            ax.set_title(f"Missing file(s)\n{stub}", fontsize=9)
            ax.axis("off")
            continue

        image = tifffile.imread(tif_path)
        if image.ndim > 2:
            image = image[0]

        masks = _load_masks(seg_path)
        roi_mask = masks == roi_id

        ax.imshow(image, cmap="gray")
        ax.contour(roi_mask.astype(float), levels=[0.5], colors="lime", linewidths=2.0)

        x0, x1, y0, y1 = _roi_view_window(roi_mask, image.shape, pad=90, min_size=220)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

        if np.any(roi_mask):
            ax.set_title(f"{Path(stub).name}\nROI {roi_id}", fontsize=9)
            plotted += 1
        else:
            ax.set_title(f"{Path(stub).name}\nROI {roi_id} (not in mask)", fontsize=9)
            missing_roi += 1

        ax.axis("off")

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Plotted outlines: {plotted}/{n_samples}")
    print(f"Missing TIFF: {missing_tif}")
    print(f"Missing segmentation: {missing_seg}")
    print(f"ROI id not present in mask: {missing_roi}")

# %%
summary_df.columns

# %%
plt.scatter(summary_df.mean_diameter_area_nm.values, summary_df.mean_cluster_neighbor_distance_nm.values)

# %%
rois_df_no_outliers.query("condition == 'control'")

# %%
rois_df.experiment.unique()

# %%
rois_df.describe()

# %%
import numpy as np

# Convex-only subset
convex_mask = rois_df["cluster_is_convex"].astype("boolean").fillna(False)
rois_convex = rois_df.loc[convex_mask].copy()

valid = (
    rois_convex["cluster_neighbor_count"].notna()
    & rois_convex["cluster_neighbor_distance"].notna()
    & rois_convex["pixel_size"].notna()
    & rois_convex["diameter_area"].notna()
    & (rois_convex["cluster_neighbor_distance"] > 0)
    & (rois_convex["pixel_size"] > 0)
    & (rois_convex["diameter_area"] > 0)
)
rois_convex = rois_convex.loc[valid].copy()

# Convert spacing to nm (distance is stored in px).
rois_convex["neighbor_distance_nm"] = (
    rois_convex["cluster_neighbor_distance"] * rois_convex["pixel_size"]
)

# Neighbor density: rho = N / (pi * r^2), r in nm.
rois_convex["rho_per_nm2"] = (
    rois_convex["cluster_neighbor_count"]
    / (np.pi * rois_convex["neighbor_distance_nm"] ** 2)
)
rois_convex["rho_per_um2"] = rois_convex["rho_per_nm2"] * 1_000_000

# Fenestration area from diameter (diameter_area is in nm).
rois_convex["fenestration_area_nm2"] = np.pi * (rois_convex["diameter_area"] / 2.0) ** 2

# Porosity (fraction of area occupied by fenestrations):
# phi = N * A_fenestration / (pi * r^2)
rois_convex["porosity_fraction"] = (
    rois_convex["cluster_neighbor_count"] * rois_convex["fenestration_area_nm2"]
) / (np.pi * rois_convex["neighbor_distance_nm"] ** 2)
rois_convex["porosity_percent"] = rois_convex["porosity_fraction"] * 100

rho_porosity_convex_by_condition = (
    rois_convex.groupby("condition", as_index=False)
    .agg(
        n_convex_rois=("cluster_is_convex", "size"),
        mean_neighbor_count=("cluster_neighbor_count", "mean"),
        mean_neighbor_distance_nm=("neighbor_distance_nm", "mean"),
        mean_fenestration_diameter_nm=("diameter_area", "mean"),
        mean_rho_per_um2=("rho_per_um2", "mean"),
        median_rho_per_um2=("rho_per_um2", "median"),
        mean_porosity_fraction=("porosity_fraction", "mean"),
        mean_porosity_percent=("porosity_percent", "mean"),
        median_porosity_percent=("porosity_percent", "median"),
    )
)

# Porosity from means: phi = N * (pi*(d/2)^2) / (pi*r^2)
rho_porosity_convex_by_condition["porosity_from_means_fraction"] = (
    rho_porosity_convex_by_condition["mean_neighbor_count"]
    * (np.pi * (rho_porosity_convex_by_condition["mean_fenestration_diameter_nm"] / 2.0) ** 2)
) / (
    np.pi * rho_porosity_convex_by_condition["mean_neighbor_distance_nm"] ** 2
)
rho_porosity_convex_by_condition["porosity_from_means_percent"] = (
    rho_porosity_convex_by_condition["porosity_from_means_fraction"] * 100
)

rho_porosity_convex_by_condition

# %%
excluded_control_count = (rois_df_outliers["condition"] == "control").sum()
excluded_control_count

# %%
# Exact neighbor count and mean neighbor distance values (with stdev)
pd.set_option("display.precision", 15)

# ROI-level metrics (distance in raw ROI units, typically pixels)
roi_required = ["condition", "cluster_neighbor_count", "cluster_neighbor_distance"]
missing_roi = [c for c in roi_required if c not in rois_df.columns]
if missing_roi:
    raise KeyError(f"rois_df is missing required columns: {missing_roi}")

roi_exact = (
    rois_df.dropna(subset=["condition", "cluster_neighbor_count", "cluster_neighbor_distance"])
    .groupby("condition", as_index=False)
    .agg(
        n_rois=("cluster_neighbor_count", "size"),
        mean_neighbor_count=("cluster_neighbor_count", "mean"),
        std_neighbor_count=("cluster_neighbor_count", "std"),
        mean_neighbor_distance_raw=("cluster_neighbor_distance", "mean"),
        std_neighbor_distance_raw=("cluster_neighbor_distance", "std"),
    )
)

# Stub-level summary metrics (distance already in nm)
summary_required = ["condition", "mean_neighbor_count", "mean_cluster_neighbor_distance_nm"]
missing_summary = [c for c in summary_required if c not in summary_df.columns]
if missing_summary:
    raise KeyError(f"summary_df is missing required columns: {missing_summary}")

summary_exact = (
    summary_df.dropna(subset=summary_required)
    .groupby("condition", as_index=False)
    .agg(
        n_stubs=("mean_neighbor_count", "size"),
        mean_neighbor_count=("mean_neighbor_count", "mean"),
        std_neighbor_count=("mean_neighbor_count", "std"),
        mean_neighbor_distance_nm=("mean_cluster_neighbor_distance_nm", "mean"),
        std_neighbor_distance_nm=("mean_cluster_neighbor_distance_nm", "std"),
    )
)

control_roi_exact = roi_exact.loc[roi_exact["condition"] == "control"].copy()
control_summary_exact = summary_exact.loc[summary_exact["condition"] == "control"].copy()

print("ROI-level exact values by condition (mean and stdev):")
display(roi_exact)

print("Summary-level exact values by condition (mean and stdev):")
display(summary_exact)

print("Control-only exact values (ROI-level):")
display(control_roi_exact)

print("Control-only exact values (Summary-level):")
display(control_summary_exact)

# %%
# Compute ROI circularity from segmentation masks and plot histogram for control group

if "stub" not in rois_df.columns or "roi_id" not in rois_df.columns or "condition" not in rois_df.columns:
    raise KeyError("rois_df must contain 'stub', 'roi_id', and 'condition' columns.")

from pathlib import Path
from typing import Optional

repo_root = Path("C:/Users/sthui4072/Github/fenestrations")
seg_roots = [repo_root / "flatten_npy"]

def _load_masks(seg_path: Path):
    arr = np.load(seg_path, allow_pickle=True)
    payload = arr.item() if arr.ndim == 0 and arr.dtype == object else arr
    if isinstance(payload, dict) and "masks" in payload:
        return payload["masks"]
    return payload

def _stub_name_variants(stub: str) -> list[str]:
    stub_path = Path(str(stub).strip())
    stem = stub_path.stem
    base = stub_path.name

    variants = {base, stem}

    # Handle naming mismatch like FAS24_* vs FAS_24_*.
    stem_upper = stem.upper()
    if stem_upper.startswith("FAS"):
        tail = stem[3:]
        if tail and tail[0].isdigit():
            variants.add(f"FAS_{tail}")
        if tail.startswith("_") and len(tail) > 1:
            variants.add(f"FAS{tail[1:]}")

    return sorted(v for v in variants if v)

def _resolve_seg_file(stub: str) -> Optional[Path]:
    variants = _stub_name_variants(stub)

    for root in seg_roots:
        if not root.exists():
            continue

        for variant in variants:
            direct = root / f"{variant}_seg.npy"
            if direct.exists():
                return direct

        for variant in variants:
            matches = sorted(root.rglob(f"{variant}_seg.npy"))
            if matches:
                return matches[0]
    return None

def _perimeter_4_connected(mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.size == 0 or not m.any():
        return np.nan
    up = np.zeros_like(m, dtype=bool)
    up[1:, :] = m[:-1, :]
    down = np.zeros_like(m, dtype=bool)
    down[:-1, :] = m[1:, :]
    left = np.zeros_like(m, dtype=bool)
    left[:, 1:] = m[:, :-1]
    right = np.zeros_like(m, dtype=bool)
    right[:, :-1] = m[:, 1:]
    boundary_edges = (m & ~up).sum() + (m & ~down).sum() + (m & ~left).sum() + (m & ~right).sum()
    return float(boundary_edges)

circularity_df = rois_df.copy()
circularity_df["area_for_circularity_px2"] = np.nan
circularity_df["perimeter_for_circularity_px"] = np.nan

mask_cache = {}
missing_seg = 0
missing_roi = 0
processed = 0

for idx, row in circularity_df[["stub", "roi_id"]].iterrows():
    stub = row["stub"]
    roi_id = row["roi_id"]
    if pd.isna(stub) or pd.isna(roi_id):
        continue

    stub_key = str(stub)
    if stub_key not in mask_cache:
        seg_path = _resolve_seg_file(stub_key)
        if seg_path is None:
            mask_cache[stub_key] = None
        else:
            try:
                masks = _load_masks(seg_path)
                mask_cache[stub_key] = masks
            except Exception:
                mask_cache[stub_key] = None

    masks = mask_cache[stub_key]
    if masks is None:
        missing_seg += 1
        continue

    roi_mask = masks == int(roi_id)
    if not np.any(roi_mask):
        missing_roi += 1
        continue

    area_px2 = float(np.count_nonzero(roi_mask))
    perimeter_px = _perimeter_4_connected(roi_mask)

    circularity_df.at[idx, "area_for_circularity_px2"] = area_px2
    circularity_df.at[idx, "perimeter_for_circularity_px"] = perimeter_px
    processed += 1

valid = (
    circularity_df["area_for_circularity_px2"].notna()
    & circularity_df["perimeter_for_circularity_px"].notna()
    & (circularity_df["area_for_circularity_px2"] > 0)
    & (circularity_df["perimeter_for_circularity_px"] > 0)
)

circularity_df.loc[valid, "circularity"] = (
    4.0 * np.pi * circularity_df.loc[valid, "area_for_circularity_px2"]
    / (circularity_df.loc[valid, "perimeter_for_circularity_px"] ** 2)
)
circularity_df.loc[~valid, "circularity"] = np.nan

# Keep the circularity in rois_df for downstream use.
rois_df["circularity"] = circularity_df["circularity"]

control_circularity = rois_df.loc[rois_df["condition"] == "control", "circularity"].dropna()
if control_circularity.empty:
    raise ValueError("No valid control circularity values available to plot from masks.")

plt.figure(figsize=(7, 4))
plt.hist(control_circularity, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
plt.xlabel("Circularity (from masks)")
plt.ylabel("Count")
plt.title("Control ROI Circularity Distribution")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

print("Circularity source: ROI masks (area and perimeter in pixels)")
print(f"ROIs processed from masks: {processed}")
print(f"Rows with missing segmentation file: {missing_seg}")
print(f"Rows with ROI id not present in mask: {missing_roi}")
print(control_circularity.describe())

# %%
# List stubs with missing segmentation files (show directly in output, no CSV export)

from pathlib import Path

repo_root = Path("C:/Users/sthui4072/Github/fenestrations")
seg_roots = [repo_root / "flatten_npy"]

def _stub_name_variants(stub: str) -> list[str]:
    stub_path = Path(str(stub).strip())
    stem = stub_path.stem
    base = stub_path.name

    variants = {base, stem}

    # Handle naming mismatch like FAS24_* vs FAS_24_*.
    stem_upper = stem.upper()
    if stem_upper.startswith("FAS"):
        tail = stem[3:]
        if tail and tail[0].isdigit():
            variants.add(f"FAS_{tail}")
        if tail.startswith("_") and len(tail) > 1:
            variants.add(f"FAS{tail[1:]}")

    return sorted(v for v in variants if v)

def _resolve_seg_file_for_report(stub: str):
    variants = _stub_name_variants(stub)

    for root in seg_roots:
        if not root.exists():
            continue

        for variant in variants:
            direct = root / f"{variant}_seg.npy"
            if direct.exists():
                return direct

        for variant in variants:
            matches = sorted(root.rglob(f"{variant}_seg.npy"))
            if matches:
                return matches[0]
    return None

unique_stubs = sorted(rois_df["stub"].dropna().astype(str).unique())
rows = []

for stub in unique_stubs:
    resolved = _resolve_seg_file_for_report(stub)
    if resolved is None:
        variants = _stub_name_variants(stub)
        expected_names = [f"{v}_seg.npy" for v in variants]
        rows.append({
            "stub": stub,
            "expected_name_1": expected_names[0] if len(expected_names) > 0 else None,
            "expected_name_2": expected_names[1] if len(expected_names) > 1 else None,
            "expected_name_3": expected_names[2] if len(expected_names) > 2 else None,
        })

missing_seg_files_df = pd.DataFrame(rows).sort_values("stub").reset_index(drop=True)
missing_stub_list = missing_seg_files_df["stub"].tolist()

print(f"Unique stubs checked: {len(unique_stubs)}")
print(f"Missing segmentation files (unique stubs): {len(missing_seg_files_df)}")
print("\nMissing stubs:")
for i, stub in enumerate(missing_stub_list, start=1):
    print(f"{i:03d}. {stub}")

display(missing_seg_files_df)
