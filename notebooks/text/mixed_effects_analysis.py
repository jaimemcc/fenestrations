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

# %% [markdown]
# # Mixed Effects Model Analysis
#
# Statistical testing of condition differences using linear mixed effects models.
#
# **Model Structure:**
# - Fixed effect: `condition`
# - Random effects: `ID` (animal), `stub` (measurement location)
# - Dependent variables: diameter (area), diameter (profiles), porosity, density

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not available. Install with: pip install statsmodels")

sns.set_theme(style="whitegrid")
print(f"statsmodels available: {HAS_STATSMODELS}")

# %%
# Load data from same sources as main visualization notebook
repo_root = Path("C:/Users/sthui4072/Github/fenestrations")
data_path = repo_root / "roi_data.pickle"
metafile_path = repo_root / "fenestrations_metafile.xlsx"
metrics_file = repo_root / "stub_profile_metrics_major_minor.xlsx"
union_summary_file = repo_root / "union_masks" / "union_mask_summary.csv"

for p in [data_path, metafile_path, metrics_file, union_summary_file]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

data = pd.read_pickle(data_path)
rois_df = data["rois"].copy()
summary_df = data["summary"].copy()
metadata = pd.read_excel(metafile_path)
metrics_df = pd.read_excel(metrics_file)
union_area_df = pd.read_csv(union_summary_file)

def _stub_key(value):
    if pd.isna(value):
        return None
    return "".join(ch for ch in str(value).upper() if ch.isalnum())

# Add condition to metrics_df if needed
if "condition" not in metrics_df.columns:
    meta_lookup = (
        metadata[["stub", "condition"]]
        .dropna(subset=["stub", "condition"])
        .assign(_stub_key=lambda d: d["stub"].map(_stub_key))
        .drop_duplicates(subset=["_stub_key"], keep="first")
        [["_stub_key", "condition"]]
    )
    metrics_df = metrics_df.assign(_stub_key=metrics_df["stub"].map(_stub_key)).merge(
        meta_lookup,
        on="_stub_key",
        how="left"
    )

print(f"Loaded {len(rois_df)} ROI records")
print(f"Loaded {len(metrics_df)} metric records")
print(f"Loaded {len(union_area_df)} union mask records")


# %%
def extract_animal_id(stub):
    """
    Extract animal ID from stub string.
    Stub format: "FAS_XX_ME_YY" -> Animal ID: "FAS_XX"
    """
    if pd.isna(stub):
        return None
    stub_str = str(stub).upper()
    parts = stub_str.split('_')
    # Assuming format is [ANIMAL_PREFIX]_[NUMBER]_ME_[EVENT]
    # Extract [ANIMAL_PREFIX]_[NUMBER]
    if len(parts) >= 3 and 'ME' in parts:
        me_idx = parts.index('ME')
        animal_id = '_'.join(parts[:me_idx])
        return animal_id
    return stub_str

# Test the extraction
sample_stubs = rois_df['stub'].dropna().unique()[:3]
print("Sample stub -> animal_id mappings:")
for stub in sample_stubs:
    aid = extract_animal_id(stub)
    print(f"  {stub} -> {aid}")

# Extract animal IDs
rois_df['animal_id'] = rois_df['stub'].map(extract_animal_id)
metrics_df['animal_id'] = metrics_df['stub'].map(extract_animal_id)
union_area_df['animal_id'] = union_area_df['stub'].map(extract_animal_id)

print(f"\nExtracted {rois_df['animal_id'].nunique()} unique animal IDs")
print(f"Unique animal IDs: {sorted(rois_df['animal_id'].dropna().unique())}")

# %%
# Prepare data for diameter from area analysis
diameter_area_df = rois_df[['animal_id', 'stub', 'condition', 'diameter_area']].dropna().copy()
diameter_area_df = diameter_area_df[diameter_area_df['diameter_area'] > 0]

print(f"Diameter (area) analysis:")
print(f"  Total observations: {len(diameter_area_df)}")
print(f"  Unique animals: {diameter_area_df['animal_id'].nunique()}")
print(f"  Unique stubs: {diameter_area_df['stub'].nunique()}")
print(f"  Conditions: {sorted(diameter_area_df['condition'].unique())}")
print(f"\nSummary by condition:")
print(diameter_area_df.groupby('condition')['diameter_area'].describe())

# %%
# Prepare data for diameter from profiles analysis
method_specs = [
    ("p2p", "major_p2p_nm", "minor_p2p_nm"),
    ("fwhm", "major_fwhm_nm", "minor_fwhm_nm"),
    ("derivative", "major_derivative_nm", "minor_derivative_nm"),
]

profile_rows = []
for method, major_col, minor_col in method_specs:
    tmp = metrics_df[["animal_id", "stub", "condition", major_col, minor_col]].dropna().copy()
    tmp = tmp[(tmp[major_col] > 0) & (tmp[minor_col] > 0)]
    tmp["diameter_nm"] = np.sqrt(tmp[major_col] * tmp[minor_col])
    tmp["method"] = method
    profile_rows.append(tmp[["animal_id", "stub", "condition", "method", "diameter_nm"]])

diameter_profile_df = pd.concat(profile_rows, ignore_index=True)

print(f"Diameter (profile) analysis:")
print(f"  Total observations: {len(diameter_profile_df)}")
print(f"  Unique animals: {diameter_profile_df['animal_id'].nunique()}")
print(f"  Unique stubs: {diameter_profile_df['stub'].nunique()}")
print(f"  Methods: {sorted(diameter_profile_df['method'].unique())}")
print(f"  Conditions: {sorted(diameter_profile_df['condition'].unique())}")
print(f"\nSummary by method and condition:")
print(diameter_profile_df.groupby(['method', 'condition'])['diameter_nm'].describe())

# %%
# Prepare porosity and density metrics
# First, compute porosity metrics for each stub from union masks

# Get summary data grouped by stub to compute porosity
if "stub" in summary_df.columns:
    # Compute union coverage from union_area_df
    union_stub_group = union_area_df.groupby('stub')[['covered_area_nm2']].sum().reset_index()
    
    # Merge with ROI summary for total area
    area_data = rois_df.groupby(['animal_id', 'stub', 'condition']).size().reset_index(name='count')
    
    # Compute porosity as ratio
    porosity_df = area_data[["animal_id", "stub", "condition"]].drop_duplicates().copy()
    
    # Add union coverage
    porosity_df = porosity_df.merge(
        union_stub_group.rename(columns={'covered_area_nm2': 'union_area_nm2'}),
        on='stub',
        how='left'
    )
    
    print(f"Porosity analysis:")
    print(f"  Stub-level observations: {len(porosity_df)}")
    print(f"  Unique animals: {porosity_df['animal_id'].nunique()}")
    print(f"  Conditions: {sorted(porosity_df['condition'].unique())}")
    print(f"\nSummary by condition:")
    print(porosity_df.groupby('condition')['union_area_nm2'].describe())
else:
    print("WARNING: Creates simplified analysis for now; porosity computation may need adjustment")
    porosity_df = None

# %%
if HAS_STATSMODELS:
    print("="*70)
    print("LINEAR MIXED EFFECTS MODEL: Diameter from Area")
    print("="*70)
    
    # Ensure no NaN in key columns
    model_df = diameter_area_df[['animal_id', 'stub', 'condition', 'diameter_area']].dropna().copy()
    model_df['animal_stub'] = model_df['animal_id'].astype(str) + ':' + model_df['stub'].astype(str)
    
    # Target model:
    # diameter_area ~ condition + (1 | animal_id) + (1 | animal_id:stub)
    # statsmodels equivalent uses groups + vc_formula for nested random intercepts
    model = smf.mixedlm(
        "diameter_area ~ C(condition)",
        data=model_df,
        groups=model_df["animal_id"],
        re_formula="~1",
        vc_formula={"animal_stub": "0 + C(animal_stub)"}
    )
    
    result = model.fit(reml=True)
    
    print("\nModel Summary:")
    print(result.summary())
    
    print("\nRandom Effects (animal_id):")
    print(result.random_effects)
else:
    print("statsmodels not available - skipping mixed effects modeling")

# %%
if HAS_STATSMODELS:
    print("\n" + "="*70)
    print("LINEAR MIXED EFFECTS MODEL: Diameter from Derivative Method")
    print("="*70)
    
    # Filter to derivative method only
    model_df_profile = diameter_profile_df[
        (diameter_profile_df['method'] == 'derivative')
    ][['animal_id', 'stub', 'condition', 'diameter_nm']].dropna().copy()
    
    # Fit model with condition only
    model_profile = smf.mixedlm(
        "diameter_nm ~ C(condition)",
        data=model_df_profile,
        groups=model_df_profile["animal_id"],
        re_formula="~1"
    )
    result_profile = model_profile.fit(reml=True)
    
    print("\nModel Summary:")
    print(result_profile.summary())
else:
    print("statsmodels not available")

# %%
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS BY CONDITION")
print("="*70)

print("\n--- Diameter from Area ---")
for condition in sorted(diameter_area_df['condition'].unique()):
    data = diameter_area_df[diameter_area_df['condition'] == condition]['diameter_area']
    print(f"\n{condition.upper()}:")
    print(f"  n = {len(data)}")
    print(f"  Mean ± SD: {data.mean():.2f} ± {data.std():.2f}")
    print(f"  Median [IQR]: {data.median():.2f} [{data.quantile(0.25):.2f} - {data.quantile(0.75):.2f}]")
    print(f"  Range: {data.min():.2f} - {data.max():.2f}")

print("\n--- Diameter from Profiles ---")
for method in sorted(diameter_profile_df['method'].unique()):
    print(f"\n{method.upper()} Method:")
    for condition in sorted(diameter_profile_df['condition'].unique()):
        data = diameter_profile_df[
            (diameter_profile_df['method'] == method) & 
            (diameter_profile_df['condition'] == condition)
        ]['diameter_nm']
        print(f"  {condition}: Mean={data.mean():.2f} ± {data.std():.2f} (n={len(data)})")

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Diameter area by condition
sns.boxplot(data=diameter_area_df, x='condition', y='diameter_area', ax=axes[0, 0])
axes[0, 0].set_title('Diameter from Area by Condition')
axes[0, 0].set_ylabel('Diameter (nm)')

# Diameter profile by condition and method
sns.boxplot(data=diameter_profile_df, x='condition', y='diameter_nm', hue='method', ax=axes[0, 1])
axes[0, 1].set_title('Diameter from Profiles by Condition and Method')
axes[0, 1].set_ylabel('Diameter (nm)')
axes[0, 1].legend(title='Method', loc='upper right')

# Violin plot for area
sns.violinplot(data=diameter_area_df, x='condition', y='diameter_area', ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Diameter from Area')
axes[1, 0].set_ylabel('Diameter (nm)')

# Strip plot with jitter
sns.stripplot(data=diameter_area_df, x='condition', y='diameter_area', 
              jitter=True, alpha=0.5, ax=axes[1, 1])
axes[1, 1].set_title('Individual Observations: Diameter from Area')
axes[1, 1].set_ylabel('Diameter (nm)')

plt.tight_layout()
plt.show()

print("Visualization complete")

# %%
# Create summary statistics table
summary_area = diameter_area_df.groupby('condition')['diameter_area'].agg([
    ('n', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('Min', 'min'),
    ('Median', 'median'),
    ('Max', 'max')
]).round(2)

print("\nDiameter Area - Summary Statistics:")
print(summary_area)

# Summary for profile diameters
summary_profile = diameter_profile_df.groupby(['method', 'condition'])['diameter_nm'].agg([
    ('n', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('Min', 'min'),
    ('Median', 'median'),
    ('Max', 'max')
]).round(2)

print("\nDiameter Profile - Summary Statistics:")
print(summary_profile)

# Save to CSV
output_dir = Path("C:/Users/sthui4072/Github/fenestrations/results")
output_dir.mkdir(exist_ok=True)

summary_area.to_csv(output_dir / "diameter_area_summary.csv")
summary_profile.to_csv(output_dir / "diameter_profile_summary.csv")
diameter_area_df.to_csv(output_dir / "diameter_area_analysis.csv", index=False)
diameter_profile_df.to_csv(output_dir / "diameter_profile_analysis.csv", index=False)

print(f"\nResults saved to {output_dir}")

# %% [markdown]
# ## Model Details
#
# ### Linear Mixed Effects Model Specification
#
# **Diameter from Area:**
# ```
# diameter_area ~ C(condition) + (1 | animal_id) + (1 | animal_id:stub)
# ```
#
# **Diameter from Derivative Method:**
# ```
# diameter_nm ~ C(condition) + (1 | animal_id)
# ```
#
# ### Interpretation
#
# - **Fixed effects**: Condition (control, fasted, protein restricted) coefficients show mean differences relative to the reference level
# - **Random intercepts**:
#   - `animal_id` captures between-animal variability
#   - `animal_id:stub` captures additional within-animal, between-stub variability
# - **REML estimation**: Restricted maximum likelihood for unbiased variance estimates
# - **Model summary**: Provides parameter estimates (fixed effects), standard errors, t-values, and p-values for individual parameters
# - **NOTE**: ROIs are not independent (clustered within stubs and animals), so overall significance tests (ANOVA/LRT) are not appropriate. Instead, interpret individual parameter estimates and their confidence intervals from the model summary.
#
# ### Output Files
#
# - `diameter_area_summary.csv`: Descriptive statistics for area-based diameter
# - `diameter_profile_summary.csv`: Descriptive statistics for profile-based diameter
# - `diameter_area_analysis.csv`: Full analysis data for diameter from area
# - `diameter_profile_analysis.csv`: Full analysis data for diameter from profiles
