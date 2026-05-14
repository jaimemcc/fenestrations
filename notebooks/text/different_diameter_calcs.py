# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
# ---

# %% [markdown]
# # Different Diameter Calculation Methods
#
# This notebook explores alternative methods for calculating ROI diameter from intensity profiles:
# - **FWHM (Full Width Half Maximum)**: Based on the width of the dip at 50% of the baseline-to-minimum depth
# - **Derivative**: Uses Savitzky-Golay filtering and derivative to find steep flanks of the central dip
# - **Baseline**: Identifies the width of the region below an estimated baseline profile

# %%
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# %% [markdown]
# ## Helper Functions for Diameter Estimation

# %%
from src.diameter_calcs import (
    _compute_baseline_dip_width,
    _compute_dip_width,
    _compute_dip_width_derivative,
    compute_profiles_df,
)

# %%
# Moved to src.diameter_calcs

# %%
# Moved to src.diameter_calcs

# %% [markdown]
# ## Alternative Diameter Calculation from Profiles
#
# These were the original methods used in `compute_profiles_df()` before consolidating on the principal axes profile-based diameter calculation.

# %%
# Moved to src.diameter_calcs

# %% [markdown]
# ## Notes on Method Comparison
#
# These three methods represent different approaches to extracting diameter information from intensity profiles:
#
# 1. **FWHM (Full Width Half Maximum)**:
#    - Classic approach from spectroscopy
#    - Finds width at 50% of dip depth
#    - Robust but may not capture full feature extent
#
# 2. **Derivative**:
#    - Identifies steepest slopes on either side of dip
#    - Sensitive to noise but captures feature boundaries well
#    - Uses Savitzky-Golay smoothing for stability
#
# 3. **Baseline**:
#    - Estimates linear baseline between edges
#    - Finds maximum continuous region below baseline
#    - More conservative, captures the deepest part of the dip
#
# The current implementation uses **principal axes profiles** with a simpler edge-based diameter method
# that focuses on profile half-width maxima, which has shown better correlation with fenestration geometry.
