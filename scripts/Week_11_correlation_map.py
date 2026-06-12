"""
new-file.py — Gamma-SSI Drought Analysis Pipeline
==================================================
Full pipeline from raw data to daily SSI values for African maize districts.

STEPS:
  1. Load GADM districts + GEOGLAM V1.4 crop calendar shapefiles
  2. Filter GEOGLAM to Maize 1; spatial-join with districts via centroids
  3. Calculate risk window per district (skip first 25% of season)
  4. Load ERA5 daily soil moisture; merge with district risk windows
  5. Compute daily SSI via Gamma-CDF transformation per (district, DOY)
  6. Count drought days per district per crop-year

METHOD — Gamma-CDF SSI (per AghaKouchak 2014, McKee et al. 1993):
  For each (district, day-of-year):
    1. Collect all years of daily soil moisture for that DOY
    2. Fit a Gamma distribution (shape α, scale β) to non-zero values
    3. For each observation: p = q_zero + (1 − q_zero) × Gamma_CDF(x; α, β)
    4. SSI = Φ⁻¹(p)  (inverse standard normal)

  This ensures SSI = −1.5 universally means "6.7th percentile event"
  regardless of the local soil moisture distribution shape.

References:
  - McKee, Doesken & Kleist (1993), 8th Conf. Applied Climatology
  - AghaKouchak (2014), HESS, 18(7), 2515–2526
  - GEOGLAM CM4EW Calendars V1.4 (Zenodo)
  - GADM Agricultural Domain 2019
"""

import os
import json
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import stats as scipy_stats
import plotly.express as px

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up two levels (from scripts -> dissertation_work -> Africa-Drought-Study)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
# Allow overriding the outputs root via env var; default to <project>/outputs
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_ROOT = os.environ.get(
  "OUTPUT_ROOT",
  os.path.join(PROJECT_ROOT, "dissertation_work", "outputs")
)
# Create a subfolder named after this script (e.g., outputs/new-file)
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GADM agricultural districts shapefile (polygon boundaries for ~3,800 districts)
GADM_PATH = os.path.join(DATA_DIR, "africa_agricultural_domain_2019.shp")

# GEOGLAM V1.4 crop calendar shapefile (polygons with planting/harvest DOY per crop)
GEOGLAM_PATH = os.path.join(DATA_DIR,"GEOGLAM_CM4EW_Calendars_V1.4.shp")

# ERA5-Land daily soil moisture CSV (one row per district per day, 2000–2026)
ERA5_PATH = os.path.join(DATA_DIR, "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv")

# SSI threshold: −1.0 = onset of "Moderate Drought" per WMO (2012) / McKee et al. (1993)
# This corresponds to the ~15.9th percentile (1-in-6-year dryness)
SSI_THRESHOLD = -1.0

# Minimum observations per (district, DOY) to fit a Gamma distribution reliably
MIN_OBS_FOR_FIT = 10

# Pixel-overlap quality filtering (optional — skips if file not found)
# Run utils/overlapping_pixel_solver.py first to generate this file
QUALITY_PATH = os.path.join(DATA_DIR, "..", "outputs", "overlap_analysis", "districts_with_quality.geojson")
MIN_QUALITY = os.environ.get("MIN_QUALITY", None)  # None = no filter, "Medium" or "High"


# ---------------------------------------------------------------------------
# STEP 0: Optional pixel-overlap quality filter
# ---------------------------------------------------------------------------
def filter_districts_by_quality(gdf_districts, min_quality="Medium"):
    """
    Filter districts by pixel-overlap quality class.

    Loads the GeoJSON produced by utils/overlapping_pixel_solver.py.
    Districts below the minimum quality threshold are excluded from
    subsequent analysis. If the quality file is not found, all districts
    are kept and a warning is printed.

    Quality classes:
        High   → >= 80% of pixels fully contained in district
        Medium → >= 50% but < 80%
        Low    -> <  50%

    Reference: Fisher & Langford (1995) — area-weighted areal interpolation
    """
    print("\n" + "=" * 70)
    print("STEP 0: Optional pixel-overlap quality filter")
    print("=" * 70)

    if not os.path.exists(QUALITY_PATH):
        print(f"\n  ⚠ Quality file not found: {QUALITY_PATH}")
        print(f"  → Skipping quality filter (all {len(gdf_districts)} districts kept)")
        print(f"  → Run utils/overlapping_pixel_solver.py to generate quality data")
        # Add placeholder columns for downstream compatibility
        gdf_districts["quality_class"] = "Unknown"
        gdf_districts["pct_full_pixels"] = np.nan
        return gdf_districts

    print(f"\n  Loading quality metrics from: {QUALITY_PATH}")
    quality_gdf = gpd.read_file(QUALITY_PATH)
    quality = quality_gdf[["ADM_NAME", "quality_class", "pct_full_pixels"]]

    # Merge quality with districts
    gdf_districts["ADM_NAME"] = gdf_districts["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_districts = gdf_districts.merge(quality, on="ADM_NAME", how="left")

    # Fill NaN quality for districts not in the solver output
    gdf_districts["quality_class"] = gdf_districts["quality_class"].fillna("Unknown")
    gdf_districts["pct_full_pixels"] = gdf_districts["pct_full_pixels"].fillna(np.nan)

    # Report distribution
    print(f"\n  Quality distribution before filtering:")
    for qc in ["High", "Medium", "Low", "Unknown"]:
        count = (gdf_districts["quality_class"] == qc).sum()
        if count > 0:
            print(f"    {qc:<10s}: {count} districts")

    # Apply filter
    # Districts with no quality data (Unknown) are kept by default
    quality_order = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 2}
    min_val = quality_order.get(min_quality, 1)

    before_counts = gdf_districts["quality_class"].value_counts()
    mask = gdf_districts["quality_class"].map(quality_order) >= min_val
    n_before = len(gdf_districts)
    gdf_districts = gdf_districts[mask].copy()
    n_after = len(gdf_districts)

    print(f"\n  Filter: keep ≥ {min_quality} quality")
    print(f"    Before: {n_before} districts")
    print(f"    After:  {n_after} districts ({100*n_after/n_before:.1f}%)")
    print(f"    Removed: {n_before - n_after} districts")
    if n_before != n_after:
        print(f"\n  Excluded districts by class:")
        after_counts = gdf_districts["quality_class"].value_counts()
        for qc in ["Low", "Medium", "High", "Unknown"]:
            removed = before_counts.get(qc, 0) - after_counts.get(qc, 0)
            if removed > 0:
                print(f"    Removed {int(removed)} {qc} districts")

    return gdf_districts


# ---------------------------------------------------------------------------
# STEP 1: Load shapefiles
# ---------------------------------------------------------------------------
def load_shapefiles():
    """
    Load both shapefiles and standardise the district name column.

    Returns:
        gdf_districts: GeoDataFrame of GADM agricultural districts
        gdf_geoglam:   GeoDataFrame of GEOGLAM crop calendar polygons
    """
    print("=" * 70)
    print("STEP 1: Loading shapefiles")
    print("=" * 70)

    # Load GADM districts
    print(f"\n  Loading GADM districts from:\n    {GADM_PATH}")
    gdf_districts = gpd.read_file(GADM_PATH)

    # Standardise district names to uppercase (for consistent matching later)
    gdf_districts["ADM_NAME"] = (
        gdf_districts["ADM_NAME"].astype(str).str.strip().str.upper()
    )
    print(f"  → {len(gdf_districts)} district polygons loaded")
    print(f"  → CRS: {gdf_districts.crs}")
    print(f"  → Columns: {gdf_districts.columns.tolist()}")

    # Load GEOGLAM crop calendar
    print(f"\n  Loading GEOGLAM V1.4 from:\n    {GEOGLAM_PATH}")
    gdf_geoglam = gpd.read_file(GEOGLAM_PATH)
    print(f"  → {len(gdf_geoglam)} calendar polygons loaded")
    print(f"  → CRS: {gdf_geoglam.crs}")
    print(f"  → Crops available: {sorted(gdf_geoglam['crop'].unique())}")

    return gdf_districts, gdf_geoglam


# ---------------------------------------------------------------------------
# STEP 2: Filter GEOGLAM to Maize 1 only
# ---------------------------------------------------------------------------
def filter_maize1(gdf_geoglam):
    """
    Filter the GEOGLAM calendar to only 'Maize 1' entries.

    GEOGLAM contains calendars for many crops (Wheat, Rice, Maize 1, Maize 2, etc.).
    We isolate Maize 1 because it is the primary maize growing season across Africa.

    Returns:
        gdf_maize1: GeoDataFrame containing only Maize 1 calendar polygons
    """
    print("\n" + "=" * 70)
    print("STEP 2: Filtering GEOGLAM to Maize 1")
    print("=" * 70)

    gdf_maize1 = gdf_geoglam[gdf_geoglam["crop"] == "Maize 1"].copy()

    print(f"\n  Total GEOGLAM polygons: {len(gdf_geoglam)}")
    print(f"  Maize 1 polygons:      {len(gdf_maize1)}")
    print(f"\n  Maize 1 calendar columns:")
    print(f"    planting    — DOY when planting begins (1–365)")
    print(f"    vegetative  — DOY when vegetative growth begins")
    print(f"    harvest     — DOY when harvest begins")
    print(f"    endofseaso  — DOY when season ends")
    print(f"\n  Countries covered by Maize 1:")
    countries = sorted(gdf_maize1["country"].unique())
    for c in countries:
        n_regions = gdf_maize1[gdf_maize1["country"] == c]["region"].nunique()
        print(f"    • {c} ({n_regions} regions)")

    return gdf_maize1


# ---------------------------------------------------------------------------
# STEP 3: Compute district centroids and spatial join with Maize 1
# ---------------------------------------------------------------------------
def join_districts_to_maize1(gdf_districts, gdf_maize1):
    """
    Spatially join GADM districts to Maize 1 calendar zones using centroids.

    Why centroids?
      - District polygons may overlap multiple GEOGLAM zones
      - Using the centroid gives a single, deterministic assignment
      - Centroid is computed in EPSG:3857 (metric) for accuracy, then
        reprojected back to WGS84 (EPSG:4326) for the spatial join

    After the join, we filter to districts with VALID calendar data:
      - planting > 0  (zero means "no data" in GEOGLAM)
      - harvest > 0

    Returns:
        valid_maize: GeoDataFrame of districts with valid Maize 1 calendar data
    """
    print("\n" + "=" * 70)
    print("STEP 3: Spatial join — districts to Maize 1 calendar via centroids")
    print("=" * 70)

    # Compute centroids in a projected CRS (metres) for geometric accuracy
    print("\n  Computing district centroids (EPSG:3857 → EPSG:4326) …")
    gdf_centroids = gdf_districts.to_crs(epsg=3857).copy()
    gdf_centroids["geometry"] = gdf_centroids.centroid
    gdf_centroids = gdf_centroids.to_crs(epsg=4326)

    # Ensure GEOGLAM is also in WGS84
    if gdf_maize1.crs != gdf_centroids.crs:
        gdf_maize1 = gdf_maize1.to_crs(epsg=4326)

    # Spatial join: find which Maize 1 polygon contains each district centroid
    print("  Performing spatial join (centroid 'within' Maize 1 polygon) …")
    districts_maize = gpd.sjoin(
        gdf_centroids, gdf_maize1, how="inner", predicate="within"
    )

    print(f"\n  Results:")
    print(f"    Total GADM districts:               {len(gdf_districts)}")
    print(f"    Districts matched to Maize 1:       {len(districts_maize)}")
    print(f"    Districts NOT matched (no Maize 1): {len(gdf_districts) - len(districts_maize)}")

    # Filter out districts where GEOGLAM has placeholder zeros
    valid_maize = districts_maize[
        (districts_maize["planting"] > 0) & (districts_maize["harvest"] > 0)
    ].copy()

    n_invalid = len(districts_maize) - len(valid_maize)
    print(f"\n    Districts with valid planting/harvest dates: {len(valid_maize)}")
    print(f"    Districts with zero (placeholder) dates:     {n_invalid}")

    # Summary of calendar dates
    print(f"\n  Calendar date summary (DOY) for valid districts:")
    print(f"    Planting    — min: {valid_maize['planting'].min()}, "
          f"max: {valid_maize['planting'].max()}, "
          f"mean: {valid_maize['planting'].mean():.0f}")
    print(f"    Harvest     — min: {valid_maize['harvest'].min()}, "
          f"max: {valid_maize['harvest'].max()}, "
          f"mean: {valid_maize['harvest'].mean():.0f}")
    print(f"    End-of-season — min: {valid_maize['endofseaso'].min()}, "
          f"max: {valid_maize['endofseaso'].max()}, "
          f"mean: {valid_maize['endofseaso'].mean():.0f}")

    # Show a few example districts
    print(f"\n  Sample districts (first 10):")
    sample_cols = ["ADM_NAME", "COUNTRY", "country", "region", "planting", "harvest", "endofseaso"]
    available_cols = [c for c in sample_cols if c in valid_maize.columns]
    print(valid_maize[available_cols].head(10).to_string(index=False))

    return valid_maize


# ---------------------------------------------------------------------------
# STEP 4: Calculate risk window for each district
# ---------------------------------------------------------------------------
def calculate_risk_windows(valid_maize):
    """
    For each district, compute the RISK WINDOW within the growing season.

    Why skip the first 25%?
      - The early season (planting → early vegetative) is when crops are
        establishing roots. Soil moisture stress during this phase is less
        impactful on final yield than stress during flowering/grain-fill.
      - The remaining 75% of the season (vegetative → end-of-season) is
        the "risk window" where drought has the most impact on yield.

    How it works:
      - Season duration = endofseaso − planting (handles year-wrap)
      - Skip = 25% of duration
      - risk_start_doy = planting + skip (mod 365)
      - risk_end_doy = endofseaso

    Returns:
        valid_maize with two new columns: risk_start_doy, risk_end_doy
    """
    print("\n" + "=" * 70)
    print("STEP 4: Calculating risk windows (skip first 25% of season)")
    print("=" * 70)

    valid_maize = valid_maize.copy()

    def _calc_risk(row):
        p = row["planting"]
        h = row["endofseaso"]

        # Season duration, handling year-wrap (e.g., planting in Nov, harvest in Apr)
        duration = (h - p) if h >= p else (365 - p) + h

        # Skip first 25% of season
        delay = duration * 0.25
        risk_start = int(round((p + delay) % 365))
        risk_end = h

        return pd.Series([risk_start, risk_end], index=["risk_start_doy", "risk_end_doy"])

    valid_maize[["risk_start_doy", "risk_end_doy"]] = valid_maize.apply(_calc_risk, axis=1)

    # Summary
    print(f"\n  Risk windows computed for {len(valid_maize)} districts")
    print(f"\n  Example (first 5):")
    print(valid_maize[["ADM_NAME", "planting", "endofseaso", "risk_start_doy", "risk_end_doy"]].head().to_string(index=False))

    # Check for year-wrapped seasons
    n_wrapped = (valid_maize["risk_start_doy"] > valid_maize["risk_end_doy"]).sum()
    n_standard = len(valid_maize) - n_wrapped
    print(f"\n  Standard seasons (start ≤ end): {n_standard}")
    print(f"  Year-wrapped seasons (start > end, e.g. Nov→Apr): {n_wrapped}")

    return valid_maize


# ---------------------------------------------------------------------------
# STEP 5: Load ERA5 and merge with district risk windows
# ---------------------------------------------------------------------------
def load_and_merge_era5(valid_maize):
    """
    Load the ERA5-Land daily soil moisture CSV and merge with district
    risk windows. Keep only rows where the day-of-year falls WITHIN
    the district's risk window.

    Also assigns crop_year:
      - For standard seasons (e.g. May→Oct): crop_year = calendar year
      - For year-wrapped seasons (e.g. Nov→Apr): days in Jan–Apr are
        assigned to the PREVIOUS year's crop season

    Returns:
        df_filtered: DataFrame with columns [feature_id, year, doy, crop_year,
                     volumetric_soil_water_layer_2, risk_start_doy, risk_end_doy]
    """
    print("\n" + "=" * 70)
    print("STEP 5: Loading ERA5 soil moisture & merging with risk windows")
    print("=" * 70)

    # Load ERA5 CSV (with memory-optimized dtypes)
    print(f"\n  Loading ERA5 from:\n    {ERA5_PATH}")
    dtype = {
        "year": "int16", "month": "int8", "day": "int8", "doy": "int16",
        "volumetric_soil_water_layer_2": "float32",
    }
    df_era5 = pd.read_csv(ERA5_PATH, dtype=dtype)
    df_era5["feature_id"] = df_era5["feature_id"].astype(str).str.strip().str.upper()
    print(f"  → {len(df_era5):,} rows, {df_era5['feature_id'].nunique()} districts")
    print(f"  → Year range: {df_era5['year'].min()}–{df_era5['year'].max()}")
    print(f"  → Columns: {df_era5.columns.tolist()}")

    # Prepare the lookup table from valid_maize
    valid_maize_lookup = valid_maize.copy()
    valid_maize_lookup["ADM_NAME"] = valid_maize_lookup["ADM_NAME"].astype(str).str.strip().str.upper()

    # Merge: each ERA5 row gets its district's risk window dates
    print(f"\n  Merging ERA5 rows with {len(valid_maize_lookup)} district risk windows …")
    df_merged = df_era5.merge(
        valid_maize_lookup[["ADM_NAME", "risk_start_doy", "risk_end_doy"]],
        left_on="feature_id",
        right_on="ADM_NAME",
        how="inner",
    )
    print(f"  → Rows after merge (districts with Maize 1 calendar): {len(df_merged):,}")

    # Filter: keep only days that fall within the risk window
    # Two cases: standard season (start ≤ end) and wrapped season (start > end)
    print("\n  Filtering to days within risk window …")

    mask_standard = (
        (df_merged["risk_start_doy"] <= df_merged["risk_end_doy"])
        & (df_merged["doy"] >= df_merged["risk_start_doy"])
        & (df_merged["doy"] <= df_merged["risk_end_doy"])
    )
    mask_wrapped = (
        (df_merged["risk_start_doy"] > df_merged["risk_end_doy"])
        & (
            (df_merged["doy"] >= df_merged["risk_start_doy"])
            | (df_merged["doy"] <= df_merged["risk_end_doy"])
        )
    )
    df_filtered = df_merged[mask_standard | mask_wrapped].copy()
    print(f"  → Rows within risk window: {len(df_filtered):,}")
    print(f"    (Dropped {len(df_merged) - len(df_filtered):,} out-of-season rows)")

    # Assign crop_year
    # For wrapped seasons: days early in the calendar year (Jan–Apr) belong
    # to the PREVIOUS year's planting season
    df_filtered["crop_year"] = df_filtered["year"].copy()
    wrapped_mask = df_filtered["risk_start_doy"] > df_filtered["risk_end_doy"]
    early_in_year = df_filtered["doy"] <= df_filtered["risk_end_doy"]
    df_filtered.loc[wrapped_mask & early_in_year, "crop_year"] = (
        df_filtered.loc[wrapped_mask & early_in_year, "year"] - 1
    )

    print(f"\n  Crop year assigned:")
    print(f"    Year range in data: {df_filtered['year'].min()}–{df_filtered['year'].max()}")
    print(f"    Crop-year range:    {df_filtered['crop_year'].min()}–{df_filtered['crop_year'].max()}")
    print(f"    Districts:          {df_filtered['feature_id'].nunique()}")

    return df_filtered


# ---------------------------------------------------------------------------
# STEP 6: Compute daily SSI via Gamma-CDF transformation
# ---------------------------------------------------------------------------
def compute_daily_gamma_ssi(df_filtered):
    """
    Compute the Standardised Soil-moisture Index (SSI) for each daily
    observation using a Gamma-CDF transformation.

    METHOD (per AghaKouchak 2014, McKee et al. 1993):
      For each unique (district, day-of-year) combination:
        1. Gather ALL years of soil moisture values for that DOY
           (this is the full 25-year record — no separate "baseline")
        2. Separate into zero and non-zero values
        3. Fit a Gamma distribution to the non-zero values:
             Gamma(α, β) where α = shape, β = scale
        4. For each observed value x:
             If x ≤ 0:  p = q_zero / 2  (probability mass at zero)
             If x > 0:  p = q_zero + (1 − q_zero) × Gamma_CDF(x; α, β)
        5. Transform to standard normal:
             SSI = Φ⁻¹(p)  (inverse normal CDF)

    WHY Gamma?
      - Soil moisture is bounded below by 0, often right-skewed
      - Gamma naturally models non-negative, skewed data
      - This is the same approach used for SPI (precipitation) since 1993

    WHY this gives universal meaning to SSI values:
      - SSI = 0   → median conditions (50th percentile)
      - SSI = −1  → ~15.9th percentile (moderate drought onset per WMO)
      - SSI = −1.5 → ~6.7th percentile (severe drought per WMO)
      - SSI = −2  → ~2.3rd percentile (extreme drought per WMO)
      These percentile meanings hold for EVERY district regardless of
      its local climate, because we fitted the distribution to each
      district's own data.

    Returns:
        df_filtered with a new 'SSI' column (daily SSI values)
    """
    print("\n" + "=" * 70)
    print("STEP 6: Computing daily Gamma-CDF SSI")
    print("=" * 70)
    print(f"\n  Method: Gamma distribution fitted per (district, DOY)")
    print(f"  Baseline: Full record (all available years for each district)")
    print(f"  Min observations for reliable fit: {MIN_OBS_FOR_FIT}")

    df_filtered = df_filtered.copy()
    df_filtered["SSI"] = np.nan

    # Group by (district, DOY) — each group gets its own Gamma fit
    groups = df_filtered.groupby(["feature_id", "doy"])
    n_groups = len(groups)
    n_fitted = 0
    n_skipped = 0

    print(f"  Total (district × DOY) groups to fit: {n_groups:,}")
    print(f"\n  Fitting Gamma distributions …")

    for (district, doy), group in groups:
        values = group["volumetric_soil_water_layer_2"].dropna().values

        # Skip if not enough observations for a reliable fit
        if len(values) < MIN_OBS_FOR_FIT:
            n_skipped += 1
            continue

        # Separate zero and non-zero values
        # (Gamma is defined only for x > 0; zeros handled via mixed distribution)
        nonzero = values[values > 0]
        n_zeros = len(values) - len(nonzero)
        q_zero = n_zeros / len(values)  # probability of observing zero

        if len(nonzero) < 5:
            n_skipped += 1
            continue

        # Fit Gamma distribution to non-zero values
        # floc=0 forces the location parameter to 0 (standard for SSI)
        try:
            alpha, loc, beta = scipy_stats.gamma.fit(nonzero, floc=0)
        except Exception:
            n_skipped += 1
            continue

        # Validate: shape and scale must be positive and finite
        if alpha <= 0 or beta <= 0 or np.isnan(alpha) or np.isnan(beta):
            n_skipped += 1
            continue

        n_fitted += 1

        # Compute SSI for every observation in this group
        idx = group.index
        sm_vals = df_filtered.loc[idx, "volumetric_soil_water_layer_2"].values
        ssi_vals = np.full(len(sm_vals), np.nan)

        for i, sm_val in enumerate(sm_vals):
            if pd.isna(sm_val):
                continue

            # Mixed distribution CDF
            if sm_val <= 0:
                # For zero values: assign probability = half the zero mass
                # (convention from McKee et al. 1993)
                p = q_zero / 2.0
            else:
                # For positive values: combine zero mass + Gamma CDF
                # P(X ≤ x) = P(X=0) + P(X>0) × P(X≤x | X>0)
                p = q_zero + (1.0 - q_zero) * scipy_stats.gamma.cdf(
                    sm_val, alpha, loc=0, scale=beta
                )

            # Clamp probability to avoid ±infinity from ppf
            p = np.clip(p, 0.001, 0.999)

            # Inverse standard normal: transform probability → SSI
            ssi_vals[i] = scipy_stats.norm.ppf(p)

        df_filtered.loc[idx, "SSI"] = ssi_vals

    # Drop rows where SSI could not be computed
    n_before = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=["SSI"])
    n_dropped = n_before - len(df_filtered)

    print(f"\n  Results:")
    print(f"    (district × DOY) groups fitted:  {n_fitted:,}")
    print(f"    Groups skipped (too few obs):    {n_skipped:,}")
    print(f"    Rows dropped (no SSI):           {n_dropped:,}")
    print(f"    Total daily SSI values:          {len(df_filtered):,}")
    print(f"\n  SSI distribution:")
    print(f"    Mean:  {df_filtered['SSI'].mean():.4f}  (should be ≈ 0)")
    print(f"    Std:   {df_filtered['SSI'].std():.4f}   (should be ≈ 1)")
    print(f"    Min:   {df_filtered['SSI'].min():.2f}")
    print(f"    Max:   {df_filtered['SSI'].max():.2f}")
    print(f"\n  Drought days (SSI ≤ {SSI_THRESHOLD}):")
    n_drought = (df_filtered["SSI"] <= SSI_THRESHOLD).sum()
    pct_drought = 100 * n_drought / len(df_filtered)
    print(f"    Count: {n_drought:,} ({pct_drought:.1f}% of all in-season days)")

    return df_filtered


# ---------------------------------------------------------------------------
# STEP 7: Aggregate to annual drought-day counts
# ---------------------------------------------------------------------------
def aggregate_annual_drought_days(df_filtered):
    """
    Count the number of drought days per district per crop-year.

    A "drought day" is any day where SSI ≤ −1.0 (15.9th percentile),
    the WMO-defined onset of "Moderate Drought" (McKee et al. 1993; WMO 2012).

    Returns:
        df_annual: DataFrame with columns [feature_id, year, Drought_Days]
    """
    print("\n" + "=" * 70)
    print("STEP 7: Aggregating to annual drought-day counts")
    print("=" * 70)
    print(f"\n  Threshold: SSI ≤ {SSI_THRESHOLD} (= 6.7th percentile)")

    df_filtered = df_filtered.copy()
    df_filtered["is_drought"] = (df_filtered["SSI"] <= SSI_THRESHOLD).astype(int)

    df_annual = (
        df_filtered.groupby(["feature_id", "crop_year"])["is_drought"]
        .sum()
        .reset_index()
        .rename(columns={"is_drought": "Drought_Days", "crop_year": "year"})
    )

    # Keep only 2000–2025 (ERA5 data may produce crop_year 1999 or 2026 at edges)
    df_annual = df_annual[(df_annual["year"] >= 2000) & (df_annual["year"] <= 2025)].copy()

    print(f"\n  Results:")
    print(f"    District-year rows:              {len(df_annual):,}")
    print(f"    Unique districts:                {df_annual['feature_id'].nunique()}")
    print(f"    Year range:                      {df_annual['year'].min()}–{df_annual['year'].max()}")
    print(f"    Mean drought days/district/year: {df_annual['Drought_Days'].mean():.1f}")
    print(f"    Median:                          {df_annual['Drought_Days'].median():.0f}")
    print(f"    Max:                             {df_annual['Drought_Days'].max()}")
    print(f"    Districts with 0 drought days (all years): "
          f"{(df_annual.groupby('feature_id')['Drought_Days'].sum() == 0).sum()}")

    return df_annual


# ---------------------------------------------------------------------------
# STEP 8: Bin drought days into severity categories
# ---------------------------------------------------------------------------
"""
DROUGHT SEVERITY CLASSIFICATION — SCIENTIFIC JUSTIFICATION
===========================================================

We classify annual drought-day counts into severity bins. Our SSI threshold
is −1.0, which marks the onset of "Moderate Drought" in the WMO framework.

PUBLISHED REFERENCES:
  1. WMO (2012). "Standardized Precipitation Index User Guide."
     WMO-No. 1090, Geneva. Table 2, p.13.
     → SSI/SPI ≤ −1.0 = "Moderately Dry" (≈15.9th percentile)
     → This is the internationally accepted threshold for drought onset

  2. McKee, T.B., Doesken, N.J., Kleist, J. (1993).
     "The relationship of drought frequency and duration to time scales."
     8th Conf. Applied Climatology, Anaheim, CA.
     → Original classification: −1.0 = moderate drought onset

  3. WMO & GWP (2016). "Handbook of Drought Indicators and Indices."
     WMO-No. 1173, Geneva.
     → Confirms −1.0 as operational threshold for moderate drought
       across all standardised indices (SPI, SPEI, SSI)

  4. AghaKouchak, A. (2014). "A baseline probabilistic drought forecasting
     framework using standardized soil moisture index."
     HESS, 18(7), 2515–2526.
     → Applies WMO SPI thresholds to SSI; uses −1.0 for drought onset

  5. Svoboda, M. et al. (2002). "The Drought Monitor."
     Bull. Amer. Meteor. Soc., 83(8), 1181–1190.
     → US Drought Monitor D0–D4 scale; D1 (moderate) onset at ~15th percentile

TRANSLATING TO DROUGHT-DAY COUNTS:
  With SSI ≤ −1.0 as threshold (15.9th percentile), each day below this
  value represents a day of moderate-or-worse soil moisture deficit.

  For a typical Maize 1 risk window of ~90–120 days, the bins represent:
    - 0 days:     No drought stress at all during risk window
    - 1–10 days:  Brief stress (<10% of season) — "Abnormally Dry" (D0)
    - 11–20 days: ~10–20% of season — "Moderate Drought" (D1)
    - 21–35 days: ~20–30% of season — "Severe Drought" (D2)
    - 36–55 days: ~30–50% of season — "Extreme Drought" (D3)
    - >55 days:   >50% of season — "Exceptional Drought" (D4)

  These align with the US Drought Monitor's D0–D4 scale (Svoboda et al. 2002)
  where severity doubles at each class. The wider bins (vs. −1.5 threshold)
  reflect that SSI = −1.0 captures more frequent events, so more days
  are expected per season.
"""

# Drought severity bins: (lower, upper, label)
# Calibrated for SSI ≤ −1.0 threshold (WMO "Moderate Drought" onset)
# Based on WMO (2012) + US Drought Monitor (Svoboda et al. 2002) frameworks
DROUGHT_BINS = [
    (0,    0,   "No Drought"),          # Will be rendered as neutral grey (background)
    (1,   10,   "D1 – Abnormally Dry"),
    (11,  20,   "D2 – Moderate Drought"),
    (21,  35,   "D3 – Severe Drought"),
    (36,  55,   "D4 – Extreme Drought"),
    (56, None,  "D5 – Exceptional Drought"),
]


def classify_drought_severity(drought_days):
    """
    Map a drought-day count to a severity category.

    Classification follows WMO (2012) SPI User Guide thresholds
    adapted to drought-day counts per growing season, aligned with
    the US Drought Monitor D0–D4 scale (Svoboda et al. 2002).
    """
    for lower, upper, label in DROUGHT_BINS:
        if upper is None:
            return label
        if lower <= drought_days <= upper:
            return label
    return "D5 – Exceptional Drought"


def get_category_order():
    """Return ordered list of severity labels (for Plotly category ordering)."""
    return [label for _, _, label in DROUGHT_BINS]


def get_color_map(category_order):
    """
    Assign hand-picked colours to severity bins for maximum readability.
    Uses a colorblind-safe sequential warm palette (inspired by USDM),
    with "No Drought" as transparent grey.
    """
    # Professional drought severity palette (light → dark, warm tones)
    SEVERITY_COLORS = {
        "No Drought":                "rgba(220,220,220,0.25)",
        "D1 – Abnormally Dry":       "#FFEDA0",   # pale yellow
        "D2 – Moderate Drought":     "#FEB24C",   # orange-yellow
        "D3 – Severe Drought":       "#FC4E2A",   # red-orange
        "D4 – Extreme Drought":      "#BD0026",   # dark red
        "D5 – Exceptional Drought":  "#4A0010",   # near-black red
    }
    return {cat: SEVERITY_COLORS.get(cat, "#999999") for cat in category_order}


def bin_drought_days(df_annual):
    """
    Add a 'Drought_Category' column to df_annual based on published severity bins.

    Returns:
        df_annual with new 'Drought_Category' column
    """
    print("\n" + "=" * 70)
    print("STEP 8: Classifying drought severity (WMO 2012 / Svoboda et al. 2002)")
    print("=" * 70)

    df_annual = df_annual.copy()
    df_annual["Drought_Category"] = df_annual["Drought_Days"].apply(classify_drought_severity)

    # Print distribution
    category_order = get_category_order()
    print(f"\n  Severity distribution across all district-years:")
    for cat in category_order:
        count = (df_annual["Drought_Category"] == cat).sum()
        pct = 100 * count / len(df_annual)
        print(f"    {cat:<30s}  {count:>6,} ({pct:>5.1f}%)")

    return df_annual


# ---------------------------------------------------------------------------
# STEP 9: Generate animated choropleth map (25-year time series)
# ---------------------------------------------------------------------------
def generate_animated_map(df_annual, gdf_districts):
    """
    Produce an animated Plotly choropleth HTML map showing drought severity
    per district per year (2000–2025), with a year slider/play button.

    This is the same style as Africa_Maize_Drought_10Day_Bins.html but
    computed with the proper Gamma-CDF SSI methodology.

    Output: Africa_Maize_GammaSSI_Drought_Map.html
    """
    print("\n" + "=" * 70)
    print("STEP 9: Generating animated choropleth map")
    print("=" * 70)

    category_order = get_category_order()
    color_map = get_color_map(category_order)

    # Simplify district geometry for faster rendering
    print("\n  Simplifying geometries for map …")
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.1, preserve_topology=True)

    # Keep only districts present in df_annual
    valid_names = df_annual["feature_id"].unique()
    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(valid_names)].copy()
    gdf_plot = gdf_plot.drop_duplicates("ADM_NAME")
    print(f"  Districts in map: {len(gdf_plot)}")

    # Merge annual data with geometry
    final_gdf = gdf_plot[["ADM_NAME", "geometry"]].merge(
        df_annual, left_on="ADM_NAME", right_on="feature_id", how="inner"
    )
    final_gdf = final_gdf.to_crs(epsg=4326)

    # Build GeoJSON for Plotly
    temp_gdf = (
        final_gdf[["ADM_NAME", "geometry"]]
        .drop_duplicates("ADM_NAME")
        .set_index("ADM_NAME")
    )
    geojson_payload = json.loads(temp_gdf.geometry.to_json())

    # Generate the animated choropleth
    print(f"  Building Plotly choropleth ({final_gdf['ADM_NAME'].nunique()} districts, "
          f"{final_gdf['year'].nunique()} years) …")

    fig = px.choropleth(
        final_gdf,
        geojson=geojson_payload,
        locations="ADM_NAME",
        color="Drought_Category",
        animation_frame="year",
        category_orders={"Drought_Category": category_order},
        color_discrete_map=color_map,
        title=(
            "<b>African Maize Drought Severity (Gamma-SSI Method, 2000–2025)</b>"
            "<br><sup>Days with SSI ≤ −1.0 (15.9th percentile, WMO Moderate Drought onset) "
            "during Maize 1 risk window | Classification: WMO 2012 / Svoboda et al. 2002</sup>"
        ),
        labels={"Drought_Category": "Drought Severity"},
    )

    # Style the map
    fig.update_traces(marker_line_color="white", marker_line_width=0.2)
    fig.update_geos(
        visible=True,
        scope="africa",
        showland=True,
        landcolor="#D3D3D3",
        showocean=True,
        oceancolor="white",
        showcountries=False,
        showcoastlines=False,
        fitbounds="locations",
    )
    fig.update_layout(
        margin={"r": 0, "t": 80, "l": 0, "b": 60},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="Drought Severity",
        font=dict(family="Inter, sans-serif"),
        annotations=[
            dict(
                text=(
                    "<b>References:</b> "
                    "McKee, T.B. et al. (1993) <i>The relationship of drought frequency and duration to time scales</i>, "
                    "8th Conf. Applied Climatology, AMS; "
                    "WMO (2012) <i>Standardized Precipitation Index User Guide</i>, WMO-No. 1090, Table 2; "
                    "Svoboda, M. et al. (2002) <i>The Drought Monitor</i>, Bull. Amer. Meteor. Soc., 83(8), 1181–1190; "
                    "AghaKouchak, A. (2014) <i>A baseline probabilistic drought forecasting framework</i>, "
                    "Hydrol. Earth Syst. Sci., 18, 2485–2492."
                ),
                xref="paper", yref="paper",
                x=0.0, y=-0.04,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=9, color="#555555"),
                align="left",
            )
        ],
    )

    # Save
    output_path = os.path.join(OUTPUT_DIR, "Africa_Maize_GammaSSI_Drought_Map.html")
    print(f"  Writing HTML (this may take several minutes for {len(final_gdf):,} rows) …", flush=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  ✅ Map saved: {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")
    print(f"     Open in browser to view animated time series")

    return output_path


# ---------------------------------------------------------------------------
# GROUP COLOUR PALETTE (consistent across all maps)
# ---------------------------------------------------------------------------
GROUP_COLOR_MAP = {
    "Jan–Feb planters": "#1b9e77",
    "Mar–Apr planters": "#d95f02",
    "May–Jun planters": "#7570b3",
    "Jul–Aug planters": "#e7298a",
    "Sep–Oct planters": "#66a61e",
    "Nov–Dec planters": "#e6ab02",
}


def _build_group_legend_html(district_to_group):
    """Build legend HTML items for only the groups actually present in data."""
    present_groups = sorted(set(district_to_group.values()))
    lines = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        lines.append(
            f'  <div class="legend-item"><div class="legend-color" style="background:{color}"></div> {g}</div>'
        )
    return "\n".join(lines)


def _build_group_colors_js(district_to_group):
    """Build JS object literal for only the groups actually present in data."""
    present_groups = sorted(set(district_to_group.values()))
    entries = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        entries.append(f'  "{g}": "{color}"')
    return "{\n" + ",\n".join(entries) + "\n}"


def _build_short_group_legend_html(district_to_group):
    """Build legend for inter-group map using shortened names (no 'planters')."""
    present_groups = sorted(set(district_to_group.values()))
    lines = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        short = g.replace(" planters", "")
        lines.append(
            f'  <div class="legend-item"><div class="legend-color" style="background:{color}"></div> {short}</div>'
        )
    return "\n".join(lines)


def _build_group_filter_buttons_html(district_to_group):
    """Build HTML buttons for filtering map by season group."""
    present_groups = sorted(set(district_to_group.values()))
    buttons = ['<span class="group-btn active" onclick="filterGroup(\'all\')" style="background:#f8f8f8;">Show All</span>']
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        short = g.replace(" planters", "")
        buttons.append(
            f'<span class="group-btn" onclick="filterGroup(\'{g}\')" style="background:{color}22;border-color:{color};">{short}</span>'
        )
    return "\n  ".join(buttons)


    """Build JS object for inter-group map using shortened group keys, with optional hex suffix."""
    present_groups = sorted(set(district_to_group.values()))
    entries = []
    for g in present_groups:
        color = GROUP_COLOR_MAP.get(g, "#999999")
        short = g.replace(" planters", "")
        entries.append(f'  "{short}": "{color}{suffix}"')
    return "{\n" + ",\n".join(entries) + "\n}"


# ---------------------------------------------------------------------------
# STEP 10–11: Group districts by crop-season type → within-group correlation
# ---------------------------------------------------------------------------
#
# METHODOLOGY & REFERENCES
# -------------------------
# Grouping districts by crop calendar planting month follows the operational
# parametric drought insurance approach and published co-variability paradigm:
#
#   1. Africa Risk Capacity (ARC) — African Union (2024)
#      "Drought Model" — www.arc.int/drought
#      → Operational sovereign parametric insurance that groups admin units
#        by crop calendar timing to assess co-varying drought risk and
#        trigger payouts. Direct precedent for our planting-month grouping.
#
#   2. Carrão, H., Naumann, G. & Barbosa, P. (2016)
#      "Mapping global patterns of drought risk: An empirical framework
#       based on sub-national estimates of hazard, exposure and vulnerability"
#      Global Environmental Change, 39, 108–124.
#      → Groups regions by shared growing-season timing to assess co-occurring risk
#
#   3. Tovihoudji, P.G. et al. (2025)
#      "Agroclimatic analysis: a coping strategy for reducing planting risks"
#      Theor. Appl. Climatol. (Springer)
#      → Groups West African districts by rain onset / planting window;
#        correlates with drought risk for insurance design
#
#   4. Vicente-Serrano, S.M. et al. (2012)
#      "Performance of drought indices for ecological, agricultural,
#       and hydrological applications"
#      Earth Interactions, 16(10), 1–27.
#      → Demonstrates that drought correlation is season-specific; districts
#        sharing growing-season timing show coherent drought patterns
#
#   5. Masante, D. et al. (2020)
#      "A pan-African high-resolution drought index dataset"
#      Earth Syst. Sci. Data, 12, 753–769.
#      → Pan-African SPEI-HR at 5km aggregated by admin boundaries aligned
#        to growing seasons — validates seasonal stratification approach
#
#   6. Zargar, A., Sadiq, R., Naser, B. & Khan, F.I. (2011)
#      "A review of drought indices"
#      Environmental Reviews, 19, 333–349.
#      → Recommends grouping by agro-climatic zones (defined by crop calendar)
#        before computing inter-station drought correlations
#
#   7. Karume, K. et al. (2024)
#      "Crop calendar optimization for climate change adaptation in yam"
#      PLOS ONE, doi:10.1371/journal.pone.0309775
#      → Maps planting windows across agro-ecological zones; correlates with
#        climate variability for adaptive risk management
#
# WHY GROUP BY PLANTING MONTH?
#   - Districts with the same planting window are exposed to the same
#     large-scale climate drivers (ITCZ position, monsoon onset, ENSO)
#   - Correlating Nov planters (Southern) with May planters (Sahel)
#     is climatologically meaningless — they respond to different
#     seasonal rainfall regimes
#   - Within-group correlation reveals which districts co-experience drought
#     under the same climate forcing — critical for parametric insurance
#     portfolio design and early warning trigger zones
#   - This is exactly how ARC operates at sovereign level (African Union 2024)
#
# APPROACH:
#   1. Convert each district's GEOGLAM planting DOY to a planting month,
#      then cluster into bimonthly groups (6 groups across Africa)
#
#   2. Within each group, compute Pearson correlation of annual drought-day
#      time series between all district pairs (25-year record)
#
#   3. Generate interactive HTML: click a district → see all correlated
#      districts in the same group, colour-coded by correlation strength
# ---------------------------------------------------------------------------


def classify_season_groups(valid_maize):
    """
    STEP 10: Assign each district to a season group based on GEOGLAM
    planting month.

    Groups districts by the MONTH in which planting begins (from GEOGLAM
    V1.4 planting DOY). This creates natural agro-climatic clusters that
    reflect distinct rainfall regimes across Africa:

      - Jan–Feb planters: Southern Africa second season / bimodal East
      - Mar–Apr planters: East Africa long rains (Masika)
      - May–Jun planters: West Africa / Sahel onset
      - Jul–Aug planters: Late Sahel / Horn short rains prep
      - Sep–Oct planters: East Africa short rains (Vuli)
      - Nov–Dec planters: Southern Africa main season

    Districts within the same planting-month group share large-scale
    climate forcing (ITCZ position, monsoon onset) and are therefore
    meaningful to correlate (Carrão et al. 2016; ARC Drought Model 2024).

    Returns:
        dict mapping group_name → list of district ADM_NAMEs
    """
    print("\n" + "=" * 70)
    print("STEP 10: Grouping districts by crop-season type")
    print("  (Grouping derived from GEOGLAM V1.4 crop calendar planting DOY)")
    print("=" * 70)

    # Convert planting DOY to planting month, then group into bimonthly windows
    import math
    valid = valid_maize.copy()
    # Approximate month from DOY: month = ceil(DOY / 30.44)
    valid["plant_month"] = valid["planting"].apply(
        lambda d: min(12, math.ceil(d / 30.44)) if d > 0 else 0
    )

    # Bimonthly grouping for meaningful cluster sizes
    BIMONTH_LABELS = {
        (1, 2):   "Jan–Feb planters",
        (3, 4):   "Mar–Apr planters",
        (5, 6):   "May–Jun planters",
        (7, 8):   "Jul–Aug planters",
        (9, 10):  "Sep–Oct planters",
        (11, 12): "Nov–Dec planters",
    }

    def _get_bimonth_group(month):
        for (m1, m2), label in BIMONTH_LABELS.items():
            if month in (m1, m2):
                return label
        return "Unknown"

    valid["season_group"] = valid["plant_month"].apply(_get_bimonth_group)

    # Build groups dict (only include groups with ≥2 districts)
    groups = {}
    for grp_name, sub in valid.groupby("season_group"):
        if grp_name == "Unknown":
            continue
        members = sub["ADM_NAME"].tolist()
        if len(members) >= 2:
            groups[grp_name] = members

    print(f"\n  Season groups identified: {len(groups)}")
    for name, members in sorted(groups.items()):
        print(f"    {name:<22s}: {len(members):>5} districts")

    total = sum(len(m) for m in groups.values())
    print(f"\n  Total districts assigned: {total}")

    # Build district → group lookup for downstream functions
    district_to_group = {}
    for grp_name, members in groups.items():
        for d in members:
            district_to_group[d] = grp_name

    return groups, district_to_group


def compute_within_group_correlation(df_annual, groups):
    """
    STEP 11a: Compute Pearson correlation of drought-day time series
    between all district pairs WITHIN each season group.

    Pearson correlation on standardised drought indices between stations
    is the standard approach for parametric drought co-variability
    (Vicente-Serrano et al. 2012; Carrão et al. 2016).

    Min 5 overlapping years required to include a district.

    Returns:
        dict: group_name → (corr_matrix, district_list)
    """
    print("\n" + "=" * 70)
    print("STEP 11: Computing within-group district drought correlation")
    print("=" * 70)
    print("  Method: Pearson correlation of annual drought-day time series")
    print("  Ref: Vicente-Serrano et al. (2012); Carrão et al. (2016)")

    min_years = 5  # Include districts with as few as 5 years (maximise coverage)
    results = {}

    for group_name, district_list in groups.items():
        print(f"\n  Processing group: {group_name} ({len(district_list)} districts)")

        # Filter to group districts
        df_grp = df_annual[df_annual["feature_id"].isin(district_list)].copy()

        # Pivot: rows=years, cols=districts
        pivot = df_grp.pivot_table(
            index="year", columns="feature_id", values="Drought_Days"
        )

        # Drop districts with < min_years
        valid_cols = pivot.columns[pivot.notna().sum() >= min_years]
        pivot = pivot[valid_cols]
        print(f"    Districts with ≥{min_years} years data: {len(valid_cols)}")

        if len(valid_cols) < 2:
            print(f"    ⚠ Too few districts for correlation — skipping")
            continue

        # Fill NaN with column mean (conservative approach)
        pivot_filled = pivot.fillna(pivot.mean())

        # Pearson correlation
        corr = pivot_filled.corr()
        results[group_name] = (corr, valid_cols.tolist())
        print(f"    Correlation matrix: {corr.shape[0]}×{corr.shape[1]}")

        # Summary stats (exclude diagonal)
        mask = np.ones(corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        vals = corr.values[mask]
        print(f"    Mean correlation: {vals.mean():.3f}")
        print(f"    Median correlation: {np.median(vals):.3f}")
        print(f"    Highly correlated pairs (r≥0.7): {(vals >= 0.7).sum() // 2}")

    return results


def generate_correlation_map(df_annual, valid_maize, gdf_districts, groups, district_to_group, district_to_country):
    """
    Generate interactive HTML map showing within-group district correlations.

    Click a district → all districts in the same group colour-coded by
    Pearson correlation with the selected district.

    References embedded in map:
      - ARC / African Union (2024): operational planting-window grouping
      - Carrão et al. (2016): drought risk grouping framework
      - Tovihoudji et al. (2025): district grouping by planting window
      - Vicente-Serrano et al. (2012): season-specific correlation
      - Zargar et al. (2011): agro-climatic zone grouping rationale
    """
    # Compute correlations
    corr_results = compute_within_group_correlation(df_annual, groups)

    if not corr_results:
        print("  ⚠ No correlation results — skipping map generation")
        return

    # Prepare geometry
    print("\n  Preparing map geometry …")
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.05, preserve_topology=True)
    gdf_plot = gdf_plot.to_crs(epsg=4326)

    # Combine all districts from all groups
    all_districts = set()
    for _, (_, dlist) in corr_results.items():
        all_districts.update(dlist)

    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(all_districts)].drop_duplicates("ADM_NAME")
    print(f"  Districts in map: {len(gdf_plot)}")

    gdf_plot["season_group"] = gdf_plot["ADM_NAME"].map(district_to_group)

    # Build GeoJSON with country info
    gdf_plot["COUNTRY_LABEL"] = gdf_plot["ADM_NAME"].map(district_to_country).fillna("Unknown")
    geojson = json.loads(
        gdf_plot[["ADM_NAME", "COUNTRY_LABEL", "geometry", "season_group"]].to_json()
    )

    # Build correlation lookup for JavaScript
    # Include ALL district pairs — even low/negative correlations — so every
    # district shows its relationship to every other district in its group
    corr_lookup = {}
    for group_name, (corr_matrix, dlist) in corr_results.items():
        for dist in dlist:
            row = corr_matrix.loc[dist].drop(dist)  # exclude self
            # Include every pair (replace NaN with 0.0)
            corr_lookup[dist] = {
                k: round(v if not np.isnan(v) else 0.0, 2)
                for k, v in row.items()
            }

    # Build time series data for tooltip
    ts_data = {}
    for dist, sub in df_annual.groupby("feature_id"):
        if dist in all_districts:
            ts_data[dist] = dict(zip(
                sub["year"].astype(int).tolist(),
                sub["Drought_Days"].astype(int).tolist()
            ))

    # Generate HTML
    print("  Generating interactive HTML …")
    _write_correlation_html(geojson, corr_lookup, ts_data, district_to_group, district_to_country)


def _write_correlation_html(geojson, corr_lookup, ts_data, district_to_group, district_to_country):
    """Write the standalone interactive correlation HTML map."""

    geojson_str = json.dumps(geojson)
    corr_str = json.dumps(corr_lookup)
    ts_str = json.dumps(ts_data)
    groups_str = json.dumps(district_to_group)
    countries_str = json.dumps(district_to_country)

    # Dynamic legend and colors from actual data
    group_legend_html = _build_group_legend_html(district_to_group)
    group_colors_js = _build_group_colors_js(district_to_group)
    group_filter_html = _build_group_filter_buttons_html(district_to_group)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Maize District Drought Correlation (Gamma-SSI, Grouped by Crop Season)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; font-family: 'Inter', 'Segoe UI', sans-serif; }}
  #map {{ width:100%; height:100vh; }}
  .info-panel {{
    position:absolute; top:10px; right:10px; z-index:1000;
    background:white; padding:14px 18px; border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.15); max-width:360px;
    font-size:13px; line-height:1.5;
  }}
  .info-panel h3 {{ margin:0 0 8px 0; font-size:15px; }}
  .legend {{
    position:absolute; bottom:30px; left:10px; z-index:1000;
    background:white; padding:12px 16px; border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.15); font-size:12px;
  }}
  .legend-item {{ display:flex; align-items:center; margin:3px 0; }}
  .legend-color {{ width:20px; height:14px; margin-right:8px; border:1px solid #ccc; }}
  .references {{
    position:absolute; bottom:10px; right:10px; z-index:1000;
    background:rgba(255,255,255,0.95); padding:10px 14px; border-radius:6px;
    box-shadow:0 1px 6px rgba(0,0,0,0.1); font-size:10px; max-width:500px;
    color:#555; line-height:1.4;
  }}
  .ts-chart {{ margin-top:10px; }}
  .ts-bar {{ display:inline-block; width:8px; margin:0 1px; background:#e74c3c;
             vertical-align:bottom; border-radius:2px 2px 0 0; }}
  .group-filter {{
    position:absolute; top:10px; left:50%; transform:translateX(-50%); z-index:1000;
    background:white; padding:10px 16px; border-radius:8px;
    box-shadow:0 2px 12px rgba(0,0,0,0.15); font-size:12px; text-align:center;
  }}
  .group-filter b {{ display:block; margin-bottom:6px; font-size:11px; color:#333; }}
  .group-btn {{
    display:inline-block; margin:3px 4px; padding:4px 10px; border-radius:14px;
    border:1px solid #ccc; cursor:pointer; font-size:11px; font-weight:500;
    transition: all 0.2s;
  }}
  .group-btn:hover {{ box-shadow:0 2px 6px rgba(0,0,0,0.15); }}
  .group-btn.active {{ border-color:#333; box-shadow:0 0 0 2px rgba(0,0,0,0.15); }}
</style>
</head>
<body>
<div id="map"></div>

<div class="group-filter">
  <b>Focus on Season Group</b>
  {group_filter_html}
</div>

<div class="info-panel" id="info">
  <h3>🌽 Maize District Drought Correlation</h3>
  <p><b>Click a district</b> to see correlated districts within its crop-season group.</p>
  <p>Districts are grouped by growing-season type (Southern wrap vs Sahelian standard)
  so that correlations reflect shared climate drivers.</p>
  <div id="details"></div>
</div>

<div class="legend">
  <b>Correlation with selected district</b>
  <div class="legend-item"><div class="legend-color" style="background:#a50f15"></div> r ≥ 0.8 (very high)</div>
  <div class="legend-item"><div class="legend-color" style="background:#fb6a4a"></div> 0.6 ≤ r < 0.8 (high)</div>
  <div class="legend-item"><div class="legend-color" style="background:#fcae91"></div> 0.4 ≤ r < 0.6 (moderate)</div>
  <div class="legend-item"><div class="legend-color" style="background:#fee0d2"></div> 0.2 ≤ r < 0.4 (low)</div>
  <div class="legend-item"><div class="legend-color" style="background:#f0f0f0"></div> 0 ≤ r < 0.2 (negligible)</div>
  <div class="legend-item"><div class="legend-color" style="background:#bdd7e7"></div> −0.2 ≤ r < 0 (weak inverse)</div>
  <div class="legend-item"><div class="legend-color" style="background:#6baed6"></div> −0.4 ≤ r < −0.2 (moderate inverse)</div>
  <div class="legend-item"><div class="legend-color" style="background:#08306b"></div> r < −0.4 (strong inverse)</div>
  <div style="margin-top:8px;"><b>Season groups (no selection)</b></div>
  {group_legend_html}
</div>

<div class="references">
  <b>References:</b><br>
  African Union / ARC (2024) <i>Drought Model</i>, www.arc.int/drought — operational planting-window grouping for parametric insurance.<br>
  Carrão, H. et al. (2016) <i>Global Environmental Change</i>, 39, 108–124 — drought risk grouping by season.<br>
  Tovihoudji, P.G. et al. (2025) <i>Theor. Appl. Climatol.</i> (Springer) — district grouping by planting window for insurance.<br>
  Vicente-Serrano, S.M. et al. (2012) <i>Earth Interactions</i>, 16(10) — season-specific drought correlation.<br>
  Masante, D. et al. (2020) <i>ESSD</i>, 12, 753–769 — pan-African drought index aligned to growing seasons.<br>
  Zargar, A. et al. (2011) <i>Environmental Reviews</i>, 19, 333–349 — agro-climatic zone grouping rationale.<br>
  McKee, T.B. et al. (1993) 8th Conf. Applied Climatology — SPI/SSI classification.<br>
  AghaKouchak, A. (2014) <i>HESS</i>, 18, 2485–2492 — Gamma-CDF SSI framework.
</div>

<script>
const geojson = {geojson_str};
const corrData = {corr_str};
const tsData = {ts_str};
const districtGroups = {groups_str};
const districtCountries = {countries_str};

const map = L.map('map').setView([0, 20], 4);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
  attribution: '© CARTO'
}}).addTo(map);

const groupColors = {group_colors_js};

function corrColor(r) {{
  if (r >= 0.8) return "#a50f15";
  if (r >= 0.6) return "#fb6a4a";
  if (r >= 0.4) return "#fcae91";
  if (r >= 0.2) return "#fee0d2";
  if (r >= 0)   return "#f0f0f0";
  if (r >= -0.2) return "#bdd7e7";
  if (r >= -0.4) return "#6baed6";
  return "#08306b";
}}

function displayName(name) {{
  const country = districtCountries[name] || "";
  return country ? name + " (" + country + ")" : name;
}}

function defaultStyle(feature) {{
  const grp = feature.properties.season_group || "";
  return {{
    fillColor: groupColors[grp] || "#d3d3d3",
    weight: 0.5, color: "#999", fillOpacity: 0.6
  }};
}}

let selected = null;
const layerLookup = {{}};

const geoLayer = L.geoJSON(geojson, {{
  style: defaultStyle,
  onEachFeature: function(feature, layer) {{
    const name = feature.properties.ADM_NAME;
    layerLookup[name] = layer;
    layer.bindTooltip(displayName(name), {{sticky:true, className:'tooltip'}});
    layer.on('click', function() {{ selectDistrict(name); }});
  }}
}}).addTo(map);

function makeSparkline(dist) {{
  if (!tsData[dist]) return "";
  const years = Object.keys(tsData[dist]).sort();
  const vals = years.map(y => tsData[dist][y]);
  const maxV = Math.max(...vals, 1);
  let html = '<div class="ts-chart">';
  vals.forEach((v, i) => {{
    const h = Math.max(2, (v / maxV) * 40);
    html += `<div class="ts-bar" style="height:${{h}}px" title="${{years[i]}}: ${{v}} days"></div>`;
  }});
  html += '</div><div style="font-size:10px;color:#888;margin-top:2px;">Drought days/year (2000–2025)</div>';
  return html;
}}

function selectDistrict(name) {{
  selected = name;
  const grp = districtGroups[name];
  const corrs = corrData[name] || {{}};

  // Reset all styles
  geoLayer.eachLayer(l => l.setStyle(defaultStyle(l.feature)));

  // Highlight selected
  if (layerLookup[name]) {{
    layerLookup[name].setStyle({{fillColor:"#ffd700", weight:2, color:"#333", fillOpacity:0.9}});
    layerLookup[name].bringToFront();
  }}

  // Colour correlated districts
  const sorted = Object.entries(corrs).sort((a,b) => b[1]-a[1]);
  for (const [dist, r] of sorted) {{
    if (layerLookup[dist]) {{
      layerLookup[dist].setStyle({{fillColor: corrColor(r), weight:0.5, color:"#666", fillOpacity:0.8}});
    }}
  }}

  // Info panel
  const topN = sorted.slice(0, 8);
  let html = `<h3>${{displayName(name)}}</h3>`;
  html += `<div style="color:#666;font-size:11px;">Group: ${{grp || "Unknown"}}</div>`;
  html += makeSparkline(name);
  html += `<div style="margin-top:10px;"><b>Top correlated districts:</b></div>`;
  html += '<table style="width:100%;font-size:11px;margin-top:4px;">';
  for (const [d, r] of topN) {{
    const col = corrColor(r);
    html += `<tr><td style="padding:2px 0;">${{displayName(d)}}</td>`;
    html += `<td style="text-align:right;color:${{col}};font-weight:bold;">${{r.toFixed(2)}}</td></tr>`;
  }}
  html += '</table>';
  if (sorted.length > 8) html += `<div style="font-size:10px;color:#888;margin-top:4px;">+ ${{sorted.length-8}} more districts</div>`;

  document.getElementById('details').innerHTML = html;
}}

// Double-click to reset
map.on('dblclick', function() {{
  selected = null;
  activeGroup = 'all';
  geoLayer.eachLayer(l => {{
    l.setStyle(defaultStyle(l.feature));
    l.getElement() && (l.getElement().style.display = '');
  }});
  document.getElementById('details').innerHTML = '<p><b>Click a district</b> to see correlations.</p>';
  document.querySelectorAll('.group-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('.group-btn').classList.add('active');
}});

// Group filter
let activeGroup = 'all';
function filterGroup(group) {{
  activeGroup = group;
  selected = null;
  document.querySelectorAll('.group-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');

  geoLayer.eachLayer(function(layer) {{
    const grp = layer.feature.properties.season_group || "";
    const el = layer.getElement();
    if (!el) return;
    if (group === 'all') {{
      el.style.display = '';
      layer.setStyle(defaultStyle(layer.feature));
    }} else if (grp === group) {{
      el.style.display = '';
      layer.setStyle({{ fillColor: groupColors[grp] || "#d3d3d3", weight: 0.8, color: "#444", fillOpacity: 0.75 }});
    }} else {{
      el.style.display = 'none';
    }}
  }});
  document.getElementById('details').innerHTML = group === 'all'
    ? '<p><b>Click a district</b> to see correlations.</p>'
    : `<p>Showing <b>${{group}}</b> districts only. Click one to see correlations.</p>`;
}}
</script>
</body>
</html>"""

    output_path = os.path.join(OUTPUT_DIR, "Africa_Maize_GammaSSI_Correlation_Map.html")
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(html)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  ✅ Correlation map saved: {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")
    print(f"     Click any district to see co-varying districts in its season group")


# ---------------------------------------------------------------------------
# STEP 11b: Static season-group map (visualise the groupings only)
# ---------------------------------------------------------------------------

def generate_season_group_map(valid_maize, gdf_districts, district_to_group, district_to_country):
    """
    Generate a static choropleth map showing only the planting-month
    season groups — no correlation, just the grouping itself.

    Output: Africa_Maize_Season_Groups_Map.html
    """
    print("\n" + "=" * 70)
    print("STEP 11b: Generating static season-group map")
    print("=" * 70)

    # Prepare geometry
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.05, preserve_topology=True)
    gdf_plot = gdf_plot.to_crs(epsg=4326)

    valid_names = set(valid_maize["ADM_NAME"])
    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(valid_names)].drop_duplicates("ADM_NAME")
    gdf_plot["season_group"] = gdf_plot["ADM_NAME"].map(district_to_group).fillna("Unknown")
    gdf_plot["COUNTRY_LABEL"] = gdf_plot["ADM_NAME"].map(district_to_country).fillna("Unknown")

    print(f"  Districts in map: {len(gdf_plot)}")
    for grp, count in gdf_plot["season_group"].value_counts().items():
        print(f"    {grp:<22s}: {count}")

    # Build GeoJSON
    geojson = json.loads(gdf_plot[["ADM_NAME", "COUNTRY_LABEL", "geometry", "season_group"]].to_json())
    geojson_str = json.dumps(geojson)

    # Dynamic legend and colors from actual data
    group_legend_html = _build_group_legend_html(district_to_group)
    group_colors_js = _build_group_colors_js(district_to_group)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Africa Maize — Season Groups (GEOGLAM Planting Month)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; font-family:'Inter','Segoe UI',sans-serif; }}
  #map {{ width:100%; height:100vh; }}
  .legend {{
    position:absolute; bottom:30px; left:10px; z-index:1000;
    background:white; padding:16px 20px; border-radius:10px;
    box-shadow:0 4px 16px rgba(0,0,0,0.12); font-size:12px;
  }}
  .legend h4 {{ margin:0 0 10px 0; font-size:14px; }}
  .legend-item {{ display:flex; align-items:center; margin:6px 0; }}
  .legend-color {{ width:24px; height:16px; margin-right:10px; border:1px solid #ccc; border-radius:3px; }}
  .title-panel {{
    position:absolute; top:10px; left:50%; transform:translateX(-50%); z-index:1000;
    background:white; padding:12px 24px; border-radius:10px;
    box-shadow:0 4px 16px rgba(0,0,0,0.12); text-align:center;
  }}
  .title-panel h2 {{ margin:0; font-size:16px; color:#1a1a2e; }}
  .title-panel p {{ margin:4px 0 0 0; font-size:11px; color:#666; }}
</style>
</head>
<body>
<div id="map"></div>

<div class="title-panel">
  <h2>🌽 African Maize Season Groups</h2>
  <p>Districts grouped by GEOGLAM V1.4 planting month (bimonthly clusters)<br>
  <span style="font-size:10px;">Source: GEOGLAM Crop Monitor for Early Warning (CM4EW) Calendars V1.4</span></p>
</div>

<div class="legend">
  <h4>Planting Season Group</h4>
  {group_legend_html}
</div>

<script>
const geojson = {geojson_str};

const map = L.map('map').setView([0, 20], 4);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
  attribution: '© CARTO'
}}).addTo(map);

const groupColors = {group_colors_js};

L.geoJSON(geojson, {{
  style: function(feature) {{
    const grp = feature.properties.season_group || "";
    return {{
      fillColor: groupColors[grp] || "#d3d3d3",
      weight: 0.6,
      color: "#444",
      fillOpacity: 0.75
    }};
  }},
  onEachFeature: function(feature, layer) {{
    const name = feature.properties.ADM_NAME;
    const grp = feature.properties.season_group || "Unknown";
    const country = feature.properties.COUNTRY_LABEL || "";
    const label = country ? name + " (" + country + ")" : name;
    layer.bindTooltip(`<b>${{label}}</b><br>${{grp}}`, {{sticky:true}});
  }}
}}).addTo(map);
</script>
</body>
</html>"""

    output_path = os.path.join(OUTPUT_DIR, "Africa_Maize_Season_Groups_Map.html")
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(html)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  ✅ Season group map saved: {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# STEP 12: Inter-group correlation map
# ---------------------------------------------------------------------------
# Shows how districts ACROSS different planting-month groups correlate.
# This reveals teleconnection patterns — e.g., does drought in Nov–Dec
# planters (Southern Africa) predict drought in Mar–Apr planters (East Africa)
# in subsequent seasons? Critical for portfolio diversification in parametric
# insurance (ARC 2024; Carrão et al. 2016).
# ---------------------------------------------------------------------------

def generate_intergroup_correlation_map(df_annual, valid_maize, gdf_districts, district_to_group, district_to_country):
    """
    STEP 12: Compute and map correlation between ALL district pairs
    regardless of season group (inter-group + intra-group).

    This map shows the full continental correlation structure:
    - Same-group districts (expected high correlation)
    - Cross-group districts (reveals teleconnections / lagged relationships)

    The colour of each district shows its mean correlation with the selected
    district. Group membership is shown via hatching/border colour.

    Output: Africa_Maize_GammaSSI_InterGroup_Correlation_Map.html
    """
    print("\n" + "=" * 70)
    print("STEP 12: Computing inter-group (continental) correlation matrix")
    print("=" * 70)
    print("  Correlating ALL district pairs across all season groups")
    print("  Ref: ARC (2024) portfolio diversification; Carrão et al. (2016)")

    # All districts in df_annual
    all_districts = df_annual["feature_id"].unique()
    print(f"  Total districts: {len(all_districts)}")

    # Pivot full matrix
    pivot = df_annual.pivot_table(
        index="year", columns="feature_id", values="Drought_Days"
    )

    # Keep districts with ≥5 years
    min_years = 5
    valid_cols = pivot.columns[pivot.notna().sum() >= min_years]
    pivot = pivot[valid_cols]
    print(f"  Districts with ≥{min_years} years data: {len(valid_cols)}")

    # Fill NaN with column mean
    pivot_filled = pivot.fillna(pivot.mean())

    # Compute full Pearson correlation
    corr = pivot_filled.corr()
    print(f"  Full correlation matrix: {corr.shape[0]}×{corr.shape[1]}")

    # Stats
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    vals = corr.values[mask]
    print(f"  Mean inter-district correlation: {vals.mean():.3f}")
    print(f"  Cross-group pairs with r≥0.5: {(vals >= 0.5).sum() // 2:,}")

    # Prepare geometry
    print("\n  Preparing map geometry …")
    gdf_plot = gdf_districts.copy()
    gdf_plot["ADM_NAME"] = gdf_plot["ADM_NAME"].astype(str).str.strip().str.upper()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(tolerance=0.05, preserve_topology=True)
    gdf_plot = gdf_plot.to_crs(epsg=4326)

    valid_set = set(valid_cols)
    gdf_plot = gdf_plot[gdf_plot["ADM_NAME"].isin(valid_set)].drop_duplicates("ADM_NAME")
    gdf_plot["season_group"] = gdf_plot["ADM_NAME"].map(district_to_group).fillna("Unknown")
    gdf_plot["COUNTRY_LABEL"] = gdf_plot["ADM_NAME"].map(district_to_country).fillna("Unknown")
    print(f"  Districts in map: {len(gdf_plot)}")

    # Build data for HTML
    geojson = json.loads(
        gdf_plot[["ADM_NAME", "COUNTRY_LABEL", "geometry", "season_group"]].to_json()
    )

    # Full correlation lookup
    corr_lookup = {}
    for dist in valid_cols:
        row = corr.loc[dist].drop(dist)
        corr_lookup[dist] = {
            k: round(v if not np.isnan(v) else 0.0, 2)
            for k, v in row.items()
        }

    # Time series
    ts_data = {}
    for dist, sub in df_annual.groupby("feature_id"):
        if dist in valid_set:
            ts_data[dist] = dict(zip(
                sub["year"].astype(int).tolist(),
                sub["Drought_Days"].astype(int).tolist()
            ))

    # Write HTML
    print("  Generating interactive HTML …")
    _write_intergroup_html(geojson, corr_lookup, ts_data, district_to_group, district_to_country)


def _write_intergroup_html(geojson, corr_lookup, ts_data, district_to_group, district_to_country):
    """Write a clean interactive district correlation map (Pearson, all pairs)."""

    geojson_str = json.dumps(geojson)
    corr_str = json.dumps(corr_lookup)
    ts_str = json.dumps(ts_data)
    groups_str = json.dumps(district_to_group)
    countries_str = json.dumps(district_to_country)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>African Maize — District Drought Correlation (Pearson r)</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:'Inter','Segoe UI',system-ui,sans-serif; background:#f0f2f5; }}
  #map {{ width:100%; height:100vh; }}

  .info-panel {{
    position:absolute; top:12px; right:12px; z-index:1000;
    background:white; padding:20px 24px; border-radius:12px;
    box-shadow:0 8px 32px rgba(0,0,0,0.12); max-width:400px; width:380px;
    font-size:13px; line-height:1.6; max-height:90vh; overflow-y:auto;
  }}
  .info-panel h3 {{ margin:0 0 4px 0; font-size:16px; color:#1a1a2e; }}
  .info-panel .subtitle {{ color:#666; font-size:11px; margin-bottom:12px; }}

  .section-title {{
    font-size:12px; font-weight:700; color:#333; margin:14px 0 6px 0;
    border-bottom:1px solid #eee; padding-bottom:4px;
  }}
  .corr-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  .corr-table td {{ padding:4px 6px; border-bottom:1px solid #f5f5f5; }}
  .corr-table .dist-name {{ max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .corr-table .country {{ color:#888; font-size:10px; }}
  .corr-table .rval {{ text-align:right; font-weight:700; font-family:monospace; font-size:12px; }}

  .legend {{
    position:absolute; bottom:30px; left:12px; z-index:1000;
    background:white; padding:14px 18px; border-radius:10px;
    box-shadow:0 4px 16px rgba(0,0,0,0.1); font-size:11px;
  }}
  .legend h4 {{ margin:0 0 8px 0; font-size:12px; color:#1a1a2e; }}
  .legend-item {{ display:flex; align-items:center; margin:3px 0; }}
  .legend-color {{ width:24px; height:12px; margin-right:10px; border-radius:2px; }}

  .sparkline {{ display:flex; align-items:flex-end; gap:1px; height:35px; margin:8px 0; }}
  .spark-bar {{ width:8px; background:#4361ee; border-radius:2px 2px 0 0; min-height:2px; }}

  .tag {{ display:inline-block; padding:2px 8px; border-radius:10px;
          font-size:10px; font-weight:600; color:white; background:#666; }}

  .references {{
    position:absolute; bottom:8px; right:12px; z-index:1000;
    background:rgba(255,255,255,0.95); padding:8px 12px; border-radius:8px;
    font-size:9px; color:#777; max-width:480px; line-height:1.5;
  }}
</style>
</head>
<body>
<div id="map"></div>

<div class="info-panel" id="info">
  <h3>🌍 District Drought Correlation</h3>
  <div class="subtitle">Pearson r on annual SSI drought-day counts (2000–2025).<br>
  Click any district to see its correlation with all others.</div>
  <div id="details">
    <p style="color:#999;">Select a district on the map to view correlations.</p>
  </div>
</div>

<div class="legend">
  <h4>Pearson Correlation (r)</h4>
  <div class="legend-item"><div class="legend-color" style="background:#a50f15"></div> r ≥ 0.8 — Very Strong</div>
  <div class="legend-item"><div class="legend-color" style="background:#fb6a4a"></div> 0.6 ≤ r < 0.8 — Strong</div>
  <div class="legend-item"><div class="legend-color" style="background:#fcae91"></div> 0.4 ≤ r < 0.6 — Moderate</div>
  <div class="legend-item"><div class="legend-color" style="background:#fee0d2"></div> 0.2 ≤ r < 0.4 — Weak</div>
  <div class="legend-item"><div class="legend-color" style="background:#f0f0f0"></div> −0.2 < r < 0.2 — Negligible</div>
  <div class="legend-item"><div class="legend-color" style="background:#bdd7e7"></div> −0.4 ≤ r < −0.2 — Weak Inverse</div>
  <div class="legend-item"><div class="legend-color" style="background:#6baed6"></div> −0.6 ≤ r < −0.4 — Moderate Inverse</div>
  <div class="legend-item"><div class="legend-color" style="background:#08306b"></div> r < −0.6 — Strong Inverse</div>
</div>

<div class="references">
  <b>Method:</b> Pearson r on annual drought-day counts (SSI ≤ −1.0) per district, 2000–2025. &nbsp;
  <b>Refs:</b> AghaKouchak (2014) HESS; McKee et al. (1993); ARC/African Union (2024); Carrão et al. (2016).
</div>

<script>
const geojson = {geojson_str};
const corrData = {corr_str};
const tsData = {ts_str};
const districtGroups = {groups_str};
const districtCountries = {countries_str};

const map = L.map('map').setView([0, 20], 4);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}@2x.png', {{
  attribution: '© CARTO'
}}).addTo(map);

function corrColor(r) {{
  if (r >= 0.8) return "#a50f15";
  if (r >= 0.6) return "#fb6a4a";
  if (r >= 0.4) return "#fcae91";
  if (r >= 0.2) return "#fee0d2";
  if (r > -0.2) return "#f0f0f0";
  if (r >= -0.4) return "#bdd7e7";
  if (r >= -0.6) return "#6baed6";
  return "#08306b";
}}

function displayName(name) {{
  const country = districtCountries[name] || "";
  return country ? name + " (" + country + ")" : name;
}}

function defaultStyle(feature) {{
  return {{
    fillColor: "#d9d9d9",
    weight: 0.4,
    color: "#aaa",
    fillOpacity: 0.5
  }};
}}

let selected = null;
const layerLookup = {{}};

const geoLayer = L.geoJSON(geojson, {{
  style: defaultStyle,
  onEachFeature: function(feature, layer) {{
    const name = feature.properties.ADM_NAME;
    layerLookup[name] = layer;
    layer.bindTooltip(displayName(name), {{sticky:true, direction:'top', className:'tooltip'}});
    layer.on('click', function() {{ selectDistrict(name); }});
  }}
}}).addTo(map);

function makeSparkline(dist) {{
  if (!tsData[dist]) return "";
  const years = Object.keys(tsData[dist]).sort();
  const vals = years.map(y => tsData[dist][y]);
  const maxV = Math.max(...vals, 1);
  let html = '<div class="sparkline">';
  vals.forEach((v, i) => {{
    const h = Math.max(2, (v / maxV) * 35);
    html += `<div class="spark-bar" style="height:${{h}}px" title="${{years[i]}}: ${{v}} days"></div>`;
  }});
  html += '</div>';
  return html;
}}

function selectDistrict(name) {{
  selected = name;
  const grp = districtGroups[name] || "—";
  const country = districtCountries[name] || "";
  const corrs = corrData[name] || {{}};

  // Reset styles
  geoLayer.eachLayer(l => l.setStyle(defaultStyle(l.feature)));

  // Highlight selected district
  if (layerLookup[name]) {{
    layerLookup[name].setStyle({{fillColor:"#ffd700", weight:2.5, color:"#333", fillOpacity:0.95}});
    layerLookup[name].bringToFront();
  }}

  // Colour all districts by correlation
  const sorted = Object.entries(corrs).sort((a,b) => b[1]-a[1]);
  for (const [dist, r] of sorted) {{
    if (layerLookup[dist]) {{
      layerLookup[dist].setStyle({{
        fillColor: corrColor(r),
        weight: 0.3,
        color: "#888",
        fillOpacity: 0.8
      }});
    }}
  }}

  // Build info panel
  const topPos = sorted.filter(([d,r]) => r > 0).slice(0, 8);
  const topNeg = sorted.filter(([d,r]) => r < 0).reverse().slice(0, 5);

  let html = `<h3>${{name}}</h3>`;
  if (country) html += `<div class="subtitle">${{country}} · ${{grp}}</div>`;
  else html += `<div class="subtitle">${{grp}}</div>`;

  html += makeSparkline(name);
  html += `<div style="font-size:10px;color:#888;margin-bottom:8px;">Annual drought days (2000–2025)</div>`;

  // Positive correlations
  html += '<div class="section-title">🔴Highest Positive Correlations</div>';
  html += '<table class="corr-table">';
  for (const [d, r] of topPos) {{
    const c = districtCountries[d] || "";
    html += `<tr>`;
    html += `<td class="dist-name">${{d}}<br><span class="country">${{c}} · ${{districtGroups[d] || ""}}</span></td>`;
    html += `<td class="rval" style="color:${{corrColor(r)}}">${{r.toFixed(2)}}</td>`;
    html += `</tr>`;
  }}
  if (topPos.length === 0) html += '<tr><td colspan="2" style="color:#ccc;">None</td></tr>';
  html += '</table>';

  // Negative correlations
  html += '<div class="section-title">🔵 Most Negative Correlations</div>';
  html += '<table class="corr-table">';
  for (const [d, r] of topNeg) {{
    const c = districtCountries[d] || "";
    html += `<tr>`;
    html += `<td class="dist-name">${{d}}<br><span class="country">${{c}} · ${{districtGroups[d] || ""}}</span></td>`;
    html += `<td class="rval" style="color:${{corrColor(r)}}">${{r.toFixed(2)}}</td>`;
    html += `</tr>`;
  }}
  if (topNeg.length === 0) html += '<tr><td colspan="2" style="color:#ccc;">None (all positive)</td></tr>';
  html += '</table>';

  // Summary stats
  const allR = sorted.map(([d,r]) => r);
  const meanR = allR.length > 0 ? (allR.reduce((a,b) => a+b, 0) / allR.length).toFixed(3) : "—";
  const nPos = allR.filter(r => r >= 0.4).length;
  const nNeg = allR.filter(r => r < -0.2).length;
  html += `<div style="margin-top:12px;font-size:10px;color:#666;border-top:1px solid #eee;padding-top:8px;">`;
  html += `Mean r: ${{meanR}} · Strong positive (r≥0.4): ${{nPos}} · Negative (r<−0.2): ${{nNeg}}`;
  html += `</div>`;

  document.getElementById('details').innerHTML = html;
}}

// Double-click to deselect
map.on('dblclick', function() {{
  selected = null;
  geoLayer.eachLayer(l => l.setStyle(defaultStyle(l.feature)));
  document.getElementById('details').innerHTML = '<p style="color:#999;">Select a district on the map to view correlations.</p>';
}});
</script>
</body>
</html>"""

    output_path = os.path.join(OUTPUT_DIR, "Africa_Maize_GammaSSI_InterGroup_Correlation_Map.html")
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(html)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n  ✅ Inter-group correlation map saved: {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")
    print(f"     Click any district to see cross-group teleconnections")


# ---------------------------------------------------------------------------
# STEP 13: Group-level correlation matrix (Spearman)
# ---------------------------------------------------------------------------
# WHY SPEARMAN (not Pearson)?
#   - Drought-day distributions are typically skewed and bounded (0 to ~season_length)
#   - Spearman rank correlation is robust to outliers and non-linear monotonic
#     relationships — appropriate for hydroclimatic indices
#   - Recommended by: Wilhite & Glantz (1985); Vicente-Serrano et al. (2012)
#   - ARC uses rank-based measures for portfolio diversification assessment
#
# WHAT THIS SHOWS:
#   - Aggregate each group's drought time series (mean drought days across
#     all districts in the group per year) → 25-year group-level series
#   - Compute Spearman ρ between all group pairs → 6×6 matrix
#   - High ρ = groups experience drought simultaneously (systemic risk)
#   - Low/negative ρ = natural hedge (portfolio diversification opportunity)
# ---------------------------------------------------------------------------

def generate_group_correlation_matrix(df_annual, district_to_group):
    """
    STEP 13: Compute Spearman rank correlation matrix at group level
    and generate a static HTML heatmap.

    Aggregates mean drought days per group per year, then computes
    pairwise Spearman ρ. Visualised as an interactive heatmap.

    Output: Africa_Maize_GammaSSI_Group_Correlation_Matrix.png
    """
    print("\n" + "=" * 70)
    print("STEP 13: Group-level Spearman correlation matrix")
    print("=" * 70)
    print("  Method: Spearman ρ on mean drought-days per group per year")
    print("  Rationale: rank-based, robust to skewed drought distributions")
    print("  Ref: Vicente-Serrano et al. (2012); Wilhite & Glantz (1985)")

    # Attach group to df_annual (using pre-computed GEOGLAM-derived mapping)
    df = df_annual.copy()
    df["season_group"] = df["feature_id"].map(district_to_group)
    df = df.dropna(subset=["season_group"])

    # Aggregate: mean drought days per group per year
    group_annual = df.groupby(["season_group", "year"])["Drought_Days"].mean().reset_index()
    pivot = group_annual.pivot_table(index="year", columns="season_group", values="Drought_Days")

    # Only include groups that actually exist in GEOGLAM data (data-driven)
    groups_present = [g for g in pivot.columns if pivot[g].notna().sum() >= 3]
    pivot = pivot[groups_present]

    # Count districts per group for display
    group_counts = df.groupby("season_group")["feature_id"].nunique()
    print(f"  Groups present in GEOGLAM data: {len(groups_present)}")
    for i, g in enumerate(groups_present):
        count = group_counts.get(g, 0)
        print(f"    Group {i+1}: {g} planters — {count} districts")
    print(f"  Years of data: {len(pivot)}")

    # Spearman correlation
    from scipy.stats import spearmanr
    n = len(groups_present)
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                x = pivot[groups_present[i]].values
                y = pivot[groups_present[j]].values
                # Drop pairs where either is NaN
                valid_mask = ~(np.isnan(x) | np.isnan(y))
                if valid_mask.sum() >= 5:
                    rho, pval = spearmanr(x[valid_mask], y[valid_mask])
                    corr_matrix[i, j] = rho
                    pval_matrix[i, j] = pval
                else:
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = 1.0

    # Print matrix
    print(f"\n  Spearman ρ correlation matrix:")
    print(f"  {'':>10s}", end="")
    for i in range(n):
        print(f"  Grp{i+1:>2d}", end="")
    print()
    for i in range(n):
        print(f"  Grp {i+1:>2d}  ", end="")
        for j in range(n):
            v = corr_matrix[i, j]
            if np.isnan(v):
                print(f"    N/A ", end="")
            else:
                star = "*" if pval_matrix[i, j] < 0.05 else " "
                print(f" {v:>6.3f}{star}", end="")
        print()
    print(f"\n  (* = p < 0.05)")

    # Generate static PNG heatmap
    _write_group_matrix_png(groups_present, corr_matrix, pval_matrix, group_counts)


def _write_group_matrix_png(groups, corr_matrix, pval_matrix, group_counts):
    """Generate a publication-quality static PNG heatmap of group-level Spearman ρ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    n = len(groups)
    group_labels = [f"Group {i+1}" for i in range(n)]
    group_descriptions = list(groups)  # e.g. "Jan–Feb planters"
    group_district_counts = [int(group_counts.get(g, 0)) for g in groups]

    # Y-axis: full detail (horizontal, plenty of room)
    y_labels = [
        f"Grp {i+1}: {group_descriptions[i]}  ({group_district_counts[i]} districts)"
        for i in range(n)
    ]
    # X-axis: short labels only (rotated 45° to avoid overlap)
    x_labels = [f"Grp {i+1}: {group_descriptions[i].replace(' planters','')}" for i in range(n)]

    # Colour map: diverging blue-white-red centred at 0
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # Group colours for label accents
    group_colors = [GROUP_COLOR_MAP.get(g, "#999999") for g in groups]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Plot heatmap cells
    im = ax.imshow(corr_matrix, cmap=cmap, norm=norm, aspect="equal")

    # Annotate each cell with ρ value and significance
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "#888"
            else:
                sig = "**" if pval_matrix[i, j] < 0.01 else ("*" if pval_matrix[i, j] < 0.05 else "")
                text = f"{val:.2f}{sig}"
                color = "white" if abs(val) > 0.5 else "#1a1a2e"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # X-axis: rotated 45° for readability
    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha="left")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    # Y-axis: horizontal, full detail
    ax.set_yticks(range(n))
    ax.set_yticklabels(y_labels, fontsize=10)

    # Color the tick labels by group
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(group_colors[i])
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(group_colors[i])

    # Grid lines between cells
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2.5)
        ax.axvline(i - 0.5, color="white", linewidth=2.5)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Title
    ax.set_title(
        "Season Group Correlation Matrix (Spearman ρ)\n"
        "African Maize — Gamma-CDF SSI Drought Days",
        fontsize=13, fontweight="bold", pad=50
    )

    # Footnotes
    footnote = (
        "* p < 0.05  |  ** p < 0.01  |  "
        "Spearman rank correlation on mean drought-days/group/year (2000–2025)\n"
        "Refs: Vicente-Serrano et al. (2012); Wilhite & Glantz (1985); "
        "McKee et al. (1993); AghaKouchak (2014)"
    )
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    output_path = os.path.join(OUTPUT_DIR, "Africa_Maize_GammaSSI_Group_Correlation_Matrix.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"\n  ✅ Group correlation matrix saved: {output_path}")
    print(f"     File size: {file_size_kb:.0f} KB")
    print(f"     Format: static PNG (publication-ready)")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  GAMMA-SSI DROUGHT ANALYSIS PIPELINE                                ║
║  Daily SSI via Gamma-CDF (AghaKouchak 2014, McKee et al. 1993)     ║
║  Threshold: SSI ≤ −1.0 = 15.9th percentile (WMO Moderate Drought) ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Load data
    gdf_districts, gdf_geoglam = load_shapefiles()

    # Optional: filter districts by pixel-overlap quality
    # Set env var MIN_QUALITY="Medium" to enable, or change the config above
    if MIN_QUALITY is not None:
        gdf_districts = filter_districts_by_quality(gdf_districts, min_quality=MIN_QUALITY)

    # Filter to Maize 1
    gdf_maize1 = filter_maize1(gdf_geoglam)

    # Spatial join
    valid_maize = join_districts_to_maize1(gdf_districts, gdf_maize1)

    # Risk windows
    valid_maize = calculate_risk_windows(valid_maize)

    # Load ERA5 and merge
    df_filtered = load_and_merge_era5(valid_maize)

    # Compute daily Gamma-CDF SSI
    df_filtered = compute_daily_gamma_ssi(df_filtered)

    # Aggregate to annual drought days
    df_annual = aggregate_annual_drought_days(df_filtered)

    df_annual.to_csv(os.path.join(OUTPUT_DIR, "drought_annual.csv"), index=False)
    valid_maize.to_file(os.path.join(OUTPUT_DIR, "valid_maize.geojson"), driver="GeoJSON")
    
    # Classify severity using published bins
    df_annual = bin_drought_days(df_annual)

    # Generate animated map
    generate_animated_map(df_annual, gdf_districts)

    # Classify season groups ONCE from GEOGLAM data (used by all subsequent maps)
    groups, district_to_group = classify_season_groups(valid_maize)

    # Build district → country lookup for display in maps
    district_to_country = {}
    country_col = "COUNTRY" if "COUNTRY" in valid_maize.columns else "country"
    for _, row in valid_maize[["ADM_NAME", country_col]].drop_duplicates("ADM_NAME").iterrows():
        district_to_country[row["ADM_NAME"]] = str(row[country_col]).strip().title()
    print(f"\n  District-to-country mapping: {len(district_to_country)} districts across "
          f"{len(set(district_to_country.values()))} countries")

    # Generate static season-group map
    generate_season_group_map(valid_maize, gdf_districts, district_to_group, district_to_country)

    # Compute within-group district correlations and generate map
    generate_correlation_map(df_annual, valid_maize, gdf_districts, groups, district_to_group, district_to_country)

    # Compute inter-group (continental) correlations and generate map
    generate_intergroup_correlation_map(df_annual, valid_maize, gdf_districts, district_to_group, district_to_country)

    # Generate group-level Spearman correlation matrix (PNG)
    generate_group_correlation_matrix(df_annual, district_to_group)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  ✅ {df_annual['feature_id'].nunique()} districts processed")
    print(f"  ✅ {len(df_annual):,} district-year records")
    print(f"  ✅ Daily SSI computed via Gamma-CDF (WMO moderate drought threshold: −1.0)")
    print(f"\n  Generated outputs:")
    print(f"     - Africa_Maize_GammaSSI_Drought_Map.html")
    print(f"     - Africa_Maize_Season_Groups_Map.html")
    print(f"     - Africa_Maize_GammaSSI_Correlation_Map.html")
    print(f"     - Africa_Maize_GammaSSI_InterGroup_Correlation_Map.html")
    print(f"     - Africa_Maize_GammaSSI_Group_Correlation_Matrix.png")

    return df_annual


if __name__ == "__main__":
    main()
