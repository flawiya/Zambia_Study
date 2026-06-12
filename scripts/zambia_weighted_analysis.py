"""
zambia_weighted_analysis.py — Weighted vs Centroid SSI Comparison for Zambia
=============================================================================
Purpose:
  Apply the overlapping pixel solver to Zambia (70 districts) and quantify
  how much the SSI results change when using area-weighted pixel assignment
  versus the current centroid-based GEOGLAM calendar assignment.

Pipeline:
  1. Load Zambia districts + ERA5 data
  2. Run overlap analysis → district quality metrics
  3. Compare weighted vs centroid GEOGLAM calendar assignment
  4. Re-compute SSI with both calendars → compare drought day counts
  5. Sensitivity: exclude "Low" quality districts → do results hold?

Outputs:
  - zambia_calendar_comparison.csv    — per-district calendar dates
  - zambia_ssi_comparison.csv         — per-district-year drought days
  - zambia_comparison_report.html     — interactive scatter/bar charts

References:
  Openshaw (1984) — MAUP theoretical framework
  Fisher & Langford (1995) — area-weighted interpolation
  McKee et al. (1993) — SSI/SPI methodology
  AghaKouchak (2014) — SSI for drought monitoring
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from utils.overlapping_pixel_solver import (
    create_era5_fishnet,
    compute_pixel_district_overlaps,
    cache_overlap_table,
    load_cached_overlaps,
    compute_district_quality_metrics,
    assign_geoglam_weighted,
    filter_by_quality,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "zambia_weighted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GADM_PATH = DATA_DIR / "africa-agricultural-domain-2019" / "africa_agricultural_domain_2019.shp"
GEOGLAM_PATH = DATA_DIR / "GEOGLAM" / "GEOGLAM_CM4EW_Calendars_V1.4.shp"
ERA5_PATH = DATA_DIR / "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"

SSI_THRESHOLD = -1.0


# ===========================================================================
# STEP 1: Load data for Zambia
# ===========================================================================
def load_zambia_data():
    """
    Load and prepare Zambia-specific data.

    Returns
    -------
    zmb_districts : gpd.GeoDataFrame — 70 Zambia districts
    zmb_era5     : pd.DataFrame — daily ERA5 for Zambia
    geoglam_maize: gpd.GeoDataFrame — Maize 1 calendar zones
    """
    print("=" * 70)
    print("STEP 1: Loading Zambia data")
    print("=" * 70)

    # Load districts
    districts = gpd.read_file(GADM_PATH)
    districts["ADM_NAME"] = districts["ADM_NAME"].astype(str).str.strip().str.upper()
    zmb_districts = districts[districts["ISO3"] == "ZMB"].copy()
    print(f"\n  Zambia districts:       {len(zmb_districts)}")
    print(f"  Crop area range:        {zmb_districts['crop_pct'].min():.1f}–{zmb_districts['crop_pct'].max():.1f}%")
    print(f"  Mean crop area:         {zmb_districts['crop_pct'].mean():.1f}%")

    # Load GEOGLAM
    geoglam = gpd.read_file(GEOGLAM_PATH)
    geoglam_maize = geoglam[geoglam["crop"] == "Maize 1"].copy()
    print(f"\n  Maize 1 polygons:       {len(geoglam_maize)}")

    # Load ERA5 (only Zambia districts — filter during read for memory efficiency)
    print(f"\n  Loading ERA5 from: {ERA5_PATH}")
    dtype = {"year": "int16", "month": "int8", "day": "int8", "doy": "int16",
             "volumetric_soil_water_layer_2": "float32"}
    zmb_names = set(zmb_districts["ADM_NAME"])
    # Read in chunks to avoid memory issues with the 1.4GB file
    chunks = []
    for chunk in pd.read_csv(ERA5_PATH, dtype=dtype, chunksize=500000):
        chunk["feature_id"] = chunk["feature_id"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["feature_id"].isin(zmb_names)]
        if not chunk.empty:
            chunks.append(chunk)
    zmb_era5 = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"\n  ERA5 rows for Zambia:   {len(zmb_era5):,}")
    print(f"  Year range:             {zmb_era5['year'].min()}–{zmb_era5['year'].max()}")

    return zmb_districts, zmb_era5, geoglam_maize


# ===========================================================================
# STEP 2: Run overlap analysis
# ===========================================================================
def run_zambia_overlap_analysis(zmb_districts):
    """
    Apply the overlapping pixel solver to Zambia.

    Returns
    -------
    zmb_with_quality : gpd.GeoDataFrame — districts with quality metrics
    overlap_df       : pd.DataFrame — pixel-district weight table
    """
    print("\n" + "=" * 70)
    print("STEP 2: Overlap analysis for Zambia")
    print("=" * 70)

    # Try cached overlaps first
    overlap_df = load_cached_overlaps()
    if overlap_df is not None:
        valid = set(zmb_districts["ADM_NAME"])
        overlap_df = overlap_df[overlap_df["ADM_NAME"].isin(valid)].copy()
        print(f"  Loaded cached: {len(overlap_df)} overlap pairs")
    else:
        # Create fishnet for Zambia
        fishnet = create_era5_fishnet(zmb_districts.total_bounds)
        overlap_df = compute_pixel_district_overlaps(fishnet, zmb_districts)
        cache_overlap_table(overlap_df)

    # Quality metrics
    zmb_with_quality = compute_district_quality_metrics(overlap_df, zmb_districts)

    # Summary for Zambia
    print(f"\n  Zambia quality distribution:")
    for qc in ["High", "Medium", "Low"]:
        count = (zmb_with_quality["quality_class"] == qc).sum()
        if count > 0:
            pct = (zmb_with_quality[zmb_with_quality["quality_class"] == qc]["crop_pct"].mean()
                   if count > 0 else 0)
            print(f"    {qc:<8s}: {count:>2} districts, mean crop_pct = {pct:.1f}%")

    return zmb_with_quality, overlap_df


# ===========================================================================
# STEP 3: Compare centroid vs weighted GEOGLAM calendar
# ===========================================================================
def compare_calendars(zmb_districts, geoglam_maize):
    """
    Compare centroid-based vs area-weighted GEOGLAM calendar assignment.

    The "centroid" method: each district's centroid → one GEOGLAM zone.
    The "weighted" method: area-weighted average of all overlapping zones.

    Returns
    -------
    pd.DataFrame with both calendar assignments per district.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Centroid vs Weighted GEOGLAM calendar")
    print("=" * 70)

    # --- Centroid method (current Week_11 approach) ---
    print("\n  Computing centroid-based assignment …")
    zmb_proj = zmb_districts.to_crs(epsg=3857)
    zmb_proj["geometry"] = zmb_proj.centroid
    zmb_centroids = zmb_proj.to_crs(epsg=4326)

    # Ensure GEOGLAM is in WGS84
    if geoglam_maize.crs != zmb_centroids.crs:
        geoglam_wgs84 = geoglam_maize.to_crs(epsg=4326)
    else:
        geoglam_wgs84 = geoglam_maize

    centroid_join = gpd.sjoin(
        zmb_centroids, geoglam_wgs84, how="left", predicate="within"
    )
    centroid_cal = centroid_join[["ADM_NAME", "planting", "harvest", "endofseaso"]].copy()
    centroid_cal.columns = ["ADM_NAME", "planting_centroid", "harvest_centroid", "endofseaso_centroid"]

    # --- Weighted method ---
    print("  Computing area-weighted assignment …")
    weighted_cal = assign_geoglam_weighted(zmb_districts, geoglam_maize)
    weighted_cols = ["ADM_NAME", "planting", "harvest", "endofseaso", "n_geoglam_zones", "dominant_crop"]
    weighted_cal = weighted_cal[[c for c in weighted_cols if c in weighted_cal.columns]].copy()
    # Merge keeps the weighted version with _weighted suffix already

    # --- Merge and compare ---
    comparison = zmb_districts[["ADM_NAME"]].merge(
        centroid_cal, on="ADM_NAME", how="left"
    )
    comparison = comparison.merge(weighted_cal, on="ADM_NAME", how="left")

    # Handle NaNs (districts where weighted didn't find GEOGLAM overlap)
    for col in ["planting", "planting_centroid"]:
        if col in comparison.columns:
            comparison[col] = comparison[col].fillna(0).astype(int)
    for col in ["harvest", "harvest_centroid", "endofseaso", "endofseaso_centroid"]:
        if col in comparison.columns:
            comparison[col] = comparison[col].fillna(0).astype(int)

    # Calculate absolute difference (only for districts with both)
    mask = (comparison["planting"] > 0) & (comparison["planting_centroid"] > 0)
    if mask.any():
        comparison["planting_diff"] = np.abs(
            comparison["planting"] - comparison["planting_centroid"]
        )
        comparison["planting_diff"].mask(~mask, np.nan, inplace=True)
        mean_diff = comparison["planting_diff"].mean()
        max_diff = comparison["planting_diff"].max()
        print(f"\n  Planting date comparison:")
        print(f"    Mean absolute diff: {mean_diff:.1f} DOY")
        print(f"    Max absolute diff:  {max_diff:.0f} DOY")
        print(f"    Districts with difference > 5 DOY: "
              f"{(comparison['planting_diff'] > 5).sum()}")

        # Pearson correlation
        valid = comparison[mask]
        r, p = scipy_stats.pearsonr(valid["planting"], valid["planting_centroid"])
        print(f"    Pearson r:          {r:.4f} (p={p:.4f})")

    # Show biggest differences
    print(f"\n  Districts with largest calendar differences:")
    if "planting_diff" in comparison.columns:
        top_diff = comparison.dropna(subset=["planting_diff"]).nlargest(5, "planting_diff")
        for _, row in top_diff.iterrows():
            print(f"    {row['ADM_NAME']:<30s}: centroid={row['planting_centroid']:.0f}, "
                  f"weighted={row['planting']:.0f}, diff={row['planting_diff']:.0f} DOY")

    # Save
    comp_path = OUTPUT_DIR / "zambia_calendar_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")

    return comparison


# ===========================================================================
# STEP 4: Compare SSI outcomes
# ===========================================================================
def compute_ssi_comparison(zmb_era5, calendar_comparison, zmb_with_quality):
    """
    Compute annual drought days using both centroid and weighted calendars,
    then compare the results per district.

    Simplified SSI: uses z-score method (like Week_10) for speed.
    The Gamma-CDF method (Week_11) can be substituted later.

    Risk window: skip first 25% of the season (matching Week_11 methodology).
    """
    print("\n" + "=" * 70)
    print("STEP 4: SSI comparison — centroid vs weighted calendar")
    print("=" * 70)

    if zmb_era5.empty:
        print("  ⚠ No ERA5 data loaded — skipping SSI comparison")
        return pd.DataFrame()

    # Compute SSI via z-score method (fast, adequate for comparison)
    print("\n  Computing daily SSI (z-score method) …")
    clim = zmb_era5.groupby(["feature_id", "doy"])[
        "volumetric_soil_water_layer_2"
    ].agg(["mean", "std"]).reset_index()
    df = zmb_era5.merge(clim, on=["feature_id", "doy"], how="left")
    df["SSI"] = (df["volumetric_soil_water_layer_2"] - df["mean"]) / (df["std"] + 1e-6)

    # Ensure crop_year alignment (Nov-Dec planting → next year harvest)
    df["crop_year"] = np.where(df["month"] >= 11, df["year"] + 1, df["year"])

    def _risk_window_doy(planting, endofseaso):
        """Calculate risk start DOY (skip first 25% of season)."""
        if planting <= 0 or endofseaso <= 0:
            return -1, -1
        if endofseaso >= planting:
            duration = endofseaso - planting
        else:
            duration = (365 - planting) + endofseaso
        delay = duration * 0.25
        risk_start = int(round((planting + delay) % 365))
        return risk_start, int(endofseaso)

    def _is_in_risk_window(doy, risk_start, risk_end):
        """Check if DOY falls in risk window (handles year-wrapped seasons)."""
        if risk_start < 0:
            return False
        if risk_start <= risk_end:
            return risk_start <= doy <= risk_end
        else:
            return doy >= risk_start or doy <= risk_end

    # ---- Scenario A: centroid calendar ----
    print("  Scenario A: centroid-based risk window …")
    centroids_ok = calendar_comparison[
        calendar_comparison["planting_centroid"] > 0
    ][["ADM_NAME", "planting_centroid", "endofseaso_centroid"]].dropna()
    centroids_ok[["risk_start", "risk_end"]] = centroids_ok.apply(
        lambda r: _risk_window_doy(r["planting_centroid"], r["endofseaso_centroid"]),
        axis=1, result_type="expand"
    )

    df_a = df.merge(
        centroids_ok[["ADM_NAME", "risk_start", "risk_end"]],
        left_on="feature_id", right_on="ADM_NAME", how="inner"
    )
    in_window_a = df_a.apply(
        lambda r: _is_in_risk_window(r["doy"], r["risk_start"], r["risk_end"]),
        axis=1
    )
    df_a = df_a[in_window_a]
    annual_a = (
        df_a.groupby(["feature_id", "crop_year"])["SSI"]
        .apply(lambda x: (x <= SSI_THRESHOLD).sum())
        .reset_index(name="drought_days_centroid")
    )
    print(f"    Districts: {annual_a['feature_id'].nunique()}, "
          f"years: {annual_a['crop_year'].nunique()}")

    # ---- Scenario B: weighted calendar ----
    print("  Scenario B: weighted risk window …")
    weighted_ok = calendar_comparison[
        calendar_comparison["planting"] > 0
    ][["ADM_NAME", "planting", "endofseaso"]].dropna()
    weighted_ok[["risk_start", "risk_end"]] = weighted_ok.apply(
        lambda r: _risk_window_doy(r["planting"], r["endofseaso"]),
        axis=1, result_type="expand"
    )

    df_b = df.merge(
        weighted_ok[["ADM_NAME", "risk_start", "risk_end"]],
        left_on="feature_id", right_on="ADM_NAME", how="inner"
    )
    in_window_b = df_b.apply(
        lambda r: _is_in_risk_window(r["doy"], r["risk_start"], r["risk_end"]),
        axis=1
    )
    df_b = df_b[in_window_b]
    annual_b = (
        df_b.groupby(["feature_id", "crop_year"])["SSI"]
        .apply(lambda x: (x <= SSI_THRESHOLD).sum())
        .reset_index(name="drought_days_weighted")
    )
    print(f"    Districts: {annual_b['feature_id'].nunique()}, "
          f"years: {annual_b['crop_year'].nunique()}")

    # ---- Merge and compare ----
    comparison = annual_a.merge(annual_b, on=["feature_id", "crop_year"], how="outer").fillna(0)
    comparison["difference"] = comparison["drought_days_weighted"] - comparison["drought_days_centroid"]
    comparison["abs_diff"] = comparison["difference"].abs()

    print(f"\n  Results:")
    print(f"    Districts compared:       {comparison['feature_id'].nunique()}")
    print(f"    District-years compared:  {len(comparison)}")
    print(f"    Mean drought days (centroid):  {comparison['drought_days_centroid'].mean():.1f}")
    print(f"    Mean drought days (weighted):  {comparison['drought_days_weighted'].mean():.1f}")
    print(f"    Mean absolute difference:      {comparison['abs_diff'].mean():.2f} days")
    print(f"    Max absolute difference:       {comparison['abs_diff'].max():.0f} days")

    # Quality class merge
    quality = zmb_with_quality[["ADM_NAME", "quality_class", "pct_full_pixels"]]
    comparison = comparison.merge(quality, left_on="feature_id", right_on="ADM_NAME", how="left")

    # Correlation between methods
    valid = comparison[(comparison["drought_days_centroid"] > 0) | (comparison["drought_days_weighted"] > 0)]
    if len(valid) > 5:
        r, p = scipy_stats.pearsonr(valid["drought_days_centroid"], valid["drought_days_weighted"])
        print(f"    Pearson r: {r:.4f} (p={p:.4f})")
        if r > 0.95:
            print(f"    ✅ Methods are essentially identical")
        elif r > 0.8:
            print(f"    ✅ Methods are strongly correlated")

    # Per-quality-class breakdown
    print(f"\n  Breakdown by quality class:")
    for qc in ["High", "Medium", "Low"]:
        subset = comparison[comparison["quality_class"] == qc]
        if len(subset) > 0:
            print(f"    {qc:<8s}: mean abs_diff = {subset['abs_diff'].mean():.2f} days "
                  f"(n={len(subset)})")

    comp_path = OUTPUT_DIR / "zambia_ssi_comparison.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")

    return comparison


# ===========================================================================
# STEP 5: Sensitivity — exclude low-quality districts
# ===========================================================================
def run_quality_sensitivity(zmb_era5, calendar_comparison, zmb_with_quality):
    """
    Test whether excluding "Low" quality districts changes the SSI conclusions.
    This is the critical validation for the professor: if results hold with
    only clean districts, the centroid method is validated.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Sensitivity analysis — filter by quality")
    print("=" * 70)

    # Full analysis
    print("\n  --- Full Zambia (all 70 districts) ---")
    full = compute_ssi_comparison(zmb_era5, calendar_comparison, zmb_with_quality)

    if full.empty or full["feature_id"].nunique() < 5:
        print("\n  ⚠ SSI data insufficient for quality sensitivity — "
              "run zambia_weighted_analysis.py directly with full ERA5 data.")
        return pd.DataFrame()

    # Filtered: exclude Low quality
    print("\n  --- Clean districts (High + Medium quality only) ---")
    clean_districts = zmb_with_quality[
        zmb_with_quality["quality_class"].isin(["High", "Medium"])
    ]["ADM_NAME"].tolist()
    print(f"  Districts kept: {len(clean_districts)} / {len(zmb_with_quality)}")

    cal_clean = calendar_comparison[calendar_comparison["ADM_NAME"].isin(clean_districts)].copy()
    quality_clean = zmb_with_quality[zmb_with_quality["ADM_NAME"].isin(clean_districts)].copy()
    clean = compute_ssi_comparison(zmb_era5, cal_clean, quality_clean)

    # Compare mean drought days
    if full.empty or clean.empty:
        print("\n  ⚠ Insufficient data for quality sensitivity comparison")
        return pd.DataFrame()

    full_mean = full.groupby("crop_year")["drought_days_centroid"].mean().reset_index()
    full_mean.columns = ["crop_year", "full_all"]
    clean_mean = clean.groupby("crop_year")["drought_days_centroid"].mean().reset_index()
    clean_mean.columns = ["crop_year", "full_clean"]

    merged = full_mean.merge(clean_mean, on="crop_year", how="outer")
    merged["diff"] = (merged["full_all"] - merged["full_clean"]).abs()

    mean_diff = merged["diff"].mean()
    max_diff = merged["diff"].max()

    print(f"\n  --- Quality sensitivity result ---")
    print(f"  Mean annual difference:  {mean_diff:.2f} drought days")
    print(f"  Max annual difference:   {max_diff:.2f} drought days")

    if len(merged) >= 2:
        r, p = scipy_stats.pearsonr(merged["full_all"], merged["full_clean"])
        print(f"  Pearson r (all vs clean): {r:.4f} (p={p:.4f})")
        if r > 0.95 and mean_diff < 5:
            print(f"  ✅ CONCLUSION: Results are robust to quality filtering.")
            print(f"     The centroid method is validated — low-quality districts")
            print(f"     do not bias the SSI signal.")
        else:
            print(f"  ⚠ CONCLUSION: Quality filtering changes results.")
            print(f"     Recommend using weighted GEOGLAM assignment.")

    merged.to_csv(OUTPUT_DIR / "zambia_quality_sensitivity.csv", index=False)
    return merged


# ===========================================================================
# STEP 6: Generate figures
# ===========================================================================
def generate_comparison_figures(calendar_comp, ssi_comp, quality_sens):
    """
    Generate dissertation-quality comparison figures.

    Figure 1: Calendar scatter (centroid vs weighted planting DOY)
    Figure 2: SSI scatter (centroid vs weighted drought days)
    Figure 3: Time series comparison (all vs clean districts)
    Figure 4: Difference map of Zambia (if possible with simple plot)
    """
    print("\n" + "=" * 70)
    print("STEP 6: Generating comparison figures")
    print("=" * 70)

    # --- Figure 1: Calendar comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    valid_cal = calendar_comp.dropna(subset=["planting", "planting_centroid"])
    valid_cal = valid_cal[(valid_cal["planting"] > 0) & (valid_cal["planting_centroid"] > 0)]
    ax.scatter(valid_cal["planting_centroid"], valid_cal["planting"],
               c="steelblue", alpha=0.7, edgecolors="white", linewidth=0.5)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Centroid-based planting DOY")
    ax.set_ylabel("Area-weighted planting DOY")
    ax.set_title("GEOGLAM Calendar: Centroid vs Weighted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Figure 2: SSI comparison ---
    ax = axes[0, 1]
    valid_ssi = ssi_comp[(ssi_comp["drought_days_centroid"] > 0) | (ssi_comp["drought_days_weighted"] > 0)]
    scatter = ax.scatter(
        valid_ssi["drought_days_centroid"],
        valid_ssi["drought_days_weighted"],
        c=valid_ssi["quality_class"].map({"High": "green", "Medium": "orange", "Low": "red"}),
        alpha=0.6, edgecolors="white", linewidth=0.3,
        label="Color = quality class"
    )
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="1:1 line")
    ax.set_xlabel("Drought days (centroid calendar)")
    ax.set_ylabel("Drought days (weighted calendar)")
    ax.set_title(f"SSI Drought Days: Centroid vs Weighted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Figure 3: Quality distribution ---
    ax = axes[1, 0]
    if "quality_class" in calendar_comp.columns:
        qc_counts = calendar_comp["quality_class"].value_counts()
        bars = ax.bar(
            ["High", "Medium", "Low"],
            [qc_counts.get("High", 0), qc_counts.get("Medium", 0), qc_counts.get("Low", 0)],
            color=["#1b9e77", "#d95f02", "#e7298a"]
        )
        ax.set_xlabel("Pixel-overlap quality class")
        ax.set_ylabel("Number of districts")
        ax.set_title("Zambia: District Quality Distribution")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{int(height)}", ha="center", va="bottom")

    # --- Figure 4: Time series sensitivity ---
    ax = axes[1, 1]
    if quality_sens is not None and len(quality_sens) > 0 and "full_all" in quality_sens.columns:
        ax.plot(quality_sens["crop_year"], quality_sens["full_all"],
                "o-", color="steelblue", label="All districts", linewidth=2)
        if "full_clean" in quality_sens.columns:
            ax.plot(quality_sens["crop_year"], quality_sens["full_clean"],
                    "s--", color="darkorange", label="Clean districts", linewidth=2)
        ax.set_xlabel("Crop year")
        ax.set_ylabel("Mean drought days")
        ax.set_title("Sensitivity to quality filtering")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "zambia_weighted_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: zambia_weighted_comparison.png")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("ZAMBIA WEIGHTED ANALYSIS — Centroid vs Area-Weighted Comparison")
    print("=" * 70)

    # Step 1: Load data
    zmb_districts, zmb_era5, geoglam_maize = load_zambia_data()

    # Step 2: Overlap analysis
    zmb_with_quality, overlap_df = run_zambia_overlap_analysis(zmb_districts)

    # Save quality data for later filtering
    zmb_with_quality.to_file(OUTPUT_DIR / "zambia_with_quality.geojson", driver="GeoJSON")

    # Step 3: Calendar comparison
    calendar_comp = compare_calendars(zmb_districts, geoglam_maize)

    # Add quality class to calendar comparison
    calendar_comp = calendar_comp.merge(
        zmb_with_quality[["ADM_NAME", "quality_class", "pct_full_pixels"]],
        on="ADM_NAME", how="left"
    )

    # Step 4: SSI comparison
    ssi_comp = compute_ssi_comparison(zmb_era5, calendar_comp, zmb_with_quality)

    # Step 5: Quality sensitivity
    quality_sens = run_quality_sensitivity(zmb_era5, calendar_comp, zmb_with_quality)

    # Step 6: Figures
    generate_comparison_figures(calendar_comp, ssi_comp, quality_sens)

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name:<50s} {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
