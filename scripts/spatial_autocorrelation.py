"""
spatial_autocorrelation.py — Moran's I & LISA Analysis for Zambia
==================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG - ROBUST PATHS
# ---------------------------------------------------------------------------
current_path = Path(__file__).resolve()
# Go up 2 levels from scripts/ to reach Africa-Drought-Study/
PROJECT_ROOT = current_path.parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Standardize Output Directory
OUTPUT_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "spatial_autocorrelation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Correct Path to your GADM data
GADM_PATH = DATA_DIR / "africa_agricultural_domain_2019.shp"

# Use the primary output from your Week 11 run
WEEK_11_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "Week_11_correlation_map"
DROUGHT_PATH = WEEK_11_DIR / "drought_annual.csv"
QUALITY_PATH = WEEK_11_DIR / "valid_maize.geojson"

COUNTRY_ISO3 = "ZMB"  # Zambia
SSI_THRESHOLD = -1.0

# LISA cluster type labels and colors
LISA_LABELS = {
    1: "High-High (drought cluster)",
    2: "Low-Low (safe zone)",
    3: "Low-High (drought surround)",
    4: "High-Low (isolated risk)",
    0: "Not significant",
}

LISA_COLORS = {
    "High-High (drought cluster)": "#e31a1c",
    "Low-Low (safe zone)": "#3182bd",
    "Low-High (drought surround)": "#b2df8a",
    "High-Low (isolated risk)": "#ffd92f",
    "Not significant": "#f0f0f0",
}

# ===========================================================================
# STEP 1: Load and merge data
# ===========================================================================
def load_zambia_data():
    print("=" * 70)
    print("STEP 1: Loading Zambia district + drought data")
    print("=" * 70)

    # Load district shapefile
    gdf = gpd.read_file(GADM_PATH)
    gdf["ADM_NAME"] = gdf["ADM_NAME"].astype(str).str.strip().str.upper()
    zmb = gdf[gdf["ISO3"] == COUNTRY_ISO3].copy()
    print(f"\n  Zambia districts loaded: {len(zmb)}")

    # Load annual drought data
    df = pd.read_csv(DROUGHT_PATH)
    df["feature_id"] = df["feature_id"].astype(str).str.strip().str.upper()
    zmb_drought = df[df["feature_id"].isin(zmb["ADM_NAME"])].copy()
    print(f"  Drought records for Zambia: {len(zmb_drought)}")

    # Safely load quality metrics if they exist
    if QUALITY_PATH.exists():
        quality_gdf = gpd.read_file(QUALITY_PATH)
        quality_gdf["ADM_NAME"] = quality_gdf["ADM_NAME"].astype(str).str.strip().str.upper()
        
        # Only take columns that actually exist
        target_cols = ["quality_class", "pct_full_pixels", "n_boundary_pixels"]
        available_cols = [c for c in target_cols if c in quality_gdf.columns]
        
        if available_cols:
            quality_sub = quality_gdf[["ADM_NAME"] + available_cols]
            zmb = zmb.merge(quality_sub, on="ADM_NAME", how="left")
            print(f"  Quality metrics merged: {available_cols}")

    # Ensure placeholder columns exist if merge didn't provide them
    if "quality_class" not in zmb.columns:
        zmb["quality_class"] = "Unknown"

    # Compute mean drought days per district
    mean_drought = (
        zmb_drought.groupby("feature_id")["Drought_Days"]
        .mean()
        .reset_index()
        .rename(columns={"feature_id": "ADM_NAME", "Drought_Days": "mean_drought_days"})
    )
    zmb = zmb.merge(mean_drought, on="ADM_NAME", how="left")
    zmb["mean_drought_days"] = zmb["mean_drought_days"].fillna(0)

    return zmb, zmb_drought

# ===========================================================================
# STEP 2: Build spatial weights
# ===========================================================================
def build_spatial_weights(gdf):
    print("\n" + "=" * 70)
    print("STEP 2: Building spatial weights matrix (Queen contiguity)")
    print("=" * 70)
    from libpysal.weights import Queen
    gdf = gdf.reset_index(drop=True)
    w = Queen.from_dataframe(gdf, silence_warnings=True)
    w.transform = "R" 
    return w

# ===========================================================================
# STEP 3: Global Moran's I
# ===========================================================================
def compute_global_morans_i(gdf, df_annual, w):
    print("\n" + "=" * 70)
    print("STEP 3: Global Moran's I")
    print("=" * 70)
    from esda.moran import Moran
    gdf = gdf.reset_index(drop=True)
    y = gdf["mean_drought_days"].values
    mi_mean = Moran(y, w)
    
    yearly_results = []
    for yr in sorted(df_annual["year"].unique()):
        yr_data = df_annual[df_annual["year"] == yr]
        yr_map = dict(zip(yr_data["feature_id"], yr_data["Drought_Days"]))
        y_yr = np.array([yr_map.get(name, 0) for name in gdf["ADM_NAME"]])
        mi = Moran(y_yr, w)
        yearly_results.append({"year": yr, "morans_i": mi.I, "p_value": mi.p_sim, "significant": mi.p_sim < 0.05})
    
    df_yearly = pd.DataFrame(yearly_results)
    df_yearly.to_csv(OUTPUT_DIR / "zambia_global_morans_i_yearly.csv", index=False)
    return df_yearly, mi_mean

# ===========================================================================
# STEP 4: Local Moran's I (LISA)
# ===========================================================================
def compute_lisa(gdf, w):
    print("\n" + "=" * 70)
    print("STEP 4: Local Moran's I (LISA)")
    print("=" * 70)
    from esda.moran import Moran_Local
    gdf = gdf.reset_index(drop=True)
    y = gdf["mean_drought_days"].values
    lisa = Moran_Local(y, w, permutations=999)
    sig = lisa.p_sim < 0.05
    gdf["lisa_cluster"] = np.where(sig, lisa.q, 0)
    gdf["lisa_label"] = gdf["lisa_cluster"].map(LISA_LABELS)
    return gdf

# ===========================================================================
# STEP 5: Generate maps and figures
# ===========================================================================
def generate_morans_figures(gdf, df_yearly, mi_mean):
    print("\n" + "=" * 70)
    print("STEP 5: Generating maps and figures")
    print("=" * 70)
    
    # Figure 1: Static LISA Map
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(column="lisa_label", categorical=True, cmap="Set1", legend=True, ax=ax, edgecolor="black", linewidth=0.3)
    ax.set_title("Zambia Drought Risk Clusters (LISA)", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    fig.savefig(OUTPUT_DIR / "Zambia_LISA_Static_Map.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 2: Moran's I time series
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(df_yearly["year"].astype(str), df_yearly["morans_i"], color="steelblue")
    ax.set_title("Spatial Clustering (Global Moran's I) Over Time")
    plt.xticks(rotation=45)
    fig.savefig(OUTPUT_DIR / "Zambia_Global_MoransI_by_Year.png", dpi=200, bbox_inches="tight")
    plt.close()

# ===========================================================================
# STEP 6: Insurance Recommendation
# ===========================================================================
def generate_insurance_table(gdf):
    print("\n" + "=" * 70)
    print("STEP 6: Insurance decision table")
    print("=" * 70)
    
    recs = {
        "High-High (drought cluster)": "LIMIT exposure — covarying risk",
        "Low-Low (safe zone)": "PREFERRED — stable premium base",
        "High-Low (isolated risk)": "CAUTIOUS — isolated risk",
        "Low-High (drought surround)": "OPPORTUNITY — safe pocket",
        "Not significant": "STANDARD",
    }
    gdf["portfolio_rec"] = gdf["lisa_label"].apply(lambda x: recs.get(x, "STANDARD"))
    
    cols = ["ADM_NAME", "mean_drought_days", "lisa_label", "portfolio_rec"]
    existing = [c for c in cols if c in gdf.columns]
    summary = gdf[existing].sort_values("mean_drought_days", ascending=False)
    summary.to_csv(OUTPUT_DIR / "zambia_insurance_portfolio_table.csv", index=False)
    return summary

def main():
    gdf, df_annual = load_zambia_data()
    w = build_spatial_weights(gdf)
    df_yearly, mi_mean = compute_global_morans_i(gdf, df_annual, w)
    gdf = compute_lisa(gdf, w)
    generate_morans_figures(gdf, df_yearly, mi_mean)
    generate_insurance_table(gdf)

    # Save final spatial file
    gdf.to_file(OUTPUT_DIR / "zambia_lisa_clusters.geojson", driver="GeoJSON")

    report = f"Analysis Complete.\nGlobal Moran's I: {mi_mean.I:.4f}\np-value: {mi_mean.p_sim:.4f}"
    print("\n" + report)
    
    # WINDOWS ENCODING SAFE SAVE
    with open(OUTPUT_DIR / "zambia_lisa_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()