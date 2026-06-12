"""
overlapping_pixel_solver.py — Area-Weighted Pixel-to-District Harmonization
============================================================================
METHOD (the "Distinction" contribution):

Problem:
  ERA5-Land has ~9km (0.1°) pixels. A pixel straddling a district boundary
  gets assigned to only one district under centroid-based methods, creating
  systematic bias at borders. This is a manifestation of the Modifiable
  Areal Unit Problem (MAUP) applied to climate data.

Solution:
  1. Create a 0.1° fishnet matching ERA5-Land's native grid
  2. Compute geometric overlap between each fishnet cell and each district
  3. Build a weight matrix: W[pixel, district] = overlap_area / pixel_area
  4. For crop calendars: area-weighted voting across GEOGLAM zones
  5. For SSI: quality metrics tell us which districts have "clean" signals

References:
  - Openshaw (1984) "The Modifiable Areal Unit Problem" — MAUP theoretical
    framework for why pixel-to-polygon aggregation matters
  - Fisher & Langford (1995) "Areal Interpolation" — area-weighting as the
    standard solution for polygon-to-polygon interpolation
  - Flowerdew, Green & Kehris (1991) "Areal interpolation" — the
    pycnophylactic (volume-preserving) property of area-weighting
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import box
from shapely import wkt
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ERA5_RESOLUTION_DEG = 0.1

current_path = Path(__file__).resolve()
# Go up 2 levels from scripts/ to reach Africa-Drought-Study/
PROJECT_ROOT = next(p for p in current_path.parents if (p / "data").exists())
DATA_DIR = PROJECT_ROOT / "data"

# Standardize Output Directory
OUTPUT_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "overlap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Correct Path to your GADM data
GADM_PATH = DATA_DIR / "africa_agricultural_domain_2019.shp"

# Correct Path to your GEOGLAM data (Fixing the subfolder issue)
GEOGLAM_PATH = DATA_DIR / "GEOGLAM_CM4EW_Calendars_V1.4.shp"

# Cache location
OVERLAP_CACHE = DATA_DIR / "era5_fishnet_overlaps.parquet"


# ===========================================================================
# STAGE 1: Create the ERA5 fishnet grid
# ===========================================================================
def create_era5_fishnet(bounds, resolution=ERA5_RESOLUTION_DEG):
    """
    Create a GeoDataFrame of square cells matching ERA5-Land's grid.

    Parameters
    ----------
    bounds : tuple of (minx, miny, maxx, maxy)
        Bounding box in EPSG:4326.
    resolution : float
        Cell size in degrees. ERA5-Land uses 0.1°.

    Returns
    -------
    gpd.GeoDataFrame with columns: ['cell_id', 'geometry']
        Where cell_id = f"{row}_{col}" for provenance tracking.

    Reference: ECMWF ERA5-Land grid is 0.1° × 0.1° regular lat-lon.
    """
    minx, miny, maxx, maxy = bounds

    # Snap bounds to the ERA5 grid to avoid partial edge cells
    # ERA5 grid nodes are at 0.0, 0.1, 0.2, ... degrees
    minx = np.floor(minx / resolution) * resolution
    maxx = np.ceil(maxx / resolution) * resolution
    miny = np.floor(miny / resolution) * resolution
    maxy = np.ceil(maxy / resolution) * resolution

    xs = np.arange(minx, maxx, resolution)
    ys = np.arange(miny, maxy, resolution)

    cells = []
    cell_ids = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            cells.append(box(x, y, x + resolution, y + resolution))
            cell_ids.append(f"{i}_{j}")

    fishnet = gpd.GeoDataFrame(
        {"cell_id": cell_ids, "geometry": cells},
        crs="EPSG:4326"
    )
    print(f"  Fishnet created: {len(fishnet)} cells at {resolution}°")
    print(f"    X range: {minx:.2f} to {maxx:.2f} ({len(xs)} columns)")
    print(f"    Y range: {miny:.2f} to {maxy:.2f} ({len(ys)} rows)")
    return fishnet


# ===========================================================================
# STAGE 2: Compute pixel-district overlap weights
# ===========================================================================
def compute_pixel_district_overlaps(fishnet, districts):
    """
    For each ERA5 pixel, compute what fraction of its area overlaps each district.

    This is the core geometric operation. It produces a sparse weight matrix
    where each row is (pixel_id, district_name, weight).

    Parameters
    ----------
    fishnet : gpd.GeoDataFrame
        ERA5 grid cells with 'cell_id' column.
    districts : gpd.GeoDataFrame
        District polygons with 'ADM_NAME' column.

    Returns
    -------
    pd.DataFrame with columns: [cell_id, ADM_NAME, overlap_area, weight]
        Where weight = overlap_area / pixel_area (sums to 1.0 per pixel).

    Reference: Fisher & Langford (1995) — area-weighting preserves the
    pycnophylactic property: total SSI mass is conserved across the
    pixel-to-district transformation.
    """
    print("\n=== STAGE 2: Computing pixel-district overlap weights ===")

    # Ensure both are in a projected CRS for accurate area calculations
    # Use Africa Albers Equal Area (ESRI:102022) for continental accuracy
    PROJ_CRS = "ESRI:102022"
    fishnet_proj = fishnet.to_crs(PROJ_CRS)
    districts_proj = districts.to_crs(PROJ_CRS)

    # Compute each pixel's total area (all pixels in the grid are the same)
    pixel_areas = fishnet_proj.geometry.area
    if pixel_areas.nunique() == 1:
        pixel_area = pixel_areas.iloc[0]
        print(f"  Pixel area: {pixel_area / 1e6:.2f} km²")
    else:
        # Should not happen with regular grid, but handle gracefully
        print(f"  Warning: pixels vary in area ({pixel_areas.min():.0f}–{pixel_areas.max():.0f} m²)")

    # Spatial join: find all (pixel, district) pairs that intersect
    print("  Running spatial join (fishnet × districts) …")
    joined = gpd.sjoin(
        fishnet_proj,
        districts_proj,
        how="inner",
        predicate="intersects"
    )
    print(f"  Total overlap pairs: {len(joined)}")
    print(f"  Unique pixels:       {joined['cell_id'].nunique()}")
    print(f"  Unique districts:    {joined['ADM_NAME'].nunique()}")

    # For each pair, compute the ACTUAL intersection area
    print("  Computing precise intersection areas …")

    # Create temporary GeoDataFrame for intersection computation
    temp = joined[["cell_id", "ADM_NAME", "geometry"]].copy()

    # Merge geometries for intersection: need district geometry
    districts_geom = districts_proj[["ADM_NAME", "geometry"]].set_index("ADM_NAME")
    temp["district_geom"] = temp["ADM_NAME"].map(districts_geom["geometry"])

    # Compute intersection area
    overlaps = []
    batch_size = 5000
    for start in range(0, len(temp), batch_size):
        batch = temp.iloc[start:start + batch_size]
        for _, row in batch.iterrows():
            intersection = row["geometry"].intersection(row["district_geom"])
            overlap_area = intersection.area
            pixel_area_actual = row["geometry"].area
            weight = overlap_area / pixel_area_actual if pixel_area_actual > 0 else 0.0
            overlaps.append({
                "cell_id": row["cell_id"],
                "ADM_NAME": row["ADM_NAME"],
                "overlap_area": overlap_area,
                "weight": weight
            })

    result = pd.DataFrame(overlaps)

    # Validate: each pixel's weights should sum to ~1.0
    weight_sums = result.groupby("cell_id")["weight"].sum()
    bad_pixels = (weight_sums < 0.99).sum()
    if bad_pixels > 0:
        print(f"  ⚠ {bad_pixels} pixels have weights summing < 0.99 "
              f"(likely partial coverage at study area boundary)")

    print(f"  Overlap table: {len(result)} rows")
    print(f"  Weights range: {result['weight'].min():.4f} – {result['weight'].max():.4f}")

    return result


def cache_overlap_table(overlap_df, path=None):
    """Save the overlap table to disk for reuse."""
    if path is None:
        path = OVERLAP_CACHE
    overlap_df.to_parquet(path.with_suffix(".parquet"), index=False)
    print(f"  Cached to: {path.with_suffix('.parquet')}")


def load_cached_overlaps(path=None):
    """Load cached overlap table."""
    if path is None:
        path = OVERLAP_CACHE
    path = path.with_suffix(".parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


# ===========================================================================
# STAGE 3: District quality metrics
# ===========================================================================
def compute_district_quality_metrics(overlap_df, districts):
    """
    For each district, compute metrics describing how well ERA5 pixels
    represent it. These metrics answer the professor's question:
    "How reliable is the SSI signal for this district?"

    Metrics
    -------
    n_total_pixels   : total ERA5 pixels overlapping the district
    n_boundary_pixels: pixels with weight < 0.95 (not fully contained)
    n_full_pixels    : pixels fully contained (weight >= 0.95)
    pct_full_pixels  : percentage of pixels that are fully contained
    mean_weight      : average weight across all overlapping pixels
    quality_class    : 'High' if >= 80% full pixels, 'Medium' if >= 50%, else 'Low'

    Returns
    -------
    gpd.GeoDataFrame: districts with quality metrics appended.

    Reference: The Nyquist-Shannon sampling theorem applied to spatial data:
    a feature is reliably represented when the sampling grid cell (9km) is at
    most half the feature's width. Our quality metrics operationalize this.
    """
    print("\n=== STAGE 3: Computing district quality metrics ===")

    # Per-district aggregation
    district_stats = []
    for name, grp in overlap_df.groupby("ADM_NAME"):
        n_total = len(grp)
        n_full = (grp["weight"] >= 0.95).sum()
        n_boundary = n_total - n_full
        pct_full = 100 * n_full / n_total if n_total > 0 else 0
        mean_w = grp["weight"].mean()

        if pct_full >= 80:
            qclass = "High"
        elif pct_full >= 50:
            qclass = "Medium"
        else:
            qclass = "Low"

        district_stats.append({
            "ADM_NAME": name,
            "n_total_pixels": n_total,
            "n_boundary_pixels": n_boundary,
            "n_full_pixels": n_full,
            "pct_full_pixels": round(pct_full, 1),
            "mean_overlap_weight": round(mean_w, 4),
            "quality_class": qclass,
        })

    quality_df = pd.DataFrame(district_stats)

    # Merge with districts
    result = districts.merge(quality_df, on="ADM_NAME", how="left")

    # Fill NaN for districts with no overlap (shouldn't happen)
    for col in ["n_total_pixels", "n_boundary_pixels", "n_full_pixels"]:
        result[col] = result[col].fillna(0).astype(int)
    result["pct_full_pixels"] = result["pct_full_pixels"].fillna(0.0)
    result["mean_overlap_weight"] = result["mean_overlap_weight"].fillna(0.0)
    result["quality_class"] = result["quality_class"].fillna("No Data")

    print(f"\n  Quality distribution:")
    for qc in ["High", "Medium", "Low", "No Data"]:
        count = (result["quality_class"] == qc).sum()
        if count > 0:
            print(f"    {qc:<10s}: {count} districts")

    return result


# ===========================================================================
# STAGE 4: Weighted GEOGLAM crop calendar assignment
# ===========================================================================
def assign_geoglam_weighted(districts, geoglam):
    """
    Assign crop calendar parameters to each district using area-weighted
    voting across overlapping GEOGLAM zones, instead of centroid-in-polygon.

    The current method (Week_11) uses centroid-in-polygon: a district's
    centroid falls in one GEOGLAM zone → that zone's calendar is assigned
    to the entire district. This is biased when a district spans multiple
    crop calendar zones.

    This method:
    1. For each GEOGLAM zone, compute its overlap with each district
    2. For each district, compute the area-weighted mean of planting/harvest
       dates from all overlapping GEOGLAM zones

    Parameters
    ----------
    districts : gpd.GeoDataFrame
        District polygons.
    geoglam : gpd.GeoDataFrame
        GEOGLAM crop calendar polygons (filtered to desired crop).
    overlap_df : pd.DataFrame or None
        Pre-computed pixel-district overlaps. If None, computes on-the-fly.

    Returns
    -------
    gpd.GeoDataFrame: districts with weighted planting/harvest dates.
    """
    print("\n=== STAGE 4: Area-weighted GEOGLAM assignment ===")

    PROJ_CRS = "ESRI:102022"

    # Strip to only essential columns to avoid merge conflicts
    dist_clean = districts[["ADM_NAME", "geometry"]].copy()
    geoglam_clean = geoglam[["crop", "planting", "harvest", "endofseaso", "geometry"]].copy()

    dist_proj = dist_clean.to_crs(PROJ_CRS)
    geoglam_proj = geoglam_clean.to_crs(PROJ_CRS)

    # Spatial join: find all (district, GEOGLAM) overlaps
    joined = gpd.sjoin(
        dist_proj,
        geoglam_proj,
        how="inner",
        predicate="intersects"
    )
    print(f"  (District x GEOGLAM) overlap pairs: {len(joined)}")

    if joined.empty:
        print("  ⚠ No district-GEOGLAM overlaps found. Check CRS alignment.")
        return districts

    # Compute intersection area for each pair
    results = []
    for (adm_name, crop), grp in joined.groupby(["ADM_NAME", "crop"]):
        district_geom = dist_proj[dist_proj["ADM_NAME"] == adm_name].geometry.iloc[0]
        crop_geom = grp.geometry.unary_union
        overlap_area = district_geom.intersection(crop_geom).area
        results.append({
            "ADM_NAME": adm_name,
            "crop": crop,
            "planting": int(grp["planting"].iloc[0]),
            "harvest": int(grp["harvest"].iloc[0]),
            "endofseaso": int(grp["endofseaso"].iloc[0]),
            "overlap_km2": overlap_area / 1e6
        })

    df_overlap = pd.DataFrame(results)
    print(f"  Unique district-crop pairs: {len(df_overlap)}")

    # For each district, compute area-weighted mean calendar dates
    weighted = []
    for adm_name, grp in df_overlap.groupby("ADM_NAME"):
        total_area = grp["overlap_km2"].sum()
        if total_area <= 0:
            weighted.append({
                "ADM_NAME": adm_name,
                "planting": int(grp["planting"].mean()),
                "harvest": int(grp["harvest"].mean()),
                "endofseaso": int(grp["endofseaso"].mean()),
                "n_geoglam_zones": grp["crop"].nunique(),
                "dominant_crop": grp.loc[grp["overlap_km2"].idxmax(), "crop"],
            })
        else:
            weighted.append({
                "ADM_NAME": adm_name,
                "planting": int(np.average(grp["planting"], weights=grp["overlap_km2"])),
                "harvest": int(np.average(grp["harvest"], weights=grp["overlap_km2"])),
                "endofseaso": int(np.average(grp["endofseaso"], weights=grp["overlap_km2"])),
                "n_geoglam_zones": grp["crop"].nunique(),
                "dominant_crop": grp.loc[grp["overlap_km2"].idxmax(), "crop"],
            })

    result = pd.DataFrame(weighted)
    print(f"  Districts with weighted calendar: {len(result)}")

    # Compare with centroid-based planting (if the original districts have it)
    if "planting" in districts.columns:
        merged = result.merge(
            districts[["ADM_NAME", "planting"]].rename(columns={"planting": "planting_centroid"}),
            on="ADM_NAME"
        )
        if len(merged) > 5:
            diff = (merged["planting"] - merged["planting_centroid"]).abs()
            print(f"\n  Comparison with centroid assignment:")
            print(f"    Mean diff in planting DOY: {diff.mean():.1f}")
            print(f"    Max diff:                  {diff.max():.0f}")
            print(merged[["ADM_NAME", "planting", "planting_centroid"]].head().to_string(index=False))

    return districts.merge(result, on="ADM_NAME", how="left", suffixes=("", "_weighted"))


# ===========================================================================
# STAGE 5: Comparison report and figures
# ===========================================================================
def generate_quality_report(districts_with_quality, output_dir=None):
    """
    Generate the figures and tables for the dissertation results chapter.

    Figure 1: Map of district quality classes (High/Medium/Low)
    Figure 2: Distribution of pct_full_pixels across districts
    Figure 3: Scatter: crop_pct vs quality_class
    Table 1: Summary statistics per quality class
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = districts_with_quality.copy()

    # --- Figure 1: Quality map ---
    print("\n  Generating quality map …")
    try:
        m = gdf.explore(
            column="quality_class",
            categories=["High", "Medium", "Low"],
            cmap=["#1b9e77", "#d95f02", "#e7298a"],
            legend=True,
            tiles="CartoDB positron",
            title="District Pixel-Overlap Quality",
            style_kwds={"weight": 0.5},
        )
        m.save(os.path.join(output_dir, "district_quality_map.html"))
        print(f"    Saved: district_quality_map.html")
    except Exception as e:
        print(f"    ⚠ Map generation failed: {e}")

    # --- Figure 2: Quality distribution histogram ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    gdf["pct_full_pixels"].hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(80, color="red", ls="--", label="High quality threshold (80%)")
    ax.set_xlabel("% of full pixels per district")
    ax.set_ylabel("Number of districts")
    ax.set_title("Distribution of Pixel-Overlap Quality")
    ax.legend()

    ax = axes[1]
    gdf["quality_class"].value_counts().plot(kind="bar", ax=ax, color=["#1b9e77", "#d95f02", "#e7298a"])
    ax.set_xlabel("Quality class")
    ax.set_ylabel("Number of districts")
    ax.set_title("Districts per quality class")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[2]
    gdf.boxplot(column="crop_pct", by="quality_class", ax=ax)
    ax.set_xlabel("Quality class")
    ax.set_ylabel("Crop area (%)")
    ax.set_title("Crop area by quality class")
    fig.suptitle("")

    plt.tight_layout()
    fig.savefig(output_dir / "district_quality_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: district_quality_analysis.png")

    # --- Summary table ---
    summary = gdf.groupby("quality_class").agg(
        District_Count=("ADM_NAME", "count"),
        Mean_Crop_Pct=("crop_pct", "mean"),
        Mean_Total_Pixels=("n_total_pixels", "mean"),
        Mean_Boundary_Pixels=("n_boundary_pixels", "mean"),
        Mean_Full_Pixels=("n_full_pixels", "mean"),
    ).round(1)
    summary.to_csv(output_dir / "quality_summary.csv")
    print(f"\n  Summary table saved: quality_summary.csv")
    print(summary.to_string())


# ===========================================================================
# STAGE 6: Run the full pipeline (end-to-end)
# ===========================================================================
def run_overlap_pipeline(country_iso3=None):
    """
    Run the full overlap analysis pipeline for all or a subset of districts.

    Parameters
    ----------
    country_iso3 : str or None
        If provided (e.g. "ZMB"), only processes that country's districts.
        If None, processes all 3,333 districts (can be slow).

    Returns
    -------
    gpd.GeoDataFrame with quality metrics appended.
    """
    print("=" * 70)
    print("OVERLAPPING PIXEL SOLVER — Full Pipeline")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading districts …")
    districts = gpd.read_file(GADM_PATH)
    districts["ADM_NAME"] = districts["ADM_NAME"].astype(str).str.strip().str.upper()

    if country_iso3:
        districts = districts[districts["ISO3"] == country_iso3].copy()
        print(f"  Filtered to {country_iso3}: {len(districts)} districts")

    # --- Try to load cached overlaps ---
    overlap_df = load_cached_overlaps()
    if overlap_df is not None:
        # Filter cached overlaps to our districts
        valid_names = set(districts["ADM_NAME"])
        overlap_df = overlap_df[overlap_df["ADM_NAME"].isin(valid_names)].copy()
        print(f"\nLoaded cached overlaps: {len(overlap_df)} pairs")
    else:
        # Create fishnet and compute overlaps
        print("\nCreating ERA5 fishnet …")
        fishnet = create_era5_fishnet(districts.total_bounds)
        overlap_df = compute_pixel_district_overlaps(fishnet, districts)
        cache_overlap_table(overlap_df)

    # --- Compute quality metrics ---
    districts_with_quality = compute_district_quality_metrics(overlap_df, districts)

    # --- GEOGLAM weighted assignment ---
    print("\nLoading GEOGLAM crop calendar …")
    geoglam = gpd.read_file(GEOGLAM_PATH)
    geoglam_maize = geoglam[geoglam["crop"] == "Maize 1"].copy()
    print(f"  Maize 1 polygons: {len(geoglam_maize)}")

    districts_calendar = assign_geoglam_weighted(districts_with_quality, geoglam_maize)

    # --- Save results ---
    output_path = OUTPUT_DIR / "districts_with_quality.geojson"
    districts_calendar.to_file(output_path, driver="GeoJSON")
    print(f"\nSaved results: {output_path}")

    # --- Generate report ---
    generate_quality_report(districts_calendar)

    return districts_calendar


# ===========================================================================
# Sensitivity filtering helper (for use by other scripts)
# ===========================================================================
def filter_by_quality(districts_df, min_quality="Medium"):
    """
    Filter districts by pixel-overlap quality class.

    This is the key function that connects the overlap analysis to the
    SSI sensitivity analysis. Districts with poor overlap quality can
    be excluded to produce more reliable SSI signals.

    Parameters
    ----------
    districts_df : gpd.GeoDataFrame
        Districts with quality_class column (from run_overlap_pipeline).
    min_quality : str
        Minimum quality: "High" (>=80% full pixels), "Medium" (>=50%), or "Low".

    Returns
    -------
    gpd.GeoDataFrame of filtered districts, plus exclusion counts.
    """
    quality_order = {"High": 3, "Medium": 2, "Low": 1}
    min_val = quality_order.get(min_quality, 1)

    mask = districts_df["quality_class"].map(quality_order) >= min_val
    filtered = districts_df[mask].copy()

    n_total = len(districts_df)
    n_kept = len(filtered)
    n_excluded = n_total - n_kept

    print(f"\nQuality filter (≥ {min_quality}):")
    print(f"  Total districts: {n_total}")
    print(f"  Kept:           {n_kept} ({100*n_kept/n_total:.1f}%)")
    print(f"  Excluded:       {n_excluded} ({100*n_excluded/n_total:.1f}%)")

    if n_excluded > 0:
        print(f"\n  Excluded districts (first 10):")
        excluded = districts_df[~mask]["ADM_NAME"].head(10).tolist()
        for d in excluded:
            print(f"    • {d}")

    return filtered


# ===========================================================================
# EFFICIENT ALL-AFRICA RUN (country-by-country)
# ===========================================================================
def run_africa_by_country():
    """
    Run the overlap pipeline for ALL African districts, processing one
    country at a time. This is more memory-efficient than a single
    continent-wide fishnet.

    Strategy:
      1. For each country, create a fishnet over that country's bounds
      2. Compute overlaps within each country
      3. Concatenate results into one quality GeoDataFrame
      4. Save cumulative cache to parquet

    Returns
    -------
    gpd.GeoDataFrame: all districts with quality metrics
    """
    print("=" * 70)
    print("ALL-AFRICA OVERLAP ANALYSIS (by country)")
    print("=" * 70)

    districts = gpd.read_file(GADM_PATH)
    districts["ADM_NAME"] = districts["ADM_NAME"].astype(str).str.strip().str.upper()

    countries = sorted(districts["ISO3"].dropna().unique())
    print(f"\nProcessing {len(countries)} countries ({len(districts)} total districts)")

    all_overlaps = []
    all_quality_dfs = []

    for i, iso3 in enumerate(countries):
        ctry = districts[districts["ISO3"] == iso3]
        n_dist = len(ctry)
        print(f"\n[{i+1}/{len(countries)}] {iso3}: {n_dist} districts")

        if ctry.total_bounds.size == 0:
            print("  ⚠ No bounds — skipping")
            continue

        # Create fishnet for this country's bounding box
        fishnet = create_era5_fishnet(ctry.total_bounds)

        # Compute overlaps
        overlap_df = compute_pixel_district_overlaps(fishnet, ctry)
        all_overlaps.append(overlap_df)

        # Compute quality metrics
        quality_df = compute_district_quality_metrics(overlap_df, ctry)
        all_quality_dfs.append(quality_df)

    if not all_quality_dfs:
        print("❌ No results — aborting")
        return gpd.GeoDataFrame()

    # Combine all results
    combined_overlaps = pd.concat(all_overlaps, ignore_index=True)
    combined_quality = pd.concat(all_quality_dfs, ignore_index=True)

    # Cache
    cache_overlap_table(combined_overlaps)
    combined_quality.to_file(OUTPUT_DIR / "africa_districts_with_quality.geojson", driver="GeoJSON")

    print(f"\n{'='*70}")
    print(f"✅ ALL-AFRICA ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"  Districts processed: {len(combined_quality)}")
    print(f"  Total overlap pairs: {len(combined_overlaps)}")
    print(f"  Quality distribution:")
    for qc in ["High", "Medium", "Low", "Unknown"]:
        count = (combined_quality["quality_class"] == qc).sum()
        if count > 0:
            print(f"    {qc:<10s}: {count}")

    return combined_quality


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--africa":
        # Run for all Africa (country-by-country)
        print("Running ALL-AFRICA overlap analysis …")
        result = run_africa_by_country()
    else:
        # Run for Zambia (fast ~30s) as a test
        print("Running for Zambia (ZMB) …")
        result = run_overlap_pipeline(country_iso3="ZMB")

        print("\n" + "=" * 70)
        print("✅ OVERLAP ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Outputs saved to: {OUTPUT_DIR}")
        print(f"\nNext step: Run zambia_weighted_analysis.py to compare")
        print(f"weighted vs centroid SSI results.")
