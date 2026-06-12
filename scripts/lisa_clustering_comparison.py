"""
lisa_clustering_comparison.py — LISA vs Agglomerative Clustering (Strategy 5.3)
================================================================================
Purpose:
  Compare Moran's I LISA zones with spatially-constrained agglomerative
  clusters to answer: do districts with the same spatial autocorrelation
  pattern (HH, LL, HL, LH) fall into the same agglomerative group?

  This bridges spatial autocorrelation (strategy 5.1-5.2) with the
  climate-based clustering (spatial_agglomerative_clustering.py) for
  insurance portfolio interpretation.

Method:
  1. Run AgglomerativeClustering on Zambia districts with k-NN spatial
     connectivity constraint (matching spatial_agglomerative_clustering.py)
  2. Cross-tabulate LISA labels × agglomerative clusters
  3. Chi-squared test for association
  4. Visualize with grouped bar chart + side-by-side maps

References:
  - spatial_agglomerative_clustering.py: uses AgglomerativeClustering with
    k-NN graph, auto-detects K via silhouette, λ=0.1 weather/geography mix
  - zambia_spatial_autocorrelation.py: computes LISA labels

Outputs:
  - zambia_lisa_vs_clusters.csv — cross-tabulation
  - zambia_lisa_vs_clusters_figure.png — comparison figure
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- STANDALONE UTILS (Embedded to prevent ModuleNotFoundError) ---
def compute_zscore_ssi(df):
    """Computes a simple Z-Score based SSI for clustering comparison."""
    df = df.copy()
    stats = df.groupby(['feature_id', 'doy'])['volumetric_soil_water_layer_2'].agg(['mean', 'std']).reset_index()
    df = df.merge(stats, on=['feature_id', 'doy'], how='left')
    df['SSI'] = (df['volumetric_soil_water_layer_2'] - df['mean']) / df['std']
    df['SSI'] = df['SSI'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def aggregate_annual_drought_days(df, ssi_threshold=-1.0):
    """Counts days per year where SSI is below threshold."""
    df['is_drought'] = (df['SSI'] <= ssi_threshold).astype(int)
    annual = df.groupby(['feature_id', 'year'])['is_drought'].sum().reset_index()
    return annual.rename(columns={'is_drought': 'Drought_Days'})

# --- ROBUST PATH LOGIC ---
current_path = Path(__file__).resolve()
# Find Africa-Drought-Study project root
PROJECT_ROOT = next(p for p in current_path.parents if (p / "data").exists())
DATA_DIR = PROJECT_ROOT / "data"

# Standardize output directory
OUTPUT_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "spatial_autocorrelation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Point to your Week 11 results (The primary data source)
WEEK_11_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "Week_11_correlation_map"
LISA_PATH = OUTPUT_DIR / "zambia_lisa_clusters.geojson" 
# Update this line in lisa_clustering_comparison.py
QUALITY_PATH = PROJECT_ROOT / "dissertation_work" / "outputs" / "overlap_analysis" / "districts_with_quality.geojson" 

# ERA5 Path
ERA5_PATH = DATA_DIR / "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"

SSI_THRESHOLD = -1.0
SPATIAL_NEIGHBORS = 10
K_MAX = 7
LAMBDA = 0.1
RANDOM_STATE = 42

# --- UPDATED LOAD_DATA ---
def load_data():
    """Load Zambia districts, ERA5, and check for LISA output."""
    if not LISA_PATH.exists():
        print(f"ERROR: LISA clusters not found at {LISA_PATH}")
        print("Please run zambia_spatial_autocorrelation.py first.")
        sys.exit(1)

    lisa = gpd.read_file(LISA_PATH)
    lisa["ADM_NAME"] = lisa["ADM_NAME"].astype(str).str.strip().str.upper()

    districts = gpd.read_file(QUALITY_PATH)
    # Filter for Zambia pilot
    if 'COUNTRY' in districts.columns:
        districts = districts[districts['COUNTRY'].str.upper() == 'ZAMBIA']
    districts["ADM_NAME"] = districts["ADM_NAME"].astype(str).str.strip().str.upper()

    print(f"Loading ERA5 for Zambia districts ...")
    zmb_names = set(districts["ADM_NAME"])
    dtype = {"year": "int16", "month": "int8", "day": "int8", "doy": "int16",
             "volumetric_soil_water_layer_2": "float32"}
    
    df_era5 = pd.read_csv(ERA5_PATH, dtype=dtype)
    df_era5["feature_id"] = df_era5["feature_id"].astype(str).str.strip().str.upper()
    df_era5 = df_era5[df_era5["feature_id"].isin(zmb_names)]

    df = compute_zscore_ssi(df_era5)
    df_risk = df[df["month"].between(1, 4)].copy()
    annual = aggregate_annual_drought_days(df_risk, ssi_threshold=SSI_THRESHOLD)

    return lisa, districts, annual


def run_agglomerative_clustering(annual_df, districts):
    """
    Run spatially-constrained agglomerative clustering for Zambia districts.

    Follows spatial_agglomerative_clustering.py methodology:
    - k-NN spatial connectivity graph (k=10)
    - Auto-detect optimal K via silhouette score
    - λ=0.1 (90% weather, 10% geography)
    """
    pivot = annual_df.pivot_table(
        index="year", columns="feature_id", values="Drought_Days"
    ).fillna(0)

    common = sorted(set(pivot.columns) & set(districts["ADM_NAME"]))

    # Weather features
    weather = StandardScaler().fit_transform(pivot[common].T)

    # Spatial features (centroids)
    centroids = districts.set_index("ADM_NAME").to_crs(epsg=3857).centroid
    coords = np.column_stack([centroids.loc[common].x, centroids.loc[common].y])
    geo = StandardScaler().fit_transform(coords)

    # Combined with lambda weighting
    X = np.column_stack([weather * (1 - LAMBDA), geo * LAMBDA])

    # Build k-NN spatial connectivity constraint
    spatial_graph = kneighbors_graph(coords, n_neighbors=SPATIAL_NEIGHBORS,
                                     mode="connectivity")

    # Auto-detect optimal K
    print(f"  Auto-detecting K (1–{K_MAX}) …")
    scores = []
    for k in range(2, K_MAX + 1):
        model = AgglomerativeClustering(
            n_clusters=k, connectivity=spatial_graph, linkage="ward"
        )
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores.append((k, sil))
        print(f"    K={k}: silhouette={sil:.4f}")

    best_k = max(scores, key=lambda x: x[1])[0]
    best_sil = max(scores, key=lambda x: x[1])[1]
    print(f"  Optimal K={best_k} (silhouette={best_sil:.4f})")

    # Final model
    model = AgglomerativeClustering(
        n_clusters=best_k, connectivity=spatial_graph, linkage="ward"
    )
    labels = model.fit_predict(X)

    return dict(zip(common, labels)), best_k, best_sil, scores


def main():
    print("=" * 70)
    print("LISA vs AGGLOMERATIVE CLUSTERING — Strategy 5.3")
    print("=" * 70)

    lisa, districts, annual = load_data()
    print(f"\nZambia districts: {len(districts)}")
    print(f"LISA distribution:\n{lisa['lisa_label'].value_counts().to_string()}")

    # ---- Step 1: Agglomerative clustering ----
    print("\n" + "=" * 70)
    print("STEP 1: Spatially-constrained agglomerative clustering")
    print("=" * 70)
    cluster_map, best_k, best_sil, sil_scores = run_agglomerative_clustering(annual, districts)

    # Save cluster assignments
    df_clusters = pd.DataFrame([
        {"ADM_NAME": d, "agglomerative_cluster": f"G{c+1}"}
        for d, c in cluster_map.items()
    ])
    df_clusters.to_csv(OUTPUT_DIR / "zambia_agglomerative_clusters.csv", index=False)
    print(f"\n  Saved: zambia_agglomerative_clusters.csv")

    # Print cluster profiles
    print(f"\n  Cluster profiles:")
    for cid in range(best_k):
        members = [d for d, c in cluster_map.items() if c == cid]
        mean_dd = annual[annual["feature_id"].isin(members)]["Drought_Days"].mean()
        print(f"    G{cid+1}: {len(members):>2} districts, "
              f"{mean_dd:.1f} mean drought days/yr")

    # ---- Step 2: Cross-tabulation ----
    print("\n" + "=" * 70)
    print("STEP 2: LISA × Agglomerative Cross-Tabulation")
    print("=" * 70)

    merged = lisa.merge(df_clusters, on="ADM_NAME", how="left")

    # Simplified LISA labels for grouping
    lisa_short = {
        "High-High (drought cluster)": "HH",
        "Low-Low (safe zone)": "LL",
        "High-Low (isolated risk)": "HL",
        "Low-High (drought surround)": "LH",
        "Not significant": "NS",
    }
    merged["lisa_short"] = merged["lisa_label"].map(lisa_short)

    # Cross-tabulation
    ct = pd.crosstab(
        merged["lisa_short"],
        merged["agglomerative_cluster"],
        margins=True,
        margins_name="Total"
    )
    print(f"\n  Cross-tabulation (count):")
    print(f"  {ct.to_string()}")

    ct.to_csv(OUTPUT_DIR / "zambia_lisa_vs_clusters.csv")
    print(f"\n  Saved: zambia_lisa_vs_clusters.csv")

    # Conditional probabilities: given LISA label, which agglomerative cluster?
    ct_pct = pd.crosstab(
        merged["lisa_short"],
        merged["agglomerative_cluster"],
        margins=True,
        margins_name="Total",
        normalize="index"
    )
    print(f"\n  Conditional probability P(cluster | LISA):")
    print(f"  {ct_pct.round(3).to_string()}")

    # Chi-squared test (no margins)
    ct_no_margins = ct.loc[ct.index != "Total", ct.columns != "Total"]
    chi2, p_val, dof, expected = chi2_contingency(ct_no_margins)
    print(f"\n  Chi-squared test:")
    print(f"    χ² = {chi2:.2f}, df = {dof}, p = {p_val:.4f}")
    if p_val < 0.05:
        print(f"    ✅ Significant association (p < 0.05)")
        print(f"       → LISA zones and agglomerative clusters are related")
    else:
        print(f"    ⚠ No significant association (p >= 0.05)")
        print(f"       → LISA and agglomerative clusters capture different structure")

    # Cramer's V (effect size)
    n = ct_no_margins.sum().sum()
    cramer_v = np.sqrt(chi2 / (n * min(ct_no_margins.shape) - 1))
    print(f"    Cramér's V = {cramer_v:.3f}")

    # ---- Step 3: Figure ----
    print("\n" + "=" * 70)
    print("STEP 3: Generating comparison figure")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("LISA Zones vs Agglomerative Clusters (Zambia)",
                 fontweight="bold", fontsize=13)

    lisa_colors = {
        "HH": "#d73027", "LL": "#1a9850", "HL": "#fee08b",
        "LH": "#4575b4", "NS": "#f0f0f0"
    }
    cluster_colors = [plt.cm.Set1(i / best_k) for i in range(best_k)]
    cluster_color_map = {f"G{i+1}": cluster_colors[i] for i in range(best_k)}

    # Panel 1: Grouped bar — LISA × Clusters
    ax = axes[0]
    ct_plot = ct_no_margins
    lisa_cats = [l for l in ["HH", "LL", "HL", "LH", "NS"] if l in ct_plot.index]
    cluster_cats = sorted(ct_plot.columns)

    x = np.arange(len(lisa_cats))
    w = 0.8 / len(cluster_cats)
    for i, cl in enumerate(cluster_cats):
        vals = [ct_plot.loc[l, cl] if cl in ct_plot.columns else 0 for l in lisa_cats]
        offset = (i - len(cluster_cats) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=cl, color=cluster_color_map[cl],
                      edgecolor="black", linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(int(v)), ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(lisa_cats, fontsize=10)
    ax.set_xlabel("LISA zone")
    ax.set_ylabel("Number of districts")
    ax.set_title("LISA × Agglomerative\nCross-Tabulation", fontsize=11)
    ax.legend(fontsize=8, title="Cluster", title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Conditional probability heatmap
    ax = axes[1]
    pct_data = ct_pct.loc[ct_pct.index != "Total", ct_pct.columns != "Total"]
    im = ax.imshow(pct_data.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pct_data.columns)))
    ax.set_xticklabels(pct_data.columns, fontsize=9)
    ax.set_yticks(range(len(pct_data.index)))
    ax.set_yticklabels(pct_data.index, fontsize=9)
    ax.set_xlabel("Agglomerative cluster")
    ax.set_ylabel("LISA zone")
    ax.set_title("P(cluster | LISA label)", fontsize=11)

    for i in range(len(pct_data.index)):
        for j in range(len(pct_data.columns)):
            val = pct_data.iloc[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Probability")

    # Panel 3: Summary stats
    ax = axes[2]
    ax.axis("off")
    summary_lines = [
        f"Agglomerative Clustering (K={best_k}, λ={LAMBDA})",
        f"Optimal silhouette: {best_sil:.3f}",
        f"",
        f"Chi-squared test:",
        f"  χ² = {chi2:.1f}, p = {p_val:.4f}",
        f"  Cramér's V = {cramer_v:.3f}",
        f"",
        f"Clusters:",
    ]
    for cid in range(best_k):
        members = [d for d, c in cluster_map.items() if c == cid]
        mean_dd = annual[annual["feature_id"].isin(members)]["Drought_Days"].mean()
        n_hh = len([d for d in members
                    if merged.loc[merged["ADM_NAME"] == d, "lisa_short"].values[0] == "HH"] if len(members) > 0 else 0)

        summary_lines.append(f"  G{cid+1}: n={len(members)}, "
                           f"{mean_dd:.1f} days/yr")

    # Insurance interpretation
    summary_lines.extend([
        f"",
        f"Insurance Interpretation:",
    ])
    for cid in range(best_k):
        members = [d for d, c in cluster_map.items() if c == cid]
        n_ns = sum(1 for d in members
                   if merged.loc[merged["ADM_NAME"] == d, "lisa_short"].values[0] == "NS")
        n_hh = sum(1 for d in members
                   if merged.loc[merged["ADM_NAME"] == d, "lisa_short"].values[0] == "HH")
        n_ll = sum(1 for d in members
                   if merged.loc[merged["ADM_NAME"] == d, "lisa_short"].values[0] == "LL")
        dominant = max(
            [("NS", n_ns), ("HH", n_hh), ("LL", n_ll)] +
            [(l, sum(1 for d in members
                     if merged.loc[merged["ADM_NAME"] == d, "lisa_short"].values[0] == l))
             for l in ["HL", "LH"]],
            key=lambda x: x[1]
        )
        summary_lines.append(f"  G{cid+1}: mostly {dominant[0]} ({dominant[1]}/{len(members)})")

    if p_val < 0.05:
        if cramer_v > 0.3:
            summary_lines.append(f"  ✅ Strong association — LISA zones predict cluster membership")
            summary_lines.append(f"     → Insurance zones should consider both")
        else:
            summary_lines.append(f"  ⚠ Weak but significant association")
            summary_lines.append(f"     → Clusters and LISA capture complementary structure")
    else:
        summary_lines.append(f"  → Clusters and LISA are independent")
        summary_lines.append(f"    Use LISA for spatial risk pooling,")
        summary_lines.append(f"    clusters for temporal diversification")

    y = 0.95
    for line in summary_lines:
        ax.text(0.05, y, line, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace")
        y -= 0.045

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "zambia_lisa_vs_clusters_figure.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: zambia_lisa_vs_clusters_figure.png")

    # ---- Step 4: Report ----
    report_lines = [
        "=" * 70,
        "LISA vs AGGLOMERATIVE CLUSTERING — REPORT (Strategy 5.3)",
        "=" * 70,
        "",
        f"Agglomerative clustering: K={best_k} (auto-detected, λ={LAMBDA})",
        f"Optimal silhouette: {best_sil:.4f}",
        f"Spatial neighbors (k-NN): {SPATIAL_NEIGHBORS}",
        "",
        "Cross-tabulation (LISA × Cluster):",
        ct.round(1).to_string(),
        "",
        f"Chi-squared: χ²={chi2:.2f}, p={p_val:.4f}, Cramér's V={cramer_v:.3f}",
    ]

    if p_val < 0.05:
        report_lines.append("")
        report_lines.append("Result: Significant association between LISA zones")
        report_lines.append("and agglomerative clusters.")
        if cramer_v > 0.3:
            report_lines.append("The association is moderate-to-strong.")
        else:
            report_lines.append("The association is weak.")
    else:
        report_lines.append("")
        report_lines.append("Result: No significant association.")
        report_lines.append("LISA zones and clusters capture different structure.")

    report_lines.extend([
        "",
        "Insurance portfolio implications:",
    ])
    for cid in range(best_k):
        members = [d for d, c in cluster_map.items() if c == cid]
        member_str = ", ".join(sorted(members)[:5])
        if len(members) > 5:
            member_str += f" … (+{len(members) - 5} more)"
        report_lines.append(f"  G{cid+1} ({len(members)} districts): {member_str}")

    report_lines.extend([
        "",
        "Files: zambia_lisa_vs_clusters.csv, zambia_lisa_vs_clusters_figure.png,",
        "       zambia_agglomerative_clusters.csv",
        "=" * 70,
    ])

    report = "\n".join(report_lines)
    print("\n" + report)
    with open(OUTPUT_DIR / "zambia_lisa_vs_clusters_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved: zambia_lisa_vs_clusters_report.txt")
    print("\n✅ LISA vs CLUSTERING COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
