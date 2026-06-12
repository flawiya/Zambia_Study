"""
cluster_stability.py — Cluster Stability Matrix (Strategy 3.3)
===============================================================
Purpose:
  Measure how stable K-Means cluster assignments are across different
  crop area thresholds. The strategy asks for a Jaccard similarity
  matrix between cluster assignments.

Method:
  1. Compute z-score SSI and annual drought days for Zambia districts
  2. For each crop area threshold (0%, 10%, 20%, 25%, 30%, 35%, 40%):
     a. Filter districts above the threshold
     b. Run K-Means (n_clusters=3, matching Week_10 methodology)
  3. For each pair of thresholds:
     a. Restrict to districts common to both
     b. Compute co-membership Jaccard similarity
     c. Compute Adjusted Rand Index (for comparison)
  4. Output: stability matrix + heatmap figure

Reference:
  Week_10 uses K-Means with n_clusters=3, random_state=42.
  Jaccard similarity on co-membership follows strategy item 3.3.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- EMBEDDED UTILS (Replaces the need for utils.ssi_lib) ---
def compute_zscore_ssi(df):
    """Computes a simple Z-Score based SSI for clustering stability."""
    df = df.copy()
    # Group by District and Day of Year to get local means/stds
    stats = df.groupby(['feature_id', 'doy'])['volumetric_soil_water_layer_2'].agg(['mean', 'std']).reset_index()
    df = df.merge(stats, on=['feature_id', 'doy'], how='left')
    # Z-Score formula
    df['SSI'] = (df['volumetric_soil_water_layer_2'] - df['mean']) / df['std']
    # Handle cases where std is 0
    df['SSI'] = df['SSI'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def aggregate_annual_drought_days(df, ssi_threshold=-1.0):
    """Counts days per year where SSI is below threshold."""
    df['is_drought'] = (df['SSI'] <= ssi_threshold).astype(int)
    annual = df.groupby(['feature_id', 'year'])['is_drought'].sum().reset_index()
    return annual.rename(columns={'is_drought': 'Drought_Days'})

# --- ROBUST PATH LOGIC ---
current_path = Path(__file__).resolve()
# Find Africa-Drought-Study root
PROJECT_ROOT = next(p for p in current_path.parents if (p / "data").exists())
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "cluster_stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Point to your Week 11 results
WEEK_11_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "Week_11_correlation_map"
QUALITY_PATH = WEEK_11_DIR / "valid_maize.geojson" 
ERA5_PATH = DATA_DIR / "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"

SSI_THRESHOLD = -1.0
N_CLUSTERS = 3
CROP_AREA_THRESHOLDS = [0, 10, 20, 25, 30, 35, 40]
RANDOM_STATE = 42

# --- REST OF THE SCRIPT (load_data, etc.) ---
def load_data():
    """Load Zambia districts with quality + ERA5, return z-score SSI annual drought days."""
    if not QUALITY_PATH.exists():
        print(f"ERROR: Quality file not found at {QUALITY_PATH}")
        sys.exit(1)

    districts = gpd.read_file(QUALITY_PATH)
    # Ensure it's filtered to Zambia for the pilot stability check
    if 'COUNTRY' in districts.columns:
        districts = districts[districts['COUNTRY'].str.upper() == 'ZAMBIA']
    
    districts["ADM_NAME"] = districts["ADM_NAME"].astype(str).str.strip().str.upper()

    print(f"Loading ERA5 for Zambia districts ({len(districts)} total) ...")
    zmb_names = set(districts["ADM_NAME"])
    
    dtype = {"year": "int16", "month": "int8", "day": "int8", "doy": "int16",
             "volumetric_soil_water_layer_2": "float32"}
    
    # Read ERA5 (Filter for Zambia only to save RAM)
    df_era5 = pd.read_csv(ERA5_PATH, dtype=dtype)
    df_era5["feature_id"] = df_era5["feature_id"].astype(str).str.strip().str.upper()
    df_era5 = df_era5[df_era5["feature_id"].isin(zmb_names)]

    print(f"Computing Z-Score SSI ...")
    df = compute_zscore_ssi(df_era5)

    # Risk window: Jan-Apr
    df_risk = df[df["month"].between(1, 4)].copy()

    # Annual drought days
    annual = aggregate_annual_drought_days(df_risk, ssi_threshold=SSI_THRESHOLD)
    print(f"  Districts: {annual['feature_id'].nunique()}, "
          f"Years: {annual['year'].min()}–{annual['year'].max()}")

    return districts, annual

# (Keep your existing run_clustering_for_threshold, co_membership_jaccard, and main functions below)


def run_clustering_for_threshold(annual_df, district_list, n_clusters=N_CLUSTERS):
    """
    Run K-Means on drought-day profiles for a set of districts.

    Follows Week_10 methodology: pivot to (year × district), transpose,
    standardize, K-Means.

    Returns
    -------
    dict: {district_name: cluster_label}
    """
    pivot = annual_df.pivot_table(
        index="year", columns="feature_id", values="Drought_Days"
    )
    available = [d for d in district_list if d in pivot.columns]
    if len(available) < n_clusters:
        return None

    data = pivot[available].T.fillna(pivot.mean())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(scaled)
    return dict(zip(available, labels))


def co_membership_jaccard(labels1, labels2):
    """
    Compute Jaccard similarity between two cluster assignments using
    co-membership. Measures fraction of district pairs that agree
    in both clusterings.

    J = |same_pairs_in_both| / |same_pairs_in_either|
    """
    n = len(labels1)
    # Co-membership vectors (upper triangle of n×n matrix)
    pairs_same_1 = set()
    pairs_same_2 = set()
    for i in range(n):
        for j in range(i + 1, n):
            if labels1[i] == labels1[j]:
                pairs_same_1.add((i, j))
            if labels2[i] == labels2[j]:
                pairs_same_2.add((i, j))

    intersection = pairs_same_1 & pairs_same_2
    union = pairs_same_1 | pairs_same_2
    if not union:
        return 1.0
    return len(intersection) / len(union)


def main():
    print("=" * 70)
    print("CLUSTER STABILITY ANALYSIS — Strategy 3.3")
    print("=" * 70)

    districts, annual = load_data()

    # ---- Step 1: Run clustering at each threshold ----
    print("\n" + "=" * 70)
    print("STEP 1: Running K-Means at each crop area threshold")
    print("=" * 70)

    threshold_labels = {}
    n_districts = {}
    for t in CROP_AREA_THRESHOLDS:
        names = districts[districts["crop_pct"] >= t]["ADM_NAME"].unique()
        labels = run_clustering_for_threshold(annual, names)
        if labels is not None:
            threshold_labels[t] = labels
            n_districts[t] = len(labels)
            print(f"  Threshold {t:>2}%: {len(labels):>2} districts, "
                  f"{len(set(labels.values()))} clusters")
        else:
            print(f"  Threshold {t:>2}%: too few districts ({len(names)}), skipping")

    # ---- Step 2: Compute stability matrix ----
    print("\n" + "=" * 70)
    print("STEP 2: Computing pairwise stability metrics")
    print("=" * 70)

    valid_thresholds = sorted(threshold_labels.keys())
    k = len(valid_thresholds)
    jaccard_matrix = np.ones((k, k))
    ari_matrix = np.ones((k, k))

    for i, t1 in enumerate(valid_thresholds):
        for j, t2 in enumerate(valid_thresholds):
            if i >= j:
                continue

            # Common districts (intersection)
            common = list(set(threshold_labels[t1].keys()) &
                          set(threshold_labels[t2].keys()))
            if len(common) < 3:
                jaccard_matrix[i, j] = jaccard_matrix[j, i] = np.nan
                ari_matrix[i, j] = ari_matrix[j, i] = np.nan
                continue

            l1 = [threshold_labels[t1][d] for d in common]
            l2 = [threshold_labels[t2][d] for d in common]

            jac = co_membership_jaccard(np.array(l1), np.array(l2))
            ari = adjusted_rand_score(l1, l2)

            jaccard_matrix[i, j] = jaccard_matrix[j, i] = jac
            ari_matrix[i, j] = ari_matrix[j, i] = ari

            print(f"  {t1:>2}% vs {t2:>2}%: Jaccard={jac:.3f}, ARI={ari:.3f} "
                  f"(n_common={len(common)})")

    # ---- Step 3: Export ----
    print("\n" + "=" * 70)
    print("STEP 3: Saving results")
    print("=" * 70)

    labels_list = [f"{t}%" for t in valid_thresholds]

    df_jac = pd.DataFrame(jaccard_matrix, index=labels_list, columns=labels_list)
    df_ari = pd.DataFrame(ari_matrix, index=labels_list, columns=labels_list)
    df_jac.to_csv(OUTPUT_DIR / "zambia_cluster_stability_jaccard.csv")
    df_ari.to_csv(OUTPUT_DIR / "zambia_cluster_stability_ari.csv")
    print(f"  Saved: zambia_cluster_stability_jaccard.csv")
    print(f"  Saved: zambia_cluster_stability_ari.csv")

    # Per-threshold cluster assignments
    assignment_rows = []
    for t in valid_thresholds:
        for district, cluster in threshold_labels[t].items():
            assignment_rows.append({
                "threshold": t,
                "district": district,
                "cluster": cluster
            })
    df_assign = pd.DataFrame(assignment_rows)
    df_assign.to_csv(OUTPUT_DIR / "zambia_cluster_assignments.csv", index=False)
    print(f"  Saved: zambia_cluster_assignments.csv")
    print(f"  {len(df_assign)} total (district, threshold) assignments")

    # ---- Step 4: Figure ----
    print("\n" + "=" * 70)
    print("STEP 4: Generating stability heatmap figure")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Cluster Stability Across Crop Area Thresholds\n"
                 "(K-Means, n=3, z-score SSI drought days)",
                 fontweight="bold", fontsize=13)

    common_kw = dict(annot=True, fmt=".3f", cmap="YlOrRd",
                     vmin=0, vmax=1,
                     square=True, linewidths=0.5,
                     cbar_kws={"shrink": 0.75, "label": "Stability"})

    # Panel 1: Jaccard matrix
    ax = axes[0]
    im = ax.imshow(jaccard_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels_list, fontsize=8)
    ax.set_yticklabels(labels_list, fontsize=8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Threshold")
    ax.set_title("Co-membership Jaccard", fontsize=11)
    for i in range(k):
        for j in range(k):
            val = jaccard_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Jaccard similarity")

    # Panel 2: ARI matrix
    ax = axes[1]
    im = ax.imshow(ari_matrix, cmap="YlOrRd", vmin=-0.2, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels_list, fontsize=8)
    ax.set_yticklabels(labels_list, fontsize=8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Threshold")
    ax.set_title("Adjusted Rand Index", fontsize=11)
    for i in range(k):
        for j in range(k):
            val = ari_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Adjusted Rand Index")

    # Panel 3: District count and stability summary
    ax = axes[2]
    thresholds_plot = valid_thresholds
    n_dists = [n_districts[t] for t in thresholds_plot]

    # Mean Jaccard vs highest threshold (as measure of how stable each is)
    mean_jac_vs_highest = []
    for i, t in enumerate(thresholds_plot):
        vals = [jaccard_matrix[i, j] for j in range(k)
                if j != i and not np.isnan(jaccard_matrix[i, j])]
        mean_jac_vs_highest.append(np.mean(vals) if vals else np.nan)

    ax2 = ax.twinx()
    bars = ax.bar(range(len(thresholds_plot)), n_dists, color="steelblue",
                  alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(thresholds_plot)))
    ax.set_xticklabels([f"{t}%" for t in thresholds_plot], fontsize=9)
    ax.set_ylabel("Number of districts", color="steelblue", fontsize=10)
    ax.set_xlabel("Crop area threshold")
    ax.set_title("District Count & Mean Stability", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    line = ax2.plot(range(len(thresholds_plot)), mean_jac_vs_highest,
                    "o-", color="darkred", linewidth=2.5, label="Mean Jaccard")
    ax2.set_ylabel("Mean Jaccard similarity", color="darkred", fontsize=10)
    ax2.set_ylim(0, 1.1)
    if not np.isnan(np.nanmean(mean_jac_vs_highest)):
        ax2.axhline(np.nanmean(mean_jac_vs_highest), color="darkred",
                    linestyle="--", alpha=0.4, linewidth=1)

    # Bar labels
    for bar, n in zip(bars, n_dists):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(n), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "zambia_cluster_stability.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: zambia_cluster_stability.png")

    # ---- Step 5: Report ----
    report_lines = [
        "=" * 70,
        "CLUSTER STABILITY ANALYSIS — REPORT (Strategy 3.3)",
        "=" * 70,
        "",
        f"Method: K-Means (k={N_CLUSTERS}), z-score SSI drought days, Jan-Apr risk window",
        f"Thresholds: {valid_thresholds}",
        f"Stable metric: Co-membership Jaccard similarity",
        "",
        "Jaccard Matrix:",
    ]
    report_lines.append(df_jac.round(3).to_string())
    report_lines.append("")
    report_lines.append("ARI Matrix:")
    report_lines.append(df_ari.round(3).to_string())

    report_lines.extend([
        "",
        "Interpretation:",
        f"  Jaccard ~1.0 = identical cluster assignments (perfect stability)",
        f"  Jaccard ~0.3 = random agreement",
        f"  Jaccard ~0.0 = completely different assignments",
        "",
        "Key finding:",
    ])

    # Find the best and worst pairs
    valid_jac = [(valid_thresholds[i], valid_thresholds[j], jaccard_matrix[i, j])
                 for i in range(k) for j in range(i + 1, k)
                 if not np.isnan(jaccard_matrix[i, j])]
    if valid_jac:
        best = max(valid_jac, key=lambda x: x[2])
        worst = min(valid_jac, key=lambda x: x[2])
        report_lines.append(f"  Most stable pair:   {best[0]}% vs {best[1]}%  (Jaccard = {best[2]:.3f})")
        report_lines.append(f"  Least stable pair:  {worst[0]}% vs {worst[1]}%  (Jaccard = {worst[2]:.3f})")

    mean_jac = np.nanmean([x[2] for x in valid_jac])
    report_lines.append(f"  Mean Jaccard across all pairs: {mean_jac:.3f}")
    if mean_jac > 0.7:
        report_lines.append("  ✅ Strong stability — clustering is robust to threshold choice.")
    elif mean_jac > 0.4:
        report_lines.append("  ⚠ Moderate stability — some sensitivity to threshold.")
    else:
        report_lines.append("  ❌ Weak stability — clustering changes substantially with threshold.")

    report_lines.append("")
    report_lines.append(f"Files: zambia_cluster_stability_jaccard.csv, zambia_cluster_stability_ari.csv,")
    report_lines.append(f"       zambia_cluster_assignments.csv, zambia_cluster_stability.png")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)
    print("\n" + report)
    with open(OUTPUT_DIR / "zambia_cluster_stability_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved: zambia_cluster_stability_report.txt")
    print("\n✅ CLUSTER STABILITY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
