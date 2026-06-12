"""
nyquist_sampling_check.py — Nyquist-Shannon Sampling Robustness Check
=====================================================================
Purpose:
  Apply the Nyquist-Shannon sampling theorem to validate whether districts'
  agricultural extent is sufficient for 9km ERA5-Land sampling.

  Nyquist criterion for spatial data:
    sampling_interval ≤ 0.5 × feature_width
  → For 9km ERA5 pixels, minimum resolvable feature width = 4.5km
  → Minimum agricultural area (square approx) = 4.5² = 20.25 km²

  Three thresholds tested:
    - 20  km²  (relaxed: half-pixel width)
    - 80  km²  (moderate: one-pixel width)
    - 324 km²  (strict: two-pixel width, full Nyquist)

  Compares Nyquist filter against existing pixel-overlap quality_class
  to answer: do they capture different aspects of data quality?

Outputs:
  - zambia_nyquist_check.csv     — per-district Nyquist results
  - zambia_nyquist_figure.png    — comparison figure
  - zambia_nyquist_report.txt    — text summary
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

current_path = Path(__file__).resolve()
# Find Africa-Drought-Study project root (up 2 levels)
PROJECT_ROOT = next(p for p in current_path.parents if (p / "data").exists())

# Standardize Output Directory
OUTPUT_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "nyquist_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Point to your actual analysis results
WEEK_11_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "Week_11_correlation_map"
SPATIAL_DIR = PROJECT_ROOT / "dissertation_work" / "outputs" / "spatial_autocorrelation"

# Mapping Zambia pilot names to your actual files
# Update this line in nyquist_sampling_check.py
QUALITY_PATH = PROJECT_ROOT / "dissertation_work" / "outputs" / "overlap_analysis" / "districts_with_quality.geojson"
SSI_COMPARISON_PATH = WEEK_11_DIR / "drought_annual.csv"
INSURANCE_PATH = SPATIAL_DIR / "zambia_insurance_portfolio_table.csv"

NYQUIST_THRESHOLDS = [20, 80, 324]
LABELS = {
    20:  "Relaxed (≥4.5km, half-pixel)",
    80:  "Moderate (≥9km, one-pixel)",
    324: "Strict (≥18km, two-pixel Nyquist)",
}

CRITICAL_PIXEL_SIZE_KM = 9
HALF_PIXEL_KM = CRITICAL_PIXEL_SIZE_KM / 2


def load_data():
    """Load Zambia districts with quality metrics and SSI drought data."""
    districts = gpd.read_file(QUALITY_PATH)
    # Filter for Zambia
    if 'COUNTRY' in districts.columns:
        districts = districts[districts['COUNTRY'].str.upper() == 'ZAMBIA']
    districts.columns = [c.strip() for c in districts.columns]

    ssi = pd.read_csv(SSI_COMPARISON_PATH)
    # Map Week 11 column names to what this script expects
    ssi = ssi.rename(columns={"feature_id": "ADM_NAME", "Drought_Days": "drought_days_weighted"})
    ssi["ADM_NAME"] = ssi["ADM_NAME"].astype(str).str.strip().str.upper()

    ins = pd.read_csv(INSURANCE_PATH)
    ins["ADM_NAME"] = ins["ADM_NAME"].astype(str).str.strip().str.upper()

    return districts, ssi, ins


def compute_nyquist_metrics(districts):
    """
    For each district, compute Nyquist-relevant metrics.

    Parameters
    ----------
    districts : gpd.GeoDataFrame with crop_km2, total_km2, crop_pct

    Returns
    -------
    districts with added Nyquist columns
    """
    d = districts.copy()

    # Equivalent square side length of agricultural area
    d["ag_side_km"] = np.sqrt(d["crop_km2"])

    # Ratio of ag side to pixel size (how many pixels wide is the ag area)
    d["ag_pixels_wide"] = d["ag_side_km"] / CRITICAL_PIXEL_SIZE_KM

    # Nyquist pass/fail for each threshold
    for t in NYQUIST_THRESHOLDS:
        d[f"nyquist_pass_{t}"] = d["crop_km2"] >= t

    return d


def main():
    print("=" * 70)
    print("NYQUIST-SHANNON SAMPLING CHECK")
    print("=" * 70)

    print(f"\nPixel size: {CRITICAL_PIXEL_SIZE_KM}km")
    print(f"Nyquist limit (half-pixel): {HALF_PIXEL_KM}km feature width")
    print(f"Minimum ag area (relaxed):  {HALF_PIXEL_KM**2:.1f} km²")
    print(f"Minimum ag area (strict):   {(CRITICAL_PIXEL_SIZE_KM*2)**2:.0f} km²")
    print()

    districts, ssi, ins = load_data()
    districts = compute_nyquist_metrics(districts)

    # ---- Step 1: How many pass/fail each threshold ----
    print("=" * 70)
    print("STEP 1: Nyquist pass/fail by threshold")
    print("=" * 70)

    for t in NYQUIST_THRESHOLDS:
        passed = districts[f"nyquist_pass_{t}"].sum()
        failed = len(districts) - passed
        min_crop = districts.loc[districts[f"nyquist_pass_{t}"], "crop_km2"].min() if passed > 0 else 0
        print(f"  {LABELS[t]:>40s}: {passed:>2d} pass, {failed:>2d} fail  "
              f"(min crop_km2 = {min_crop:.0f})")

    # ---- Step 2: Overlap with quality_class ----
    print("\n" + "=" * 70)
    print("STEP 2: Nyquist vs Pixel-Overlap Quality")
    print("=" * 70)

    low_quality = set(districts[districts["quality_class"] == "Low"]["ADM_NAME"])
    for t in NYQUIST_THRESHOLDS:
        nyquist_fail = set(districts[~districts[f"nyquist_pass_{t}"]]["ADM_NAME"])
        both = low_quality & nyquist_fail
        only_low = low_quality - nyquist_fail
        only_nyquist = nyquist_fail - low_quality
        jaccard = len(both) / len(low_quality | nyquist_fail) if (low_quality | nyquist_fail) else 0

        print(f"\n  {LABELS[t]}:")
        print(f"    Low quality districts:       {len(low_quality):>2d}")
        print(f"    Nyquist-failed districts:     {len(nyquist_fail):>2d}")
        print(f"    Both fail:                    {len(both):>2d}")
        print(f"    Only Low quality:             {len(only_low):>2d}")
        print(f"    Only Nyquist-failed:          {len(only_nyquist):>2d}")
        print(f"    Jaccard similarity:           {jaccard:.2f}")

    # ---- Step 3: Drought day comparison ----
    print("\n" + "=" * 70)
    print("STEP 3: Drought Day Comparison (Nyquist-filtered)")
    print("=" * 70)

    # Merge SSI annual data with district info
    annual = ssi.groupby(["ADM_NAME"])["drought_days_weighted"].mean().reset_index()
    annual = annual.rename(columns={"drought_days_weighted": "mean_drought_days"})
    merged = districts.merge(annual, on="ADM_NAME", how="left")

    for t in NYQUIST_THRESHOLDS:
        pass_df = merged[merged[f"nyquist_pass_{t}"]]["mean_drought_days"]
        fail_df = merged[~merged[f"nyquist_pass_{t}"]]["mean_drought_days"]
        print(f"\n  {LABELS[t]}:")
        print(f"    Pass: n={len(pass_df):>2d}, mean={pass_df.mean():.1f}±{pass_df.std():.1f}")
        if len(fail_df) > 0:
            print(f"    Fail: n={len(fail_df):>2d}, mean={fail_df.mean():.1f}±{fail_df.std():.1f}")

    # ---- Step 4: Correlation between crop_km2 and quality ----
    print("\n" + "=" * 70)
    print("STEP 4: Correlation Analysis")
    print("=" * 70)

    qual_order = {"High": 2, "Medium": 1, "Low": 0}
    districts["qual_order"] = districts["quality_class"].map(qual_order)

    corr_km2_qual = districts["crop_km2"].corr(districts["qual_order"])
    corr_km2_pct = districts["crop_km2"].corr(districts["pct_full_pixels"])
    print(f"  crop_km2 vs quality_class:       r = {corr_km2_qual:.3f}")
    print(f"  crop_km2 vs pct_full_pixels:     r = {corr_km2_pct:.3f}")

    # Export results
    out_cols = ["ADM_NAME", "crop_km2", "crop_pct", "quality_class",
                "ag_side_km", "ag_pixels_wide", "pct_full_pixels"] + \
               [f"nyquist_pass_{t}" for t in NYQUIST_THRESHOLDS]
    out = districts[out_cols].copy()
    for t in NYQUIST_THRESHOLDS:
        out[f"nyquist_pass_{t}"] = out[f"nyquist_pass_{t}"].astype(int)
    out.to_csv(OUTPUT_DIR / "zambia_nyquist_check.csv", index=False)
    print(f"\n  Exported: zambia_nyquist_check.csv")

    # =======================================================================
    # FIGURE
    # =======================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Generating figure")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Nyquist-Shannon Sampling Check for Zambia Districts\n"
                 f"(ERA5-Land pixel = {CRITICAL_PIXEL_SIZE_KM}km, "
                 f"Nyquist limit = {HALF_PIXEL_KM}km)",
                 fontweight="bold", fontsize=13)

    colors = {"High": "#1b9e77", "Medium": "#d95f02", "Low": "#e7298a"}

    # --- Panel 1: Pie of quality classes ---
    ax = axes[0, 0]
    qual_counts = districts["quality_class"].value_counts()
    qual_vals = [qual_counts.get(q, 0) for q in ["High", "Medium", "Low"]]
    qual_cols = [colors.get(q, "gray") for q in ["High", "Medium", "Low"]]
    wedges, texts, autotexts = ax.pie(
        qual_vals, labels=["High", "Medium", "Low"], autopct="%1.0f%%",
        colors=qual_cols, startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("Pixel-Overlap Quality", fontsize=11)

    # --- Panel 2: Scatter crop_km2 vs pct_full_pixels ---
    ax = axes[0, 1]
    for q in ["High", "Medium", "Low"]:
        subset = districts[districts["quality_class"] == q]
        ax.scatter(subset["crop_km2"], subset["pct_full_pixels"],
                   c=colors.get(q, "gray"), label=q, alpha=0.7, edgecolors="k", s=40)
    for t in NYQUIST_THRESHOLDS:
        ax.axvline(t, color="gray", linestyle="--", alpha=0.4)
        ax.text(t, 5, f"{t}km²", rotation=90, fontsize=7, alpha=0.6)
    ax.set_xlabel("Agricultural area (km²)")
    ax.set_ylabel("Pixels fully contained (%)")
    ax.set_title("crop_km² vs Pixel-Overlap Quality", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # --- Panel 3: Ag side length distribution (Nyquist relevance) ---
    ax = axes[0, 2]
    ax.hist(districts["ag_side_km"], bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(HALF_PIXEL_KM, color="red", linestyle="--", linewidth=2,
               label=f"Nyquist limit ({HALF_PIXEL_KM}km)")
    ax.axvline(CRITICAL_PIXEL_SIZE_KM, color="orange", linestyle="--", linewidth=2,
               label=f"Pixel size ({CRITICAL_PIXEL_SIZE_KM}km)")
    ax.axvline(CRITICAL_PIXEL_SIZE_KM * 2, color="darkred", linestyle="--", linewidth=2,
               label=f"2× pixel ({CRITICAL_PIXEL_SIZE_KM*2}km)")
    ax.set_xlabel("Agricultural area equivalent side (km)")
    ax.set_ylabel("Number of districts")
    ax.set_title("Distribution of Ag Extent Width", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 4: Venn — Nyquist (strict) vs Low quality ---
    ax = axes[1, 0]
    low_set = set(districts[districts["quality_class"] == "Low"]["ADM_NAME"])
    strict_fail = set(districts[~districts["nyquist_pass_324"]]["ADM_NAME"])
    if low_set or strict_fail:
        v = venn2(
            subsets=(len(low_set - strict_fail),
                     len(strict_fail - low_set),
                     len(low_set & strict_fail)),
            set_labels=("Low quality\n(pixel-overlap)", "Nyquist fail\n(strict, 324km²)"),
            ax=ax
        )
        if v:
            for label in v.set_labels:
                if label:
                    label.set_fontsize(9)
    ax.set_title("Overlap: Quality vs Nyquist", fontsize=11)

    # --- Panel 5: Mean drought days by Nyquist threshold ---
    ax = axes[1, 1]
    x = np.arange(len(NYQUIST_THRESHOLDS))
    w = 0.3
    pass_means = []
    fail_means = []
    for t in NYQUIST_THRESHOLDS:
        pass_df = merged[merged[f"nyquist_pass_{t}"]]["mean_drought_days"]
        fail_df = merged[~merged[f"nyquist_pass_{t}"]]["mean_drought_days"]
        pass_means.append(pass_df.mean())
        fail_means.append(fail_df.mean() if len(fail_df) > 0 else np.nan)
    ax.bar(x - w/2, pass_means, w, label="Pass", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, fail_means, w, label="Fail", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Relaxed\n(20km²)", "Moderate\n(80km²)", "Strict\n(324km²)"], fontsize=9)
    ax.set_ylabel("Mean drought days/year")
    ax.set_title("Drought Days: Pass vs Fail", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 6: Nyquist pass/fail at each threshold ---
    ax = axes[1, 2]
    x = np.arange(len(NYQUIST_THRESHOLDS))
    pass_counts = [districts[f"nyquist_pass_{t}"].sum() for t in NYQUIST_THRESHOLDS]
    fail_counts = [len(districts) - c for c in pass_counts]
    ax.bar(x - w/2, pass_counts, w, label="Pass", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, fail_counts, w, label="Fail", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Relaxed\n(20km²)", "Moderate\n(80km²)", "Strict\n(324km²)"], fontsize=9)
    ax.set_ylabel("Number of districts")
    ax.set_title("Districts passing Nyquist", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "zambia_nyquist_figure.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: zambia_nyquist_figure.png")

    # =======================================================================
    # REPORT
    # =======================================================================
    report_lines = [
        "=" * 70,
        "NYQUIST-SHANNON SAMPLING CHECK — REPORT",
        "=" * 70,
        "",
        f"Pixel size:            {CRITICAL_PIXEL_SIZE_KM} km",
        f"Nyquist limit:         {HALF_PIXEL_KM} km (half-pixel width)",
        f"Total districts:       {len(districts)}",
        "",
        "Thresholds:",
    ]
    for t in NYQUIST_THRESHOLDS:
        passed = districts[f"nyquist_pass_{t}"].sum()
        failed = len(districts) - passed
        min_ag = districts.loc[districts[f"nyquist_pass_{t}"], "crop_km2"].min() if passed > 0 else 0
        report_lines.append(f"  {LABELS[t]}: {passed} pass, {failed} fail")
        report_lines.append(f"    Minimum crop_km² in pass group: {min_ag:.0f}")

    report_lines.extend([
        "",
        "Overlap with Low quality:",
    ])
    for t in NYQUIST_THRESHOLDS:
        nyquist_fail = set(districts[~districts[f"nyquist_pass_{t}"]]["ADM_NAME"])
        both = low_quality & nyquist_fail
        only_low = low_quality - nyquist_fail
        only_nyquist = nyquist_fail - low_quality
        report_lines.append(f"  {LABELS[t]}:")
        report_lines.append(f"    Low ∩ Nyquist-fail: {len(both)}")
        report_lines.append(f"    Only Low quality:   {len(only_low)}")
        report_lines.append(f"    Only Nyquist-fail:  {len(only_nyquist)}")

    report_lines.extend([
        "",
        f"Correlation crop_km² vs quality_class:  r = {corr_km2_qual:.3f}",
        f"Correlation crop_km² vs pct_full_pixels: r = {corr_km2_pct:.3f}",
        "",
        "CONCLUSION:",
    ])

    # Conclusion
    if districts["nyquist_pass_20"].all():
        report_lines.append("  ✅ ALL districts pass the relaxed Nyquist threshold (20 km²).")
        report_lines.append("     Zambia's districts have sufficient agricultural extent")
        report_lines.append("     for the half-pixel Nyquist criterion.")
    else:
        f20 = len(districts) - districts["nyquist_pass_20"].sum()
        report_lines.append(f"  ⚠ {f20} districts fail the relaxed Nyquist threshold.")

    strict_pass = districts["nyquist_pass_324"].sum()
    strict_fail = len(districts) - strict_pass
    report_lines.append(f"")
    report_lines.append(f"  At the strict Nyquist threshold (324 km²):")
    report_lines.append(f"    {strict_pass} districts pass, {strict_fail} fail.")
    report_lines.append(f"    The Nyquist filter identifies different districts than")
    report_lines.append(f"    the pixel-overlap quality_class, making them complementary.")

    report_lines.append("")
    report_lines.append(f"  Files: zambia_nyquist_check.csv, zambia_nyquist_figure.png")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)
    print(report)

    with open(OUTPUT_DIR / "zambia_nyquist_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved: zambia_nyquist_report.txt")
    print("\n NYQUIST CHECK COMPLETE")


if __name__ == "__main__":
    main()
