"""
ssi_lib.py — Shared SSI Computation & Drought Analysis Utilities
=================================================================
Extracted from Week_11_correlation_map.py to provide a clean,
reproducible API for the dissertation pipeline.

Functions:
  - compute_daily_gamma_ssi     — Full Gamma-CDF SSI (AghaKouchak 2014)
  - compute_zscore_ssi          — Fast z-score SSI for sensitivity analysis
  - calculate_risk_windows      — Risk window with skip-first-25% rule
  - aggregate_annual_drought_days — Count drought days per district-year
  - classify_drought_severity   — Map drought-day count to WMO severity category
  - get_category_order          — Ordered severity labels
  - get_color_map               — Colorblind-safe severity colors
  - bin_drought_days            — Add Drought_Category column

References:
  - McKee, Doesken & Kleist (1993), 8th Conf. Applied Climatology
  - AghaKouchak (2014), HESS, 18(7), 2515–2526
  - WMO (2012) SPI User Guide
  - Svoboda et al. (2002) USDM Drought Monitor classification
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DROUGHT_BINS = [
    (0,    0,   "No Drought"),
    (1,   10,   "D1 – Abnormally Dry"),
    (11,  20,   "D2 – Moderate Drought"),
    (21,  35,   "D3 – Severe Drought"),
    (36,  55,   "D4 – Extreme Drought"),
    (56, None,  "D5 – Exceptional Drought"),
]

SSI_THRESHOLD_DEFAULT = -1.0
MIN_OBS_FOR_FIT_DEFAULT = 10


def calculate_risk_windows(valid_maize, skip_fraction=0.25):
    """
    For each district, compute the RISK WINDOW within the growing season.

    Why skip the first 25%?
      The early season (planting -> early vegetative) is when crops are
      establishing roots. Soil moisture stress during this phase is less
      impactful on final yield than stress during flowering/grain-fill.
      The remaining 75% of the season is the "risk window".

    Parameters
    ----------
    valid_maize : pd.DataFrame — must have columns [planting, endofseaso]
    skip_fraction : float — fraction of season to skip (default 0.25)

    Returns
    -------
    pd.DataFrame with added columns [risk_start_doy, risk_end_doy]
    """
    result = valid_maize.copy()

    def _calc_risk(row):
        p = row["planting"]
        h = row["endofseaso"]
        duration = (h - p) if h >= p else (365 - p) + h
        delay = duration * skip_fraction
        risk_start = int(round((p + delay) % 365))
        risk_end = h
        return pd.Series([risk_start, risk_end], index=["risk_start_doy", "risk_end_doy"])

    result[["risk_start_doy", "risk_end_doy"]] = result.apply(_calc_risk, axis=1)
    return result


def compute_daily_gamma_ssi(df, ssi_threshold=SSI_THRESHOLD_DEFAULT,
                            min_obs_for_fit=MIN_OBS_FOR_FIT_DEFAULT):
    """
    Compute the Standardised Soil-moisture Index (SSI) for each daily
    observation using a Gamma-CDF transformation.

    METHOD (per AghaKouchak 2014, McKee et al. 1993):
      For each unique (district, day-of-year) combination:
        1. Gather ALL years of soil moisture values for that DOY
        2. Separate into zero and non-zero values
        3. Fit a Gamma distribution to the non-zero values
        4. For observed value x:
              If x <= 0:  p = q_zero / 2
              If x > 0:   p = q_zero + (1 - q_zero) * Gamma_CDF(x)
        5. Transform:  SSI = Phi^{-1}(p)

    Parameters
    ----------
    df : pd.DataFrame — must have columns [feature_id, doy,
        volumetric_soil_water_layer_2]
    ssi_threshold : float — SSI threshold for drought (default -1.0)
    min_obs_for_fit : int — minimum observations for Gamma fit (default 10)

    Returns
    -------
    pd.DataFrame with new 'SSI' column
    """
    df = df.copy()
    df["SSI"] = np.nan

    groups = df.groupby(["feature_id", "doy"])

    for (district, doy), group in groups:
        values = group["volumetric_soil_water_layer_2"].dropna().values

        if len(values) < min_obs_for_fit:
            continue

        nonzero = values[values > 0]
        n_zeros = len(values) - len(nonzero)
        q_zero = n_zeros / len(values)

        if len(nonzero) < 5:
            continue

        try:
            alpha, loc, beta = scipy_stats.gamma.fit(nonzero, floc=0)
        except Exception:
            continue

        if alpha <= 0 or beta <= 0 or np.isnan(alpha) or np.isnan(beta):
            continue

        idx = group.index
        sm_vals = df.loc[idx, "volumetric_soil_water_layer_2"].values
        ssi_vals = np.full(len(sm_vals), np.nan)

        for i, sm_val in enumerate(sm_vals):
            if pd.isna(sm_val):
                continue
            if sm_val <= 0:
                p = q_zero / 2.0
            else:
                p = q_zero + (1.0 - q_zero) * scipy_stats.gamma.cdf(
                    sm_val, alpha, loc=0, scale=beta
                )
            p = np.clip(p, 0.001, 0.999)
            ssi_vals[i] = scipy_stats.norm.ppf(p)

        df.loc[idx, "SSI"] = ssi_vals

    df = df.dropna(subset=["SSI"])
    return df


def compute_zscore_ssi(df, ssi_threshold=SSI_THRESHOLD_DEFAULT):
    """
    Fast z-score approximation of SSI per (district, DOY).

    Computes (observation - mean) / std for each DOY per district.
    This is a computationally cheaper approximation of the full
    Gamma-CDF method, suitable for sensitivity analysis where
    relative comparisons matter more than exact quantiles.

    Parameters
    ----------
    df : pd.DataFrame — must have columns [feature_id, doy,
        volumetric_soil_water_layer_2, month]
    ssi_threshold : float — SSI threshold for drought (default -1.0)

    Returns
    -------
    pd.DataFrame with new 'SSI' column
    """
    df = df.copy()
    clim = df.groupby(["feature_id", "doy"])[
        "volumetric_soil_water_layer_2"
    ].agg(["mean", "std"]).reset_index()
    df = df.merge(clim, on=["feature_id", "doy"], how="left")
    df["SSI"] = (df["volumetric_soil_water_layer_2"] - df["mean"]) / (df["std"] + 1e-6)
    return df


def aggregate_annual_drought_days(df, ssi_threshold=SSI_THRESHOLD_DEFAULT,
                                  ssi_col="SSI"):
    """
    Count the number of drought days per district per crop-year.

    A "drought day" is any day where SSI <= -1.0 (15.9th percentile),
    the WMO-defined onset of "Moderate Drought".

    Parameters
    ----------
    df : pd.DataFrame — must have columns [feature_id, SSI, crop_year or year, month]
    ssi_threshold : float — SSI threshold (default -1.0)
    ssi_col : str — name of SSI column (default "SSI")

    Returns
    -------
    pd.DataFrame with columns [feature_id, year, Drought_Days]
    """
    df = df.copy()

    # Determine year column
    if "crop_year" in df.columns:
        year_col = "crop_year"
    else:
        df["crop_year"] = np.where(df["month"] >= 11, df["year"] + 1, df["year"])
        year_col = "crop_year"

    is_drought = (df[ssi_col] <= ssi_threshold).astype(int)
    df["_drought"] = is_drought

    annual = (
        df.groupby(["feature_id", year_col])["_drought"]
        .sum()
        .reset_index()
        .rename(columns={"_drought": "Drought_Days", year_col: "year"})
    )
    annual = annual[(annual["year"] >= 2000) & (annual["year"] <= 2025)]
    return annual


def classify_drought_severity(drought_days, bins=None):
    """
    Map a drought-day count to a severity category.

    Follows WMO (2012) SPI User Guide thresholds adapted to drought-day
    counts, aligned with the US Drought Monitor D0-D4 scale.

    Parameters
    ----------
    drought_days : int — number of drought days in a season
    bins : list of tuples — (lower, upper, label), defaults to DROUGHT_BINS

    Returns
    -------
    str — severity category label
    """
    if bins is None:
        bins = DROUGHT_BINS
    for lower, upper, label in bins:
        if upper is None:
            return label
        if lower <= drought_days <= upper:
            return label
    return bins[-1][2]


def get_category_order(bins=None):
    """
    Return ordered list of severity labels (for Plotly category ordering).

    Parameters
    ----------
    bins : list of tuples, defaults to DROUGHT_BINS

    Returns
    -------
    list of str — severity labels in order
    """
    if bins is None:
        bins = DROUGHT_BINS
    return [label for _, _, label in bins]


def get_color_map(category_order):
    """
    Assign colorblind-safe sequential warm palette to severity bins.

    Parameters
    ----------
    category_order : list of str — severity labels in order

    Returns
    -------
    dict — {category: color}
    """
    SEVERITY_COLORS = {
        "No Drought":                "rgba(220,220,220,0.25)",
        "D1 – Abnormally Dry":       "#FFEDA0",
        "D2 – Moderate Drought":     "#FEB24C",
        "D3 – Severe Drought":       "#FC4E2A",
        "D4 – Extreme Drought":      "#BD0026",
        "D5 – Exceptional Drought":  "#4A0010",
    }
    return {cat: SEVERITY_COLORS.get(cat, "#999999") for cat in category_order}


def bin_drought_days(df_annual, bins=None):
    """
    Add a 'Drought_Category' column to df_annual based on severity bins.

    Parameters
    ----------
    df_annual : pd.DataFrame — must have column 'Drought_Days'
    bins : list of tuples, defaults to DROUGHT_BINS

    Returns
    -------
    pd.DataFrame with new 'Drought_Category' column
    """
    if bins is None:
        bins = DROUGHT_BINS
    df_annual = df_annual.copy()
    df_annual["Drought_Category"] = df_annual["Drought_Days"].apply(
        lambda x: classify_drought_severity(x, bins)
    )
    return df_annual
