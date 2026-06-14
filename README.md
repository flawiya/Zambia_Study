# Zambia Agricultural Drought Multi-Crop Analysis
## Parametric Insurance Risk Modeling using Gamma-SSI

### 1. Project Overview
This repository contains a high-performance data science pipeline designed to quantify sub-national agricultural drought risk across Africa. The framework utilizes the **Standardized Soil-moisture Index (SSI)** derived from 25 years of ERA5-Land daily data (2000–2025). By standardizing soil moisture through probability distributions, the pipeline identifies "Drought Hubs" and teleconnected risks, providing a data-driven foundation for parametric insurance design.

---

### 2. Methodological Framework & Validation
To ensure scientific validity and address the limitations of coarse-resolution climate data, the following methodologies were implemented:

| Technical Requirement | Methodological Solution | Script Name |
| :--- | :--- | :--- |
| **Solve "Overlapping Pixel Problem"** | Implemented **Area-Weighted Areal Interpolation**. A 0.1° fishnet matching the ERA5-Land grid was constructed, calculating precise intersection areas for every pixel/district pair to prevent boundary bias. | `overlapping_pixel_solver.py` |
| **Spatial Sampling Validation** | Applied the **Nyquist-Shannon Sampling Theorem**. Features must be $\geq$ 4.5km wide ($20.25 km^2$) to be resolved by a 9km grid; this check ensures district sizes are sufficient for the sensor resolution. | `nyquist_sampling_check.py` |
| **Clustering Stability** | Conducted co-membership stability testing on K-Means clusters across multiple crop-area thresholds (0% to 40%) to ensure "Drought Hubs" are consistent regardless of agricultural density. | `cluster_stability.py` |
| **Spatial Risk Concentration** | Utilized **Global Moran's I** and **Local Indicators of Spatial Association (LISA)** to identify "High-High" systemic risk clusters and "High-Low" diversification opportunities. | `spatial_autocorrelation.py` |
| **Crop Footprint Expansion** | Developed a multi-crop envelope tool to identify all secondary crops sharing the primary Maize growing window, ensuring the soil moisture signal represents total agricultural risk. | `all_crops_footprint.py` |
| **Indicator Lead-Time** | Implemented **Gamma-CDF SSI** (AghaKouchak, 2014). SSI serves as a leading indicator of plant stress, unlike NDVI, which is a lagging indicator reflecting damage only after it has occurred. | `ssi_lib.py` |

---

### 3. Key Justifications

#### Addressing the Modifiable Areal Unit Problem (MAUP)
A 9km ERA5-Land pixel often straddles multiple district boundaries. To resolve this, we move beyond simple centroid-based assignment. By computing a weight matrix, we harmonize pixels to polygons (districts) based on actual geometric contribution.
*   **Robustness Proof:** A comparison of area-weighted vs. centroid methods for 70 Zambia districts showed that while planting calendars varied, the resulting drought-day counts were highly correlated ($r=0.96$). This proves the methodology is robust for continental-scale deployment.

#### Nyquist-Shannon Theorem for Spatial Data
To validate the use of 9km data for sub-national districts, we apply signal processing theory. According to the Nyquist criterion, the sampling interval must be $\leq 0.5 \times$ the feature width. Our check confirms that the agricultural extent in the study areas exceeds these requirements, minimizing "aliasing" in the climate signal and ensuring that the extracted SSI values are truly representative of local conditions.

---

### 4. Execution Sequence
To reproduce the full analysis, scripts should be executed in the following order:

**Phase 1: Foundational Processing**
1.  **`Week_11_correlation_map.py`**: The primary engine. Processes raw ERA5 data, computes Gamma-CDF SSI, and builds the 25-year continental drought database.

**Phase 2: Reliability & Quality Control**
2.  **`overlapping_pixel_solver.py`**: Computes geometric weights and assigns quality classes (High/Medium/Low) to every district.
3.  **`zambia_weighted_analysis.py`**: Quantifies the variation between area-weighted results and centroid results to provide a "Robustness Benchmark."
4.  **`nyquist_sampling_check.py`**: Validates the spatial compatibility between district sizes and climate sensor resolution.

**Phase 3: Sensitivity & Portfolio Strategy**
5.  **`all_crops_footprint.py`**: Maps multi-crop overlaps to confirm the "Maize-only" index is an effective proxy for general agricultural risk.
6.  **`cluster_stability.py`**: Proves that spatial drought groupings remain stable across different agricultural intensity filters.
7.  **`spatial_autocorrelation.py`**: Identifies geographic hotspots of risk for insurance underwriting.
8.  **`lisa_clustering_comparison.py`**: Statistically correlates geographic hotspots with climate-timing clusters to define optimal risk pools.

---

### 5. Technical Requirements
*   **Python 3.10+**
*   **Geospatial:** `geopandas`, `libpysal`, `esda`, `shapely`
*   **Statistics:** `pandas`, `numpy`, `scipy`, `scikit-learn`
*   **Visualization:** `plotly`, `matplotlib`, `matplotlib-venn`

## Structure

- `scripts/` - working analysis scripts for dissertation tasks.
- `data/` - derived or focused data files for dissertation analysis.
- `outputs/` - charts, tables, and results generated during dissertation work.

## Initial goals

1. Keep dissertation code and outputs separate from the main AFRICA-DROUGHT-STUDY office work.
2. Build analysis around maize sensitivity, district-level aggregation, and ERA5-Land resolution questions.
3. Prepare for later GitHub upload with a clean project structure.

## Next steps

- Run `python scripts/maize_threshold_sensitivity.py` to generate threshold sensitivity outputs.
- Run `python scripts/visualize_threshold_sensitivity.py` to create interactive HTML charts and maps.
- Open the generated HTML files in `dissertation_work/outputs/`:
  - `threshold_sensitivity_crop_summary.html`
  - `threshold_sensitivity_calendar_period_summary.html`
  - `threshold_sensitivity_maize1_map.html`
- Use the generated CSVs in `outputs/` to review how district inclusion changes by crop-area threshold.
- Document data sources and literature on pixel-polygon overlap.
- Confirm whether 9 km ERA5-Land is appropriate for district-level maize analysis.
