# Strategic Roadmap: SSI-Based Drought Study for Parametric Insurance

## Phase 0: Infrastructure & Version Control (Do First)

| Step | Action | Details |
|------|--------|---------|
| 0.1 | Create private GitHub repo | Initialize locally, push code (not data). Use .gitignore for large data files (CSVs, shapefiles). |
| 0.2 | Switch to Overleaf (LaTeX) | Migrate Word draft for better version control, math rendering, and table/citation management. |
| 0.3 | Restructure code into modules | Extract shared functions into utils/ssi_lib.py (gamma fit, risk windows, drought binning) so the dissertation pipeline is clean and reproducible. |

---

## Phase 1: Literature & Problem Definition (Weeks 1-2)

| Step | Action | Papers/Tools |
|------|--------|-------------|
| 1.1 | Overlapping Pixel Problem (MAUP) | Read Liping Di (agro-geoinformatics), Donglian Sun (GMU), and Openshaw (1984) on the Modifiable Areal Unit Problem. This is your methodological novelty. |
| 1.2 | Centroid-based proximity matching | Week_11 already uses centroid spatial join. Justify in lit review citing Flowerdew et al. (1991) and Fisher & Langford (1995) on areal interpolation. |
| 1.3 | Nyquist-Shannon sampling | For a 9km ERA5 pixel, resolvable feature size is ~4.5km. Address why district-level aggregation is valid (your sensitivity analysis proves it). |
| 1.4 | Crop data sources | Search: Liping Di global crop layers, GEOGLAM (already have V1.4), GLAD (already have), USDA FAS, ESA CCI Land Cover, MapSPAM. Build lit review comparison table. |
| 1.5 | SSI justification | Build comparative drought index table the professor requested: SSI (soil moisture, leading), SPI (precipitation only), SPEI (precip + PET), NDVI (lagging 2-4 weeks). Cite AghaKouchak (2014) and McKee et al. (1993). |

---

## Phase 2: Overlapping Pixel Solution (Your "Distinction" Work)

The professor said solving this properly separates a Distinction.

| Step | Action | Implementation |
|------|--------|----------------|
| 2.1 | Weighted pixel-district assignment | Instead of centroid-in-polygon (current), implement area-weighted averaging: 1) Overlay 9km ERA5 grid with district polygons 2) Compute weight = overlap_area / pixel_area for each (pixel, district) pair 3) Assign SSI_weighted = sum(SSI_pixel x weight) / sum(weight) per district. Code: utils/overlapping_pixel_solver.py |
| 2.2 | Alternative: Nyquist-based sampling | Test whether a district needs to be >= 4.5km in agricultural extent to get a clean signal. Filter districts by crop_pct * area > 20 km^2 as robustness check. |
| 2.3 | Compare both methods | Run SSI with (a) current centroid method and (b) weighted method. Show scatter plot: Centroid SSI vs Weighted SSI — correlation should be high (R^2 > 0.95) if centroid is valid. |
| 2.4 | Write the one-page narrative | Step 1: Overlay 9km grid with districts. Step 2: Compute overlap weights. Step 3: Pixel-to-polygon aggregation. This goes in your Methods chapter. |

---

## Phase 3: Sensitivity Analysis (Zambia District Level)

Already started in Week_10.py. Refine for the dissertation:

| Step | Action | Details |
|------|--------|---------|
| 3.1 | Crop area thresholds | Re-run Zambia temporal grouping at: 0%, 10%, 25%, 50%, 75% (expand from current 0/10/20/30%). |
| 3.2 | Metric: Temporal grouping stability | For each threshold, compute mean inter-district Pearson correlation (get_group_synchronicity in Week_10.py). Key finding: At >=25% crop area, correlation stabilizes at r approx 0.7. |
| 3.3 | Metric: Cluster stability | Run K-Means at each threshold using clustering_experiment.py, compare cluster changes. Output a cluster stability matrix (Jaccard similarity between assignments). |
| 3.4 | Visualization | Produce Sensitivity Heatmap: x-axis = thresholds, y-axis = districts, color = cluster assignment change. |

---

## Phase 4: All-Crops Footprint (Scaling Beyond Maize)

| Step | Action | Implementation |
|------|--------|----------------|
| 4.1 | GEOGLAM: Find all same-calendar crops | Week_10.py already identifies these for Zambia (lines 56-65). Extend to all African districts. |
| 4.2 | Create crop_group column | For each district, use the envelope of all crops within 30 days of the Maize 1 window. |
| 4.3 | Re-run SSI pipeline | Feed combined crop group into Week_11 pipeline instead of just Maize 1. |
| 4.4 | Compare results | Produce table: Scenario | Mean SSI | Drought Days/Year | Correlation Strength — Maize 1 only vs All same-cycle crops. |

---

## Phase 5: Spatial Correlation & Risk Balancing

| Step | Action | Details |
|------|--------|---------|
| 5.1 | Moran's I | Compute Global Moran's I on district-level drought-day counts. Test for significant spatial autocorrelation. |
| 5.2 | LISA clusters | Map High-High (severe drought clusters) and High-Low (drought islands) districts. Tells insurance company which districts co-vary in risk. |
| 5.3 | Integration with clustering | Compare Moran's I zones with spatial agglomerative clusters from spatial_agglomerative_clustering.py. LISA clusters give insurance-specific risk interpretation. |

---

## Phase 6: GitHub & Code Quality

| Step | Action |
|------|--------|
| 6.1 | Create .gitignore | Exclude: data/ (large shapefiles/CSVs), .venv/, outputs/, __pycache__/, *.html (maps) |
| 6.2 | Push code structure | core_analysis/ (pipeline_ssi.py, sensitivity_zambia.py, clustering_*.py), utils/ (ssi_lib.py, overlapping_pixel_solver.py), data_acquisition/ (gee scripts), notebooks/, outputs/ (gitignored), requirements.txt, README.md |
| 6.3 | Add docstrings | Each function should have brief docstring with purpose, inputs/outputs, and citation reference. |

---

## Phase 7: Writing Schedule (Aligned to Dissertation Timeline)

| Week | Writing Milestone | Technical Milestone |
|------|-------------------|---------------------|
| 1 | Introduction — CARS model. Add Maize dependency stats (% African GDP, calories). | Phase 0: GitHub, Overleaf |
| 2 | Lit Review — Drought index comparison table. MAUP/overlapping pixel lit review. | Phase 1: Literature search |
| 3 | Methods Section 2 — Pixel-to-Polygon Weighted Aggregation narrative | Phase 2: Weighted pixel solver |
| 4 | Methods Section 3 — SSI Gamma-CDF methodology | Phase 3: Sensitivity analysis |
| 5 | Results Section 1 — Sensitivity tables and figures | Phase 4: All-crops pipeline |
| 6 | Results Section 2 — Spatial clusters and Moran's I maps | Phase 5: LISA analysis |
| 7 | Discussion — Connect results to insurance use case | Phase 6: GitHub cleanup |
| 8 | Conclusion — Limitations, future work | Buffer week |
| 9 | Full draft review | |
| 10 | Final polish | Submit |

---

## Critical Decision Points (Ask Your Professor)

1. Weighted vs centroid matching — Present both options, ask which he prefers for the novelty claim.
2. Threshold selection — After running sensitivity at 0/10/25/50/75%, present the sweet spot plot and ask which is more defensible for insurance.
3. Moran's I interpretation — Ask if he wants insurance-specific High-High risk clusters map or general spatial autocorrelation.

---

## Immediate Actions (Today)

1. Create GitHub repo and push current code (with .gitignore).
2. Read 2 key papers: Openshaw (1984) on MAUP + AghaKouchak (2014) on SSI.
3. Refactor Week_11_correlation_map.py into core_analysis/pipeline_ssi.py with a main() function.
4. Run Week_10.py with expanded thresholds (0/10/25/50/75) and save the sensitivity CSV as baseline result.
