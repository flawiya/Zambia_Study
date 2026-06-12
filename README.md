

This folder contains the dedicated dissertation analysis workspace for Zambia district-level drought and maize crop sensitivity research.

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
