import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
import os

# 1. SETUP PATHS
base_path = r"C:\Users\FlawiyaShirishMore\Downloads\Africa-Drought-Study"
# This points to the specific shapefile in your data folder
gadm_path = os.path.join(base_path, "data", "africa_agricultural_domain_2019.shp")

# 2. LOAD DATA & FILTER FOR ZAMBIA
print("Loading Africa-wide Agricultural Domain...")
gdf_africa = gpd.read_file(gadm_path)

# Ensure column names match (Standardize to COUNTRY and ADM_NAME)
# In your dataset, it is likely 'COUNTRY' or 'country'
gdf_zambia = gdf_africa[gdf_africa['COUNTRY'].str.upper() == 'ZAMBIA'].copy()

if gdf_zambia.empty:
    print("Error: Could not find Zambia in the dataset. Check column names.")
    exit()

# 3. FIND THE SMALLEST DISTRICT IN ZAMBIA
# We project to a metric system (EPSG:3857) to calculate area in km2 accurately
gdf_zambia['area_km2'] = gdf_zambia.to_crs(epsg=3857).area / 1e6
smallest_district = gdf_zambia.nsmallest(1, 'area_km2').iloc[0]
dist_name = smallest_district['ADM_NAME']
dist_geom = smallest_district.geometry

print(f"\nTargeting Smallest District for Validation: {dist_name}")
print(f"Total District Area: {smallest_district['area_km2']:.2f} km2")

# 4. SIMULATE THE 9KM ERA5 GRID (Approx 0.1 degree)
bounds = dist_geom.bounds
lon_min, lat_min, lon_max, lat_max = bounds

# Create the pixel grid covering the district
pixels = []
# 0.1 is the decimal degree equivalent for ~9-11km
grid_step = 0.1 
for lon in np.arange(lon_min - grid_step, lon_max + grid_step, grid_step):
    for lat in np.arange(lat_min - grid_step, lat_max + grid_step, grid_step):
        pixels.append(box(lon, lat, lon + grid_step, lat + grid_step))

gdf_grid = gpd.GeoDataFrame(geometry=pixels, crs=gdf_zambia.crs)
gdf_grid['pixel_id'] = [f"Pixel_{i}" for i in range(len(gdf_grid))]

# Assign a fake SSI value to each pixel to show the math clearly
# We will use a sequence so you can see which pixel contributes what
gdf_grid['ssi_value'] = np.linspace(-2.0, 1.0, len(gdf_grid))

# 5. CALCULATION A: THE CENTROID METHOD (Insurance Company Approach)
centroid = dist_geom.centroid
# Find which single pixel contains the center point
centroid_pixel = gdf_grid[gdf_grid.contains(centroid)]
centroid_val = centroid_pixel['ssi_value'].values[0]
centroid_id = centroid_pixel['pixel_id'].values[0]

# 6. CALCULATION B: AREA WEIGHTED INTERPOLATION (Your New Approach)
# Intersect the district with the grid
intersections = gpd.overlay(gdf_grid, gdf_zambia[gdf_zambia['ADM_NAME'] == dist_name], how='intersection')

# Calculate area of each intersection in km2
intersections['inter_area_km2'] = intersections.to_crs(epsg=3857).area / 1e6
total_district_area = intersections['inter_area_km2'].sum()

# Calculate weights and weighted contribution
intersections['weight'] = intersections['inter_area_km2'] / total_district_area
intersections['contribution'] = intersections['ssi_value'] * intersections['weight']

final_weighted_ssi = intersections['contribution'].sum()

# 7. OUTPUT RESULTS FOR SUPERVISOR
print("\n" + "="*50)
print(f" MANUAL VALIDATION TABLE: {dist_name}")
print("="*50)
print(intersections[['pixel_id', 'ssi_value', 'inter_area_km2', 'weight']].to_string(index=False))

print("\n" + "="*50)
print(" COMPARISON SUMMARY")
print("="*50)
print(f"Centroid Method (Pixel {centroid_id}): {centroid_val:.4f}")
print(f"Area-Weighted Method (Weighted Avg):  {final_weighted_ssi:.4f}")
print(f"Numerical Difference:                {abs(centroid_val - final_weighted_ssi):.4f}")
print("="*50)

if abs(centroid_val - final_weighted_ssi) < 0.0001:
    print("\nINSIGHT: The methods yield identical results for this district.")
    print("This explains why your overall correlation was 1.0.")
else:
    print(f"\nINSIGHT: Your new method adjusted the value by {abs(centroid_val - final_weighted_ssi):.4f}")
    print("This proves the Centroid method was 'ignoring' boundary information.")