#!/usr/bin/env python3
"""
Debug shapefile bounds issue
"""

import geopandas as gpd
import numpy as np

# Load the shapefile
shapefile_path = "data/shapefiles/processed/mississippi_river_basin.shp"
print(f"Loading shapefile: {shapefile_path}")

# Load it
gdf = gpd.read_file(shapefile_path)
print(f"\nOriginal shapefile info:")
print(f"  Shape: {gdf.shape}")
print(f"  CRS: {gdf.crs}")
print(f"  Bounds: {gdf.total_bounds}")
print(f"  Geometry type: {type(gdf.geometry.iloc[0])}")
print(f"  Is valid: {gdf.is_valid.all()}")

# Try converting to EPSG:4326
print(f"\nConverting to EPSG:4326...")
gdf_4326 = gdf.to_crs('EPSG:4326')
print(f"  Converted CRS: {gdf_4326.crs}")
print(f"  Converted bounds: {gdf_4326.total_bounds}")
print(f"  Bounds are finite: {np.all(np.isfinite(gdf_4326.total_bounds))}")

# Check geometry after conversion
print(f"\nGeometry check after conversion:")
print(f"  Is valid: {gdf_4326.is_valid.all()}")
print(f"  Is empty: {gdf_4326.is_empty.all()}")

# Try to access bounds differently
print(f"\nAlternative bounds calculation:")
try:
    # Get bounds from geometry
    geom = gdf_4326.geometry.iloc[0]
    print(f"  Geometry bounds: {geom.bounds}")
    
    # Get bounds from unary_union
    union = gdf_4326.unary_union
    print(f"  Union bounds: {union.bounds}")
except Exception as e:
    print(f"  Error: {e}")

# Check if it's already in WGS84 but with different naming
print(f"\nCRS comparison:")
print(f"  Original CRS EPSG code: {gdf.crs.to_epsg()}")
print(f"  Is geographic: {gdf.crs.is_geographic}")
print(f"  CRS name: {gdf.crs.name}")

# Try different approach
print(f"\nTrying explicit bounds calculation:")
try:
    minx = gdf_4326.geometry.bounds.minx.min()
    miny = gdf_4326.geometry.bounds.miny.min()
    maxx = gdf_4326.geometry.bounds.maxx.max()
    maxy = gdf_4326.geometry.bounds.maxy.max()
    print(f"  Calculated bounds: [{minx}, {miny}, {maxx}, {maxy}]")
except Exception as e:
    print(f"  Error: {e}") 