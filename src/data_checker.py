import rasterio
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

summary_lines = []

# Base directory
base_dir = Path("data/raw")

# Dataset directories based on data_loader.py logic
datasets = {
    "grace": base_dir / "grace",
    "gldas": {
        "SoilMoi0_10cm_inst": base_dir / "gldas" / "SoilMoi0_10cm_inst",
        "SoilMoi10_40cm_inst": base_dir / "gldas" / "SoilMoi10_40cm_inst",
        "SoilMoi40_100cm_inst": base_dir / "gldas" / "SoilMoi40_100cm_inst",
        "SoilMoi100_200cm_inst": base_dir / "gldas" / "SoilMoi100_200cm_inst",
        "Evap_tavg": base_dir / "gldas" / "Evap_tavg",
        "SWE_inst": base_dir / "gldas" / "SWE_inst"
    },
    "chirps": base_dir / "chirps",
    "modis_land_cover": base_dir / "modis_land_cover",
    "terraclimate": {
        "tmmx": base_dir / "terraclimate" / "tmmx",
        "tmmn": base_dir / "terraclimate" / "tmmn",
        "pr": base_dir / "terraclimate" / "pr",
        "aet": base_dir / "terraclimate" / "aet",
        "def": base_dir / "terraclimate" / "def"
    },
    "usgs_dem": base_dir / "usgs_dem",
    "usgs_well_data": base_dir / "usgs_well_data",
    "openlandmap": base_dir / "openlandmap"
}

def analyze_rasters(folder, expected_minimum=100):
    rasters = list(folder.glob("*.tif"))
    summary = [f"  Total GeoTIFF files: {len(rasters)}"]
    if len(rasters) < expected_minimum:
        summary.append(f"  ⚠️ Expected at least {expected_minimum} files, found only {len(rasters)}")

    for i, f in enumerate(rasters[:5]):  # Only sample 5 for detailed stats
        try:
            with rasterio.open(f) as src:
                summary.append(f"    Sample {i+1}: {f.name}")
                summary.append(f"      Shape: {src.shape}")
                summary.append(f"      Number of bands (layers): {src.count}")
                summary.append(f"      Bounds: {src.bounds}")
                summary.append(f"      Resolution: {src.res}")
                summary.append(f"      CRS: {src.crs}")
        except Exception as e:
            summary.append(f"    Could not read {f.name}: {e}")
    return summary

def analyze_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        summary = [
            f"  File: {csv_path.name}",
            f"  Shape: {df.shape}",
            f"  Columns: {list(df.columns)}",
            f"  Date Range: {df.index.min()} to {df.index.max()}",
            f"  Missing values: {df.isnull().sum().sum()} total"
        ]
        return summary
    except Exception as e:
        return [f"  Could not read CSV file {csv_path.name}: {e}"]

# Start summary
summary_lines.append("==== DATA CHECK SUMMARY ====")
summary_lines.append(f"Generated on: {datetime.now()}")
summary_lines.append("")

# Loop through datasets
for name, path in datasets.items():
    summary_lines.append(f"--- {name.upper()} ---")
    if isinstance(path, dict):  # e.g., GLDAS or TERRACLIMATE with subfolders
        for subname, subpath in path.items():
            summary_lines.append(f" Subcomponent: {subname}")
            if not subpath.exists():
                summary_lines.append(f"  ❌ Folder not found: {subpath}")
                continue
            summary_lines.extend(analyze_rasters(subpath))
    else:
        if not path.exists():
            summary_lines.append("  ❌ Folder not found.")
            continue
        if name == "usgs_well_data":
            csvs = list(path.glob("*.csv"))
            if csvs:
                for csv in csvs:
                    summary_lines.extend(analyze_csv(csv))
            else:
                summary_lines.append("  ❌ No CSV files found.")
        else:
            summary_lines.extend(analyze_rasters(path))

    summary_lines.append("")

# Write to summary.txt
summary_file = Path("data_checker_summary.txt")
with open(summary_file, "w") as f:
    f.write("\n".join(summary_lines))

summary_file.name
