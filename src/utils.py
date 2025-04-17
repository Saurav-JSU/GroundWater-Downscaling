# src/utils.py

import ee
import os
import re
import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd

def create_date_list(start="2003-01", end="2022-12"):
    from datetime import datetime, timedelta
    dates = []
    current = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")
    while current <= end_date:
        dates.append(current.strftime("%Y-%m"))
        current += timedelta(days=32)
        current = current.replace(day=1)
    return dates

def bbox_to_geometry(region):
    return ee.Geometry.Rectangle([
        region["lon_min"],
        region["lat_min"],
        region["lon_max"],
        region["lat_max"]
    ])

def reproject_match(src, match):
    """Reproject src to match CRS of reference raster"""
    return src.rio.reproject_match(match)

def resample_match(src, match):
    """Resample src to match resolution and shape of reference raster"""
    return src.rio.reproject_match(match)  # resampling handled within reproject_match

def match_resolution(src, match):
    """Alias: Match CRS, resolution, and alignment"""
    return reproject_match(src, match)

# 🆕 NEW: Parse valid GRACE months from filename ranges
def parse_grace_months(grace_dir):
    months = set()
    pattern = re.compile(r"(\d{8})_(\d{8})\.tif$")
    for fname in os.listdir(grace_dir):
        match = pattern.match(fname)
        if match:
            start = pd.to_datetime(match.group(1), format="%Y%m%d")
            end = pd.to_datetime(match.group(2), format="%Y%m%d")
            # Include months between start and end (but pick mid-month)
            mid = start + (end - start) / 2
            months.add(mid.strftime("%Y-%m"))
    return sorted(months)

# 🆕 NEW: Load grace-based timestamp map for masking/filtering
def load_timestamp_map(grace_dir):
    valid_months = parse_grace_months(grace_dir)
    return set(valid_months)
