# src/test_features.py

import xarray as xr
import numpy as np
import re
from datetime import datetime

def test_feature_stack(nc_path, expected_months=240, start="2003-01", end="2022-12"):
    ds = xr.open_dataset(nc_path)
    features = ds.features
    names = ds.feature.values.astype(str)

    print("\n✅ Feature Stack Summary")
    print(f"Total features: {len(names)}")
    print(f"Shape: {features.shape}")

    # Check for duplicates
    assert len(set(names)) == len(names), "❌ Duplicate feature names found!"

    # Check static features
    static_features = [n for n in names if re.match(r".*_(sand|clay|srtm).*", n)]
    print(f"Static features (DEM, soil): {len(static_features)}")

    # Check monthly features
    monthly_features = [n for n in names if re.match(r".*_\d{4}-\d{2}", n)]
    print(f"Monthly features: {len(monthly_features)}")

    # Count per dataset
    print("\n📊 Feature counts by dataset prefix:")
    prefix_counts = {}
    for n in names:
        prefix = n.split("_")[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    for k, v in prefix_counts.items():
        print(f"  - {k}: {v}")

    # Check for expected time coverage
    time_range = [
        datetime.strptime(start, "%Y-%m").strftime("%Y-%m")
    ]
    dt = datetime.strptime(start, "%Y-%m")
    while dt.strftime("%Y-%m") < end:
        dt = dt.replace(day=1)
        if dt.month == 12:
            dt = dt.replace(year=dt.year + 1, month=1)
        else:
            dt = dt.replace(month=dt.month + 1)
        time_range.append(dt.strftime("%Y-%m"))

    for prefix in prefix_counts:
        if prefix in ["openlandmap", "usgs"]:  # skip static
            continue
        count = sum(1 for n in names if n.startswith(prefix))
        if count < expected_months:
            print(f"⚠️ Missing months in {prefix}: only {count} of {expected_months}")

    # Check for NaN values
    nan_fraction = float(np.isnan(features).sum()) / features.size
    print(f"\n🧪 NaN fraction: {nan_fraction:.4f}")
    assert nan_fraction < 0.05, "❌ Too many NaNs!"

    print("\n✅ All checks passed!")

if __name__ == "__main__":
    test_feature_stack("data/processed/feature_stack.nc")
