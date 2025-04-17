from datetime import datetime
import os

def generate_index_based_timestamp_map(start_date="2003-01", num_months=240):
    """Generate a mapping: '0.tif' -> '2003-01', ..., '239.tif' -> '2022-12'"""
    start = datetime.strptime(start_date, "%Y-%m")
    timestamp_map = {
        f"{i}.tif": (start.replace(day=1) if i == 0 else (start.replace(day=1) + pd.DateOffset(months=i))).strftime("%Y-%m")
        for i in range(num_months)
    }
    return timestamp_map

def validate_file_timestamps(folder_path, expected_count=240, start_date="2003-01"):
    timestamp_map = generate_index_based_timestamp_map(start_date=start_date, num_months=expected_count)
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".tif") and f.split('.')[0].isdigit())
    
    missing = [f"{i}.tif" for i in range(expected_count) if f"{i}.tif" not in files]
    extra = [f for f in files if f not in timestamp_map]
    aligned = [f for f in files if f in timestamp_map]

    print(f"\n📂 Checking folder: {folder_path}")
    print(f"✅ Total expected: {expected_count}")
    print(f"✅ Found: {len(files)}")
    print(f"✅ Aligned with timestamp map: {len(aligned)}")
    print(f"❌ Missing files: {len(missing)}")
    if missing:
        print(f"   ⤷ Sample missing: {missing[:5]}")
    if extra:
        print(f"❌ Extra/unmapped files found: {extra[:5]}")

    return {
        "aligned_files": aligned,
        "missing_files": missing,
        "extra_files": extra,
        "timestamp_map": timestamp_map
    }
