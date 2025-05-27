#!/usr/bin/env python3
"""
Quick script to inspect the downloaded well data files and find the mismatch
"""
import pandas as pd
import os

print("🔍 INSPECTING DOWNLOADED WELL DATA FILES")
print("=" * 50)

# Check if files exist
data_file = "data/raw/usgs_well_data/monthly_groundwater_anomalies.csv"
metadata_file = "data/raw/usgs_well_data/well_metadata.csv"

print(f"📁 Checking file existence:")
print(f"   Data file: {os.path.exists(data_file)} ({data_file})")
print(f"   Metadata file: {os.path.exists(metadata_file)} ({metadata_file})")

if not os.path.exists(data_file):
    print("❌ Data file not found!")
    exit(1)

if not os.path.exists(metadata_file):
    print("❌ Metadata file not found!")
    exit(1)

# Load and inspect the data file
print(f"\n📊 INSPECTING DATA FILE: {data_file}")
try:
    data_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"   ✅ Successfully loaded")
    print(f"   📏 Shape: {data_df.shape} (rows × columns)")
    print(f"   📅 Index (dates): {data_df.index[0]} to {data_df.index[-1]}")
    print(f"   🔢 Number of wells (columns): {len(data_df.columns)}")
    
    # Show column names/well IDs
    print(f"\n   📋 COLUMN NAMES (Well IDs in data):")
    for i, col in enumerate(data_df.columns):
        print(f"      {i+1:2d}: '{col}' (type: {type(col).__name__})")
        if i >= 9:  # Show first 10 only
            print(f"      ... and {len(data_df.columns)-10} more")
            break
    
    # Check for data content
    print(f"\n   📈 Data content check:")
    non_null_counts = data_df.count()
    print(f"      Total values: {data_df.size}")
    print(f"      Non-null values: {non_null_counts.sum()}")
    print(f"      Data coverage: {non_null_counts.sum()/data_df.size*100:.1f}%")
    
    # Show sample data
    print(f"\n   🔬 Sample data (first well, first 5 dates):")
    first_well = data_df.columns[0]
    sample_data = data_df[first_well].head()
    for date, value in sample_data.items():
        print(f"      {date}: {value}")

except Exception as e:
    print(f"   ❌ Error loading data file: {e}")

# Load and inspect the metadata file
print(f"\n📊 INSPECTING METADATA FILE: {metadata_file}")
try:
    metadata_df = pd.read_csv(metadata_file)
    print(f"   ✅ Successfully loaded")
    print(f"   📏 Shape: {metadata_df.shape} (rows × columns)")
    print(f"   📋 Columns: {list(metadata_df.columns)}")
    
    # Show well IDs from metadata
    print(f"\n   📋 WELL IDs FROM METADATA:")
    for i, well_id in enumerate(metadata_df['well_id']):
        print(f"      {i+1:2d}: '{well_id}' (type: {type(well_id).__name__})")
        if i >= 9:  # Show first 10 only
            print(f"      ... and {len(metadata_df)-10} more")
            break
    
    # Show sample metadata
    print(f"\n   🔬 Sample metadata (first well):")
    first_row = metadata_df.iloc[0]
    for col, value in first_row.items():
        print(f"      {col}: {value}")

except Exception as e:
    print(f"   ❌ Error loading metadata file: {e}")

# CRITICAL: Check for exact matches
print(f"\n🔍 CRITICAL MISMATCH CHECK:")
try:
    data_columns = set(str(col) for col in data_df.columns)
    metadata_wells = set(str(well_id) for well_id in metadata_df['well_id'])
    
    print(f"   Data columns (as strings): {len(data_columns)} wells")
    print(f"   Metadata well IDs (as strings): {len(metadata_wells)} wells")
    
    # Find matches
    matches = data_columns.intersection(metadata_wells)
    only_in_data = data_columns - metadata_wells
    only_in_metadata = metadata_wells - data_columns
    
    print(f"\n   ✅ MATCHES: {len(matches)} wells")
    if matches:
        print(f"      First few matches: {list(matches)[:5]}")
    
    if only_in_data:
        print(f"\n   📊 ONLY IN DATA (not in metadata): {len(only_in_data)} wells")
        print(f"      First few: {list(only_in_data)[:5]}")
    
    if only_in_metadata:
        print(f"\n   📋 ONLY IN METADATA (not in data): {len(only_in_metadata)} wells")
        print(f"      First few: {list(only_in_metadata)[:5]}")
    
    if len(matches) == 0:
        print(f"\n   ❌ NO MATCHES FOUND! This is the problem!")
        print(f"   🔧 Sample comparison:")
        print(f"      Data column sample: '{list(data_columns)[0]}'")
        print(f"      Metadata ID sample: '{list(metadata_wells)[0]}'")
        print(f"      Are they equal? {list(data_columns)[0] == list(metadata_wells)[0]}")
    
except Exception as e:
    print(f"   ❌ Error during comparison: {e}")

print(f"\n✅ INSPECTION COMPLETE!")