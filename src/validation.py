# src/validation.py - DEBUG VERSION
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import yaml
from tqdm import tqdm

def validate_with_wells(config_path="src/config.yaml"):
    """Validate derived groundwater against USGS well observations - DEBUG VERSION"""
    # Create output directories
    results_dir = Path("results")
    figures_dir = Path("figures/validation")
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Load groundwater predictions
    print("Loading groundwater storage predictions...")
    gws_path = "results/groundwater_storage_anomalies.nc"
    if not os.path.exists(gws_path):
        print(f"❌ Groundwater data file not found: {gws_path}")
        return pd.DataFrame()
    
    gws_ds = xr.open_dataset(gws_path)
    print(f"✅ Loaded groundwater data with {len(gws_ds.time)} time steps")
    print(f"   Spatial grid: {len(gws_ds.lat)} × {len(gws_ds.lon)}")
    print(f"   Lat range: {gws_ds.lat.min().values:.2f} to {gws_ds.lat.max().values:.2f}")
    print(f"   Lon range: {gws_ds.lon.min().values:.2f} to {gws_ds.lon.max().values:.2f}")
    print(f"   Time range: {gws_ds.time.values[0]} to {gws_ds.time.values[-1]}")
    
    # Load well data
    print("\nLoading well data...")
    well_data_path = "data/raw/usgs_well_data/monthly_groundwater_anomalies.csv"
    well_data = pd.read_csv(well_data_path, index_col=0, parse_dates=True)
    print(f"✅ Loaded well data with {len(well_data.columns)} wells")
    print(f"   Time range: {well_data.index[0]} to {well_data.index[-1]}")
    print(f"   Sample well IDs: {list(well_data.columns[:3])}")
    
    # Load well coordinates
    well_locations_path = "data/raw/usgs_well_data/well_metadata.csv"
    well_locations = pd.read_csv(well_locations_path)
    print(f"✅ Have coordinates for {len(well_locations)} wells")
    
    # Debug: Check coordinate ranges
    print(f"   Well lat range: {well_locations['lat'].min():.2f} to {well_locations['lat'].max():.2f}")
    print(f"   Well lon range: {well_locations['lon'].min():.2f} to {well_locations['lon'].max():.2f}")
    
    # Debug: Check time alignment
    gws_times = pd.DatetimeIndex(gws_ds.time.values)
    well_times = well_data.index
    common_times = gws_times.intersection(well_times)
    print(f"\n🔍 Time alignment check:")
    print(f"   GWS time points: {len(gws_times)}")
    print(f"   Well time points: {len(well_times)}")
    print(f"   Common time points: {len(common_times)}")
    if len(common_times) > 0:
        print(f"   First common: {common_times[0]}")
        print(f"   Last common: {common_times[-1]}")
    else:
        print("   ❌ NO COMMON TIME POINTS!")
        print(f"   GWS sample times: {gws_times[:3].tolist()}")
        print(f"   Well sample times: {well_times[:3].tolist()}")
    
    # Debug each well individually
    results = []
    debug_info = []
    
    print(f"\n🔍 Processing wells individually (showing details for first 5):")
    for idx, well in tqdm(well_locations.iterrows(), total=len(well_locations)):
        well_id = str(well['well_id'])  # Convert to string to match CSV columns
        show_details = idx < 5  # Show details for first 5 wells
        
        debug_entry = {'well_id': well_id, 'step_failed': 'unknown'}
        
        if show_details: print(f"\n  Well {idx+1}: {well_id}")
        
        # Check if well is in data
        if well_id not in well_data.columns:
            debug_entry['step_failed'] = 'not_in_data'
            if show_details: print(f"    ❌ Well ID not found in data")
            debug_info.append(debug_entry)
            continue
        
        # Check coordinates are within grid bounds
        if (well['lat'] < gws_ds.lat.min() or well['lat'] > gws_ds.lat.max() or
            well['lon'] < gws_ds.lon.min() or well['lon'] > gws_ds.lon.max()):
            debug_entry['step_failed'] = 'outside_grid'
            if show_details: print(f"    ❌ Outside grid bounds: ({well['lat']:.2f}, {well['lon']:.2f})")
            debug_info.append(debug_entry)
            continue
        
        # Try to extract GWS at well location
        try:
            gws_at_well = gws_ds.groundwater.sel(
                lat=well['lat'], lon=well['lon'], method='nearest'
            )
            if show_details: print(f"    ✅ Extracted GWS at location")
            
            # Convert to pandas Series
            pred_series = pd.Series(
                gws_at_well.values, 
                index=pd.DatetimeIndex(gws_ds.time.values)
            )
            if show_details: print(f"    ✅ Created prediction series: {len(pred_series)} points")
            
        except Exception as e:
            debug_entry['step_failed'] = 'extraction_error'
            debug_entry['error'] = str(e)
            if show_details: print(f"    ❌ Extraction error: {e}")
            debug_info.append(debug_entry)
            continue
        
        # Get observed data
        observed = well_data[well_id].copy()
        if show_details: print(f"    ✅ Got observed data: {len(observed)} points")
        
        # Find common dates
        common_dates = pred_series.index.intersection(observed.index)
        if show_details: print(f"    🔍 Common dates: {len(common_dates)}")
        
        if len(common_dates) < 12:  # Need at least 1 year
            debug_entry['step_failed'] = 'insufficient_overlap'
            debug_entry['common_dates'] = len(common_dates)
            if show_details: print(f"    ❌ Only {len(common_dates)} common dates, need ≥12")
            debug_info.append(debug_entry)
            continue
            
        # Align data
        obs_aligned = observed.loc[common_dates]
        pred_aligned = pred_series.loc[common_dates]
        
        # Remove NaN values
        valid_mask = ~(obs_aligned.isna() | pred_aligned.isna())
        valid_count = valid_mask.sum()
        if show_details: print(f"    🔍 Valid (non-NaN) pairs: {valid_count}")
        
        if valid_count < 6:  # Need at least 6 valid pairs
            debug_entry['step_failed'] = 'insufficient_valid_data'
            debug_entry['valid_pairs'] = valid_count
            if show_details: print(f"    ❌ Only {valid_count} valid pairs, need ≥6")
            debug_info.append(debug_entry)
            continue
            
        obs_valid = obs_aligned[valid_mask]
        pred_valid = pred_aligned[valid_mask]
        
        # Calculate metrics
        try:
            correlation, p_value = pearsonr(pred_valid, obs_valid)
            rmse = np.sqrt(mean_squared_error(obs_valid, pred_valid))
            nse = 1 - (np.sum((obs_valid - pred_valid)**2) / 
                      np.sum((obs_valid - obs_valid.mean())**2))
            
            results.append({
                'well_id': well_id,
                'lat': well['lat'],
                'lon': well['lon'],
                'correlation': correlation,
                'p_value': p_value,
                'rmse': rmse,
                'nse': nse,
                'n_observations': len(obs_valid),
                'n_common_dates': len(common_dates)
            })
            
            debug_entry['step_failed'] = 'success'
            debug_entry['correlation'] = correlation
            if show_details: print(f"    ✅ SUCCESS! Correlation: {correlation:.3f}, RMSE: {rmse:.2f}")
            
        except Exception as e:
            debug_entry['step_failed'] = 'metrics_calculation'
            debug_entry['error'] = str(e)
            if show_details: print(f"    ❌ Metrics calculation error: {e}")
        
        debug_info.append(debug_entry)
    
    # Print debug summary
    debug_df = pd.DataFrame(debug_info)
    print(f"\n📊 DEBUG SUMMARY:")
    failure_counts = debug_df['step_failed'].value_counts()
    for step, count in failure_counts.items():
        print(f"   {step}: {count} wells")
    
    if len(results) == 0:
        print(f"\n❌ No wells passed validation. Most common failure: {failure_counts.index[0]}")
        
        # Save debug info for analysis
        debug_df.to_csv("results/validation_debug.csv", index=False)
        print(f"   Saved debug info to: results/validation_debug.csv")
        return pd.DataFrame()
    
    # Success! Process results
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("results/well_validation_metrics.csv", index=False)
    
    print(f"\n✅ Validation successful: {len(results)} wells")
    print(f"   Average correlation: {metrics_df['correlation'].mean():.3f}")
    print(f"   Average RMSE: {metrics_df['rmse'].mean():.2f} cm")
    print(f"   Average NSE: {metrics_df['nse'].mean():.3f}")
    
    return metrics_df

if __name__ == "__main__":
    validate_with_wells()