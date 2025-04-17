import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def check_netcdf(filepath, save_report=True, plot_samples=True, verbose=True):
    """
    Comprehensive check of a NetCDF file for data quality and structure.
    
    Parameters:
    -----------
    filepath : str
        Path to the NetCDF file
    save_report : bool
        Whether to save a report to disk
    plot_samples : bool
        Whether to plot sample data for visual inspection
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    dict
        A dictionary containing the check results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if verbose:
        print(f"🔍 Checking NetCDF file: {filepath}")
    
    # Open the dataset
    ds = xr.open_dataset(filepath)
    
    # Initialize results dictionary
    results = {
        "filename": filepath,
        "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "checks": {},
        "stats": {},
        "issues": []
    }
    
    # Basic structure check
    results["structure"] = {
        "data_vars": list(ds.data_vars),
        "coords": list(ds.coords),
        "dims": dict(ds.dims)
    }
    
    if verbose:
        print(f"\n📊 Dataset structure:")
        print(f"  Variables: {', '.join(results['structure']['data_vars'])}")
        print(f"  Coordinates: {', '.join(results['structure']['coords'])}")
        print(f"  Dimensions: {results['structure']['dims']}")
    
    # Check for temporal features
    if "features" in ds.data_vars:
        temporal_data = ds["features"]
        results["checks"]["has_temporal"] = True
        results["stats"]["temporal"] = {
            "shape": temporal_data.shape,
            "time_periods": len(ds.time),
            "features": len(ds.feature) if "feature" in ds.dims else None
        }
        
        # Check for NaN/missing values in temporal data
        nan_count = np.isnan(temporal_data.values).sum()
        nan_percentage = (nan_count / temporal_data.size) * 100
        results["stats"]["temporal"]["nan_count"] = nan_count
        results["stats"]["temporal"]["nan_percentage"] = nan_percentage
        
        if nan_percentage > 20:
            results["issues"].append(f"High percentage of NaN values in temporal data: {nan_percentage:.2f}%")
        
        if verbose:
            print(f"\n⏱️ Temporal features:")
            print(f"  Shape: {temporal_data.shape}")
            print(f"  Time periods: {len(ds.time)}")
            print(f"  Features: {len(ds.feature) if 'feature' in ds.dims else 'N/A'}")
            print(f"  NaN values: {nan_count} ({nan_percentage:.2f}%)")
            
            # Print basic statistics for each feature if not too many
            if "feature" in ds.dims and len(ds.feature) <= 20:
                print("\n📈 Feature statistics:")
                for i in range(len(ds.feature)):
                    feature_data = temporal_data.sel(feature=i)
                    non_nan = feature_data.values[~np.isnan(feature_data.values)]
                    if len(non_nan) > 0:
                        print(f"  Feature {i}: min={non_nan.min():.4f}, max={non_nan.max():.4f}, mean={non_nan.mean():.4f}")
                    else:
                        print(f"  Feature {i}: ALL NaN VALUES")
                        results["issues"].append(f"Feature {i} contains all NaN values")
    else:
        results["checks"]["has_temporal"] = False
        results["issues"].append("No temporal features found in the dataset")
    
    # Check for static features
    if "static_features" in ds.data_vars:
        static_data = ds["static_features"]
        results["checks"]["has_static"] = True
        results["stats"]["static"] = {
            "shape": static_data.shape,
            "features": len(ds.static_feature) if "static_feature" in ds.dims else None
        }
        
        # Check for NaN/missing values in static data
        nan_count = np.isnan(static_data.values).sum()
        nan_percentage = (nan_count / static_data.size) * 100
        results["stats"]["static"]["nan_count"] = nan_count
        results["stats"]["static"]["nan_percentage"] = nan_percentage
        
        if nan_percentage > 20:
            results["issues"].append(f"High percentage of NaN values in static data: {nan_percentage:.2f}%")
        
        if verbose:
            print(f"\n🗺️ Static features:")
            print(f"  Shape: {static_data.shape}")
            print(f"  Features: {len(ds.static_feature) if 'static_feature' in ds.dims else 'N/A'}")
            print(f"  NaN values: {nan_count} ({nan_percentage:.2f}%)")
            
            # Print basic statistics for each static feature if not too many
            if "static_feature" in ds.dims and len(ds.static_feature) <= 20:
                print("\n📊 Static feature statistics:")
                for idx in range(len(ds.static_feature)):
                    name = str(ds.static_feature.values[idx])
                    feature_data = static_data.isel(static_feature=idx)  # Use isel instead of sel
                    non_nan = feature_data.values[~np.isnan(feature_data.values)]
                    if len(non_nan) > 0:
                        print(f"  {name}: min={non_nan.min():.4f}, max={non_nan.max():.4f}, mean={non_nan.mean():.4f}")
                    else:
                        print(f"  {name}: ALL NaN VALUES")
                        results["issues"].append(f"Static feature {name} contains all NaN values")
    else:
        results["checks"]["has_static"] = False
        
    # Check spatial dimensions and coverage
    if "lat" in ds.dims and "lon" in ds.dims:
        results["checks"]["has_spatial"] = True
        results["stats"]["spatial"] = {
            "lat_range": [float(ds.lat.min().values), float(ds.lat.max().values)],
            "lon_range": [float(ds.lon.min().values), float(ds.lon.max().values)],
            "lat_resolution": float(abs(ds.lat[1] - ds.lat[0]).values) if len(ds.lat) > 1 else None,
            "lon_resolution": float(abs(ds.lon[1] - ds.lon[0]).values) if len(ds.lon) > 1 else None,
            "grid_size": [len(ds.lat), len(ds.lon)]
        }
        
        if verbose:
            print(f"\n🌐 Spatial coverage:")
            print(f"  Latitude range: {results['stats']['spatial']['lat_range'][0]:.4f} to {results['stats']['spatial']['lat_range'][1]:.4f}")
            print(f"  Longitude range: {results['stats']['spatial']['lon_range'][0]:.4f} to {results['stats']['spatial']['lon_range'][1]:.4f}")
            print(f"  Resolution: {results['stats']['spatial']['lat_resolution']:.4f}° (lat), {results['stats']['spatial']['lon_resolution']:.4f}° (lon)")
            print(f"  Grid size: {results['stats']['spatial']['grid_size'][0]} × {results['stats']['spatial']['grid_size'][1]}")
    else:
        results["checks"]["has_spatial"] = False
        results["issues"].append("No spatial dimensions (lat, lon) found in the dataset")
    
    # Check for time coordinate and its properties
    if "time" in ds.dims:
        time_data = ds.time.values
        results["checks"]["has_time"] = True
        results["stats"]["time"] = {
            "start": str(time_data[0]),
            "end": str(time_data[-1]),
            "periods": len(time_data)
        }
        
        # Check time continuity
        if len(time_data) > 1:
            try:
                time_diff = np.diff([pd.to_datetime(t) for t in time_data])
                unique_diffs = np.unique(time_diff)
                if len(unique_diffs) > 1:
                    results["issues"].append("Irregular time intervals detected")
                    results["stats"]["time"]["regular_intervals"] = False
                else:
                    results["stats"]["time"]["regular_intervals"] = True
                    results["stats"]["time"]["interval"] = str(unique_diffs[0])
            except:
                results["issues"].append("Could not analyze time intervals")
        
        if verbose:
            print(f"\n📅 Time information:")
            print(f"  Range: {results['stats']['time']['start']} to {results['stats']['time']['end']}")
            print(f"  Number of periods: {results['stats']['time']['periods']}")
            if results["stats"]["time"].get("regular_intervals") is not None:
                print(f"  Regular intervals: {results['stats']['time']['regular_intervals']}")
                if results["stats"]["time"]["regular_intervals"]:
                    print(f"  Interval: {results['stats']['time']['interval']}")
    else:
        results["checks"]["has_time"] = False
        if "features" in ds.data_vars:
            results["issues"].append("No time dimension found despite having temporal features")
    
    # Data size check
    memory_usage = ds.nbytes / (1024 * 1024)  # MB
    results["stats"]["memory_usage_mb"] = memory_usage
    
    if memory_usage > 10000:  # 10 GB
        results["issues"].append(f"Very large dataset ({memory_usage:.2f} MB), may cause memory issues")
    
    if verbose:
        print(f"\n💾 Memory usage:")
        print(f"  Dataset size: {memory_usage:.2f} MB")
    
    # Feature alignment check
    # Check if features have consistent shapes across dimensions
    for var_name in ds.data_vars:
        var = ds[var_name]
        if len(var.dims) >= 2:  # At least 2D
            shape_issues = []
            for dim in var.dims:
                if dim in ['lat', 'lon', 'time'] and len(ds[dim]) < 2:
                    shape_issues.append(f"Dimension {dim} has only {len(ds[dim])} elements")
            
            if shape_issues:
                for issue in shape_issues:
                    results["issues"].append(f"Variable {var_name}: {issue}")
    
    # Check for geographic bounds sanity
    if "lat" in ds.dims and "lon" in ds.dims:
        lat_range = results["stats"]["spatial"]["lat_range"]
        lon_range = results["stats"]["spatial"]["lon_range"]
        
        if lat_range[0] < -90 or lat_range[1] > 90:
            results["issues"].append(f"Invalid latitude range: {lat_range}")
        
        if lon_range[0] < -180 or lon_range[1] > 180:
            results["issues"].append(f"Invalid longitude range: {lon_range}")
    
    # Plotting for visual inspection
    if plot_samples:
        out_dir = os.path.dirname(filepath)
        plot_dir = os.path.join(out_dir, "nc_check_plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        if results["checks"].get("has_temporal", False):
            # Plot first time period of first feature
            plt.figure(figsize=(10, 8))
            if "feature" in ds.dims:
                for feature_idx in range(min(3, len(ds.feature))):  # Plot first 3 features max
                    plt.figure(figsize=(10, 8))
                    ds.features.isel(time=0, feature=feature_idx).plot()
                    plt.title(f"Temporal Feature {feature_idx} (First Time Period)")
                    plt.savefig(os.path.join(plot_dir, f"temporal_feature_{feature_idx}.png"))
                    plt.close()
            
            # Plot time series for a random point
            if "lat" in ds.dims and "lon" in ds.dims:
                mid_lat_idx = len(ds.lat) // 2
                mid_lon_idx = len(ds.lon) // 2
                
                # Find a non-NaN point if possible
                if "feature" in ds.dims:
                    found_valid = False
                    for lat_idx in range(len(ds.lat)):
                        for lon_idx in range(len(ds.lon)):
                            point_data = ds.features.isel(lat=lat_idx, lon=lon_idx, feature=0)
                            if not np.isnan(point_data.values).all():
                                mid_lat_idx = lat_idx
                                mid_lon_idx = lon_idx
                                found_valid = True
                                break
                        if found_valid:
                            break
                
                # Plot time series for a single point and first 3 features
                if "feature" in ds.dims:
                    plt.figure(figsize=(12, 6))
                    for feature_idx in range(min(3, len(ds.feature))):
                        point_data = ds.features.isel(lat=mid_lat_idx, lon=mid_lon_idx, feature=feature_idx)
                        plt.plot(ds.time, point_data, label=f"Feature {feature_idx}")
                    
                    plt.title(f"Time Series at Point (lat={ds.lat[mid_lat_idx].values:.4f}, lon={ds.lon[mid_lon_idx].values:.4f})")
                    plt.legend()
                    plt.savefig(os.path.join(plot_dir, "time_series.png"))
                    plt.close()
        
        if results["checks"].get("has_static", False):
            # Plot first few static features
            if "static_feature" in ds.dims:
                for idx in range(min(3, len(ds.static_feature))):  # Plot first 3 static features max
                    name = str(ds.static_feature.values[idx])
                    plt.figure(figsize=(10, 8))
                    ds.static_features.isel(static_feature=idx).plot()
                    plt.title(f"Static Feature: {name}")
                    # Sanitize filename by removing special characters
                    safe_name = "".join(c if c.isalnum() else '_' for c in name)
                    plt.savefig(os.path.join(plot_dir, f"static_feature_{safe_name}.png"))
                    plt.close()
        
        if verbose:
            print(f"\n🖼️ Plots saved to: {plot_dir}")
    
    # Overall assessment
    if len(results["issues"]) == 0:
        results["overall_status"] = "PASS"
        if verbose:
            print("\n✅ OVERALL ASSESSMENT: NetCDF file appears to be valid and well-structured.")
    else:
        results["overall_status"] = "WARNING" if any("High percentage" in issue for issue in results["issues"]) else "PASS_WITH_ISSUES"
        if verbose:
            print("\n⚠️ OVERALL ASSESSMENT: Issues found that may need attention:")
            for issue in results["issues"]:
                print(f"  - {issue}")
    
    # Save report to disk
    if save_report:
        import json
        report_path = os.path.join(os.path.dirname(filepath), "nc_check_report.json")
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.datetime64):
                return str(obj)
            return obj
        
        # Apply conversion to the entire results dictionary
        results_dict = convert_numpy_types(results)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            if verbose:
                print(f"\n📝 Report saved to: {report_path}")
        except TypeError as e:
            print(f"\n⚠️ Error saving JSON report: {e}")
            print("The check completed successfully, but the report could not be saved.")
            # Try to save a simplified version
            try:
                simplified_results = {
                    "filename": results["filename"],
                    "overall_status": results["overall_status"],
                    "issues": results["issues"]
                }
                with open(report_path, 'w') as f:
                    json.dump(convert_numpy_types(simplified_results), f, indent=2)
                if verbose:
                    print(f"Saved simplified report instead to: {report_path}")
            except:
                print("Could not save even a simplified report.")
    
    ds.close()
    return results

def main():
    """
    Main function to check NetCDF files.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Check NetCDF files for quality and structure")
    parser.add_argument("filepath", help="Path to the NetCDF file to check")
    parser.add_argument("--no-report", action="store_true", help="Don't save a report file")
    parser.add_argument("--no-plots", action="store_true", help="Don't generate plots")
    parser.add_argument("--quiet", action="store_true", help="Minimize printed output")
    
    args = parser.parse_args()
    
    check_netcdf(
        args.filepath, 
        save_report=not args.no_report, 
        plot_samples=not args.no_plots,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()