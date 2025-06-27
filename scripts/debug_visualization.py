#!/usr/bin/env python3
"""
Debug visualization issues - check data and create simple plots
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import os

def check_data():
    """Check the groundwater data."""
    print("🔍 CHECKING GROUNDWATER DATA")
    print("="*50)
    
    # Load groundwater data
    gws_file = "results/groundwater_storage_anomalies.nc"
    if not os.path.exists(gws_file):
        print(f"❌ File not found: {gws_file}")
        return None
    
    ds = xr.open_dataset(gws_file)
    print(f"✅ Loaded dataset: {gws_file}")
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Variables: {list(ds.data_vars)}")
    
    # Check groundwater variable
    if 'groundwater' in ds:
        gws = ds.groundwater
        print(f"\n📊 Groundwater variable:")
        print(f"   Shape: {gws.shape}")
        print(f"   Min value: {float(gws.min())}")
        print(f"   Max value: {float(gws.max())}")
        print(f"   Mean value: {float(gws.mean())}")
        print(f"   NaN count: {np.sum(np.isnan(gws.values))}")
        print(f"   Total values: {gws.size}")
        print(f"   Valid values: {np.sum(~np.isnan(gws.values))}")
        
        # Check spatial extent
        print(f"\n🌍 Spatial extent:")
        print(f"   Lat range: {float(gws.lat.min())} to {float(gws.lat.max())}")
        print(f"   Lon range: {float(gws.lon.min())} to {float(gws.lon.max())}")
        
        # Check time extent
        print(f"\n⏰ Time extent:")
        print(f"   Time range: {str(gws.time.values[0])[:10]} to {str(gws.time.values[-1])[:10]}")
        print(f"   Number of time steps: {len(gws.time)}")
        
    return ds

def check_shapefile():
    """Check the shapefile."""
    print("\n🗺️ CHECKING SHAPEFILE")
    print("="*50)
    
    shapefile_path = "data/shapefiles/processed/mississippi_river_basin.shp"
    if not os.path.exists(shapefile_path):
        print(f"❌ File not found: {shapefile_path}")
        return None
    
    gdf = gpd.read_file(shapefile_path)
    print(f"✅ Loaded shapefile: {shapefile_path}")
    print(f"   Shape: {gdf.shape}")
    print(f"   CRS: {gdf.crs}")
    print(f"   Bounds: {gdf.total_bounds}")
    print(f"   Columns: {list(gdf.columns)}")
    
    return gdf

def create_simple_test_plots(ds, gdf):
    """Create simple test plots to debug visualization issues."""
    print("\n🎨 CREATING SIMPLE TEST PLOTS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test 1: Simple data plot without projection
    ax1 = axes[0, 0]
    if 'groundwater' in ds:
        # Take mean over time and plot
        mean_gws = ds.groundwater.mean(dim='time')
        
        # Check if data has values
        valid_data = ~np.isnan(mean_gws.values)
        print(f"   Valid data points: {np.sum(valid_data)}")
        
        if np.sum(valid_data) > 0:
            im1 = ax1.imshow(mean_gws.values, aspect='auto', cmap='RdBu_r')
            plt.colorbar(im1, ax=ax1)
            ax1.set_title('Mean GWS (imshow)')
        else:
            ax1.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Mean GWS (NO DATA)')
    
    # Test 2: Pcolormesh plot
    ax2 = axes[0, 1]
    if 'groundwater' in ds:
        mean_gws = ds.groundwater.mean(dim='time')
        if np.sum(~np.isnan(mean_gws.values)) > 0:
            im2 = ax2.pcolormesh(ds.lon, ds.lat, mean_gws, cmap='RdBu_r')
            plt.colorbar(im2, ax=ax2)
            ax2.set_title('Mean GWS (pcolormesh)')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
        else:
            ax2.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Mean GWS (NO DATA)')
    
    # Test 3: Single time slice
    ax3 = axes[1, 0]
    if 'groundwater' in ds:
        first_time = ds.groundwater.isel(time=0)
        if np.sum(~np.isnan(first_time.values)) > 0:
            im3 = ax3.pcolormesh(ds.lon, ds.lat, first_time, cmap='RdBu_r')
            plt.colorbar(im3, ax=ax3)
            ax3.set_title(f'GWS {str(ds.time.values[0])[:10]}')
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
        else:
            ax3.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Single time (NO DATA)')
    
    # Test 4: Shapefile plot
    ax4 = axes[1, 1]
    if gdf is not None:
        try:
            gdf.plot(ax=ax4, facecolor='lightblue', edgecolor='red')
            ax4.set_title('Shapefile')
            ax4.set_xlabel('Longitude')
            ax4.set_ylabel('Latitude')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Shapefile (ERROR)')
    else:
        ax4.text(0.5, 0.5, 'No shapefile', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Shapefile (NOT FOUND)')
    
    plt.tight_layout()
    plt.savefig('debug_test_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   💾 Saved: debug_test_plots.png")

def create_cartopy_test():
    """Test cartopy visualization specifically."""
    print("\n🌍 TESTING CARTOPY VISUALIZATION")
    print("="*50)
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # Load data
        ds = xr.open_dataset("results/groundwater_storage_anomalies.nc")
        
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Set extent
        extent = [float(ds.lon.min()), float(ds.lon.max()),
                 float(ds.lat.min()), float(ds.lat.max())]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        
        # Plot data
        mean_gws = ds.groundwater.mean(dim='time')
        if np.sum(~np.isnan(mean_gws.values)) > 0:
            im = ax.pcolormesh(ds.lon, ds.lat, mean_gws, 
                             cmap='RdBu_r', transform=ccrs.PlateCarree())
            plt.colorbar(im, ax=ax, label='GWS Anomaly (cm)')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.title('Cartopy Test - Mean Groundwater Storage')
        plt.savefig('debug_cartopy_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   💾 Saved: debug_cartopy_test.png")
        
    except Exception as e:
        print(f"   ❌ Cartopy test failed: {e}")

def create_data_distribution_plots(ds):
    """Create plots to understand data distribution."""
    print("\n📊 DATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    if 'groundwater' not in ds:
        print("   ❌ No groundwater data found")
        return
    
    gws = ds.groundwater
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of all values
    ax1 = axes[0, 0]
    valid_data = gws.values[~np.isnan(gws.values)]
    if len(valid_data) > 0:
        ax1.hist(valid_data, bins=50, alpha=0.7)
        ax1.set_title(f'Data Distribution (n={len(valid_data)})')
        ax1.set_xlabel('GWS Anomaly (cm)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Time series of spatial mean
    ax2 = axes[0, 1]
    spatial_mean = gws.mean(dim=['lat', 'lon'])
    time_index = pd.to_datetime(ds.time.values)
    ax2.plot(time_index, spatial_mean.values)
    ax2.set_title('Spatial Mean Time Series')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mean GWS Anomaly (cm)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Data availability map
    ax3 = axes[1, 0]
    data_count = (~np.isnan(gws)).sum(dim='time')
    im3 = ax3.pcolormesh(ds.lon, ds.lat, data_count, cmap='viridis')
    plt.colorbar(im3, ax=ax3, label='Number of valid time points')
    ax3.set_title('Data Availability')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    # 4. Standard deviation map
    ax4 = axes[1, 1]
    std_map = gws.std(dim='time')
    im4 = ax4.pcolormesh(ds.lon, ds.lat, std_map, cmap='plasma')
    plt.colorbar(im4, ax=ax4, label='Standard deviation (cm)')
    ax4.set_title('Temporal Variability')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig('debug_data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   💾 Saved: debug_data_distribution.png")

def main():
    """Main debugging function."""
    print("🐛 DEBUGGING VISUALIZATION ISSUES")
    print("="*70)
    
    # Check data
    ds = check_data()
    
    # Check shapefile
    gdf = check_shapefile()
    
    if ds is not None:
        # Create test plots
        create_simple_test_plots(ds, gdf)
        
        # Create cartopy test
        create_cartopy_test()
        
        # Create data distribution plots
        create_data_distribution_plots(ds)
        
        print(f"\n✅ Debug complete! Check the generated files:")
        print(f"   • debug_test_plots.png")
        print(f"   • debug_cartopy_test.png") 
        print(f"   • debug_data_distribution.png")
    else:
        print("❌ Cannot proceed without data")

if __name__ == "__main__":
    main() 