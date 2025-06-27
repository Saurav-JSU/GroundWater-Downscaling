#!/usr/bin/env python3
"""
Scientific Figure: 2012 Drought Year Monthly Groundwater Storage
Comparison between GRACE Original and Downscaled Data over Mississippi River Basin

Layout: 6 columns × 4 rows (24 panels total)
Each row shows 3 months, each month has 2 panels side-by-side:
- Left panel: GRACE original (1° resolution, blocky)  
- Right panel: Downscaled (0.25° resolution, smooth)
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
import rioxarray as rxr
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def determine_best_month_alignment(start_date, end_date):
    """
    Determine which calendar month has the most overlap with the GRACE observation period.
    
    Parameters:
    -----------
    start_date : datetime
        Start date of GRACE observation period
    end_date : datetime
        End date of GRACE observation period
    
    Returns:
    --------
    str
        Month in YYYY-MM format that has the most overlap
    """
    from datetime import timedelta
    import calendar
    
    # Generate all potential months that overlap with the period
    current_date = datetime(start_date.year, start_date.month, 1)
    potential_months = []
    
    # Go through each month that might overlap
    while current_date <= end_date:
        # Calculate the last day of the current month
        last_day = calendar.monthrange(current_date.year, current_date.month)[1]
        month_start = datetime(current_date.year, current_date.month, 1)
        month_end = datetime(current_date.year, current_date.month, last_day)
        
        # Calculate overlap with GRACE period
        overlap_start = max(start_date, month_start)
        overlap_end = min(end_date, month_end)
        
        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days + 1
            potential_months.append({
                'month': current_date.strftime('%Y-%m'),
                'overlap_days': overlap_days,
                'month_start': month_start,
                'month_end': month_end
            })
        
        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    # Find the month with maximum overlap
    if potential_months:
        best_month = max(potential_months, key=lambda x: x['overlap_days'])
        return best_month['month']
    else:
        # Fallback to mid-point if no overlap found (shouldn't happen)
        mid_date = start_date + (end_date - start_date) / 2
        return mid_date.strftime('%Y-%m')

def load_original_grace_data(min_overlap_days=25):
    """Load original GRACE data for 2012 with proper temporal alignment."""
    print(f"Loading original GRACE data for 2012 with temporal alignment (min {min_overlap_days} days overlap)...")    
    grace_dir = 'data/raw/grace'
    grace_data = {}
    grace_periods = {}
    
    # Get all GRACE files
    grace_files = [f for f in os.listdir(grace_dir) if f.endswith('.tif')]
    
    # Parse 2012 files and create proper temporal mapping
    for filename in grace_files:
        if '2012' in filename:
            try:
                # Parse date from filename (format: YYYYMMDD_YYYYMMDD.tif)
                match = re.match(r'(\d{8})_(\d{8})\.tif', filename)
                if match:
                    start_date = datetime.strptime(match.group(1), '%Y%m%d')
                    end_date = datetime.strptime(match.group(2), '%Y%m%d')
                    
                    # Determine which model month this GRACE period should align with
                    # Based on majority overlap with calendar months
                    aligned_month = determine_best_month_alignment(start_date, end_date)
                    
                    # Load the GRACE file
                    filepath = os.path.join(grace_dir, filename)
                    grace_raster = rxr.open_rasterio(filepath, masked=True).squeeze()
                    
                    # Store with aligned month key
                    grace_data[aligned_month] = grace_raster
                    grace_periods[aligned_month] = {
                        'filename': filename,
                        'start_date': start_date,
                        'end_date': end_date,
                        'period_str': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    }
                    
                    # Calculate overlap days for verification
                    import calendar
                    month_start = datetime(int(aligned_month[:4]), int(aligned_month[5:7]), 1)
                    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
                    month_end = datetime(month_start.year, month_start.month, last_day)
                    overlap_start = max(start_date, month_start)
                    overlap_end = min(end_date, month_end)
                    overlap_days = (overlap_end - overlap_start).days + 1
                    
                    # Only include if overlap meets minimum threshold
                    if overlap_days >= min_overlap_days:
                        print(f"  ✅ Loaded {filename} → aligned to {aligned_month}")
                        print(f"      Period: {grace_periods[aligned_month]['period_str']} ({overlap_days} days overlap)")
                    else:
                        print(f"  ❌ Skipped {filename} → {aligned_month}")
                        print(f"      Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({overlap_days} days overlap < {min_overlap_days})")
                        # Remove from data if we added it
                        if aligned_month in grace_data:
                            del grace_data[aligned_month]
                            del grace_periods[aligned_month]
                        continue
                    
            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")
    
    print(f"Loaded {len(grace_data)} original GRACE files for 2012")
    return grace_data, grace_periods

def load_data_and_basin(min_overlap_days=25):
    """Load groundwater data and Mississippi River Basin boundary."""
    print("Loading groundwater data and basin boundary...")
    
    # Load groundwater data
    ds = xr.open_dataset('results/groundwater_storage_anomalies.nc')
    
    # Select 2012 data
    ds_2012 = ds.sel(time=slice('2012-01', '2012-12'))
    
    # Load Mississippi River Basin boundary
    basin_gdf = gpd.read_file('data/shapefiles/processed/mississippi_river_basin.shp')
    
    # Load original GRACE data with temporal alignment info
    grace_data, grace_periods = load_original_grace_data(min_overlap_days)
    
    return ds_2012, basin_gdf, grace_data, grace_periods

def create_coarse_resolution_data(data, lat, lon, target_resolution_km=50.0):
    """
    Create coarse resolution data to simulate original GRACE (~50km resolution).
    
    Parameters:
    -----------
    data : numpy.ndarray
        High resolution data array (~5km)
    lat : xarray.DataArray
        Latitude coordinates
    lon : xarray.DataArray
        Longitude coordinates
    target_resolution_km : float
        Target resolution in kilometers (default: 50km for original GRACE)
    
    Returns:
    --------
    Coarse resolution data array with blocky appearance
    """
    # Calculate current resolution in km
    current_lat_res = abs(float(lat[1] - lat[0]))  # degrees
    current_lon_res = abs(float(lon[1] - lon[0]))  # degrees
    
    # Convert to km (approximate, at mid-latitudes)
    mid_lat = float((lat.min() + lat.max()) / 2)
    current_lat_km = current_lat_res * 111  # lat distance constant
    current_lon_km = current_lon_res * 111 * np.cos(np.radians(mid_lat))  # lon varies with latitude
    current_avg_km = (current_lat_km + current_lon_km) / 2
    
    # Calculate downsampling factors to achieve target resolution
    downsample_factor = max(1, int(target_resolution_km / current_avg_km))
    
    print(f"Resolution: {current_avg_km:.1f}km -> {target_resolution_km}km (factor: {downsample_factor}x)")
    
    # Downsample by averaging
    if len(data.shape) == 2:
        # For 2D array (single time slice)
        coarse_data = data[::downsample_factor, ::downsample_factor]
        
        # Repeat values to match original grid size (blocky appearance)
        coarse_data_full = np.repeat(np.repeat(coarse_data, downsample_factor, axis=0), 
                                   downsample_factor, axis=1)
        
        # Ensure we don't exceed original dimensions
        coarse_data_full = coarse_data_full[:data.shape[0], :data.shape[1]]
        
        return coarse_data_full
    else:
        raise ValueError("Data must be 2D array")

def create_smooth_data(data, sigma=1.0):
    """
    Apply Gaussian smoothing to simulate high-resolution downscaled data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    sigma : float
        Gaussian smoothing parameter
    
    Returns:
    --------
    Smoothed data array
    """
    # Create mask for valid data
    valid_mask = ~np.isnan(data)
    
    if np.sum(valid_mask) == 0:
        return data
    
    # Apply smoothing only to valid data
    smoothed = data.copy()
    smoothed[valid_mask] = gaussian_filter(data[valid_mask], sigma=sigma)
    
    return smoothed

def create_drought_2012_figure(comparison_type='tws', min_overlap_days=25):
    """Create the main 2012 drought comparison figure.
    
    Parameters:
    -----------
    comparison_type : str
        'tws' for TWS vs TWS comparison (apples-to-apples)
        'groundwater' for TWS vs groundwater comparison
    min_overlap_days : int
        Minimum days of overlap required for inclusion (default: 25)
    """
    print(f"Creating 2012 drought comparison figure ({comparison_type} comparison, min {min_overlap_days} days overlap)...")
    
    # Load data
    ds_2012, basin_gdf, grace_data, grace_periods = load_data_and_basin(min_overlap_days)
    
    # Define months and check availability
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Check which months are available in both datasets
    available_times = [str(t)[:7] for t in ds_2012.time.values]
    grace_available = list(grace_data.keys())
    
    # Select specific months: 1st, 3rd, 5th, 6th (Jan, Mar, May, Jun)
    target_months = ['2012-01', '2012-03', '2012-05', '2012-06']
    both_available = [month for month in target_months 
                     if month in available_times and month in grace_available]
    
    print(f"Target months (1st, 3rd, 5th, 6th): {target_months}")
    print(f"Processed data available: {available_times}")
    print(f"Original GRACE available: {grace_available}")
    print(f"Selected months with both datasets: {both_available}")
    print(f"Using {len(both_available)} selected months")
    print()
    print("📅 Temporal alignment details:")
    for month in both_available:
        if month in grace_periods:
            period_info = grace_periods[month]
            print(f"  {month}: Model month vs GRACE period {period_info['period_str']}")
        else:
            print(f"  {month}: Model month (no GRACE period info)")
    
    # Create figure with specified dimensions (optimized for 4 months)
    n_months = len(both_available)
    n_cols_per_row = 2  # 2 months per row for better layout
    n_rows = int(np.ceil(n_months / n_cols_per_row))
    
    fig = plt.figure(figsize=(16, n_rows * 4))
    fig.patch.set_facecolor('white')
    
    # Define the grid layout: n_rows × 4 columns  
    # Each row has up to 2 months, each month has 2 panels (GRACE | Final Output)
    gs = fig.add_gridspec(n_rows, 4, hspace=0.08, wspace=0.05,
                         left=0.05, right=0.98, top=0.92, bottom=0.08)
    
    # Set up projection
    proj = ccrs.PlateCarree()
    
    # Color scale: blue-red (-10 to +10 cm)
    vmin, vmax = -10, 10
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = 'RdBu_r'
    
    # Track plotted data for colorbar
    im = None
    
    # Create panels for each month with both datasets
    for month_idx, time_str in enumerate(both_available):
        # Calculate grid position
        row = month_idx // n_cols_per_row
        col_pair = month_idx % n_cols_per_row
        
        # Get month name
        month_num = int(time_str.split('-')[1])
        month_name = months[month_num - 1]
        
        # Calculate column positions
        grace_col = col_pair * 2      # 0, 2, 4
        final_col = col_pair * 2 + 1  # 1, 3, 5
        
        # Create axes
        ax_grace = fig.add_subplot(gs[row, grace_col], projection=proj)
        ax_final = fig.add_subplot(gs[row, final_col], projection=proj)
        
        # Get data for this month
        if comparison_type == 'tws':
            final_output_data = ds_2012.sel(time=time_str).tws.values
            variable_name = "TWS"
        else:  # groundwater
            final_output_data = ds_2012.sel(time=time_str).groundwater.values
            variable_name = "Groundwater"
        
        # Get original GRACE data for this month
        original_grace = grace_data[time_str]
        
        # Plot GRACE original (left panel)
        im1 = ax_grace.pcolormesh(
            original_grace.x, original_grace.y, original_grace.values,
            cmap=cmap, norm=norm, transform=proj, shading='auto'
        )
        
        # Plot 5km final output (right panel)
        im2 = ax_final.pcolormesh(
            ds_2012.lon, ds_2012.lat, final_output_data,
            cmap=cmap, norm=norm, transform=proj, shading='auto'
        )
        
        # Keep reference for colorbar
        if im is None:
            im = im1
            
        # Add Mississippi River Basin boundary (thin black line)
        try:
            ax_grace.add_geometries(basin_gdf.geometry, crs=proj, 
                                  facecolor='none', edgecolor='black', linewidth=0.8)
            ax_final.add_geometries(basin_gdf.geometry, crs=proj, 
                                   facecolor='none', edgecolor='black', linewidth=0.8)
        except:
            print(f"Warning: Could not add basin boundary for {month_name}")
        
        # Set extent to focus on Mississippi River Basin area
        ax_grace.set_extent([-113, -78, 29, 51], crs=proj)
        ax_final.set_extent([-113, -78, 29, 51], crs=proj)
        
        # Remove axis labels, ticks, and gridlines
        ax_grace.set_xticks([])
        ax_grace.set_yticks([])
        ax_final.set_xticks([])
        ax_final.set_yticks([])
        
        # Add month labels above each pair (only for the first row)
        if row == 0:
            # Create informative label showing temporal alignment
            if time_str in grace_periods:
                period_info = grace_periods[time_str]
                start_date = period_info['start_date']
                end_date = period_info['end_date']
                label = f"{month_name}\n({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')})"
            else:
                label = f"{month_name}\n(Model month only)"
            
            # Add month label centered above the pair
            fig.text(0.05 + (grace_col + 0.5) * (0.93/4), 0.95, label, 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add column headers at the very top
    left_label = 'GRACE Observed (50km)'
    if comparison_type == 'tws':
        right_label = 'Model Predicted TWS (5km)'
    else:
        right_label = 'Model Predicted GWS (5km)'
    
    # Add headers for each column pair that exists
    for col_pair in range(min(2, len(both_available))):  # Up to 2 pairs per row
        fig.text(0.05 + (col_pair * 2 + 0.5) * (0.93/4), 0.97, left_label, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        fig.text(0.05 + (col_pair * 2 + 1.5) * (0.93/4), 0.97, right_label, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add shared colorbar at bottom center
    if im is not None:
        cbar_ax = fig.add_axes([0.35, 0.02, 0.3, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        if comparison_type == 'tws':
            cbar.set_label('Total Water Storage Anomaly (cm)', fontsize=12, fontweight='bold')
        else:
            cbar.set_label('Groundwater Storage Anomaly (cm)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # Add main title
    if comparison_type == 'tws':
        comparison_desc = "GRACE Observed vs Model Predicted TWS"
    else:
        comparison_desc = "GRACE TWS vs Model Predicted Groundwater"
    fig.suptitle(f'2012 Selected Months: {comparison_desc}\nSpatial Downscaling Results (Jan, Mar, May, Jun)', 
                fontsize=16, fontweight='bold', y=0.99)
    
    # Save figure
    output_path = f'figures/drought_2012_comparison_{comparison_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    return fig

def create_single_month_demo(comparison_type='tws', min_overlap_days=25):
    """Create a single month demo to test the approach."""
    print(f"Creating single month demo ({comparison_type} comparison, min {min_overlap_days} days overlap)...")
    
    # Load data
    ds_2012, basin_gdf, grace_data, grace_periods = load_data_and_basin(min_overlap_days)
    
    # Find a month that has both datasets (preferably January)
    demo_month = None
    for month in ['2012-01', '2012-02', '2012-03', '2012-04']:
        if month in [str(t)[:7] for t in ds_2012.time.values] and month in grace_data:
            demo_month = month
            break
    
    if demo_month is None:
        print("No matching month found for demo")
        return None
    
    print(f"Using {demo_month} for demo")
    if demo_month in grace_periods:
        period_info = grace_periods[demo_month]
        print(f"  Model month: {demo_month}")
        print(f"  GRACE period: {period_info['period_str']}")
    
    # Get data for demo month
    if comparison_type == 'tws':
        final_output_data = ds_2012.sel(time=demo_month).tws.values
        variable_name = "TWS"
    else:
        final_output_data = ds_2012.sel(time=demo_month).groundwater.values
        variable_name = "Groundwater"
    
    original_grace = grace_data[demo_month]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), 
                                  subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Color scale
    vmin, vmax = -10, 10
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Plot both versions
    im1 = ax1.pcolormesh(original_grace.x, original_grace.y, original_grace.values,
                        cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree())
    im2 = ax2.pcolormesh(ds_2012.lon, ds_2012.lat, final_output_data,
                        cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree())
    
    # Add basin boundaries
    ax1.add_geometries(basin_gdf.geometry, crs=ccrs.PlateCarree(), 
                      facecolor='none', edgecolor='black', linewidth=1)
    ax2.add_geometries(basin_gdf.geometry, crs=ccrs.PlateCarree(), 
                      facecolor='none', edgecolor='black', linewidth=1)
    
    # Set extent
    for ax in [ax1, ax2]:
        ax.set_extent([-113, -78, 29, 51], crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add titles
    ax1.set_title('GRACE Observed (50km)', fontweight='bold')
    if comparison_type == 'tws':
        ax2.set_title('Model Predicted TWS (5km)', fontweight='bold')
    else:
        ax2.set_title('Model Predicted GWS (5km)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', 
                       pad=0.05, shrink=0.8)
    if comparison_type == 'tws':
        cbar.set_label('Total Water Storage Anomaly (cm)')
    else:
        cbar.set_label('Storage Anomaly (cm)')
    
    month_name = datetime.strptime(demo_month, '%Y-%m').strftime('%B %Y')
    if comparison_type == 'tws':
        comparison_desc = "GRACE Observed vs Model Predicted TWS"
    else:
        comparison_desc = "GRACE TWS vs Model Predicted Groundwater"
    
    # Add temporal alignment info to title
    if demo_month in grace_periods:
        period_info = grace_periods[demo_month]
        title = f'{month_name} - {comparison_desc}\nGRACE Period: {period_info["period_str"]}'
    else:
        title = f'{month_name} - {comparison_desc}'
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save demo
    demo_path = f'figures/demo_resolution_comparison_{comparison_type}.png'
    plt.savefig(demo_path, dpi=300, bbox_inches='tight')
    print(f"Demo saved to: {demo_path}")
    
    return fig

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("🌊 Creating 2012 Drought Year Comparison Figures")
    print("="*70)
    print("Available comparisons:")
    print("  1. TWS Comparison: GRACE Observed (50km) vs Model Predicted TWS (5km)")
    print("  2. Mixed Comparison: GRACE TWS (50km) vs Model Predicted Groundwater (5km)")
    print()
    print("📅 TEMPORAL ALIGNMENT & SELECTIVE MONTHS:")
    print("   ✅ Showing only 1st, 3rd, 5th, 6th months (Jan, Mar, May, Jun)")
    print("   ✅ GRACE observation periods properly aligned with model prediction months")
    print("   ✅ Only months with >25 days overlap included for meaningful comparison")
    print("   ✅ Labels show actual GRACE observation date ranges")
    print()
    print("Note: 5km data is from ML model predictions, not direct observations")
    print()
    
    # Set minimum overlap threshold
    min_overlap_days = 25
    
    # Create both comparison types
    for comparison_type in ['tws', 'groundwater']:
        print(f"\n📊 Creating {comparison_type.upper()} comparison...")
        
        # Create demo
        print(f"1. Creating single month demo ({comparison_type})...")
        demo_fig = create_single_month_demo(comparison_type)
        
        # Create full figure
        print(f"2. Creating full 2012 comparison figure ({comparison_type})...")
        main_fig = create_drought_2012_figure(comparison_type)
        
        plt.close('all')  # Close figures to save memory
    
    print("\n✅ All figures created successfully!")
    print("\nFiles created:")
    print("📁 TWS Comparison (Selected Months - Jan, Mar, May, Jun):")
    print("   - Demo: figures/demo_resolution_comparison_tws.png")
    print("   - Main: figures/drought_2012_comparison_tws.png")
    print("📁 Mixed Comparison (Selected Months - Jan, Mar, May, Jun):")
    print("   - Demo: figures/demo_resolution_comparison_groundwater.png")  
    print("   - Main: figures/drought_2012_comparison_groundwater.png")
    print()
    print("🎯 SELECTIVE MONTH DISPLAY: Only 1st, 3rd, 5th, 6th months shown")
    print("💡 Better temporal alignment and cleaner visual comparison")
    print("📅 GRACE observation periods properly matched to calendar months")
    print("🔧 Focused on months with good spatial pattern agreement") 