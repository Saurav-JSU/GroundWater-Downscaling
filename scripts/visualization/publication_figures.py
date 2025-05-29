# scripts/publication_figures.py
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.dates as mdates
from tqdm import tqdm
import seaborn as sns

def create_publication_figures():
    """Create comprehensive set of publication-quality figures"""
    # Create output directory
    figure_dir = Path("figures/publication")
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    gws_path = "results/groundwater_storage_anomalies.nc"
    if not os.path.exists(gws_path):
        print(f"❌ Groundwater data not found at {gws_path}")
        print("Please run the groundwater calculation script first.")
        return
    
    gws_ds = xr.open_dataset(gws_path)
    
    # 1. Create study area map
    create_study_area_map(gws_ds, figure_dir)
    
    # 2. Create time series of GWS for the entire region
    create_regional_timeseries(gws_ds, figure_dir)
    
    # 3. Create seasonal maps (wet vs dry season)
    create_seasonal_maps(gws_ds, figure_dir)
    
    # 4. Create drought period maps
    create_drought_comparison_maps(gws_ds, figure_dir)
    
    # 5. Create downscaling comparison visualization
    create_downscaling_comparison(gws_ds, figure_dir)
    
    # 6. Create spatial trend map
    create_trend_map(gws_ds, figure_dir)
    
    print(f"✅ Publication figures saved to {figure_dir}")

def create_study_area_map(gws_ds, figure_dir):
    """Create a map of the study area with state boundaries"""
    plt.figure(figsize=(10, 8))
    
    # Create map with state boundaries
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([gws_ds.lon.min(), gws_ds.lon.max(), 
                  gws_ds.lat.min(), gws_ds.lat.max()], 
                  crs=ccrs.PlateCarree())
    
    # Add state boundaries
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    
    # Add country borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Add rivers
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    # Add a sample month of GWS data
    sample_month = gws_ds.groundwater.isel(time=len(gws_ds.time)//2)
    
    # Create a diverging colormap centered at zero
    vmax = max(abs(np.nanmin(sample_month)), abs(np.nanmax(sample_month)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.pcolormesh(gws_ds.lon, gws_ds.lat, sample_month, 
                      norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, 
                        label='Groundwater Storage Anomaly (cm)')
    
    # Add title
    plt.title('Study Area: Mississippi River Basin\nGroundwater Storage Anomalies')
    
    # Save the figure
    plt.savefig(figure_dir / 'study_area_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_regional_timeseries(gws_ds, figure_dir):
    """Create a time series plot of regional average groundwater storage"""
    # Calculate regional average
    regional_avg = gws_ds.groundwater.mean(dim=['lat', 'lon'])
    
    # Convert to pandas Series for easier plotting
    time_index = pd.DatetimeIndex(regional_avg.time.values)
    regional_series = pd.Series(regional_avg.values, index=time_index)
    
    # Calculate 12-month moving average
    rolling_avg = regional_series.rolling(window=12, center=True).mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(regional_series.index, regional_series.values, 'b-', alpha=0.5, label='Monthly')
    plt.plot(rolling_avg.index, rolling_avg.values, 'r-', linewidth=2, label='12-month moving avg')
    
    # Add drought periods (example - modify with actual drought periods)
    drought_periods = [
        ('2007-01', '2007-12', 'Drought 2007'),
        ('2012-05', '2012-12', 'Drought 2012'),
        ('2018-06', '2018-12', 'Drought 2018')
    ]
    
    for start, end, label in drought_periods:
        plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                   color='orange', alpha=0.2)
        plt.text(pd.to_datetime(start), regional_series.min() * 0.9, label,
                rotation=90, verticalalignment='bottom')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add title and labels
    plt.title('Regional Groundwater Storage Anomaly (2003-2022)')
    plt.xlabel('Year')
    plt.ylabel('Groundwater Storage Anomaly (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis with years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    
    # Save the figure
    plt.savefig(figure_dir / 'regional_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_seasonal_maps(gws_ds, figure_dir):
    """Create maps showing wet vs dry season groundwater patterns"""
    # Extract month information
    months = pd.DatetimeIndex(gws_ds.time.values).month
    
    # Define seasons (adjust for your region if needed)
    wet_months = [3, 4, 5]  # Spring (Mar-May)
    dry_months = [7, 8, 9]  # Summer (Jul-Sep)
    
    # Calculate seasonal averages
    wet_season = gws_ds.groundwater.isel(time=[i for i, m in enumerate(months) if m in wet_months]).mean(dim='time')
    dry_season = gws_ds.groundwater.isel(time=[i for i, m in enumerate(months) if m in dry_months]).mean(dim='time')
    
    # Calculate seasonal difference
    seasonal_diff = wet_season - dry_season
    
    # Create 3-panel figure: wet season, dry season, difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set color scale limits for consistency
    vmax = max(abs(np.nanmin(wet_season)), abs(np.nanmax(dry_season)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    diff_max = max(abs(np.nanmin(seasonal_diff)), abs(np.nanmax(seasonal_diff)))
    diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
    
    # Plot wet season
    ax = axes[0]
    im0 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, wet_season, 
                       norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('Wet Season (Spring)')
    plt.colorbar(im0, ax=ax, orientation='horizontal', pad=0.05, 
                label='GWS Anomaly (cm)')
    
    # Plot dry season
    ax = axes[1]
    im1 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, dry_season, 
                       norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('Dry Season (Summer)')
    plt.colorbar(im1, ax=ax, orientation='horizontal', pad=0.05, 
                label='GWS Anomaly (cm)')
    
    # Plot difference
    ax = axes[2]
    im2 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, seasonal_diff, 
                       norm=diff_norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('Seasonal Difference (Wet - Dry)')
    plt.colorbar(im2, ax=ax, orientation='horizontal', pad=0.05, 
                label='Difference (cm)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(figure_dir / 'seasonal_maps.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_drought_comparison_maps(gws_ds, figure_dir):
    """Create comparison maps for drought vs non-drought periods"""
    # Define drought periods (adjust based on known droughts)
    drought_periods = {
        '2012 Drought': ('2012-06', '2012-09'),
        '2007 Drought': ('2007-05', '2007-08')
    }
    
    # Normal periods for comparison (same season, different year)
    normal_periods = {
        '2012 Normal': ('2010-06', '2010-09'),
        '2007 Normal': ('2005-05', '2005-08')
    }
    
    # Process each drought
    for drought_name, (drought_start, drought_end) in drought_periods.items():
        normal_name = drought_name.replace('Drought', 'Normal')
        normal_start, normal_end = normal_periods[normal_name]
        
        # Extract time slices
        drought_data = gws_ds.groundwater.sel(
            time=slice(drought_start, drought_end)).mean(dim='time')
        
        normal_data = gws_ds.groundwater.sel(
            time=slice(normal_start, normal_end)).mean(dim='time')
        
        # Calculate difference
        difference = drought_data - normal_data
        
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Set color scale
        vmax = max(abs(np.nanmin(drought_data)), abs(np.nanmax(drought_data)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        diff_max = max(abs(np.nanmin(difference)), abs(np.nanmax(difference)))
        diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
        
        # Plot drought period
        ax = axes[0]
        im0 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, drought_data, 
                           norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f'{drought_name} ({drought_start} to {drought_end})')
        plt.colorbar(im0, ax=ax, orientation='horizontal', pad=0.05, 
                    label='GWS Anomaly (cm)')
        
        # Plot normal period
        ax = axes[1]
        im1 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, normal_data, 
                           norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f'{normal_name} ({normal_start} to {normal_end})')
        plt.colorbar(im1, ax=ax, orientation='horizontal', pad=0.05, 
                    label='GWS Anomaly (cm)')
        
        # Plot difference
        ax = axes[2]
        im2 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, difference, 
                           norm=diff_norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title('Drought Impact (Drought - Normal)')
        plt.colorbar(im2, ax=ax, orientation='horizontal', pad=0.05, 
                    label='Difference (cm)')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        drought_name_file = drought_name.replace(' ', '_').lower()
        plt.savefig(figure_dir / f'{drought_name_file}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_downscaling_comparison(gws_ds, figure_dir):
    """Create visualization of downscaling resolution improvement"""
    # Load a sample month of data
    sample_month = gws_ds.groundwater.isel(time=len(gws_ds.time)//2)
    
    # Get dimensions
    nlat, nlon = sample_month.shape
    
    # Create a synthetic "coarse" version to simulate original GRACE
    coarse_factor = 4  # Simulating 4x coarser resolution
    coarse_lat = gws_ds.lat.values[::coarse_factor]
    coarse_lon = gws_ds.lon.values[::coarse_factor]
    
    # Average to coarse resolution
    coarse_data = np.zeros((len(coarse_lat), len(coarse_lon)))
    for i in range(len(coarse_lat)):
        for j in range(len(coarse_lon)):
            i_start = i * coarse_factor
            i_end = min((i + 1) * coarse_factor, nlat)
            j_start = j * coarse_factor
            j_end = min((j + 1) * coarse_factor, nlon)
            
            block = sample_month.values[i_start:i_end, j_start:j_end]
            coarse_data[i, j] = np.nanmean(block)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set color scale
    vmax = max(abs(np.nanmin(sample_month)), abs(np.nanmax(sample_month)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot coarse data
    ax = axes[0]
    mesh_lons, mesh_lats = np.meshgrid(coarse_lon, coarse_lat)
    im0 = ax.pcolormesh(mesh_lons, mesh_lats, coarse_data, 
                       norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(f'Original Resolution (≈{coarse_factor}x coarser)')
    plt.colorbar(im0, ax=ax, orientation='horizontal', pad=0.05, 
                label='GWS Anomaly (cm)')
    
    # Plot downscaled data
    ax = axes[1]
    im1 = ax.pcolormesh(gws_ds.lon, gws_ds.lat, sample_month, 
                       norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('Downscaled Resolution')
    plt.colorbar(im1, ax=ax, orientation='horizontal', pad=0.05, 
                label='GWS Anomaly (cm)')
    
    # Add overall title
    plt.suptitle(f'Resolution Comparison: Downscaling GRACE Data', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(figure_dir / 'downscaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_trend_map(gws_ds, figure_dir):
    """Create a map of long-term groundwater storage trends"""
    # Load all data
    gws_data = gws_ds.groundwater
    
    # Calculate trend at each grid cell
    # We'll use a simple linear regression for each cell
    from scipy import stats
    
    # Convert time to numeric values (months since start)
    time_numeric = np.arange(len(gws_ds.time))
    
    # Prepare arrays for results
    nlat, nlon = gws_data.shape[1:]
    trend = np.zeros((nlat, nlon))
    p_value = np.zeros((nlat, nlon))
    
    # Calculate trend for each grid cell
    for i in range(nlat):
        for j in range(nlon):
            cell_data = gws_data[:, i, j].values
            
            # Skip if all NaN
            if np.all(np.isnan(cell_data)):
                trend[i, j] = np.nan
                p_value[i, j] = np.nan
                continue
            
            # Use only non-NaN values
            valid = ~np.isnan(cell_data)
            if np.sum(valid) < 24:  # Require at least 2 years of data
                trend[i, j] = np.nan
                p_value[i, j] = np.nan
                continue
                
            # Calculate linear trend
            slope, intercept, r_value, p, std_err = stats.linregress(
                time_numeric[valid], cell_data[valid])
            
            # Convert to cm/year (assuming monthly data)
            trend[i, j] = slope * 12  # slope per month * 12 = cm/year
            p_value[i, j] = p
    
    # Create trend map
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set color scale centered at zero
    vmax = max(abs(np.nanmin(trend)), abs(np.nanmax(trend)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot trend with significance masking
    # Transparent where p-value > 0.05 (not significant)
    alpha = np.ones_like(p_value)
    alpha[p_value > 0.05] = 0.3  # Make non-significant trends transparent
    
    im = ax.pcolormesh(gws_ds.lon, gws_ds.lat, trend, 
                      norm=norm, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                      alpha=alpha)
    
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, 
                label='Trend (cm/year)')
    
    ax.set_title('Groundwater Storage Trend (2003-2022)\nTransparent areas: not statistically significant (p>0.05)')
    
    # Save the figure
    plt.savefig(figure_dir / 'groundwater_trend_map.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_publication_figures()