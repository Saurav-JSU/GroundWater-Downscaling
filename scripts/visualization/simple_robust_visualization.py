#!/usr/bin/env python3
"""
Simple Robust Visualization for GRACE Groundwater Data
======================================================

This script creates simple but robust visualizations that are guaranteed to work.
It focuses on clarity and reliability over complex features.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from pathlib import Path
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Settings
FIGURE_DPI = 300
FONT_SIZE = 12

class SimpleGRACEVisualizer:
    """Simple and robust GRACE visualization."""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "figures" / "simple_robust"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load the groundwater data."""
        print("📦 Loading data...")
        
        gws_file = self.base_dir / "results/groundwater_storage_anomalies.nc"
        if not gws_file.exists():
            raise FileNotFoundError(f"File not found: {gws_file}")
        
        self.ds = xr.open_dataset(gws_file)
        print(f"✅ Loaded dataset with shape: {self.ds.groundwater.shape}")
        
        # Load shapefile if available
        shapefile_path = self.base_dir / "data/shapefiles/processed/mississippi_river_basin.shp"
        if shapefile_path.exists():
            self.shapefile = gpd.read_file(shapefile_path)
            if self.shapefile.crs != 'EPSG:4326':
                self.shapefile = self.shapefile.to_crs('EPSG:4326')
            print(f"✅ Loaded shapefile")
        else:
            self.shapefile = None
            print("⚠️ No shapefile found")
    
    def create_basic_overview(self):
        """Create basic overview maps."""
        print("\n🎨 Creating basic overview...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        # Get data extent
        extent = [float(self.ds.lon.min()), float(self.ds.lon.max()),
                 float(self.ds.lat.min()), float(self.ds.lat.max())]
        
        # Time slices to show
        time_indices = [0, len(self.ds.time)//4, len(self.ds.time)//2, 
                       3*len(self.ds.time)//4, len(self.ds.time)-1]
        
        # Plot mean first
        ax = axes[0]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        mean_gws = self.ds.groundwater.mean(dim='time')
        
        # Use a robust color scale
        vmin, vmax = np.percentile(mean_gws.values[~np.isnan(mean_gws.values)], [5, 95])
        
        im = ax.pcolormesh(self.ds.lon, self.ds.lat, mean_gws, 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        # Add features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        
        # Add shapefile if available
        if self.shapefile is not None:
            self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                       transform=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title('Mean GWS Anomaly (2003-2022)', fontsize=FONT_SIZE, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label('GWS Anomaly (cm)', fontsize=FONT_SIZE)
        
        # Plot time slices
        for i, time_idx in enumerate(time_indices):
            ax = axes[i+1]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            gws_slice = self.ds.groundwater.isel(time=time_idx)
            time_str = str(self.ds.time.values[time_idx])[:7]
            
            im = ax.pcolormesh(self.ds.lon, self.ds.lat, gws_slice, 
                              cmap='RdBu_r', vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree())
            
            # Add features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
            
            # Add shapefile if available
            if self.shapefile is not None:
                self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                           transform=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(f'GWS Anomaly ({time_str})', fontsize=FONT_SIZE, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label('GWS Anomaly (cm)', fontsize=FONT_SIZE)
        
        plt.suptitle('Groundwater Storage Anomaly Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'groundwater_overview.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def create_time_series_analysis(self):
        """Create time series analysis."""
        print("\n📈 Creating time series analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate regional average
        regional_avg = self.ds.groundwater.mean(dim=['lat', 'lon'])
        time_index = pd.to_datetime(self.ds.time.values)
        
        # 1. Full time series
        ax1 = axes[0, 0]
        ax1.plot(time_index, regional_avg.values, 'b-', linewidth=2, alpha=0.7)
        
        # Add 12-month rolling mean
        rolling_mean = pd.Series(regional_avg.values, index=time_index).rolling(12, center=True).mean()
        ax1.plot(time_index, rolling_mean.values, 'r-', linewidth=3, label='12-month rolling mean')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Regional GWS Anomaly (cm)')
        ax1.set_title('Regional Groundwater Storage Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal cycle
        ax2 = axes[0, 1]
        monthly_clim = []
        time_index_da = xr.DataArray(time_index, dims=['time'], coords={'time': time_index})
        for month in range(1, 13):
            month_mask = time_index_da.dt.month == month
            month_data = regional_avg.sel(time=month_mask)
            if len(month_data) > 0:
                monthly_clim.append(month_data.mean().values)
            else:
                monthly_clim.append(0.0)
        
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        bars = ax2.bar(range(1, 13), monthly_clim, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Mean GWS Anomaly (cm)')
        ax2.set_title('Seasonal Cycle')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Annual means
        ax3 = axes[1, 0]
        annual_means = []
        years = []
        for year in range(2003, 2023):
            year_mask = time_index_da.dt.year == year
            year_data = regional_avg.sel(time=year_mask)
            if len(year_data) > 0:
                annual_means.append(year_data.mean().values)
                years.append(year)
        
        bars = ax3.bar(years, annual_means, color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Annual Mean GWS Anomaly (cm)')
        ax3.set_title('Annual Mean Groundwater Storage')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, annual_means, 1)
            p = np.poly1d(z)
            ax3.plot(years, p(years), 'r--', linewidth=2, 
                    label=f'Trend: {z[0]:.3f} cm/year')
            ax3.legend()
        
        # 4. Distribution
        ax4 = axes[1, 1]
        valid_data = regional_avg.values[~np.isnan(regional_avg.values)]
        ax4.hist(valid_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(valid_data), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(valid_data):.2f} cm')
        ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('GWS Anomaly (cm)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Regional GWS Values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Groundwater Storage Time Series Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'time_series_analysis.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def create_simple_trend_map(self):
        """Create a simple trend map without complex masking."""
        print("\n📊 Creating simple trend map...")
        
        # Calculate linear trends for each pixel
        print("   Calculating trends...")
        gws = self.ds.groundwater
        n_time, n_lat, n_lon = gws.shape
        
        trends = np.full((n_lat, n_lon), np.nan)
        p_values = np.full((n_lat, n_lon), np.nan)
        
        time_numeric = np.arange(n_time)
        
        # Calculate trends (sample every 10th pixel for speed)
        for i in range(0, n_lat, 2):  # Every 2nd pixel for faster calculation
            for j in range(0, n_lon, 2):
                ts = gws[:, i, j].values
                valid = ~np.isnan(ts)
                
                if np.sum(valid) > 60:  # At least 5 years of data
                    try:
                        slope, intercept, r, p, stderr = stats.linregress(
                            time_numeric[valid], ts[valid]
                        )
                        trends[i, j] = slope * 12  # Convert to annual
                        p_values[i, j] = p
                    except:
                        pass
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                     subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Get extent
        extent = [float(self.ds.lon.min()), float(self.ds.lon.max()),
                 float(self.ds.lat.min()), float(self.ds.lat.max())]
        
        # Plot 1: Trends
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Use robust color scale
        trend_max = np.nanpercentile(np.abs(trends), 90)
        if trend_max == 0:
            trend_max = 1
        
        im1 = ax1.pcolormesh(self.ds.lon[::2], self.ds.lat[::2], trends[::2, ::2], 
                           cmap='RdBu_r', vmin=-trend_max, vmax=trend_max,
                           transform=ccrs.PlateCarree())
        
        # Add features
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
        
        # Add shapefile
        if self.shapefile is not None:
            self.shapefile.boundary.plot(ax=ax1, color='black', linewidth=1.5,
                                       transform=ccrs.PlateCarree())
        
        # Add gridlines
        gl1 = ax1.gridlines(draw_labels=True, alpha=0.5)
        gl1.top_labels = False
        gl1.right_labels = False
        
        ax1.set_title('Groundwater Storage Trends (2003-2022)', fontsize=FONT_SIZE, fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                            pad=0.05, shrink=0.8)
        cbar1.set_label('Trend (cm/year)', fontsize=FONT_SIZE)
        
        # Plot 2: Significance
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Create significance categories
        sig_levels = np.full_like(p_values, 3)  # Default: not significant
        sig_levels[p_values < 0.1] = 2  # p < 0.1
        sig_levels[p_values < 0.05] = 1  # p < 0.05
        sig_levels[p_values < 0.01] = 0  # p < 0.01
        
        colors = ['darkred', 'red', 'orange', 'lightgray']
        im2 = ax2.pcolormesh(self.ds.lon[::2], self.ds.lat[::2], sig_levels[::2, ::2], 
                           cmap=plt.matplotlib.colors.ListedColormap(colors),
                           vmin=0, vmax=3, transform=ccrs.PlateCarree())
        
        # Add features
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax2.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
        
        # Add shapefile
        if self.shapefile is not None:
            self.shapefile.boundary.plot(ax=ax2, color='black', linewidth=1.5,
                                       transform=ccrs.PlateCarree())
        
        # Add gridlines
        gl2 = ax2.gridlines(draw_labels=True, alpha=0.5)
        gl2.top_labels = False
        gl2.right_labels = False
        
        ax2.set_title('Statistical Significance', fontsize=FONT_SIZE, fontweight='bold')
        
        # Add colorbar with custom labels
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', 
                            pad=0.05, shrink=0.8, ticks=[0.375, 1.125, 1.875, 2.625])
        cbar2.ax.set_xticklabels(['p<0.01', 'p<0.05', 'p<0.1', 'n.s.'])
        
        plt.suptitle('Groundwater Storage Trend Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'trend_analysis.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def create_spatial_statistics(self):
        """Create spatial statistics visualization."""
        print("\n🗺️ Creating spatial statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                               subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        # Get extent
        extent = [float(self.ds.lon.min()), float(self.ds.lon.max()),
                 float(self.ds.lat.min()), float(self.ds.lat.max())]
        
        # Statistics to plot
        stats_to_plot = [
            ('Mean', self.ds.groundwater.mean(dim='time'), 'RdBu_r'),
            ('Std Dev', self.ds.groundwater.std(dim='time'), 'viridis'),
            ('Min', self.ds.groundwater.min(dim='time'), 'Blues_r'),
            ('Max', self.ds.groundwater.max(dim='time'), 'Reds')
        ]
        
        for i, (title, data, cmap) in enumerate(stats_to_plot):
            ax = axes[i]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Use robust color scale
            if title == 'Mean':
                vmin, vmax = np.percentile(data.values[~np.isnan(data.values)], [5, 95])
            else:
                vmin, vmax = np.percentile(data.values[~np.isnan(data.values)], [1, 99])
            
            im = ax.pcolormesh(self.ds.lon, self.ds.lat, data, 
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())
            
            # Add features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
            
            # Add shapefile
            if self.shapefile is not None:
                self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                           transform=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(f'GWS {title}', fontsize=FONT_SIZE, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label(f'{title} (cm)', fontsize=FONT_SIZE)
        
        plt.suptitle('Spatial Statistics of Groundwater Storage', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / 'spatial_statistics.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def generate_all(self):
        """Generate all visualizations."""
        print("🎨 GENERATING SIMPLE ROBUST VISUALIZATIONS")
        print("="*60)
        
        # Create all visualizations
        self.create_basic_overview()
        self.create_time_series_analysis()
        self.create_simple_trend_map()
        self.create_spatial_statistics()
        
        print(f"\n✅ All visualizations complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Files created:")
        for file in sorted(self.output_dir.glob('*.png')):
            print(f"   • {file.name}")

def main():
    """Main function."""
    visualizer = SimpleGRACEVisualizer()
    visualizer.generate_all()

if __name__ == "__main__":
    main() 