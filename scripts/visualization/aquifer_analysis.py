#!/usr/bin/env python3
"""
Individual Aquifer Analysis for GRACE Groundwater Data
======================================================

This script analyzes each individual aquifer using their respective shapefiles and creates:
1. Mean GWS maps for each aquifer (subplots)
2. Trend analysis for each aquifer (subplots) 
3. Seasonal cycle analysis for each aquifer (subplots)

Author: GRACE Analysis Pipeline
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import warnings
from matplotlib.colors import TwoSlopeNorm

try:
    import regionmask
except ImportError:
    regionmask = None
    warnings.warn("⚠️ regionmask not installed; will use alternative masking")

warnings.filterwarnings('ignore')

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")

# Publication settings
FIGURE_DPI = 300
FONT_SIZE = 10


class AquiferAnalyzer:
    """Analyzer for individual aquifers using their shapefiles."""
    
    def __init__(self, base_dir=".", aquifer_dir=None):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory of the project
        aquifer_dir : str
            Directory containing individual aquifer shapefiles
        """
        self.base_dir = Path(base_dir)
        self.figures_dir = self.base_dir / "figures" / "aquifer_analysis"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set aquifer directory
        if aquifer_dir is None:
            self.aquifer_dir = self.base_dir / "data/shapefiles/processed/individual_aquifers"
        else:
            self.aquifer_dir = Path(aquifer_dir)
        
        print(f"📂 Looking for aquifer shapefiles in: {self.aquifer_dir}")
        
        # Load groundwater data
        self._load_groundwater_data()
        
        # Load and process aquifer shapefiles
        self._load_aquifer_shapefiles()
    
    def _load_groundwater_data(self):
        """Load groundwater storage data."""
        print("\n📦 Loading groundwater data...")
        
        # Try different possible groundwater files
        gws_files = [
            "results/groundwater_storage_anomalies.nc",
            "results/groundwater_storage_anomalies_corrected.nc", 
            "results/groundwater_storage_anomalies_enhanced.nc"
        ]
        
        for gws_file in gws_files:
            if (self.base_dir / gws_file).exists():
                print(f"  ✅ Loading: {gws_file}")
                self.gws_ds = xr.open_dataset(self.base_dir / gws_file)
                break
        else:
            raise FileNotFoundError("No groundwater storage file found!")
        
        print(f"     Dataset shape: {self.gws_ds.groundwater.shape}")
        print(f"     Time range: {str(self.gws_ds.time.values[0])[:10]} to {str(self.gws_ds.time.values[-1])[:10]}")
        print(f"     Spatial extent: {float(self.gws_ds.lat.min()):.2f}°N to {float(self.gws_ds.lat.max()):.2f}°N")
        print(f"                    {float(self.gws_ds.lon.min()):.2f}°E to {float(self.gws_ds.lon.max()):.2f}°E")
    
    def _load_aquifer_shapefiles(self):
        """Load all aquifer shapefiles and create masks."""
        print(f"\n🗺️ Loading aquifer shapefiles...")
        
        # Find all .shp files
        shp_files = list(self.aquifer_dir.glob("*.shp"))
        print(f"  Found {len(shp_files)} shapefiles")
        
        if len(shp_files) == 0:
            raise FileNotFoundError(f"No shapefiles found in {self.aquifer_dir}")
        
        self.aquifers = {}
        self.aquifer_masks = {}
        
        # Process each shapefile
        for shp_file in tqdm(shp_files, desc="Processing aquifers"):
            aquifer_name = shp_file.stem
            
            try:
                # Load shapefile
                gdf = gpd.read_file(shp_file)
                
                # Ensure CRS is WGS84
                if gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                
                # Check if aquifer overlaps with data domain
                data_bounds = [
                    float(self.gws_ds.lon.min()), float(self.gws_ds.lat.min()),
                    float(self.gws_ds.lon.max()), float(self.gws_ds.lat.max())
                ]
                
                aquifer_bounds = gdf.total_bounds
                
                # Check for overlap
                if (aquifer_bounds[2] < data_bounds[0] or aquifer_bounds[0] > data_bounds[2] or
                    aquifer_bounds[3] < data_bounds[1] or aquifer_bounds[1] > data_bounds[3]):
                    print(f"    ⚠️ {aquifer_name}: No overlap with data domain, skipping")
                    continue
                
                # Create mask
                mask = self._create_aquifer_mask(gdf)
                
                if mask is not None and not np.all(np.isnan(mask)):
                    self.aquifers[aquifer_name] = gdf
                    self.aquifer_masks[aquifer_name] = mask
                    
                    # Count valid pixels
                    n_pixels = np.sum(~np.isnan(mask))
                    print(f"    ✅ {aquifer_name}: {n_pixels} pixels")
                else:
                    print(f"    ⚠️ {aquifer_name}: No valid pixels, skipping")
                
            except Exception as e:
                print(f"    ❌ {aquifer_name}: Error - {e}")
                continue
        
        print(f"\n📊 Successfully loaded {len(self.aquifers)} aquifers for analysis")
        
        if len(self.aquifers) == 0:
            raise ValueError("No valid aquifers found for analysis!")
    
    def _create_aquifer_mask(self, gdf):
        """Create a mask for the aquifer region."""
        if regionmask is None:
            print("    ⚠️ regionmask not available, using basic grid masking")
            return self._create_basic_mask(gdf)
        
        try:
            # Create region using regionmask
            if len(gdf) == 1:
                # Single polygon
                region_geom = gdf.geometry.iloc[0]
            else:
                # Multiple polygons - union them
                try:
                    region_geom = gdf.union_all()
                except AttributeError:
                    region_geom = gdf.geometry.unary_union
            
            region = regionmask.Regions([region_geom], names=['aquifer'])
            mask = region.mask(self.gws_ds.lon, self.gws_ds.lat)
            
            return mask
            
        except Exception as e:
            print(f"    ⚠️ regionmask failed: {e}, using basic masking")
            return self._create_basic_mask(gdf)
    
    def _create_basic_mask(self, gdf):
        """Create a basic mask using point-in-polygon testing."""
        try:
            from shapely.geometry import Point
            
            # Create coordinate grids
            lon_grid, lat_grid = np.meshgrid(self.gws_ds.lon.values, self.gws_ds.lat.values)
            
            # Flatten for easier processing
            points = list(zip(lon_grid.flatten(), lat_grid.flatten()))
            
            # Union all geometries
            if len(gdf) == 1:
                union_geom = gdf.geometry.iloc[0]
            else:
                try:
                    union_geom = gdf.union_all()
                except AttributeError:
                    union_geom = gdf.geometry.unary_union
            
            # Test which points are inside
            mask_flat = np.array([union_geom.contains(Point(p)) for p in points])
            mask = mask_flat.reshape(lon_grid.shape)
            
            # Convert boolean to float with NaN for outside
            mask = np.where(mask, 0.0, np.nan)
            
            return mask
            
        except Exception as e:
            print(f"    ❌ Basic masking failed: {e}")
            return None
    
    def calculate_aquifer_statistics(self):
        """Calculate statistics for each aquifer."""
        print("\n📊 Calculating aquifer statistics...")
        
        self.aquifer_stats = {}
        
        for aquifer_name, mask in tqdm(self.aquifer_masks.items(), desc="Calculating stats"):
            try:
                # Apply mask to groundwater data
                gws_masked = self.gws_ds.groundwater.where(~np.isnan(mask))
                
                # Calculate regional time series
                regional_ts = gws_masked.mean(dim=['lat', 'lon'])
                
                # Calculate statistics
                stats_dict = {
                    'time_series': regional_ts,
                    'mean_gws': float(regional_ts.mean()),
                    'std_gws': float(regional_ts.std()),
                    'min_gws': float(regional_ts.min()),
                    'max_gws': float(regional_ts.max()),
                    'spatial_mean': gws_masked.mean(dim='time'),
                    'n_pixels': int(np.sum(~np.isnan(mask)))
                }
                
                # Calculate trend
                time_numeric = np.arange(len(regional_ts))
                valid = ~np.isnan(regional_ts.values)
                
                if np.sum(valid) > 24:  # At least 2 years of data
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        time_numeric[valid], regional_ts.values[valid]
                    )
                    
                    stats_dict.update({
                        'trend_slope': slope * 12,  # Convert to annual
                        'trend_p_value': p_value,
                        'trend_r2': r_value**2,
                        'trend_std_err': std_err * 12
                    })
                else:
                    stats_dict.update({
                        'trend_slope': np.nan,
                        'trend_p_value': np.nan, 
                        'trend_r2': np.nan,
                        'trend_std_err': np.nan
                    })
                
                # Calculate seasonal cycle
                time_dt = pd.to_datetime(self.gws_ds.time.values)
                monthly_clim = []
                
                for month in range(1, 13):
                    month_mask = time_dt.month == month
                    if np.any(month_mask):
                        month_data = regional_ts.values[month_mask]
                        monthly_clim.append(np.nanmean(month_data))
                    else:
                        monthly_clim.append(0.0)
                
                stats_dict['seasonal_cycle'] = monthly_clim
                
                self.aquifer_stats[aquifer_name] = stats_dict
                
            except Exception as e:
                print(f"    ❌ Error calculating stats for {aquifer_name}: {e}")
                continue
        
        print(f"  ✅ Statistics calculated for {len(self.aquifer_stats)} aquifers")
    
    def create_mean_gws_figure(self):
        """Create figure showing mean GWS for each aquifer."""
        print("\n🎨 Creating Mean GWS figure...")
        
        n_aquifers = len(self.aquifer_stats)
        if n_aquifers == 0:
            print("  ⚠️ No aquifers to plot")
            return
        
        # Determine subplot layout
        ncols = min(4, n_aquifers)
        nrows = (n_aquifers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        
        if n_aquifers == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Find global color scale for consistency
        all_means = []
        for stats in self.aquifer_stats.values():
            mean_data = stats['spatial_mean'].values
            valid_data = mean_data[~np.isnan(mean_data)]
            if len(valid_data) > 0:
                all_means.extend(valid_data)
        
        if all_means:
            vmin, vmax = np.percentile(all_means, [5, 95])
            if abs(vmin) > abs(vmax):
                vmax = abs(vmin)
            else:
                vmin = -abs(vmax)
        else:
            vmin, vmax = -10, 10
        
        # Plot each aquifer
        for i, (aquifer_name, stats) in enumerate(self.aquifer_stats.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get aquifer geometry and data
            gdf = self.aquifers[aquifer_name]
            mask = self.aquifer_masks[aquifer_name]
            mean_gws = stats['spatial_mean']
            
            # Set extent based on aquifer bounds
            bounds = gdf.total_bounds
            buffer = 0.5
            ax.set_extent([bounds[0]-buffer, bounds[2]+buffer, 
                          bounds[1]-buffer, bounds[3]+buffer],
                         crs=ccrs.PlateCarree())
            
            # Apply mask to mean data
            masked_mean = mean_gws.where(~np.isnan(mask))
            
            # Plot mean GWS
            im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, masked_mean,
                             cmap='RdBu_r', vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())
            
            # Add geographic features
            ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            # Highlight aquifer boundary
            gdf.boundary.plot(ax=ax, color='black', linewidth=2, 
                            transform=ccrs.PlateCarree())
            
            # Add title with statistics
            title = f"{aquifer_name.replace('_', ' ')}\n"
            title += f"Mean: {stats['mean_gws']:.2f} cm"
            ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold')
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
        
        # Hide unused subplots
        for i in range(n_aquifers, len(axes)):
            axes[i].set_visible(False)
        
        # Add colorbar
        if n_aquifers > 0:
            cbar = fig.colorbar(im, ax=axes[:n_aquifers], orientation='horizontal',
                               pad=0.05, shrink=0.8, aspect=40)
            cbar.set_label('Mean Groundwater Storage Anomaly (cm)', fontsize=FONT_SIZE)
        
        plt.suptitle('Mean Groundwater Storage by Aquifer (2003-2022)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.figures_dir / 'aquifer_mean_gws.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {output_path}")
    
    def create_trend_gws_figure(self):
        """Create figure showing trend analysis for each aquifer."""
        print("\n📈 Creating Trend GWS figure...")
        
        n_aquifers = len(self.aquifer_stats)
        if n_aquifers == 0:
            print("  ⚠️ No aquifers to plot")
            return
        
        # Determine subplot layout
        ncols = min(3, n_aquifers)
        nrows = (n_aquifers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        
        if n_aquifers == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each aquifer
        for i, (aquifer_name, stats) in enumerate(self.aquifer_stats.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get time series
            ts = stats['time_series']
            time_index = pd.to_datetime(self.gws_ds.time.values)
            
            # Plot time series
            ax.plot(time_index, ts.values, 'b-', linewidth=2, alpha=0.7, 
                   label='Monthly data')
            
            # Add 12-month rolling mean
            rolling_mean = pd.Series(ts.values, index=time_index).rolling(12, center=True).mean()
            ax.plot(time_index, rolling_mean.values, 'r-', linewidth=3, 
                   label='12-month rolling mean')
            
            # Add trend line if significant
            if not np.isnan(stats['trend_slope']):
                time_numeric = np.arange(len(ts))
                trend_line = stats['trend_slope']/12 * time_numeric + ts.values[0]
                
                # Color code by significance
                if stats['trend_p_value'] < 0.05:
                    trend_color = 'red'
                    trend_style = '-'
                    significance = '**'
                elif stats['trend_p_value'] < 0.10:
                    trend_color = 'orange' 
                    trend_style = '--'
                    significance = '*'
                else:
                    trend_color = 'gray'
                    trend_style = ':'
                    significance = ''
                
                ax.plot(time_index, trend_line, color=trend_color, 
                       linestyle=trend_style, linewidth=2,
                       label=f'Trend: {stats["trend_slope"]:.3f} cm/yr{significance}')
            
            # Formatting
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('GWS Anomaly (cm)')
            ax.set_title(f"{aquifer_name.replace('_', ' ')}", fontsize=FONT_SIZE, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {stats['mean_gws']:.2f} cm\n"
            stats_text += f"Std: {stats['std_gws']:.2f} cm\n"
            if not np.isnan(stats['trend_slope']):
                stats_text += f"R²: {stats['trend_r2']:.3f}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_aquifers, len(axes)):
            axes[i].set_visible(False)
        
        # Add legend for significance levels
        legend_elements = [
            plt.Line2D([0], [0], color='red', linestyle='-', label='p < 0.05 (**)'),
            plt.Line2D([0], [0], color='orange', linestyle='--', label='p < 0.10 (*)'),
            plt.Line2D([0], [0], color='gray', linestyle=':', label='Not significant')
        ]
        
        if n_aquifers > 0:
            fig.legend(handles=legend_elements, loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02), ncol=3)
        
        plt.suptitle('Groundwater Storage Trends by Aquifer', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.figures_dir / 'aquifer_trend_gws.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {output_path}")
    
    def create_seasonal_average_figure(self):
        """Create figure showing seasonal cycles for each aquifer."""
        print("\n📅 Creating Seasonal Average figure...")
        
        n_aquifers = len(self.aquifer_stats)
        if n_aquifers == 0:
            print("  ⚠️ No aquifers to plot")
            return
        
        # Determine subplot layout
        ncols = min(4, n_aquifers)
        nrows = (n_aquifers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        
        if n_aquifers == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        # Plot each aquifer
        for i, (aquifer_name, stats) in enumerate(self.aquifer_stats.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get seasonal cycle
            seasonal_cycle = stats['seasonal_cycle']
            
            # Create bar plot
            bars = ax.bar(range(1, 13), seasonal_cycle, 
                         color='skyblue', alpha=0.7, edgecolor='black')
            
            # Color bars based on values
            for j, (bar, value) in enumerate(zip(bars, seasonal_cycle)):
                if value > 0:
                    bar.set_color('lightcoral')
                else:
                    bar.set_color('lightblue')
            
            # Find peak months
            max_month = np.argmax(seasonal_cycle) + 1
            min_month = np.argmin(seasonal_cycle) + 1
            
            # Highlight peak months
            bars[max_month-1].set_edgecolor('red')
            bars[max_month-1].set_linewidth(3)
            bars[min_month-1].set_edgecolor('blue')
            bars[min_month-1].set_linewidth(3)
            
            # Formatting
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(months)
            ax.set_ylabel('Mean GWS Anomaly (cm)')
            ax.set_title(f"{aquifer_name.replace('_', ' ')}", fontsize=FONT_SIZE, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add peak/trough info
            info_text = f"Peak: {months[max_month-1]} ({seasonal_cycle[max_month-1]:.2f})\n"
            info_text += f"Trough: {months[min_month-1]} ({seasonal_cycle[min_month-1]:.2f})"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_aquifers, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Seasonal Cycle of Groundwater Storage by Aquifer', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.figures_dir / 'aquifer_seasonal_average.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {output_path}")
    
    def create_summary_table(self):
        """Create a summary table of all aquifer statistics."""
        print("\n📋 Creating summary table...")
        
        # Prepare data for table
        table_data = []
        
        for aquifer_name, stats in self.aquifer_stats.items():
            row = {
                'Aquifer': aquifer_name.replace('_', ' '),
                'N_Pixels': stats['n_pixels'],
                'Mean_GWS (cm)': f"{stats['mean_gws']:.2f}",
                'Std_GWS (cm)': f"{stats['std_gws']:.2f}",
                'Trend (cm/yr)': f"{stats['trend_slope']:.3f}" if not np.isnan(stats['trend_slope']) else 'N/A',
                'Trend_p_value': f"{stats['trend_p_value']:.3f}" if not np.isnan(stats['trend_p_value']) else 'N/A',
                'R²': f"{stats['trend_r2']:.3f}" if not np.isnan(stats['trend_r2']) else 'N/A'
            }
            table_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_path = self.figures_dir / 'aquifer_summary_statistics.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"  💾 Saved summary table: {csv_path}")
        
        return df
    
    def run_all_analyses(self):
        """Run all analyses and create all figures."""
        print("\n🚀 RUNNING ALL AQUIFER ANALYSES")
        print("="*50)
        
        # Calculate statistics
        self.calculate_aquifer_statistics()
        
        # Create all figures
        self.create_mean_gws_figure()
        self.create_trend_gws_figure() 
        self.create_seasonal_average_figure()
        
        # Create summary table
        summary_df = self.create_summary_table()
        
        print(f"\n✅ All analyses complete!")
        print(f"📁 Results saved to: {self.figures_dir}")
        print(f"📊 Analyzed {len(self.aquifer_stats)} aquifers")
        
        return summary_df


def main():
    """Main function to run aquifer analysis."""
    print("🌊 INDIVIDUAL AQUIFER ANALYSIS FOR GRACE GROUNDWATER DATA")
    print("="*60)
    
    # Initialize analyzer
    analyzer = AquiferAnalyzer(base_dir=".")
    
    # Run all analyses
    summary_df = analyzer.run_all_analyses()
    
    # Print summary
    print("\n📋 AQUIFER SUMMARY:")
    print("-" * 30)
    print(summary_df.to_string(index=False))
    
    return analyzer


if __name__ == "__main__":
    main() 