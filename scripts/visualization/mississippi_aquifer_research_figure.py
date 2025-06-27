#!/usr/bin/env python3
"""
Research Paper-Grade Figure: Mississippi River Basin Aquifers Analysis
=====================================================================

This script creates a comprehensive research figure with:
1. Map Panel: US basemap with Mississippi River Basin and numbered aquifers
2. Heatmap Panel: Quantitative metrics for each aquifer

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
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib import cm
import matplotlib.gridspec as gridspec

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
FONT_SIZE = 12
TITLE_FONT_SIZE = 14


class MississippiAquiferResearchFigure:
    """Create research paper-grade figure for Mississippi River Basin aquifers."""
    
    def __init__(self, base_dir="."):
        """
        Initialize the figure creator.
        
        Parameters:
        -----------
        base_dir : str
            Base directory of the project
        """
        self.base_dir = Path(base_dir)
        self.figures_dir = self.base_dir / "figures" / "research_paper"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths to data
        self.aquifer_dir = self.base_dir / "data/shapefiles/processed/aquifers_mississippi"
        self.basin_shapefile = self.base_dir / "data/shapefiles/processed/mississippi_river_basin.shp"
        self.states_shapefile = self.base_dir / "data/shapefiles/states/cb_2023_us_state_500k.shp"
        
        # Load data
        self._load_groundwater_data()
        self._load_shapefiles()
        self._calculate_aquifer_metrics()
    
    def _load_groundwater_data(self):
        """Load groundwater storage data."""
        print("📦 Loading groundwater data...")
        
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
        
        print(f"     Time range: {str(self.gws_ds.time.values[0])[:10]} to {str(self.gws_ds.time.values[-1])[:10]}")
    
    def _load_shapefiles(self):
        """Load all required shapefiles."""
        print("🗺️ Loading shapefiles...")
        
        # Load Mississippi River Basin
        if self.basin_shapefile.exists():
            self.basin_gdf = gpd.read_file(self.basin_shapefile)
            if self.basin_gdf.crs != 'EPSG:4326':
                self.basin_gdf = self.basin_gdf.to_crs('EPSG:4326')
            print(f"  ✅ Basin shapefile loaded")
        else:
            raise FileNotFoundError(f"Mississippi Basin shapefile not found: {self.basin_shapefile}")
        
        # Load US states for basemap
        if self.states_shapefile.exists():
            self.states_gdf = gpd.read_file(self.states_shapefile)
            if self.states_gdf.crs != 'EPSG:4326':
                self.states_gdf = self.states_gdf.to_crs('EPSG:4326')
            print(f"  ✅ States shapefile loaded")
        else:
            print("  ⚠️ States shapefile not found, will use cartopy features")
            self.states_gdf = None
        
        # Load aquifers in Mississippi Basin
        shp_files = list(self.aquifer_dir.glob("*.shp"))
        print(f"  Found {len(shp_files)} aquifer shapefiles")
        
        self.aquifers = {}
        self.aquifer_masks = {}
        
        # Define aquifer display order and colors
        self.aquifer_order = [
            'High_Plains_aquifer_mississippi',
            'Mississippi_River_Valley_alluvial_aquifer_mississippi', 
            'Mississippi_embayment_aquifer_system_mississippi',
            'Coastal_lowlands_aquifer_system_mississippi'
        ]
        
        self.aquifer_colors = {
            'High_Plains_aquifer_mississippi': '#1f77b4',  # Blue
            'Mississippi_River_Valley_alluvial_aquifer_mississippi': '#ff7f0e',  # Orange
            'Mississippi_embayment_aquifer_system_mississippi': '#2ca02c',  # Green
            'Coastal_lowlands_aquifer_system_mississippi': '#d62728'  # Red
        }
        
        self.aquifer_names = {
            'High_Plains_aquifer_mississippi': 'High Plains',
            'Mississippi_River_Valley_alluvial_aquifer_mississippi': 'Mississippi Valley Alluvial',
            'Mississippi_embayment_aquifer_system_mississippi': 'Mississippi Embayment',
            'Coastal_lowlands_aquifer_system_mississippi': 'Coastal Lowlands'
        }
        
        # Load each aquifer
        for shp_file in shp_files:
            aquifer_id = shp_file.stem
            
            if aquifer_id in self.aquifer_order:
                try:
                    gdf = gpd.read_file(shp_file)
                    if gdf.crs != 'EPSG:4326':
                        gdf = gdf.to_crs('EPSG:4326')
                    
                    # Create mask
                    mask = self._create_aquifer_mask(gdf)
                    
                    if mask is not None and not np.all(np.isnan(mask)):
                        self.aquifers[aquifer_id] = gdf
                        self.aquifer_masks[aquifer_id] = mask
                        
                        n_pixels = np.sum(~np.isnan(mask))
                        print(f"    ✅ {self.aquifer_names[aquifer_id]}: {n_pixels} pixels")
                
                except Exception as e:
                    print(f"    ❌ Error loading {aquifer_id}: {e}")
        
        print(f"  ✅ Loaded {len(self.aquifers)} aquifers for analysis")
    
    def _create_aquifer_mask(self, gdf):
        """Create a mask for the aquifer region."""
        if regionmask is None:
            return self._create_basic_mask(gdf)
        
        try:
            # Create region using regionmask
            if len(gdf) == 1:
                region_geom = gdf.geometry.iloc[0]
            else:
                try:
                    region_geom = gdf.union_all()
                except AttributeError:
                    region_geom = gdf.geometry.unary_union
            
            region = regionmask.Regions([region_geom], names=['aquifer'])
            mask = region.mask(self.gws_ds.lon, self.gws_ds.lat)
            
            return mask
            
        except Exception as e:
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
    
    def _calculate_aquifer_metrics(self):
        """Calculate comprehensive metrics for each aquifer."""
        print("📊 Calculating aquifer metrics...")
        
        self.metrics = {}
        time_dt = pd.to_datetime(self.gws_ds.time.values)
        
        for aquifer_id, mask in self.aquifer_masks.items():
            try:
                # Apply mask to groundwater data
                gws_masked = self.gws_ds.groundwater.where(~np.isnan(mask))
                
                # Calculate regional time series
                regional_ts = gws_masked.mean(dim=['lat', 'lon'])
                
                # Basic statistics
                mean_gws = float(regional_ts.mean())
                std_gws = float(regional_ts.std())
                min_gws = float(regional_ts.min())
                max_gws = float(regional_ts.max())
                
                # Trend analysis
                time_numeric = np.arange(len(regional_ts))
                valid = ~np.isnan(regional_ts.values)
                
                if np.sum(valid) > 24:  # At least 2 years
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        time_numeric[valid], regional_ts.values[valid]
                    )
                    
                    trend_slope = slope * 12  # Convert to annual
                    trend_significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                else:
                    trend_slope = np.nan
                    trend_significance = 'ns'
                    p_value = np.nan
                    r_value = np.nan
                
                # Seasonal cycle
                monthly_clim = []
                for month in range(1, 13):
                    month_mask = time_dt.month == month
                    if np.any(month_mask):
                        month_data = regional_ts.values[month_mask]
                        monthly_clim.append(np.nanmean(month_data))
                    else:
                        monthly_clim.append(0.0)
                
                seasonal_amplitude = max(monthly_clim) - min(monthly_clim)
                peak_month = np.argmax(monthly_clim) + 1
                trough_month = np.argmin(monthly_clim) + 1
                
                # Find extreme years
                annual_means = []
                years = []
                for year in range(2003, 2023):
                    year_mask = time_dt.year == year
                    if np.any(year_mask):
                        year_data = regional_ts.values[year_mask]
                        annual_means.append(np.nanmean(year_data))
                        years.append(year)
                
                if annual_means:
                    max_drought_year = years[np.argmin(annual_means)]  # Most negative = drought
                    max_wet_year = years[np.argmax(annual_means)]      # Most positive = wet
                    drought_severity = min(annual_means)
                    wet_severity = max(annual_means)
                else:
                    max_drought_year = np.nan
                    max_wet_year = np.nan
                    drought_severity = np.nan
                    wet_severity = np.nan
                
                # Recovery rate (slope after 2012 drought if applicable)
                post_2012_mask = time_dt.year >= 2013
                if np.any(post_2012_mask):
                    post_2012_data = regional_ts.values[post_2012_mask]
                    post_2012_time = np.arange(len(post_2012_data))
                    
                    if len(post_2012_data) > 12:
                        recovery_slope, _, recovery_r, recovery_p, _ = stats.linregress(
                            post_2012_time, post_2012_data
                        )
                        recovery_rate = recovery_slope * 12
                    else:
                        recovery_rate = np.nan
                        recovery_p = np.nan
                else:
                    recovery_rate = np.nan
                    recovery_p = np.nan
                
                # Store all metrics
                self.metrics[aquifer_id] = {
                    'mean_gws': mean_gws,
                    'std_gws': std_gws,
                    'min_gws': min_gws,
                    'max_gws': max_gws,
                    'trend_slope': trend_slope,
                    'trend_significance': trend_significance,
                    'trend_p_value': p_value,
                    'trend_r2': r_value**2 if not np.isnan(r_value) else np.nan,
                    'seasonal_amplitude': seasonal_amplitude,
                    'peak_month': peak_month,
                    'trough_month': trough_month,
                    'max_drought_year': max_drought_year,
                    'max_wet_year': max_wet_year,
                    'drought_severity': drought_severity,
                    'wet_severity': wet_severity,
                    'recovery_rate': recovery_rate,
                    'recovery_significance': '**' if recovery_p < 0.01 else '*' if recovery_p < 0.05 else 'ns' if not np.isnan(recovery_p) else 'ns',
                    'n_pixels': int(np.sum(~np.isnan(mask))),
                    'time_series': regional_ts
                }
                
                print(f"  ✅ {self.aquifer_names[aquifer_id]}: metrics calculated")
                
            except Exception as e:
                print(f"  ❌ Error calculating metrics for {aquifer_id}: {e}")
        
        print(f"  ✅ Metrics calculated for {len(self.metrics)} aquifers")
    
    def create_research_figure(self):
        """Create the main research paper-grade figure."""
        print("\n🎨 Creating research paper-grade figure...")
        
        # Create figure with compact layout: AB on top, C on bottom
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.8, 1.4], width_ratios=[1.4, 1.2], 
                              hspace=0.25, wspace=0.2)
        
        # Panel A: Map
        self._create_map_panel(fig, gs[0, 0])
        
        # Panel B: Individual trend plots (was Panel C)
        self._create_trend_plots_panel(fig, gs[0, 1])
        
        # Panel C: Time series overview (spans full width, was Panel D)
        self._create_timeseries_panel(fig, gs[1, :])
        
        # Save figure
        output_path = self.figures_dir / 'mississippi_aquifers_research_figure.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        
        # Also save as PDF for publications
        pdf_path = self.figures_dir / 'mississippi_aquifers_research_figure.pdf'
        plt.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
        
        print(f"  💾 Saved: {output_path}")
        print(f"  💾 Saved: {pdf_path}")
        
        plt.close()
        
        # Create supplementary table
        self._create_metrics_table()
    
    def _create_map_panel(self, fig, gs_position):
        """Create the map panel showing US with numbered aquifers."""
        ax = fig.add_subplot(gs_position, projection=ccrs.AlbersEqualArea(
            central_longitude=-96, central_latitude=39))
        
        # Set extent to cover US
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        
        # Add base features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Add states
        if self.states_gdf is not None:
            self.states_gdf.boundary.plot(ax=ax, linewidth=0.5, color='gray', 
                                        alpha=0.7, transform=ccrs.PlateCarree())
        else:
            ax.add_feature(cfeature.STATES, linewidth=0.5, color='gray', alpha=0.7)
        
        # Add major rivers
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, color='blue', alpha=0.6)
        
        # Plot Mississippi River Basin boundary
        self.basin_gdf.boundary.plot(ax=ax, color='black', linewidth=3, 
                                   alpha=0.8, transform=ccrs.PlateCarree())
        
        # Plot and number aquifers with improved positioning
        legend_elements = []
        

        
        for i, aquifer_id in enumerate(self.aquifer_order):
            if aquifer_id in self.aquifers:
                gdf = self.aquifers[aquifer_id]
                color = self.aquifer_colors[aquifer_id]
                name = self.aquifer_names[aquifer_id]
                
                # Plot aquifer
                gdf.plot(ax=ax, color=color, alpha=0.7, edgecolor='black', 
                        linewidth=1, transform=ccrs.PlateCarree())
                
                # Add to legend (no numbers needed since colors distinguish aquifers)
                legend_elements.append(mpatches.Patch(color=color, label=name))
        
        # Add legend with better formatting
        legend = ax.legend(handles=legend_elements, loc='lower left', 
                          bbox_to_anchor=(0, 0), frameon=True, fancybox=True, shadow=True,
                          fontsize=FONT_SIZE, title_fontsize=FONT_SIZE+1)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        
        # Add title
        ax.set_title('A. Mississippi River Basin Major Aquifers', 
                    fontsize=TITLE_FONT_SIZE+1, fontweight='bold', pad=15)
        
        # Add north arrow and scale (simple version)
        ax.text(0.95, 0.95, 'N ↑', transform=ax.transAxes, fontsize=12, 
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_heatmap_panel(self, fig, gs_position):
        """Create the heatmap panel showing quantitative metrics."""
        ax = fig.add_subplot(gs_position)
        
        # Prepare data for heatmap
        metrics_data = []
        aquifer_labels = []
        
        for aquifer_id in self.aquifer_order:
            if aquifer_id in self.metrics:
                metrics = self.metrics[aquifer_id]
                aquifer_num = self.aquifer_order.index(aquifer_id) + 1
                aquifer_labels.append(f"{aquifer_num}")
                
                # Selected metrics for heatmap
                row_data = [
                    metrics['mean_gws'],
                    metrics['std_gws'], 
                    metrics['trend_slope'],
                    metrics['seasonal_amplitude'],
                    metrics['drought_severity'],
                    metrics['wet_severity'],
                    metrics['recovery_rate']
                ]
                
                metrics_data.append(row_data)
        
        metrics_data = np.array(metrics_data)
        
        # Metric labels
        metric_labels = [
            'Mean GWS\n(cm)',
            'Std Dev\n(cm)', 
            'Trend\n(cm/yr)',
            'Seasonal\nAmplitude (cm)',
            'Drought\nSeverity (cm)',
            'Wet\nSeverity (cm)',
            'Recovery\nRate (cm/yr)'
        ]
        
        # Normalize data for better visualization
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            metrics_normalized = scaler.fit_transform(metrics_data)
        except ImportError:
            # Simple z-score normalization if sklearn not available
            metrics_normalized = (metrics_data - np.nanmean(metrics_data, axis=0)) / np.nanstd(metrics_data, axis=0)
        
        # Create heatmap with smaller box height
        im = ax.imshow(metrics_normalized, cmap='RdBu_r', aspect=2.0, 
                      vmin=-2, vmax=2)
        
        # Set ticks and labels with vertical text and better spacing
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=90, ha='center', fontsize=FONT_SIZE-1,
                          verticalalignment='bottom')
        ax.set_yticks(range(len(aquifer_labels)))
        ax.set_yticklabels(aquifer_labels, fontsize=FONT_SIZE)
        
        # Add more space for x-axis labels
        ax.margins(x=5)
        
        # Add text annotations with actual values (vertical text)
        for i in range(len(aquifer_labels)):
            for j in range(len(metric_labels)):
                value = metrics_data[i, j]
                if not np.isnan(value):
                    if j == 2:  # Trend - add significance
                        aquifer_id = self.aquifer_order[i]
                        sig = self.metrics[aquifer_id]['trend_significance']
                        text = f'{value:.2f}{sig}'
                    elif j in [4, 5]:  # Years
                        text = f'{value:.1f}'
                    else:
                        text = f'{value:.2f}'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           fontsize=10, fontweight='bold', rotation=90,
                           color='white' if abs(metrics_normalized[i, j]) > 1 else 'black')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Standardized Value', fontsize=FONT_SIZE)
        
        ax.set_title('B. Aquifer Metrics Heatmap', 
                    fontsize=TITLE_FONT_SIZE, fontweight='bold')
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(bottom=3)
        
        # Add grid
        ax.set_xticks(np.arange(len(metric_labels)), minor=True)
        ax.set_yticks(np.arange(len(aquifer_labels)), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    def _create_trend_plots_panel(self, fig, gs_position):
        """Create spatial trend maps for each aquifer (Panel C)."""
        # Create 2x2 grid for spatial display with room for shared colorbar
        gs_trend = gridspec.GridSpecFromSubplotSpec(3, 2, gs_position, 
                                                   height_ratios=[1, 1, 0.08], 
                                                   hspace=0.3, wspace=0.15)
        
        # Calculate pixel-wise trends for the entire domain
        print("  📈 Calculating pixel-wise trends for spatial display...")
        trend_results = self.calculate_pixel_trends(self.gws_ds.groundwater)
        
        # Calculate global color scale across all aquifers for consistency
        all_valid_trends = []
        for aquifer_id in self.aquifer_order:
            if aquifer_id in self.aquifers:
                mask = self.aquifer_masks[aquifer_id]
                masked_trend = np.where(~np.isnan(mask), trend_results['trend'], np.nan)
                valid_trends = masked_trend[~np.isnan(masked_trend)]
                if len(valid_trends) > 0:
                    all_valid_trends.extend(valid_trends)
        
        if all_valid_trends:
            global_vmax = np.percentile(np.abs(all_valid_trends), 95)
            if global_vmax == 0:
                global_vmax = 1
            global_vmin = -global_vmax
        else:
            global_vmin, global_vmax = -1, 1
        
        # Store for colorbar
        trend_im = None
        
        for i, aquifer_id in enumerate(self.aquifer_order):
            if aquifer_id in self.aquifers:
                # Get subplot position (2x2 grid)
                row = i // 2
                col = i % 2
                ax = fig.add_subplot(gs_trend[row, col], projection=ccrs.PlateCarree())
                
                gdf = self.aquifers[aquifer_id]
                mask = self.aquifer_masks[aquifer_id]
                name = self.aquifer_names[aquifer_id]
                
                # Apply aquifer mask to trend data
                masked_trend = np.where(~np.isnan(mask), trend_results['trend'], np.nan)
                masked_pvalue = np.where(~np.isnan(mask), trend_results['p_value'], np.nan)
                
                # Set extent based on aquifer bounds with buffer
                bounds = gdf.total_bounds
                buffer = 0.5
                ax.set_extent([bounds[0]-buffer, bounds[2]+buffer, 
                              bounds[1]-buffer, bounds[3]+buffer],
                             crs=ccrs.PlateCarree())
                
                # Plot trend with global color scale
                im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, masked_trend,
                                  cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax,
                                  transform=ccrs.PlateCarree())
                
                # Store first image for shared colorbar
                if trend_im is None:
                    trend_im = im
                
                # Add significance hatching
                lon_mesh, lat_mesh = np.meshgrid(self.gws_ds.lon, self.gws_ds.lat)
                
                # Create significance masks
                sig_01 = masked_pvalue < 0.01
                sig_05 = (masked_pvalue >= 0.01) & (masked_pvalue < 0.05)
                sig_10 = (masked_pvalue >= 0.05) & (masked_pvalue < 0.10)
                
                # Add hatching for different significance levels
                if np.any(sig_01 & ~np.isnan(sig_01)):
                    ax.contourf(lon_mesh, lat_mesh, sig_01.astype(float), 
                               levels=[0.5, 1.5], colors='none', 
                               hatches=['///'], transform=ccrs.PlateCarree())
                
                if np.any(sig_05 & ~np.isnan(sig_05)):
                    ax.contourf(lon_mesh, lat_mesh, sig_05.astype(float), 
                               levels=[0.5, 1.5], colors='none', 
                               hatches=[r'\\'], transform=ccrs.PlateCarree())
                
                if np.any(sig_10 & ~np.isnan(sig_10)):
                    ax.contourf(lon_mesh, lat_mesh, sig_10.astype(float), 
                               levels=[0.5, 1.5], colors='none', 
                               hatches=['...'], transform=ccrs.PlateCarree())
                
                # Add geographic features
                ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Highlight aquifer boundary
                gdf.boundary.plot(ax=ax, color='black', linewidth=2, 
                                transform=ccrs.PlateCarree())
                
                # Add gridlines with better formatting
                gl = ax.gridlines(draw_labels=True, alpha=0.6, linewidth=0.5,
                                 xlabel_style={'size': FONT_SIZE-2},
                                 ylabel_style={'size': FONT_SIZE-2})
                gl.top_labels = False
                gl.right_labels = False
                if row == 0:  # Top row
                    gl.bottom_labels = False
                if col == 1:  # Right column
                    gl.left_labels = False
                
                # Add title with aquifer name and trend info
                metrics = self.metrics[aquifer_id]
                title = f"{name}\n"
                title += f"Regional: {metrics['trend_slope']:.3f} cm/yr"
                if metrics['trend_significance'] != 'ns':
                    title += f" ({metrics['trend_significance']})"
                
                ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold', pad=8)
        
        # Add shared colorbar at bottom
        cbar_ax = fig.add_subplot(gs_trend[2, :])
        cbar = plt.colorbar(trend_im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Groundwater Storage Trend (cm/year)', fontsize=FONT_SIZE+1, fontweight='bold')
        cbar.ax.tick_params(labelsize=FONT_SIZE)
        
        # Add panel title
        fig.text(0.75, 0.92, 'B. Spatial Trends by Aquifer', 
                fontsize=TITLE_FONT_SIZE+1, fontweight='bold', ha='center')
    
    def calculate_pixel_trends(self, data_array, min_years=5):
        """Calculate linear trends for each pixel with statistical significance."""
        from tqdm import tqdm
        
        # Get dimensions
        n_time = len(data_array.time)
        n_lat = len(data_array.lat)
        n_lon = len(data_array.lon)
        
        # Initialize output arrays
        trend = np.full((n_lat, n_lon), np.nan)
        p_value = np.full((n_lat, n_lon), np.nan)
        std_error = np.full((n_lat, n_lon), np.nan)
        n_valid = np.full((n_lat, n_lon), 0)
        
        # Time array for regression (months since start)
        time_numeric = np.arange(n_time)
        
        # Calculate trends pixel by pixel
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series for this pixel
                ts = data_array[:, i, j].values
                
                # Check for sufficient valid data
                valid_mask = ~np.isnan(ts)
                n_valid_points = np.sum(valid_mask)
                
                if n_valid_points >= min_years * 12:  # Monthly data
                    # Perform linear regression
                    try:
                        slope, intercept, r_value, p_val, std_err = stats.linregress(
                            time_numeric[valid_mask], ts[valid_mask]
                        )
                        
                        # Convert to annual trend (slope is per month)
                        trend[i, j] = slope * 12
                        p_value[i, j] = p_val
                        std_error[i, j] = std_err * 12
                        n_valid[i, j] = n_valid_points
                        
                    except Exception:
                        pass
        
        return {
            'trend': trend,
            'p_value': p_value,
            'std_error': std_error,
            'n_valid': n_valid
        }
    
    def _create_timeseries_panel(self, fig, gs_position):
        """Create 2x2 individual time series subplots for each aquifer."""
        # Create 2x2 grid for individual time series
        gs_ts = gridspec.GridSpecFromSubplotSpec(2, 2, gs_position, 
                                                hspace=0.25, wspace=0.2)
        
        time_index = pd.to_datetime(self.gws_ds.time.values)
        
        for i, aquifer_id in enumerate(self.aquifer_order):
            if aquifer_id in self.metrics:
                # Get subplot position
                row = i // 2
                col = i % 2
                ax = fig.add_subplot(gs_ts[row, col])
                
                ts = self.metrics[aquifer_id]['time_series']
                color = self.aquifer_colors[aquifer_id]
                name = self.aquifer_names[aquifer_id]
                metrics = self.metrics[aquifer_id]
                
                # Plot monthly data
                ax.plot(time_index, ts.values, color=color, linewidth=1.5, 
                       alpha=0.7, label='Monthly data')
                
                # Add 12-month rolling mean
                rolling_mean = pd.Series(ts.values, index=time_index).rolling(12, center=True).mean()
                ax.plot(time_index, rolling_mean.values, color=color, linewidth=3, 
                       alpha=1.0, label='12-month rolling mean')
                
                # Add trend line if significant
                if not np.isnan(metrics['trend_slope']):
                    time_numeric = np.arange(len(ts))
                    trend_line = metrics['trend_slope']/12 * time_numeric + ts.values[0]
                    
                    # Color code by significance
                    if metrics['trend_p_value'] < 0.05:
                        trend_color = 'red'
                        trend_style = '-'
                        significance = '**' if metrics['trend_p_value'] < 0.01 else '*'
                    else:
                        trend_color = 'gray'
                        trend_style = '--'
                        significance = 'ns'
                    
                    ax.plot(time_index, trend_line, color=trend_color, 
                           linestyle=trend_style, linewidth=2.5, alpha=0.8,
                           label=f'Trend: {metrics["trend_slope"]:.3f} cm/yr ({significance})')
                
                # Formatting for each subplot
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                ax.set_title(f'{name}', fontsize=FONT_SIZE, fontweight='bold')
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Mark specific drought and wet years for this aquifer
                if not np.isnan(metrics['max_drought_year']):
                    drought_year = int(metrics['max_drought_year'])
                    ax.axvline(pd.Timestamp(f'{drought_year}-06-01'), color='darkred', 
                              linestyle='-', alpha=0.8, linewidth=2.5,
                              label=f'Drought: {drought_year}')
                    
                    # Add drought year shading (±1 year around drought)
                    drought_start = pd.Timestamp(f'{drought_year-1}-01-01')
                    drought_end = pd.Timestamp(f'{drought_year+1}-12-31')
                    ax.axvspan(drought_start, drought_end, alpha=0.1, color='red', zorder=0)
                
                if not np.isnan(metrics['max_wet_year']):
                    wet_year = int(metrics['max_wet_year'])
                    ax.axvline(pd.Timestamp(f'{wet_year}-06-01'), color='darkblue', 
                              linestyle='-', alpha=0.8, linewidth=2.5,
                              label=f'Wet: {wet_year}')
                    
                    # Add wet year shading (±1 year around wet year)
                    wet_start = pd.Timestamp(f'{wet_year-1}-01-01')
                    wet_end = pd.Timestamp(f'{wet_year+1}-12-31')
                    ax.axvspan(wet_start, wet_end, alpha=0.1, color='blue', zorder=0)
                
                # Set axis limits
                ax.set_xlim(pd.Timestamp('2003-01-01'), pd.Timestamp('2023-01-01'))
                
                # Add statistics text box (drought/wet years now shown as lines)
                stats_text = f"Mean: {metrics['mean_gws']:.2f} cm\n"
                stats_text += f"Std: {metrics['std_gws']:.2f} cm\n"
                stats_text += f"Range: {metrics['min_gws']:.1f} to {metrics['max_gws']:.1f} cm"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=FONT_SIZE-1, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
                
                # Only add x-label to bottom plots
                if row == 1:
                    ax.set_xlabel('Year', fontsize=FONT_SIZE+1, fontweight='bold')
                
                # Only add y-label to left plots
                if col == 0:
                    ax.set_ylabel('GWS Anomaly (cm)', fontsize=FONT_SIZE+1, fontweight='bold')
                
                # Add legend to bottom-right plot for better space usage
                if i == 3:  # Bottom-right position
                    # Only show main legend items (trend, monthly, rolling mean)
                    handles, labels = ax.get_legend_handles_labels()
                    # Filter to show only the first 3 items (monthly, rolling, trend)
                    main_handles = handles[:3] if len(handles) >= 3 else handles
                    main_labels = labels[:3] if len(labels) >= 3 else labels
                    
                    legend = ax.legend(main_handles, main_labels, loc='upper right', 
                                      fontsize=FONT_SIZE-1, frameon=True,
                                      fancybox=True, shadow=True)
                    legend.get_frame().set_alpha(0.9)
                
                # Adjust tick labels
                ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
                
                # Rotate x-axis labels for better fit
                if row == 1:  # Bottom row
                    ax.tick_params(axis='x', rotation=45)
        
        # Add overall title for Panel C
        fig.text(0.5, 0.48, 'C. Individual Aquifer Time Series (2003-2022)', 
                fontsize=TITLE_FONT_SIZE+1, fontweight='bold', ha='center')
    
    def _create_metrics_table(self):
        """Create supplementary table with detailed metrics."""
        print("📋 Creating metrics table...")
        
        # Prepare table data
        table_data = []
        
        for aquifer_id in self.aquifer_order:
            if aquifer_id in self.metrics:
                metrics = self.metrics[aquifer_id]
                aquifer_num = self.aquifer_order.index(aquifer_id) + 1
                name = self.aquifer_names[aquifer_id]
                
                row = {
                    'ID': aquifer_num,
                    'Aquifer': name,
                    'N_Pixels': metrics['n_pixels'],
                    'Mean_GWS_cm': f"{metrics['mean_gws']:.2f}",
                    'Std_GWS_cm': f"{metrics['std_gws']:.2f}",
                    'Trend_cm_per_yr': f"{metrics['trend_slope']:.3f}" if not np.isnan(metrics['trend_slope']) else 'N/A',
                    'Trend_Significance': metrics['trend_significance'],
                    'Trend_R2': f"{metrics['trend_r2']:.3f}" if not np.isnan(metrics['trend_r2']) else 'N/A',
                    'Seasonal_Amplitude_cm': f"{metrics['seasonal_amplitude']:.2f}",
                    'Peak_Month': metrics['peak_month'],
                    'Trough_Month': metrics['trough_month'],
                    'Max_Drought_Year': int(metrics['max_drought_year']) if not np.isnan(metrics['max_drought_year']) else 'N/A',
                    'Drought_Severity_cm': f"{metrics['drought_severity']:.2f}" if not np.isnan(metrics['drought_severity']) else 'N/A',
                    'Max_Wet_Year': int(metrics['max_wet_year']) if not np.isnan(metrics['max_wet_year']) else 'N/A',
                    'Wet_Severity_cm': f"{metrics['wet_severity']:.2f}" if not np.isnan(metrics['wet_severity']) else 'N/A',
                    'Recovery_Rate_cm_per_yr': f"{metrics['recovery_rate']:.3f}" if not np.isnan(metrics['recovery_rate']) else 'N/A',
                    'Recovery_Significance': metrics['recovery_significance']
                }
                
                table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_path = self.figures_dir / 'mississippi_aquifers_metrics_table.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"  💾 Saved metrics table: {csv_path}")
        
        return df
    
    def run_analysis(self):
        """Run the complete analysis and create the research figure."""
        print("\n🚀 CREATING MISSISSIPPI RIVER BASIN AQUIFER RESEARCH FIGURE")
        print("="*70)
        
        # Create the main research figure
        self.create_research_figure()
        
        print(f"\n✅ Research figure created successfully!")
        print(f"📁 Results saved to: {self.figures_dir}")
        print(f"📊 Analyzed {len(self.metrics)} aquifers in Mississippi River Basin")
        
        # Print summary
        print("\n📋 AQUIFER SUMMARY:")
        print("-" * 40)
        for i, aquifer_id in enumerate(self.aquifer_order):
            if aquifer_id in self.metrics:
                metrics = self.metrics[aquifer_id]
                name = self.aquifer_names[aquifer_id]
                print(f"{i+1}. {name}:")
                print(f"   Mean GWS: {metrics['mean_gws']:.2f} cm")
                print(f"   Trend: {metrics['trend_slope']:.3f} cm/yr ({metrics['trend_significance']})")
                print(f"   Drought year: {int(metrics['max_drought_year']) if not np.isnan(metrics['max_drought_year']) else 'N/A'}")
                print()


def main():
    """Main function to create the research figure."""
    print("📰 MISSISSIPPI RIVER BASIN AQUIFER RESEARCH FIGURE")
    print("="*55)
    
    # Create the research figure
    creator = MississippiAquiferResearchFigure(base_dir=".")
    creator.run_analysis()
    
    return creator


if __name__ == "__main__":
    main() 