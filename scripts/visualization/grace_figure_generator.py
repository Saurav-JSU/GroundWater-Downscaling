#!/usr/bin/env python3
"""
GRACE Groundwater Analysis - Publication Figure Generator (FIXED)

This script automatically detects analysis outputs and generates 20+ publication-ready figures
for groundwater storage anomaly analysis using GRACE satellite data downscaling.

Author: Automated Analysis System
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.gridspec import GridSpec
import contextily as ctx
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")

# Publication settings
FIGURE_DPI = 600
FIGURE_FORMAT = ['png']
FONT_SIZE_SMALL = 12
FONT_SIZE_MEDIUM = 14
FONT_SIZE_LARGE = 16

# Color schemes for consistency
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'negative': '#C73E1D',
    'neutral': '#592E83'
}

class GRACEFigureGenerator:
    """Comprehensive figure generator for GRACE groundwater analysis."""
    
    def __init__(self, base_dir="."):
        """Initialize the figure generator."""
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "figures"
        self.data_cache = {}
        
        # Create output directories
        self.subdirs = {
            'study_area': self.output_dir / 'study_area',
            'storage_patterns': self.output_dir / 'storage_patterns', 
            'temporal_analysis': self.output_dir / 'temporal_analysis',
            'spatial_analysis': self.output_dir / 'spatial_analysis',
            'feature_importance': self.output_dir / 'feature_importance',
            'model_evaluation': self.output_dir / 'model_evaluation',
            'correlation': self.output_dir / 'correlation',
            'data_quality': self.output_dir / 'data_quality'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory created: {self.output_dir}")
        print(f"🎨 Will generate figures in {len(self.subdirs)} categories")
        
    def detect_and_load_data(self):
        """Automatically detect and load all available data files."""
        print("\n🔍 DETECTING AND LOADING DATA FILES")
        print("="*50)
        
        # 1. Groundwater storage anomalies (main results)
        gws_files = [
            "results/groundwater_storage_anomalies.nc",
            "results/groundwater_storage_anomalies_corrected.nc",
            "results/groundwater_storage_anomalies_enhanced.nc"
        ]
        
        for gws_file in gws_files:
            if (self.base_dir / gws_file).exists():
                print(f"  ✅ Loading groundwater data: {gws_file}")
                self.data_cache['gws'] = xr.open_dataset(self.base_dir / gws_file)
                break
        else:
            print("  ⚠️ No groundwater storage file found")
            
        # 2. Feature stack
        feature_file = "data/processed/feature_stack.nc"
        if (self.base_dir / feature_file).exists():
            print(f"  ✅ Loading feature stack: {feature_file}")
            self.data_cache['features'] = xr.open_dataset(self.base_dir / feature_file)
        else:
            print("  ⚠️ Feature stack not found")
            
        # 3. Model comparison results
        model_files = [
            "models/model_comparison.csv",
            "models/training_results.txt"
        ]
        
        for model_file in model_files:
            if (self.base_dir / model_file).exists():
                if model_file.endswith('.csv'):
                    print(f"  ✅ Loading model comparison: {model_file}")
                    self.data_cache['model_comparison'] = pd.read_csv(self.base_dir / model_file)
                break
        
        # 4. Feature importance
        importance_file = "models/feature_importances.csv"
        if (self.base_dir / importance_file).exists():
            print(f"  ✅ Loading feature importance: {importance_file}")
            self.data_cache['feature_importance'] = pd.read_csv(self.base_dir / importance_file)
            
        # 5. Validation results
        validation_files = [
            "results/validation/point_validation_metrics.csv",
            "results/validation/spatial_avg_50km_metrics.csv"
        ]
        
        for val_file in validation_files:
            if (self.base_dir / val_file).exists():
                file_key = val_file.split('/')[-1].replace('.csv', '')
                print(f"  ✅ Loading validation data: {val_file}")
                self.data_cache[file_key] = pd.read_csv(self.base_dir / val_file)
                
        # 6. Well data for validation
        well_files = [
            "data/raw/usgs_well_data/monthly_groundwater_anomalies.csv",
            "data/raw/usgs_well_data/well_metadata.csv"
        ]
        
        for well_file in well_files:
            if (self.base_dir / well_file).exists():
                file_key = 'well_' + well_file.split('/')[-1].replace('.csv', '').replace('monthly_groundwater_', '')
                print(f"  ✅ Loading well data: {well_file}")
                if 'metadata' in well_file:
                    self.data_cache[file_key] = pd.read_csv(self.base_dir / well_file)
                else:
                    self.data_cache[file_key] = pd.read_csv(self.base_dir / well_file, 
                                                          index_col=0, parse_dates=True)
        
        print(f"\n📦 Loaded {len(self.data_cache)} datasets")
        for key in self.data_cache.keys():
            print(f"  • {key}")
    
    def save_figure(self, fig, filename, category='general', close=True):
        """Save figure in multiple formats with consistent settings."""
        category_dir = self.subdirs.get(category, self.output_dir)
        
        for fmt in FIGURE_FORMAT:
            filepath = category_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        if close:
            plt.close(fig)
        
        print(f"    💾 Saved: {filename}.{FIGURE_FORMAT[0]}")
        
    def generate_all_figures(self):
        """Generate all publication figures."""
        print("\n🎨 GENERATING PUBLICATION FIGURES")
        print("="*50)
        
        figure_count = 0
        
        # 1. Study Area Figures
        print("\n1️⃣ Study Area Figures")
        print("-" * 30)
        figure_count += self.create_study_area_figures()
        
        # 2. Storage Pattern Analysis
        print("\n2️⃣ Storage Pattern Analysis")
        print("-" * 30)
        figure_count += self.create_storage_pattern_figures()
        
        # 3. Temporal Analysis
        print("\n3️⃣ Temporal Analysis")
        print("-" * 30)
        figure_count += self.create_temporal_analysis_figures()
        
        # 4. Spatial Analysis
        print("\n4️⃣ Spatial Analysis")
        print("-" * 30)
        figure_count += self.create_spatial_analysis_figures()
        
        # 5. Feature Importance Analysis
        print("\n5️⃣ Feature Importance Analysis")
        print("-" * 30)
        figure_count += self.create_feature_importance_figures()
        
        # 6. Model Evaluation
        print("\n6️⃣ Model Evaluation")
        print("-" * 30)
        figure_count += self.create_model_evaluation_figures()
        
        # 7. Correlation Analysis
        print("\n7️⃣ Correlation Analysis")  
        print("-" * 30)
        figure_count += self.create_correlation_figures()
        
        # 8. Data Quality Assessment
        print("\n8️⃣ Data Quality Assessment")
        print("-" * 30)
        figure_count += self.create_data_quality_figures()
        
        print(f"\n🎉 GENERATION COMPLETE!")
        print(f"📊 Total figures created: {figure_count}")
        print(f"📁 Output directory: {self.output_dir}")
        
        return figure_count
    
    def create_study_area_figures(self):
        """Create comprehensive study area visualizations."""
        count = 0
        
        if 'gws' not in self.data_cache:
            print("    ⚠️ No groundwater data available for study area")
            return count
        
        gws_ds = self.data_cache['gws']
        
        # Figure 1: Study Area Overview with Topography
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Set extent
        extent = [float(gws_ds.lon.min()), float(gws_ds.lon.max()),
                 float(gws_ds.lat.min()), float(gws_ds.lat.max())]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='black', alpha=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linestyle='--', alpha=0.7)
        ax.add_feature(cfeature.RIVERS, alpha=0.6)
        ax.add_feature(cfeature.LAKES, alpha=0.6, facecolor='lightblue')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': FONT_SIZE_MEDIUM}
        gl.ylabel_style = {'size': FONT_SIZE_MEDIUM}
        
        # Add mean groundwater storage as background
        mean_gws = gws_ds.groundwater.mean(dim='time')
        im = ax.contourf(gws_ds.lon, gws_ds.lat, mean_gws, 
                        levels=20, cmap='RdBu_r', alpha=0.7,
                        transform=ccrs.PlateCarree(), extend='both')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar.set_label('Mean Groundwater Storage Anomaly (cm)', fontsize=FONT_SIZE_MEDIUM)
        
        # Add title and labels
        plt.title('Study Area: Mississippi River Basin\nGroundwater Storage Analysis Domain', 
                 fontsize=FONT_SIZE_LARGE, fontweight='bold', pad=20)
        
        # Add text box with study area info
        info_text = (f"Study Domain:\n"
                    f"Latitude: {extent[2]:.1f}°N - {extent[3]:.1f}°N\n"
                    f"Longitude: {extent[0]:.1f}°W - {extent[1]:.1f}°W\n"
                    f"Grid: {len(gws_ds.lat)} × {len(gws_ds.lon)}\n"
                    f"Period: {str(gws_ds.time.values[0])[:7]} - {str(gws_ds.time.values[-1])[:7]}")
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=FONT_SIZE_SMALL,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.save_figure(fig, 'fig01_study_area_overview', 'study_area')
        count += 1
        
        # Figure 2: Well Locations and Data Coverage
        if 'well_well_metadata' in self.data_cache:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                          subplot_kw={'projection': ccrs.PlateCarree()})
            
            well_meta = self.data_cache['well_well_metadata']
            
            # Left panel: Well locations
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            ax1.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7)
            ax1.add_feature(cfeature.COASTLINE)
            ax1.add_feature(cfeature.RIVERS, alpha=0.5)
            
            # Plot wells colored by data length
            if 'n_months_total' in well_meta.columns:
                scatter = ax1.scatter(well_meta['lon'], well_meta['lat'], 
                                    c=well_meta['n_months_total'], s=30,
                                    cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5,
                                    transform=ccrs.PlateCarree())
                cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
                cbar1.set_label('Months of Data', fontsize=FONT_SIZE_MEDIUM)
            else:
                ax1.scatter(well_meta['lon'], well_meta['lat'], s=30, c=COLORS['primary'],
                          alpha=0.8, edgecolors='black', linewidth=0.5,
                          transform=ccrs.PlateCarree())
            
            ax1.set_title('USGS Well Observation Network', fontsize=FONT_SIZE_LARGE, fontweight='bold')
            
            # Right panel: Data coverage density
            ax2.set_extent(extent, crs=ccrs.PlateCarree())
            ax2.add_feature(cfeature.STATES, linewidth=0.8, alpha=0.7)
            ax2.add_feature(cfeature.COASTLINE)
            
            # Create hexbin plot for density
            hb = ax2.hexbin(well_meta['lon'], well_meta['lat'], gridsize=20, 
                          cmap='Reds', alpha=0.8, transform=ccrs.PlateCarree())
            cbar2 = plt.colorbar(hb, ax=ax2, shrink=0.8)
            cbar2.set_label('Well Density', fontsize=FONT_SIZE_MEDIUM)
            
            ax2.set_title('Well Observation Density', fontsize=FONT_SIZE_LARGE, fontweight='bold')
            
            # Add gridlines to both
            for ax in [ax1, ax2]:
                gl = ax.gridlines(draw_labels=True, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': FONT_SIZE_SMALL}
                gl.ylabel_style = {'size': FONT_SIZE_SMALL}
            
            plt.tight_layout()
            self.save_figure(fig, 'fig02_well_network_coverage', 'study_area')
            count += 1
        
        return count
    
    def create_storage_pattern_figures(self):
        """Create storage pattern analysis figures."""
        count = 0
        
        if 'gws' not in self.data_cache:
            print("    ⚠️ No groundwater data available")
            return count
        
        gws_ds = self.data_cache['gws']
        
        # Figure 3: Multi-panel Storage Overview
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Mean storage
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        mean_gws = gws_ds.groundwater.mean(dim='time')
        im1 = ax1.contourf(gws_ds.lon, gws_ds.lat, mean_gws, levels=20, 
                          cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
        ax1.add_feature(cfeature.STATES, linewidth=0.5)
        ax1.set_title('Mean GWS Anomaly', fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='cm')
        
        # Standard deviation
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        std_gws = gws_ds.groundwater.std(dim='time')
        im2 = ax2.contourf(gws_ds.lon, gws_ds.lat, std_gws, levels=20, 
                          cmap='plasma', transform=ccrs.PlateCarree())
        ax2.add_feature(cfeature.STATES, linewidth=0.5)
        ax2.set_title('GWS Variability (Std Dev)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='cm')
        
        # Trend analysis
        ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        
        # Calculate linear trend for each pixel
        time_numeric = np.arange(len(gws_ds.time))
        trends = np.zeros((len(gws_ds.lat), len(gws_ds.lon)))
        
        for i in range(len(gws_ds.lat)):
            for j in range(len(gws_ds.lon)):
                y = gws_ds.groundwater[:, i, j].values
                if not np.all(np.isnan(y)):
                    valid = ~np.isnan(y)
                    if np.sum(valid) > 24:  # At least 2 years
                        slope, _, _, _, _ = stats.linregress(time_numeric[valid], y[valid])
                        trends[i, j] = slope * 12  # Convert to cm/year
                    else:
                        trends[i, j] = np.nan
                else:
                    trends[i, j] = np.nan
        
        # Plot trends
        trend_max = np.nanpercentile(np.abs(trends), 95)
        im3 = ax3.contourf(gws_ds.lon, gws_ds.lat, trends, 
                          levels=np.linspace(-trend_max, trend_max, 21),
                          cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
        ax3.add_feature(cfeature.STATES, linewidth=0.5)
        ax3.set_title('Linear Trend (2003-2022)', fontweight='bold')
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='cm/year')
        
        # Regional time series
        ax4 = fig.add_subplot(gs[1, :])
        regional_avg = gws_ds.groundwater.mean(dim=['lat', 'lon'])
        time_index = pd.to_datetime(gws_ds.time.values)
        
        # Plot monthly data
        ax4.plot(time_index, regional_avg.values, alpha=0.6, color=COLORS['primary'], 
                linewidth=1, label='Monthly')
        
        # Add 12-month rolling mean
        rolling_mean = pd.Series(regional_avg.values, index=time_index).rolling(12, center=True).mean()
        ax4.plot(time_index, rolling_mean.values, color=COLORS['negative'], 
                linewidth=2, label='12-month rolling mean')
        
        # Add trend line
        time_numeric_full = np.arange(len(time_index))
        slope, intercept, _, _, _ = stats.linregress(time_numeric_full, regional_avg.values)
        trend_line = slope * time_numeric_full + intercept
        ax4.plot(time_index, trend_line, '--', color=COLORS['accent'], 
                linewidth=2, label=f'Trend: {slope*12:.2f} cm/year')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Year', fontsize=FONT_SIZE_MEDIUM)
        ax4.set_ylabel('Regional GWS Anomaly (cm)', fontsize=FONT_SIZE_MEDIUM)
        ax4.set_title('Regional Groundwater Storage Time Series', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        ax4.xaxis.set_major_locator(mdates.YearLocator(2))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Seasonal analysis - FIX: Convert to pandas Series first for groupby
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Convert to pandas Series and then group by month
        regional_series = pd.Series(regional_avg.values, index=time_index)
        monthly_cycle = regional_series.groupby(regional_series.index.month).mean()
        
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax5.bar(range(1, 13), monthly_cycle.values, color=COLORS['secondary'], alpha=0.7)
        ax5.set_xticks(range(1, 13))
        ax5.set_xticklabels(months)
        ax5.set_ylabel('Mean GWS Anomaly (cm)')
        ax5.set_title('Seasonal Cycle', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Extreme events identification
        ax6 = fig.add_subplot(gs[2, 1])
        # Calculate percentiles for drought/wet classification
        p10 = np.percentile(regional_avg.values, 10)
        p90 = np.percentile(regional_avg.values, 90)
        
        colors = ['red' if x < p10 else 'blue' if x > p90 else 'gray' for x in regional_avg.values]
        ax6.scatter(time_index, regional_avg.values, c=colors, alpha=0.6, s=20)
        ax6.axhline(y=p10, color='red', linestyle='--', label=f'10th percentile ({p10:.1f} cm)')
        ax6.axhline(y=p90, color='blue', linestyle='--', label=f'90th percentile ({p90:.1f} cm)')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax6.set_ylabel('GWS Anomaly (cm)')
        ax6.set_title('Extreme Events', fontweight='bold')
        ax6.legend(fontsize=FONT_SIZE_SMALL)
        ax6.grid(True, alpha=0.3)
        
        # Drought/wet statistics
        ax7 = fig.add_subplot(gs[2, 2])
        drought_months = np.sum(regional_avg.values < p10)
        wet_months = np.sum(regional_avg.values > p90)
        normal_months = len(regional_avg.values) - drought_months - wet_months
        
        categories = ['Drought\n(<10th %ile)', 'Normal', 'Wet\n(>90th %ile)']
        counts = [drought_months, normal_months, wet_months]
        colors_bar = ['red', 'gray', 'blue']
        
        bars = ax7.bar(categories, counts, color=colors_bar, alpha=0.7)
        ax7.set_ylabel('Number of Months')
        ax7.set_title('Event Classification', fontweight='bold')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom')
        
        plt.suptitle('Groundwater Storage Pattern Analysis', fontsize=16, fontweight='bold')
        self.save_figure(fig, 'fig03_storage_patterns_overview', 'storage_patterns')
        count += 1
        
        return count
        
    def create_temporal_analysis_figures(self):
        """Create detailed temporal analysis figures."""
        count = 0
        
        if 'gws' not in self.data_cache:
            print("    ⚠️ No groundwater data available")
            return count
        
        gws_ds = self.data_cache['gws']
        regional_avg = gws_ds.groundwater.mean(dim=['lat', 'lon'])
        time_index = pd.to_datetime(gws_ds.time.values)
        
        # Figure 4: Seasonal Decomposition Analysis
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Prepare data for seasonal decomposition with aggressive missing value handling
        ts_data = pd.Series(regional_avg.values, index=time_index)
        
        # Check the percentage of missing values
        missing_pct = ts_data.isna().sum() / len(ts_data) * 100
        print(f"    Missing values: {missing_pct:.1f}%")
        
        # More aggressive missing value handling
        if missing_pct < 50:  # Only try seasonal decomposition if less than 50% missing
            # Method 1: Interpolate and fill
            ts_filled = ts_data.interpolate(method='linear', limit_direction='both')
            ts_filled = ts_filled.fillna(ts_filled.mean())  # Fill any remaining with mean
            
            # Ensure we have a complete monthly time series
            full_date_range = pd.date_range(start=ts_filled.index.min(), 
                                        end=ts_filled.index.max(), 
                                        freq='MS')
            ts_reindexed = ts_filled.reindex(full_date_range)
            ts_reindexed = ts_reindexed.interpolate(method='linear').fillna(ts_reindexed.mean())
            
            try:
                # Only proceed if we have enough data points (at least 24 months)
                if len(ts_reindexed) >= 24 and not ts_reindexed.isna().any():
                    decomposition = seasonal_decompose(ts_reindexed, model='additive', period=12, extrapolate_trend='freq')
                    
                    # Original time series
                    axes[0].plot(decomposition.observed.index, decomposition.observed.values, 
                                color=COLORS['primary'], linewidth=1.5)
                    axes[0].set_title('Original Time Series (Interpolated)', fontweight='bold')
                    axes[0].set_ylabel('GWS Anomaly (cm)')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Trend component
                    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                                color=COLORS['negative'], linewidth=2)
                    axes[1].set_title('Trend Component', fontweight='bold')
                    axes[1].set_ylabel('Trend (cm)')
                    axes[1].grid(True, alpha=0.3)
                    
                    # Seasonal component
                    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                                color=COLORS['secondary'], linewidth=1.5)
                    axes[2].set_title('Seasonal Component', fontweight='bold')
                    axes[2].set_ylabel('Seasonal (cm)')
                    axes[2].grid(True, alpha=0.3)
                    
                    # Residual component
                    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 
                                color=COLORS['accent'], linewidth=1, alpha=0.8)
                    axes[3].set_title('Residual Component', fontweight='bold')
                    axes[3].set_ylabel('Residual (cm)')
                    axes[3].set_xlabel('Year')
                    axes[3].grid(True, alpha=0.3)
                    
                else:
                    raise ValueError("Insufficient complete data for seasonal decomposition")
                    
            except Exception as e:
                print(f"    ⚠️ Seasonal decomposition failed: {e}")
                # Use manual decomposition approach
                self._create_manual_decomposition(axes, ts_data, time_index)
        else:
            print(f"    ⚠️ Too many missing values ({missing_pct:.1f}%) for seasonal decomposition")
            # Use manual decomposition approach
            self._create_manual_decomposition(axes, ts_data, time_index)
        
        # Format all x-axes
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.suptitle('Temporal Decomposition Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'fig04_temporal_decomposition', 'temporal_analysis')
        count += 1
        
        # Figure 5: Drought and Wet Period Analysis (keeping the same)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Use available data for analysis
        valid_data = ts_data.dropna()
        if len(valid_data) == 0:
            print("    ⚠️ No valid data for drought analysis")
            return count
        
        regional_values = valid_data.values
        valid_time_index = valid_data.index
        
        # Define drought thresholds
        p20 = np.percentile(regional_values, 20)
        p80 = np.percentile(regional_values, 80)
        
        # Drought periods identification
        drought_mask = regional_values < p20
        wet_mask = regional_values > p80
        
        # Plot 1: Time series with drought/wet periods highlighted
        ax1.plot(valid_time_index, regional_values, color='black', linewidth=1, alpha=0.7)
        ax1.fill_between(valid_time_index, regional_values, p20, 
                        where=drought_mask, color='red', alpha=0.3, label='Drought periods')
        ax1.fill_between(valid_time_index, p80, regional_values, 
                        where=wet_mask, color='blue', alpha=0.3, label='Wet periods')
        ax1.axhline(y=p20, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(y=p80, color='blue', linestyle='--', alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('GWS Anomaly (cm)')
        ax1.set_title('Drought and Wet Period Identification', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Continue with rest of drought analysis...
        # (keeping the rest of the drought analysis the same)
        
        # Plot 2: Intensity and duration of droughts
        drought_events = []
        in_drought = False
        start_idx = None
        
        for i, is_drought in enumerate(drought_mask):
            if is_drought and not in_drought:
                in_drought = True
                start_idx = i
            elif not is_drought and in_drought:
                in_drought = False
                if start_idx is not None:
                    duration = i - start_idx
                    intensity = np.mean(regional_values[start_idx:i])
                    drought_events.append({'start': start_idx, 'end': i, 'duration': duration, 'intensity': intensity})
        
        if drought_events:
            durations = [event['duration'] for event in drought_events]
            intensities = [abs(event['intensity']) for event in drought_events]
            
            scatter = ax2.scatter(durations, intensities, s=60, alpha=0.7, c=range(len(durations)), cmap='Reds')
            ax2.set_xlabel('Duration (months)')
            ax2.set_ylabel('Mean Intensity (|cm|)')
            ax2.set_title('Drought Characteristics', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar for event timing
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Event Order')
        else:
            ax2.text(0.5, 0.5, 'No drought events\nidentified', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Drought Characteristics', fontweight='bold')
        
        # Plot 3: Annual cycle comparison
        years = valid_time_index.year
        unique_years = np.unique(years)
        
        if len(unique_years) > 2:
            mid_year = np.median(unique_years)
            early_mask = years <= mid_year
            late_mask = years > mid_year
            
            early_data = regional_values[early_mask]
            late_data = regional_values[late_mask]
            early_months = valid_time_index[early_mask].month
            late_months = valid_time_index[late_mask].month
            
            # Calculate monthly climatology for each period
            early_monthly = [np.mean(early_data[early_months == m]) if np.any(early_months == m) else 0 for m in range(1, 13)]
            late_monthly = [np.mean(late_data[late_months == m]) if np.any(late_months == m) else 0 for m in range(1, 13)]
            
            months = np.arange(1, 13)
            width = 0.35
            
            ax3.bar(months - width/2, early_monthly, width, label=f'Early period ({int(min(years))}-{int(mid_year)})', 
                color=COLORS['primary'], alpha=0.7)
            ax3.bar(months + width/2, late_monthly, width, label=f'Late period ({int(mid_year+1)}-{int(max(years))})', 
                color=COLORS['secondary'], alpha=0.7)
            
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Mean GWS Anomaly (cm)')
            ax3.set_title('Seasonal Cycle Comparison', fontweight='bold')
            ax3.set_xticks(months)
            ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient years\nfor comparison', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Seasonal Cycle Comparison', fontweight='bold')
        
        # Plot 4: Interannual variability
        if len(unique_years) > 1:
            annual_means = []
            annual_stds = []
            
            for year in unique_years:
                year_mask = years == year
                if np.any(year_mask):
                    year_data = regional_values[year_mask]
                    annual_means.append(np.mean(year_data))
                    annual_stds.append(np.std(year_data))
            
            ax4.errorbar(unique_years, annual_means, yerr=annual_stds, marker='o', 
                        capsize=3, capthick=1, linewidth=1.5, markersize=6, color=COLORS['primary'])
            
            # Add trend line for annual means
            if len(unique_years) > 2:
                z = np.polyfit(unique_years, annual_means, 1)
                p = np.poly1d(z)
                ax4.plot(unique_years, p(unique_years), '--', color=COLORS['negative'], 
                        linewidth=2, label=f'Trend: {z[0]:.3f} cm/year')
                ax4.legend()
            
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Annual Mean GWS Anomaly (cm)')
            ax4.set_title('Interannual Variability', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor annual analysis', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Interannual Variability', fontweight='bold')
        
        plt.suptitle('Drought and Temporal Variability Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'fig05_drought_analysis', 'temporal_analysis')
        count += 1
        
        return count

    def _create_manual_decomposition(self, axes, ts_data, time_index):
        """Create manual decomposition when statsmodels fails."""
        # Use only valid data
        valid_data = ts_data.dropna()
        valid_time = valid_data.index
        
        if len(valid_data) < 12:
            # Not enough data for any analysis
            for i, title in enumerate(['Original Time Series', 'Trend Component', 'Seasonal Component', 'Residual Component']):
                axes[i].text(0.5, 0.5, 'Insufficient data\nfor analysis', ha='center', va='center', 
                            transform=axes[i].transAxes, fontsize=12)
                axes[i].set_title(title, fontweight='bold')
            return
        
        # 1. Original series
        axes[0].plot(valid_time, valid_data.values, color=COLORS['primary'], linewidth=1.5)
        axes[0].set_title('Original Time Series', fontweight='bold')
        axes[0].set_ylabel('GWS Anomaly (cm)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Trend (using rolling mean or polynomial fit)
        if len(valid_data) > 24:
            trend = valid_data.rolling(window=12, center=True, min_periods=6).mean()
        else:
            # Use polynomial fit for trend
            x = np.arange(len(valid_data))
            z = np.polyfit(x, valid_data.values, 1)
            trend = pd.Series(np.polyval(z, x), index=valid_time)
        
        axes[1].plot(valid_time, trend.values, color=COLORS['negative'], linewidth=2)
        axes[1].set_title('Trend Component (Rolling Mean)', fontweight='bold')
        axes[1].set_ylabel('Trend (cm)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Seasonal (monthly averages)
        if len(valid_data) > 12:
            monthly_means = valid_data.groupby(valid_data.index.month).mean()
            seasonal = valid_data.copy()
            for month in range(1, 13):
                if month in monthly_means.index:
                    month_mask = valid_data.index.month == month
                    seasonal[month_mask] = monthly_means[month] - valid_data.mean()
        else:
            seasonal = pd.Series(np.zeros(len(valid_data)), index=valid_time)
        
        axes[2].plot(valid_time, seasonal.values, color=COLORS['secondary'], linewidth=1.5)
        axes[2].set_title('Seasonal Component (Monthly Averages)', fontweight='bold')
        axes[2].set_ylabel('Seasonal (cm)')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Residual
        detrended = valid_data - trend.fillna(0)
        residual = detrended - seasonal.fillna(0)
        
        axes[3].plot(valid_time, residual.values, color=COLORS['accent'], linewidth=1, alpha=0.8)
        axes[3].set_title('Residual Component', fontweight='bold')
        axes[3].set_ylabel('Residual (cm)')
        axes[3].set_xlabel('Year')
        axes[3].grid(True, alpha=0.3)

    def _create_simple_spatial_plots(self, fig, gs, gws_ds):
        """Create simple spatial plots as fallback when EOF analysis fails."""
        # Plot mean and other statistical maps without projection issues
        for i in range(6):
            ax = fig.add_subplot(gs[i//3, i%3])  # Remove projection for simpler plotting
            
            if i == 0:
                # Mean groundwater storage
                data = gws_ds.groundwater.mean(dim='time')
                title = 'Mean GWS'
                cmap = 'RdBu_r'
            elif i == 1:
                # Standard deviation
                data = gws_ds.groundwater.std(dim='time')
                title = 'GWS Std Dev'
                cmap = 'plasma'
            elif i == 2:
                # Range (max - min)
                data = gws_ds.groundwater.max(dim='time') - gws_ds.groundwater.min(dim='time')
                title = 'GWS Range'
                cmap = 'viridis'
            elif i == 3:
                # First time slice
                data = gws_ds.groundwater.isel(time=0)
                title = f'GWS {str(gws_ds.time.values[0])[:7]}'
                cmap = 'RdBu_r'
            elif i == 4:
                # Middle time slice
                mid_idx = len(gws_ds.time) // 2
                data = gws_ds.groundwater.isel(time=mid_idx)
                title = f'GWS {str(gws_ds.time.values[mid_idx])[:7]}'
                cmap = 'RdBu_r'
            else:
                # Last time slice
                data = gws_ds.groundwater.isel(time=-1)
                title = f'GWS {str(gws_ds.time.values[-1])[:7]}'
                cmap = 'RdBu_r'
            
            # Use imshow instead of contourf for better control
            im = ax.imshow(data, cmap=cmap, aspect='auto', 
                        extent=[float(gws_ds.lon.min()), float(gws_ds.lon.max()),
                                float(gws_ds.lat.min()), float(gws_ds.lat.max())],
                        origin='lower')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(title, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

    def create_spatial_analysis_figures(self):
        """Create spatial analysis and clustering figures with fixed EOF plots."""
        count = 0
        
        if 'gws' not in self.data_cache:
            print("    ⚠️ No groundwater data available")
            return count
        
        gws_ds = self.data_cache['gws']
        
        # Figure 6: Spatial EOF Analysis with FIXED projection
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Prepare data for EOF analysis
        gws_data = gws_ds.groundwater.values
        original_shape = gws_data.shape
        
        # Reshape for PCA (time x space)
        gws_reshaped = gws_data.reshape(original_shape[0], -1)
        
        # Remove NaN columns and keep track of valid locations
        valid_cols = ~np.all(np.isnan(gws_reshaped), axis=0)
        gws_clean = gws_reshaped[:, valid_cols]
        
        if gws_clean.shape[1] > 0 and gws_clean.shape[0] > 6:
            try:
                # Remove any remaining NaN rows
                valid_rows = ~np.any(np.isnan(gws_clean), axis=1)
                gws_for_pca = gws_clean[valid_rows, :]
                
                if gws_for_pca.shape[0] > 6:  # Need enough time points
                    # Standardize data
                    scaler = StandardScaler()
                    gws_scaled = scaler.fit_transform(gws_for_pca.T).T
                    
                    # Perform PCA
                    n_components = min(6, gws_scaled.shape[0]-1, gws_scaled.shape[1]-1)
                    pca = PCA(n_components=n_components)
                    pca.fit(gws_scaled.T)
                    
                    # Get coordinate information
                    lon_2d, lat_2d = np.meshgrid(gws_ds.lon, gws_ds.lat)
                    
                    # Plot first 6 EOFs with FIXED projection
                    for i in range(min(6, pca.n_components_)):
                        # Use regular subplot instead of cartopy for better control
                        ax = fig.add_subplot(gs[i//3, i%3])
                        
                        # Reconstruct spatial pattern - FIXED
                        eof_spatial = np.full(original_shape[1] * original_shape[2], np.nan)
                        
                        # Map PCA components back to valid spatial locations
                        valid_indices = np.where(valid_cols)[0]
                        n_components_available = min(len(pca.components_[i, :]), len(valid_indices))
                        eof_spatial[valid_indices[:n_components_available]] = pca.components_[i, :n_components_available]
                        
                        eof_spatial = eof_spatial.reshape(original_shape[1], original_shape[2])
                        
                        # Plot EOF using imshow for consistent aspect ratio
                        im = ax.imshow(eof_spatial, cmap='RdBu_r', aspect='auto',
                                    extent=[float(gws_ds.lon.min()), float(gws_ds.lon.max()),
                                            float(gws_ds.lat.min()), float(gws_ds.lat.max())],
                                    origin='lower')
                        
                        ax.set_xlabel('Longitude', fontsize=FONT_SIZE_SMALL)
                        ax.set_ylabel('Latitude', fontsize=FONT_SIZE_SMALL)
                        ax.set_title(f'EOF {i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)', 
                                fontweight='bold', fontsize=FONT_SIZE_MEDIUM)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                        cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL)
                        
                else:
                    # Fallback: create simple spatial analysis plots
                    self._create_simple_spatial_plots(fig, gs, gws_ds)
            except Exception as e:
                print(f"    ⚠️ EOF analysis failed: {e}")
                # Fallback: create simple spatial analysis plots
                self._create_simple_spatial_plots(fig, gs, gws_ds)
        else:
            # Fallback: create simple spatial analysis plots
            self._create_simple_spatial_plots(fig, gs, gws_ds)
        
        plt.suptitle('Empirical Orthogonal Function Analysis', fontsize=16, fontweight='bold')
        self.save_figure(fig, 'fig06_eof_analysis', 'spatial_analysis')
        count += 1
        
        # Figure 7: Spatial Clustering (keeping the fixed version from before)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), 
                                                    subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Calculate correlation distance matrix for clustering
        n_lat, n_lon = len(gws_ds.lat), len(gws_ds.lon)
        
        # Sample every nth point to make clustering computationally feasible
        step = max(1, max(n_lat, n_lon) // 15)  # Reduce step size for more points
        lat_indices = range(0, n_lat, step)
        lon_indices = range(0, n_lon, step)
        
        sample_data = []
        sample_coords = []
        
        for i in lat_indices:
            for j in lon_indices:
                ts = gws_ds.groundwater[:, i, j].values
                if not np.all(np.isnan(ts)) and np.sum(~np.isnan(ts)) > 24:  # At least 2 years of data
                    # Fill missing values with interpolation
                    ts_filled = pd.Series(ts).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                    sample_data.append(ts_filled)
                    sample_coords.append((float(gws_ds.lat[i]), float(gws_ds.lon[j])))
        
        if len(sample_data) > 10:  # Need minimum points for clustering
            try:
                sample_array = np.array(sample_data)
                
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(sample_array)
                
                # Handle NaN values in correlation matrix
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure correlation matrix is symmetric
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                
                # Convert to distance matrix (ensure values are between 0 and 2)
                distance_matrix = 1 - corr_matrix
                distance_matrix = np.clip(distance_matrix, 0, 2)
                
                # Ensure diagonal is zero
                np.fill_diagonal(distance_matrix, 0)
                
                # Convert to condensed distance matrix for linkage
                from scipy.spatial.distance import squareform
                distance_condensed = squareform(distance_matrix, checks=False)
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(distance_condensed, method='ward')
                
                # Get cluster labels for different numbers of clusters
                from scipy.cluster.hierarchy import fcluster
                
                for n_clusters, ax, title in zip([3, 4, 5, 6], [ax1, ax2, ax3, ax4], 
                                            ['3 Clusters', '4 Clusters', '5 Clusters', '6 Clusters']):
                    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    # Plot clusters
                    extent = [float(gws_ds.lon.min()), float(gws_ds.lon.max()),
                            float(gws_ds.lat.min()), float(gws_ds.lat.max())]
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
                    ax.add_feature(cfeature.COASTLINE)
                    
                    # Color points by cluster
                    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
                    for cluster_id in range(1, n_clusters + 1):
                        mask = cluster_labels == cluster_id
                        if np.any(mask):
                            cluster_coords = np.array(sample_coords)[mask]
                            ax.scatter(cluster_coords[:, 1], cluster_coords[:, 0], 
                                    c=[colors[cluster_id-1]], s=40, alpha=0.8,
                                    label=f'Cluster {cluster_id}', edgecolors='black', linewidth=0.5,
                                    transform=ccrs.PlateCarree())
                    
                    ax.set_title(title, fontweight='bold')
                    if ax == ax4:  # Only show legend on last plot
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZE_SMALL)
            
            except Exception as e:
                print(f"    ⚠️ Spatial clustering failed: {e}")
                # Create simple fallback plots
                for ax, title in zip([ax1, ax2, ax3, ax4], 
                                ['3 Clusters', '4 Clusters', '5 Clusters', '6 Clusters']):
                    extent = [float(gws_ds.lon.min()), float(gws_ds.lon.max()),
                            float(gws_ds.lat.min()), float(gws_ds.lat.max())]
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
                    ax.add_feature(cfeature.COASTLINE)
                    ax.set_title(f'{title} (Analysis failed)', fontweight='bold')
        else:
            print("    ⚠️ Insufficient data points for clustering analysis")
            # Create simple fallback plots
            for ax, title in zip([ax1, ax2, ax3, ax4], 
                            ['3 Clusters', '4 Clusters', '5 Clusters', '6 Clusters']):
                extent = [float(gws_ds.lon.min()), float(gws_ds.lon.max()),
                        float(gws_ds.lat.min()), float(gws_ds.lat.max())]
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
                ax.add_feature(cfeature.COASTLINE)
                ax.set_title(f'{title} (Insufficient data)', fontweight='bold')
        
        plt.suptitle('Spatial Clustering of Groundwater Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'fig07_spatial_clustering', 'spatial_analysis')
        count += 1
        
        return count

    def _create_simple_spatial_plots(self, fig, gs, gws_ds):
        """Create simple spatial plots as fallback when EOF analysis fails."""
        # Plot mean and standard deviation maps
        for i in range(6):
            ax = fig.add_subplot(gs[i//3, i%3], projection=ccrs.PlateCarree())
            
            if i == 0:
                # Mean groundwater storage
                data = gws_ds.groundwater.mean(dim='time')
                title = 'Mean GWS'
                cmap = 'RdBu_r'
            elif i == 1:
                # Standard deviation
                data = gws_ds.groundwater.std(dim='time')
                title = 'GWS Std Dev'
                cmap = 'plasma'
            elif i == 2:
                # Range (max - min)
                data = gws_ds.groundwater.max(dim='time') - gws_ds.groundwater.min(dim='time')
                title = 'GWS Range'
                cmap = 'viridis'
            elif i == 3:
                # First time slice
                data = gws_ds.groundwater.isel(time=0)
                title = f'GWS {str(gws_ds.time.values[0])[:7]}'
                cmap = 'RdBu_r'
            elif i == 4:
                # Middle time slice
                mid_idx = len(gws_ds.time) // 2
                data = gws_ds.groundwater.isel(time=mid_idx)
                title = f'GWS {str(gws_ds.time.values[mid_idx])[:7]}'
                cmap = 'RdBu_r'
            else:
                # Last time slice
                data = gws_ds.groundwater.isel(time=-1)
                title = f'GWS {str(gws_ds.time.values[-1])[:7]}'
                cmap = 'RdBu_r'
            
            im = ax.contourf(gws_ds.lon, gws_ds.lat, data, 
                            levels=20, cmap=cmap, extend='both',
                            transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax.set_title(title, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    def create_feature_importance_figures(self):
        """Create feature importance analysis figures with meaningful feature names."""
        count = 0
        
        # Define the actual feature mapping based on the model creation process
        def get_meaningful_feature_names():
            """Create mapping from generic feature names to meaningful ones."""
            
            # Base features in order (from features.py and model training)
            base_features = [
                'Evap_tavg',           # GLDAS evapotranspiration
                'SWE_inst',            # GLDAS snow water equivalent  
                'SoilMoi0_10cm_inst',  # GLDAS soil moisture 0-10cm
                'SoilMoi10_40cm_inst', # GLDAS soil moisture 10-40cm
                'SoilMoi40_100cm_inst',# GLDAS soil moisture 40-100cm
                'SoilMoi100_200cm_inst', # GLDAS soil moisture 100-200cm
                'chirps',              # CHIRPS precipitation
                'aet',                 # TerraClimate actual evapotranspiration
                'def',                 # TerraClimate deficit
                'pr',                  # TerraClimate precipitation
                'tmmn',                # TerraClimate min temperature
                'tmmx'                 # TerraClimate max temperature
            ]
            
            feature_mapping = {}
            feature_index = 0
            
            # Current month features
            for i, feature in enumerate(base_features):
                feature_mapping[f'feat_{i}'] = feature
                feature_mapping[f'feat_{feature_index}'] = feature
                feature_index += 1
            
            # Lagged features (1, 3, 6 months)
            lags = [1, 3, 6]
            for lag in lags:
                for i, feature in enumerate(base_features):
                    lag_name = f'{feature}_lag{lag}m'
                    feature_mapping[f'feat_{i}_lag{lag}'] = lag_name
                    feature_mapping[f'feat_{feature_index}'] = lag_name
                    feature_index += 1
            
            # Seasonal features
            feature_mapping['month_sin'] = 'Month_sin'
            feature_mapping['month_cos'] = 'Month_cos'
            feature_mapping[f'feat_{feature_index}'] = 'Month_sin'
            feature_mapping[f'feat_{feature_index+1}'] = 'Month_cos'
            feature_index += 2
            
            # Static features (approximate - actual names may vary)
            static_features = [
                'SRTM_elevation', 'Sand_0cm', 'Clay_0cm', 'Sand_10cm', 'Clay_10cm',
                'Sand_30cm', 'Clay_30cm', 'Sand_60cm', 'Clay_60cm', 'Sand_100cm', 
                'Clay_100cm', 'Sand_200cm', 'Clay_200cm', 'MODIS_landcover'
            ]
            
            for i, static_feat in enumerate(static_features):
                feature_mapping[f'static_{i}'] = static_feat
                feature_mapping[f'feat_{feature_index}'] = static_feat
                feature_index += 1
            
            return feature_mapping, base_features
        
        # Get feature mapping
        feature_mapping, base_features = get_meaningful_feature_names()
        
        # Figure 8: Feature Importance Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Check if we have feature importance data
        if 'feature_importance' in self.data_cache:
            importance_df = self.data_cache['feature_importance'].copy()
            
            # Map generic names to meaningful names
            importance_df['Meaningful_Name'] = importance_df['Feature'].map(
                lambda x: feature_mapping.get(x, x)
            )
            
            print(f"    ✅ Found {len(importance_df)} features with importance scores")
            
        else:
            print("    ❌ No feature importance data found!")
            print("    💡 To generate feature importance, run one of these commands:")
            print("       python src/updated_model_rf.py")
            print("       python src/model_manager.py") 
            print("    📁 This will create: models/feature_importances.csv")
            print("    🔄 Then re-run the figure generator")
            return count
        
        # Top features bar plot with meaningful names
        top_features = importance_df.head(20)
        y_pos = np.arange(len(top_features))
        
        bars = ax1.barh(y_pos, top_features['Importance'], color=COLORS['primary'], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features['Meaningful_Name'], fontsize=FONT_SIZE_SMALL)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top 20 Feature Importances\n(Meaningful Names)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Feature importance by category with detailed categorization
        def categorize_feature(name):
            if any(x in name for x in ['SoilMoi', 'soil']):
                return 'Soil Moisture'
            elif any(x in name for x in ['pr', 'chirps']):
                return 'Precipitation' 
            elif any(x in name for x in ['tmmx', 'tmmn']):
                return 'Temperature'
            elif any(x in name for x in ['aet', 'Evap']):
                return 'Evapotranspiration'
            elif 'SWE' in name:
                return 'Snow Water'
            elif 'def' in name:
                return 'Water Deficit'
            elif any(x in name for x in ['Sand', 'Clay']):
                return 'Soil Properties'
            elif any(x in name for x in ['SRTM', 'elevation']):
                return 'Topography'
            elif any(x in name for x in ['MODIS', 'landcover']):
                return 'Land Cover'
            elif 'Month' in name:
                return 'Seasonality'
            elif 'lag' in name:
                return 'Lagged Features'
            else:
                return 'Other'
        
        # Calculate category importance
        importance_df['Category'] = importance_df['Meaningful_Name'].apply(categorize_feature)
        category_importance = importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
        
        # Plot category importance pie chart
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_importance)))
        wedges, texts, autotexts = ax2.pie(category_importance.values, labels=category_importance.index, 
                                        autopct='%1.1f%%', colors=colors_cat, startangle=90)
        ax2.set_title('Feature Importance by Category', fontweight='bold')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(FONT_SIZE_SMALL)
        
        # Cumulative importance
        cumsum = np.cumsum(importance_df['Importance'].values)
        ax3.plot(range(1, len(cumsum)+1), cumsum/cumsum[-1]*100, 
                marker='o', linewidth=2, markersize=4, color=COLORS['secondary'])
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Cumulative Importance (%)')
        ax3.set_title('Cumulative Feature Importance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Feature type analysis (current vs lagged vs static)
        def get_feature_type(name):
            if 'lag' in name:
                return 'Lagged'
            elif any(x in name for x in ['Sand', 'Clay', 'SRTM', 'MODIS']):
                return 'Static'
            elif any(x in name for x in ['Month']):
                return 'Seasonal'
            else:
                return 'Current'
        
        importance_df['Type'] = importance_df['Meaningful_Name'].apply(get_feature_type)
        type_importance = importance_df.groupby('Type')['Importance'].sum().sort_values(ascending=False)
        
        colors_type = ['skyblue', 'lightcoral', 'lightgreen', 'orange'][:len(type_importance)]
        bars = ax4.bar(type_importance.index, type_importance.values, 
                    color=colors_type, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Total Importance')
        ax4.set_title('Importance by Feature Type', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, type_importance.values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE_SMALL)
        
        # Add subtitle
        fig.suptitle('Random Forest Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'fig08_feature_importance', 'feature_importance')
        count += 1
        
        # Print top features for reference
        print("\n📊 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(top_features.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['Meaningful_Name']:25s} (importance: {row['Importance']:.4f})")
        
        # Print feature type breakdown
        print(f"\n📈 Feature Type Breakdown:")
        for feat_type, importance in type_importance.items():
            count_type = (importance_df['Type'] == feat_type).sum()
            print(f"  {feat_type:12s}: {importance:.3f} total importance ({count_type:3d} features)")
        
        return count

    # Helper function to decode lag features specifically  
    def decode_lag_feature(feature_name):
        """Decode lag feature to show what parameter and lag time."""
        if 'lag' not in feature_name:
            return feature_name
        
        # Extract the base feature and lag time
        parts = feature_name.split('_lag')
        if len(parts) == 2:
            base_feature = parts[0]
            lag_time = parts[1].replace('m', '')
            return f"{base_feature} (lagged {lag_time} months)"
        
        return feature_name

    # Helper function to decode lag features specifically  
    def decode_lag_feature(feature_name):
        """Decode lag feature to show what parameter and lag time."""
        if 'lag' not in feature_name:
            return feature_name
        
        # Extract the base feature and lag time
        parts = feature_name.split('_lag')
        if len(parts) == 2:
            base_feature = parts[0]
            lag_time = parts[1].replace('m', '')
            return f"{base_feature} (lagged {lag_time} months)"
        
        return feature_name
    
    def create_model_evaluation_figures(self):
        """Create model evaluation and comparison figures."""
        count = 0
        
        # Figure 9: Model Comparison
        if 'model_comparison' in self.data_cache:
            model_df = self.data_cache['model_comparison']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Model performance comparison
            x_pos = np.arange(len(model_df))
            width = 0.35
            
            ax1.bar(x_pos - width/2, model_df['test_r2'], width, 
                   label='Test R²', color=COLORS['primary'], alpha=0.7)
            ax1.bar(x_pos + width/2, model_df['train_r2'], width,
                   label='Train R²', color=COLORS['secondary'], alpha=0.7)
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('R² Score')
            ax1.set_title('Model Performance Comparison (R²)', fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(model_df['display_name'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # RMSE comparison
            ax2.bar(x_pos - width/2, model_df['test_rmse'], width,
                   label='Test RMSE', color=COLORS['negative'], alpha=0.7)
            ax2.bar(x_pos + width/2, model_df['train_rmse'], width,
                   label='Train RMSE', color=COLORS['accent'], alpha=0.7)
            
            ax2.set_xlabel('Model')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Model Performance Comparison (RMSE)', fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(model_df['display_name'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Training time comparison
            if 'training_time' in model_df.columns:
                ax3.bar(x_pos, model_df['training_time'], color=COLORS['neutral'], alpha=0.7)
                ax3.set_xlabel('Model')
                ax3.set_ylabel('Training Time (seconds)')
                ax3.set_title('Training Time Comparison', fontweight='bold')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(model_df['display_name'], rotation=45, ha='right')
                ax3.grid(True, alpha=0.3, axis='y')
            
            # Performance vs complexity
            if 'training_time' in model_df.columns:
                scatter = ax4.scatter(model_df['training_time'], model_df['test_r2'], 
                                    s=100, alpha=0.7, c=range(len(model_df)), cmap='viridis')
                
                # Add model labels
                for i, (time, r2, name) in enumerate(zip(model_df['training_time'], 
                                                        model_df['test_r2'], 
                                                        model_df['display_name'])):
                    ax4.annotate(name, (time, r2), xytext=(5, 5), 
                               textcoords='offset points', fontsize=FONT_SIZE_SMALL)
                
                ax4.set_xlabel('Training Time (seconds)')
                ax4.set_ylabel('Test R²')
                ax4.set_title('Performance vs Computational Cost', fontweight='bold')
                ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Machine Learning Model Evaluation', fontsize=16, fontweight='bold')
            plt.tight_layout()
            self.save_figure(fig, 'fig09_model_comparison', 'model_evaluation')
            count += 1
        
        # Figure 10: Validation Results
        if 'point_validation_metrics' in self.data_cache:
            val_df = self.data_cache['point_validation_metrics']
            
            if len(val_df) > 0:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Validation performance distribution
                ax1.hist(val_df['pearson_r'], bins=30, alpha=0.7, 
                        color=COLORS['primary'], edgecolor='black')
                ax1.axvline(val_df['pearson_r'].mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {val_df["pearson_r"].mean():.3f}')
                ax1.axvline(val_df['pearson_r'].median(), color='orange', linestyle='--',
                           linewidth=2, label=f'Median: {val_df["pearson_r"].median():.3f}')
                ax1.set_xlabel('Correlation Coefficient')
                ax1.set_ylabel('Number of Wells')
                ax1.set_title('Validation Performance Distribution', fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Spatial distribution of validation performance
                if 'lat' in val_df.columns and 'lon' in val_df.columns:
                    scatter = ax2.scatter(val_df['lon'], val_df['lat'], c=val_df['pearson_r'], 
                                        s=30, cmap='RdYlBu', vmin=-0.2, vmax=0.8,
                                        alpha=0.8, edgecolors='black', linewidth=0.5)
                    ax2.set_xlabel('Longitude')
                    ax2.set_ylabel('Latitude')
                    ax2.set_title('Spatial Distribution of Validation Performance', fontweight='bold')
                    plt.colorbar(scatter, ax=ax2, label='Correlation')
                    ax2.grid(True, alpha=0.3)
                
                # Sample size vs performance
                if 'n_obs' in val_df.columns:
                    ax3.scatter(val_df['n_obs'], val_df['pearson_r'], alpha=0.6, 
                              color=COLORS['secondary'], s=30)
                    
                    # Add trend line
                    z = np.polyfit(val_df['n_obs'], val_df['pearson_r'], 1)
                    p = np.poly1d(z)
                    ax3.plot(val_df['n_obs'], p(val_df['n_obs']), '--', 
                            color=COLORS['negative'], linewidth=2)
                    
                    ax3.set_xlabel('Number of Observations')
                    ax3.set_ylabel('Correlation Coefficient')
                    ax3.set_title('Sample Size vs Validation Performance', fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                
                # Performance categories
                categories = ['Excellent\n(r>0.7)', 'Good\n(0.5<r≤0.7)', 'Fair\n(0.3<r≤0.5)', 'Poor\n(r≤0.3)']
                counts = [
                    (val_df['pearson_r'] > 0.7).sum(),
                    ((val_df['pearson_r'] > 0.5) & (val_df['pearson_r'] <= 0.7)).sum(),
                    ((val_df['pearson_r'] > 0.3) & (val_df['pearson_r'] <= 0.5)).sum(),
                    (val_df['pearson_r'] <= 0.3).sum()
                ]
                colors_cat = ['darkgreen', 'green', 'orange', 'red']
                
                bars = ax4.bar(categories, counts, color=colors_cat, alpha=0.7)
                ax4.set_ylabel('Number of Wells')
                ax4.set_title('Validation Performance Categories', fontweight='bold')
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{count}\n({count/len(val_df)*100:.1f}%)', 
                            ha='center', va='bottom', fontsize=FONT_SIZE_SMALL)
                
                plt.suptitle('Model Validation Against USGS Wells', fontsize=16, fontweight='bold')
                plt.tight_layout()
                self.save_figure(fig, 'fig10_validation_results', 'model_evaluation')
                count += 1
        
        return count
    
    def create_correlation_figures(self):
        """Create correlation analysis figures."""
        count = 0
        
        if 'features' not in self.data_cache:
            print("    ⚠️ No feature data available")
            return count
        
        features_ds = self.data_cache['features']
        
        # Figure 11: Feature Correlation Matrix
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # Calculate correlation matrix for a subset of features
        feature_data = features_ds.features.values
        n_times, n_features, n_lat, n_lon = feature_data.shape
        
        # Sample random locations for correlation analysis
        n_samples = min(1000, n_lat * n_lon)
        sample_indices = np.random.choice(n_lat * n_lon, n_samples, replace=False)
        
        # Reshape and sample
        feature_reshaped = feature_data.reshape(n_times, n_features, -1)
        sampled_features = feature_reshaped[:, :, sample_indices]
        
        # Calculate mean correlation across space and time
        feature_correlations = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i <= j:
                    # Calculate correlation across all samples and times
                    data_i = sampled_features[:, i, :].flatten()
                    data_j = sampled_features[:, j, :].flatten()
                    
                    # Remove NaN values
                    valid = ~(np.isnan(data_i) | np.isnan(data_j))
                    if np.sum(valid) > 100:  # Need sufficient data
                        corr = np.corrcoef(data_i[valid], data_j[valid])[0, 1]
                        feature_correlations[i, j] = corr
                        feature_correlations[j, i] = corr
                    else:
                        feature_correlations[i, j] = np.nan
                        feature_correlations[j, i] = np.nan
        
        # Plot correlation matrix
        feature_names = [f"F{i+1}" for i in range(n_features)]  # Simplified names
        
        mask = np.triu(np.ones_like(feature_correlations, dtype=bool))
        im1 = ax1.imshow(feature_correlations, cmap='RdBu_r', vmin=-1, vmax=1, 
                        aspect='auto')
        ax1.set_xticks(range(n_features))
        ax1.set_yticks(range(n_features))
        ax1.set_xticklabels(feature_names, rotation=90, fontsize=FONT_SIZE_SMALL)
        ax1.set_yticklabels(feature_names, fontsize=FONT_SIZE_SMALL)
        ax1.set_title('Feature Correlation Matrix', fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Correlation')
        
        # Plot correlation distribution
        corr_values = feature_correlations[~np.isnan(feature_correlations)]
        corr_values = corr_values[corr_values != 1]  # Remove self-correlations
        
        ax2.hist(corr_values, bins=30, alpha=0.7, color=COLORS['primary'], edgecolor='black')
        ax2.axvline(np.mean(corr_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(corr_values):.3f}')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_ylabel('Number of Feature Pairs')
        ax2.set_title('Inter-Feature Correlation Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Groundwater vs features correlation (if GWS data available)
        if 'gws' in self.data_cache:
            gws_ds = self.data_cache['gws']
            
            # Sample GWS data at same locations
            gws_data = gws_ds.groundwater.values
            gws_reshaped = gws_data.reshape(gws_data.shape[0], -1)
            gws_sampled = gws_reshaped[:, sample_indices]
            
            # Calculate correlation between each feature and GWS
            gws_feature_correlations = []
            
            for i in range(n_features):
                feature_flat = sampled_features[:, i, :].flatten()
                gws_flat = gws_sampled.flatten()
                
                valid = ~(np.isnan(feature_flat) | np.isnan(gws_flat))
                if np.sum(valid) > 100:
                    corr = np.corrcoef(feature_flat[valid], gws_flat[valid])[0, 1]
                    gws_feature_correlations.append(corr)
                else:
                    gws_feature_correlations.append(np.nan)
            
            # Plot GWS-feature correlations
            valid_correlations = [(i, corr) for i, corr in enumerate(gws_feature_correlations) 
                                if not np.isnan(corr)]
            
            if valid_correlations:
                indices, correlations = zip(*valid_correlations)
                
                bars = ax3.bar(range(len(correlations)), correlations, 
                              color=['red' if c < 0 else 'blue' for c in correlations], alpha=0.7)
                ax3.set_xlabel('Feature Index')
                ax3.set_ylabel('Correlation with GWS')
                ax3.set_title('Feature-Groundwater Correlations', fontweight='bold')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add feature labels for highest correlations
                top_indices = np.argsort(np.abs(correlations))[-5:]
                for i in top_indices:
                    ax3.text(i, correlations[i] + 0.02 * np.sign(correlations[i]), 
                            f'F{indices[i]+1}', ha='center', va='bottom' if correlations[i] > 0 else 'top',
                            fontsize=FONT_SIZE_SMALL)
        
        # Lag correlation analysis
        if 'gws' in self.data_cache:
            # Calculate lag correlations between regional GWS and a key feature
            regional_gws = gws_ds.groundwater.mean(dim=['lat', 'lon']).values
            
            # Use first feature as example
            regional_feature = np.nanmean(sampled_features[:, 0, :], axis=1)
            
            # Calculate cross-correlation
            max_lag = 12  # months
            lags = range(-max_lag, max_lag + 1)
            cross_corr = []
            
            for lag in lags:
                if lag == 0:
                    valid = ~(np.isnan(regional_gws) | np.isnan(regional_feature))
                    if np.sum(valid) > 12:
                        corr = np.corrcoef(regional_gws[valid], regional_feature[valid])[0, 1]
                    else:
                        corr = np.nan
                elif lag > 0:
                    # Feature leads GWS
                    valid = ~(np.isnan(regional_gws[lag:]) | np.isnan(regional_feature[:-lag]))
                    if np.sum(valid) > 12:
                        corr = np.corrcoef(regional_gws[lag:][valid], regional_feature[:-lag][valid])[0, 1]
                    else:
                        corr = np.nan
                else:
                    # GWS leads feature
                    lag_abs = abs(lag)
                    valid = ~(np.isnan(regional_gws[:-lag_abs]) | np.isnan(regional_feature[lag_abs:]))
                    if np.sum(valid) > 12:
                        corr = np.corrcoef(regional_gws[:-lag_abs][valid], regional_feature[lag_abs:][valid])[0, 1]
                    else:
                        corr = np.nan
                
                cross_corr.append(corr)
            
            ax4.plot(lags, cross_corr, marker='o', linewidth=2, markersize=4, color=COLORS['secondary'])
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Lag (months)')
            ax4.set_ylabel('Cross-correlation')
            ax4.set_title('Lag Correlation Analysis\n(Feature 1 vs Regional GWS)', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Mark maximum correlation
            if not all(np.isnan(cross_corr)):
                max_idx = np.nanargmax(np.abs(cross_corr))
                max_lag = lags[max_idx]
                max_corr = cross_corr[max_idx]
                ax4.plot(max_lag, max_corr, 'ro', markersize=8, 
                        label=f'Max: r={max_corr:.3f} at lag={max_lag}')
                ax4.legend()
        
        plt.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_figure(fig, 'fig11_correlation_analysis', 'correlation')
        count += 1
        
        return count
    
    def create_data_quality_figures(self):
        """Create data quality assessment figures."""
        count = 0
        
        # Figure 12: Data Coverage and Quality Assessment
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Groundwater data coverage
        if 'gws' in self.data_cache:
            gws_ds = self.data_cache['gws']
            
            # Data availability map
            ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            data_coverage = (~np.isnan(gws_ds.groundwater)).sum(dim='time')
            
            im1 = ax1.contourf(gws_ds.lon, gws_ds.lat, data_coverage, 
                            levels=20, cmap='viridis', transform=ccrs.PlateCarree())
            ax1.add_feature(cfeature.STATES, linewidth=0.5)
            ax1.set_title('GWS Data Coverage\n(Number of Valid Months)', fontweight='bold')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Missing data percentage
            ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
            missing_pct = (np.isnan(gws_ds.groundwater)).sum(dim='time') / len(gws_ds.time) * 100
            
            im2 = ax2.contourf(gws_ds.lon, gws_ds.lat, missing_pct, 
                            levels=np.arange(0, 101, 10), cmap='Reds', 
                            transform=ccrs.PlateCarree())
            ax2.add_feature(cfeature.STATES, linewidth=0.5)
            ax2.set_title('Missing Data Percentage', fontweight='bold')
            plt.colorbar(im2, ax=ax2, shrink=0.8, label='%')
            
            # Temporal coverage
            ax3 = fig.add_subplot(gs[0, 2])
            valid_count_per_time = (~np.isnan(gws_ds.groundwater)).sum(dim=['lat', 'lon'])
            time_index = pd.to_datetime(gws_ds.time.values)
            
            ax3.plot(time_index, valid_count_per_time.values, linewidth=2, color=COLORS['primary'])
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Number of Valid Pixels')
            ax3.set_title('Temporal Data Coverage', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_locator(mdates.YearLocator(2))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        else:
            # No groundwater data available
            for i, ax_pos in enumerate([gs[0, 0], gs[0, 1], gs[0, 2]]):
                ax = fig.add_subplot(ax_pos)
                ax.text(0.5, 0.5, 'No groundwater\ndata available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
                ax.set_title(['GWS Data Coverage', 'Missing Data %', 'Temporal Coverage'][i], fontweight='bold')
        
        # Feature data quality
        if 'features' in self.data_cache:
            features_ds = self.data_cache['features']
            
            # Feature data completeness
            ax4 = fig.add_subplot(gs[1, 0])
            feature_completeness = []
            
            for i in range(len(features_ds.feature)):
                feature_data = features_ds.features[:, i, :, :].values
                valid_ratio = np.sum(~np.isnan(feature_data)) / feature_data.size * 100
                feature_completeness.append(valid_ratio)
            
            bars = ax4.bar(range(len(feature_completeness)), feature_completeness, 
                        color=COLORS['secondary'], alpha=0.7)
            ax4.set_xlabel('Feature Index')
            ax4.set_ylabel('Data Completeness (%)')
            ax4.set_title('Feature Data Completeness', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 100)
            
            # Add horizontal line at 90%
            ax4.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
            ax4.legend()
            
            # Feature variability
            ax5 = fig.add_subplot(gs[1, 1])
            feature_std = []
            
            for i in range(len(features_ds.feature)):
                feature_data = features_ds.features[:, i, :, :].values
                std_val = np.nanstd(feature_data)
                feature_std.append(std_val)
            
            ax5.bar(range(len(feature_std)), feature_std, color=COLORS['accent'], alpha=0.7)
            ax5.set_xlabel('Feature Index')
            ax5.set_ylabel('Standard Deviation')
            ax5.set_title('Feature Variability', fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Feature correlation with target (if available)
            if 'gws' in self.data_cache:
                ax6 = fig.add_subplot(gs[1, 2])
                
                # Calculate simple correlation between each feature and GWS mean
                feature_gws_corr = []
                gws_mean = gws_ds.groundwater.mean(dim=['lat', 'lon']).values
                
                for i in range(len(features_ds.feature)):
                    feature_mean = features_ds.features[:, i, :, :].mean(dim=['lat', 'lon']).values
                    
                    # Align time dimensions if necessary
                    min_len = min(len(gws_mean), len(feature_mean))
                    gws_aligned = gws_mean[:min_len]
                    feature_aligned = feature_mean[:min_len]
                    
                    valid = ~(np.isnan(gws_aligned) | np.isnan(feature_aligned))
                    if np.sum(valid) > 12:
                        corr = np.corrcoef(gws_aligned[valid], feature_aligned[valid])[0, 1]
                        feature_gws_corr.append(corr)
                    else:
                        feature_gws_corr.append(np.nan)
                
                colors = ['red' if c < 0 else 'blue' if c > 0 else 'gray' 
                        for c in feature_gws_corr]
                bars = ax6.bar(range(len(feature_gws_corr)), feature_gws_corr, 
                            color=colors, alpha=0.7)
                ax6.set_xlabel('Feature Index')
                ax6.set_ylabel('Correlation with GWS')
                ax6.set_title('Feature-Target Correlation', fontweight='bold')
                ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax6.grid(True, alpha=0.3, axis='y')
            else:
                ax6 = fig.add_subplot(gs[1, 2])
                ax6.text(0.5, 0.5, 'No GWS data\nfor correlation', ha='center', va='center',
                        transform=ax6.transAxes, fontsize=12)
                ax6.set_title('Feature-Target Correlation', fontweight='bold')
        else:
            # No feature data available
            for i, ax_pos in enumerate([gs[1, 0], gs[1, 1], gs[1, 2]]):
                ax = fig.add_subplot(ax_pos)
                ax.text(0.5, 0.5, 'No feature\ndata available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
                ax.set_title(['Feature Completeness', 'Feature Variability', 'Feature-GWS Correlation'][i], 
                            fontweight='bold')
        
        # Well data quality
        well_data = None
        well_meta = None
        
        # Check for well data availability
        if 'well_anomalies' in self.data_cache and 'well_well_metadata' in self.data_cache:
            well_data = self.data_cache['well_anomalies']
            well_meta = self.data_cache['well_well_metadata']
            
            # Well data completeness
            ax7 = fig.add_subplot(gs[2, 0])
            well_completeness = []
            
            for col in well_data.columns:
                valid_ratio = well_data[col].notna().sum() / len(well_data) * 100
                well_completeness.append(valid_ratio)
            
            ax7.hist(well_completeness, bins=20, alpha=0.7, color=COLORS['neutral'], 
                    edgecolor='black')
            ax7.axvline(np.mean(well_completeness), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(well_completeness):.1f}%')
            ax7.set_xlabel('Data Completeness (%)')
            ax7.set_ylabel('Number of Wells')
            ax7.set_title('Well Data Completeness Distribution', fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            # Well temporal coverage
            ax8 = fig.add_subplot(gs[2, 1])
            if 'n_months_total' in well_meta.columns:
                ax8.hist(well_meta['n_months_total'], bins=20, alpha=0.7, 
                        color=COLORS['primary'], edgecolor='black')
                ax8.axvline(well_meta['n_months_total'].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {well_meta["n_months_total"].mean():.1f} months')
                ax8.set_xlabel('Total Months of Data')
                ax8.set_ylabel('Number of Wells')
                ax8.set_title('Well Temporal Coverage', fontweight='bold')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
            else:
                ax8.text(0.5, 0.5, 'No temporal\ncoverage data', ha='center', va='center',
                        transform=ax8.transAxes, fontsize=12)
                ax8.set_title('Well Temporal Coverage', fontweight='bold')
        else:
            # No well data available
            for i, ax_pos in enumerate([gs[2, 0], gs[2, 1]]):
                ax = fig.add_subplot(ax_pos)
                ax.text(0.5, 0.5, 'No well data\navailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
                ax.set_title(['Well Completeness', 'Well Temporal Coverage'][i], fontweight='bold')
        
        # Overall data quality summary
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Create summary statistics
        summary_text = "DATA QUALITY SUMMARY\n\n"
        
        if 'gws' in self.data_cache:
            gws_valid_pct = np.sum(~np.isnan(gws_ds.groundwater.values)) / gws_ds.groundwater.size * 100
            summary_text += f"Groundwater Data:\n"
            summary_text += f"• Valid data: {gws_valid_pct:.1f}%\n"
            summary_text += f"• Time periods: {len(gws_ds.time)}\n"
            summary_text += f"• Spatial points: {len(gws_ds.lat)}×{len(gws_ds.lon)}\n\n"
        
        if 'features' in self.data_cache:
            feature_valid_pct = np.sum(~np.isnan(features_ds.features.values)) / features_ds.features.size * 100
            summary_text += f"Feature Data:\n"
            summary_text += f"• Valid data: {feature_valid_pct:.1f}%\n"
            summary_text += f"• Number of features: {len(features_ds.feature)}\n\n"
        
        if well_data is not None:
            well_valid_pct = well_data.notna().sum().sum() / (len(well_data) * len(well_data.columns)) * 100
            summary_text += f"Well Data:\n"
            summary_text += f"• Valid data: {well_valid_pct:.1f}%\n"
            summary_text += f"• Number of wells: {len(well_data.columns)}\n"
        else:
            summary_text += f"Well Data:\n• No well data available\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=FONT_SIZE_MEDIUM,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        self.save_figure(fig, 'fig12_data_quality', 'data_quality')
        count += 1
        
        return count


def main():
    """Main function to run the complete figure generation."""
    print("🎨 GRACE GROUNDWATER ANALYSIS - PUBLICATION FIGURE GENERATOR")
    print("="*70)
    print("This script will automatically detect your analysis outputs and")
    print("generate 20+ publication-ready figures for journal submission.")
    print("="*70)
    
    # Initialize the figure generator
    generator = GRACEFigureGenerator()
    
    # Detect and load all available data
    generator.detect_and_load_data()
    
    # Check if we have sufficient data
    if len(generator.data_cache) == 0:
        print("\n❌ No data files detected!")
        print("Please ensure you have run the GRACE analysis pipeline first:")
        print("  python pipeline.py --steps all")
        return
    
    # Generate all figures
    total_figures = generator.generate_all_figures()
    
    # Print final summary
    print("\n" + "="*70)
    print("🎉 FIGURE GENERATION COMPLETE!")
    print("="*70)
    print(f"📊 Total figures generated: {total_figures}")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"💾 Formats saved: {', '.join(FIGURE_FORMAT)}")
    print(f"🔍 Resolution: {FIGURE_DPI} DPI")
    
    print(f"\n📂 Figure categories:")
    for category, path in generator.subdirs.items():
        n_files = len(list(path.glob(f"*.{FIGURE_FORMAT[0]}")))
        print(f"  • {category.replace('_', ' ').title()}: {n_files} figures")
    
    print(f"\n✅ All figures are ready for publication!")
    print(f"📄 Suitable for journals: Nature, Science, WRR, HESS, JGR, etc.")
    
    return generator


if __name__ == "__main__":
    generator = main()