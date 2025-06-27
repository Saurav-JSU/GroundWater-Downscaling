#!/usr/bin/env python3
"""
Selective Figure Generator for GRACE Analysis
============================================

This script allows you to generate specific figures from the advanced spatial-temporal analysis
without running the entire pipeline. Choose which figures you want to create.

Usage:
    python scripts/visualization/selective_figure_generator.py

Author: GRACE Analysis Pipeline
Date: 2024
"""

import os
import sys
from pathlib import Path

# Add the parent directory to path to import the main class
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.visualization.advanced_spatial_temporal_analysis import AdvancedGRACEVisualizer


def print_menu():
    """Display the menu of available figures."""
    print("\n" + "="*60)
    print("🎨 SELECTIVE GRACE FIGURE GENERATOR")
    print("="*60)
    print("Choose which figures to generate:")
    print()
    print("📊 OVERVIEW FIGURES:")
    print("  1. Groundwater Overview Maps")
    print("  2. Spatial Statistics Maps") 
    print("  3. Time Series Analysis")
    print()
    print("📈 TREND ANALYSIS:")
    print("  4. Combined Trend Maps (with significance)")
    print("  5. Combined Uncertainty Maps")
    print("  6. Individual Component Trends (all)")
    print()
    print("⏰ EXTREME ANALYSIS:")
    print("  7. Groundwater Extreme Timing")
    print("  8. Total Water Storage Extreme Timing")
    print()
    print("🌍 REGIONAL ANALYSIS:")
    print("  9. HUC-based Regional Analysis")
    print("  10. State-based Regional Analysis")
    print()
    print("📄 UTILITIES:")
    print("  11. Generate Analysis Report")
    print("  12. ALL FIGURES (full pipeline)")
    print("  0. Exit")
    print()


def get_user_choices():
    """Get user's figure choices."""
    print_menu()
    
    while True:
        try:
            choices_input = input("Enter your choices (comma-separated, e.g., 1,4,7 or 'all' for everything): ").strip()
            
            if choices_input.lower() in ['0', 'exit', 'quit']:
                return []
            
            if choices_input.lower() in ['all', '12']:
                return list(range(1, 12))
            
            # Parse comma-separated choices
            choices = []
            for choice in choices_input.split(','):
                choice = choice.strip()
                if choice.isdigit():
                    num = int(choice)
                    if 1 <= num <= 12:
                        choices.append(num)
                    else:
                        print(f"⚠️ Invalid choice: {num}. Please choose 1-12.")
                        continue
                else:
                    print(f"⚠️ Invalid input: {choice}. Please enter numbers only.")
                    continue
            
            if choices:
                return sorted(list(set(choices)))  # Remove duplicates and sort
            else:
                print("⚠️ No valid choices entered. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return []
        except Exception as e:
            print(f"⚠️ Error: {e}. Please try again.")


def execute_figure_generation(visualizer, choices):
    """Execute the selected figure generation tasks."""
    
    print(f"\n🚀 Generating {len(choices)} selected figure(s)...")
    
    for choice in choices:
        print(f"\n{'='*50}")
        
        try:
            if choice == 1:
                print("1️⃣ Creating Groundwater Overview Maps...")
                visualizer.create_robust_overview_maps()
                
            elif choice == 2:
                print("2️⃣ Creating Spatial Statistics Maps...")
                visualizer.create_robust_spatial_statistics()
                
            elif choice == 3:
                print("3️⃣ Creating Time Series Analysis...")
                visualizer.create_robust_time_series_analysis()
                
            elif choice == 4:
                print("4️⃣ Creating Combined Trend Maps...")
                visualizer.create_combined_trend_maps(clip_to_shapefile=True)
                
            elif choice == 5:
                print("5️⃣ Creating Combined Uncertainty Maps...")
                visualizer.create_combined_uncertainty_maps(clip_to_shapefile=True)
                
            elif choice == 6:
                print("6️⃣ Creating Individual Component Trends...")
                # Create individual trend maps for each component
                if hasattr(visualizer, 'gws_ds'):
                    print("   📊 Groundwater Storage...")
                    visualizer.create_trend_map_with_significance(
                        visualizer.gws_ds.groundwater, 
                        'Groundwater Storage',
                        units='cm/year',
                        clip_to_shapefile=True
                    )
                
                if hasattr(visualizer, 'gws_ds') and 'soil_moisture_anomaly' in visualizer.gws_ds:
                    print("   📊 Soil Moisture...")
                    visualizer.create_trend_map_with_significance(
                        visualizer.gws_ds.soil_moisture_anomaly,
                        'Soil Moisture',
                        units='cm/year',
                        clip_to_shapefile=True
                    )
                
                if hasattr(visualizer, 'gws_ds') and 'swe_anomaly' in visualizer.gws_ds:
                    print("   📊 Snow Water Equivalent...")
                    visualizer.create_trend_map_with_significance(
                        visualizer.gws_ds.swe_anomaly,
                        'Snow Water Equivalent',
                        units='cm/year',
                        clip_to_shapefile=True
                    )
                
                if visualizer.features_ds is not None:
                    print("   📊 Precipitation...")
                    precip_indices = []
                    for i, feat in enumerate(visualizer.features_ds.feature.values):
                        if 'pr' in str(feat).lower() or 'chirps' in str(feat).lower():
                            precip_indices.append(i)
                    
                    if precip_indices:
                        precip_data = visualizer.features_ds.features[:, precip_indices[0], :, :]
                        visualizer.create_trend_map_with_significance(
                            precip_data,
                            'Precipitation',
                            units='mm/year',
                            clip_to_shapefile=True
                        )
                
            elif choice == 7:
                print("7️⃣ Creating Groundwater Extreme Timing...")
                # Create just the groundwater extreme timing
                create_single_extreme_timing(visualizer, 'groundwater')
                
            elif choice == 8:
                print("8️⃣ Creating Total Water Storage Extreme Timing...")
                # Create just the TWS extreme timing
                create_single_extreme_timing(visualizer, 'tws')
                
            elif choice == 9:
                print("9️⃣ Creating HUC-based Regional Analysis...")
                # HUC-based regions
                if os.path.exists("data/shapefiles/processed/subregions_huc"):
                    regions = {}
                    for shp in Path("data/shapefiles/processed/subregions_huc").glob("*.shp"):
                        regions[shp.stem] = str(shp)
                    if regions:
                        visualizer.create_regional_analysis(regions)
                    else:
                        print("   ⚠️ No HUC subregions found")
                else:
                    print("   ⚠️ HUC subregions directory not found")
                
            elif choice == 10:
                print("🔟 Creating State-based Regional Analysis...")
                # State-based regions (first 5)
                if os.path.exists("data/shapefiles/processed/individual_states"):
                    state_files = list(Path("data/shapefiles/processed/individual_states").glob("*.shp"))[:5]
                    if state_files:
                        state_regions = {shp.stem: str(shp) for shp in state_files}
                        visualizer.create_regional_analysis(state_regions)
                    else:
                        print("   ⚠️ No state shapefiles found")
                else:
                    print("   ⚠️ States directory not found")
                
            elif choice == 11:
                print("1️⃣1️⃣ Generating Analysis Report...")
                visualizer.create_comprehensive_report()
                
            elif choice == 12:
                print("1️⃣2️⃣ Running Full Pipeline...")
                # Run everything
                visualizer.create_robust_overview_maps()
                visualizer.create_robust_spatial_statistics()
                visualizer.create_robust_time_series_analysis()
                visualizer.analyze_all_components()
                visualizer.create_extreme_timing_maps()
                
                # Regional analysis
                if os.path.exists("data/shapefiles/processed/subregions_huc"):
                    regions = {}
                    for shp in Path("data/shapefiles/processed/subregions_huc").glob("*.shp"):
                        regions[shp.stem] = str(shp)
                    if regions:
                        visualizer.create_regional_analysis(regions)
                
                if os.path.exists("data/shapefiles/processed/individual_states"):
                    state_files = list(Path("data/shapefiles/processed/individual_states").glob("*.shp"))[:5]
                    if state_files:
                        state_regions = {shp.stem: str(shp) for shp in state_files}
                        visualizer.create_regional_analysis(state_regions)
                
                visualizer.create_comprehensive_report()
                
        except Exception as e:
            print(f"❌ Error generating figure {choice}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Completed generating {len(choices)} figure(s)!")
    print(f"📁 Check results in: {visualizer.figures_dir}")


def create_single_extreme_timing(visualizer, variable_type):
    """Create extreme timing map for a single variable."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import TwoSlopeNorm, ListedColormap
    
    # Get the appropriate data
    if variable_type == 'groundwater' and hasattr(visualizer, 'gws_ds'):
        data_array = visualizer.gws_ds.groundwater
        display_name = 'Groundwater Storage'
        var_name = 'groundwater'
    elif variable_type == 'tws' and hasattr(visualizer, 'gws_ds') and 'tws' in visualizer.gws_ds:
        data_array = visualizer.gws_ds.tws
        display_name = 'Total Water Storage'
        var_name = 'tws'
    else:
        print(f"   ⚠️ {variable_type} data not available")
        return
    
    print(f"   📊 Processing {display_name}")
    
    # Find timing of minimum and maximum for each pixel
    n_lat, n_lon = len(data_array.lat), len(data_array.lon)
    
    min_time_index = np.full((n_lat, n_lon), np.nan)
    min_month = np.full((n_lat, n_lon), np.nan)
    
    # Convert time coordinate to pandas datetime
    time_as_datetime = pd.to_datetime(data_array.time.values)
    
    for i in range(n_lat):
        for j in range(n_lon):
            ts = data_array[:, i, j].values
            if not np.all(np.isnan(ts)):
                # Find minimum
                min_idx = np.nanargmin(ts)
                min_time_index[i, j] = min_idx
                min_month[i, j] = time_as_datetime[min_idx].month
    
    # Apply mask if using shapefile
    if visualizer.region_mask is not None:
        min_time_index = np.where(~np.isnan(visualizer.region_mask), 
                                min_time_index, np.nan)
        min_month = np.where(~np.isnan(visualizer.region_mask), 
                           min_month, np.nan)
    
    # Create figure with 4 panels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12),
                                                subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Common extent
    extent = visualizer._get_safe_extent(data_array)
    
    # 1. Year of minimum (with fixes)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Convert time index to year
    min_year = np.full_like(min_time_index, np.nan)
    valid = ~np.isnan(min_time_index)
    for i in range(n_lat):
        for j in range(n_lon):
            if valid[i, j]:
                idx = int(min_time_index[i, j])
                min_year[i, j] = time_as_datetime[idx].year
    
    # Create discrete year ticks for better display
    year_min = int(np.nanmin(min_year))
    year_max = int(np.nanmax(min_year))
    year_ticks = list(range(year_min, year_max + 1, 2))  # Every 2 years
    
    im1 = ax1.pcolormesh(data_array.lon, data_array.lat, min_year,
                       cmap='viridis', vmin=year_min, vmax=year_max,
                       transform=ccrs.PlateCarree())
    ax1.add_feature(cfeature.STATES, linewidth=0.5)
    ax1.add_feature(cfeature.COASTLINE)
    
    # Add shapefile outline
    if visualizer.shapefile is not None:
        try:
            for idx, row in visualizer.shapefile.iterrows():
                geom = row.geometry
                if geom.type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax1.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                elif geom.type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax1.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
        except:
            pass
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='Year', ticks=year_ticks)
    cbar1.ax.set_yticklabels([str(y) for y in year_ticks])
    ax1.set_title(f'Year of Minimum {display_name}')
    
    # 2. Month of minimum (with improved colormap)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Use discrete colormap for months with better gradual transition
    month_colors = plt.cm.tab20c(np.linspace(0, 1, 12))
    cmap_months = ListedColormap(month_colors)
    
    im2 = ax2.pcolormesh(data_array.lon, data_array.lat, min_month,
                       cmap=cmap_months, vmin=0.5, vmax=12.5,
                       transform=ccrs.PlateCarree())
    ax2.add_feature(cfeature.STATES, linewidth=0.5)
    ax2.add_feature(cfeature.COASTLINE)
    
    # Add shapefile outline
    if visualizer.shapefile is not None:
        try:
            for idx, row in visualizer.shapefile.iterrows():
                geom = row.geometry
                if geom.type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax2.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                elif geom.type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax2.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
        except:
            pass
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='Month', ticks=range(1, 13))
    cbar2.ax.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax2.set_title(f'Month of Typical Minimum {display_name}')
    
    # 3. Minimum value magnitude
    ax3.set_extent(extent, crs=ccrs.PlateCarree())
    
    min_values = np.full((n_lat, n_lon), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            ts = data_array[:, i, j].values
            if not np.all(np.isnan(ts)):
                min_values[i, j] = np.nanmin(ts)
    
    if visualizer.region_mask is not None:
        min_values = np.where(~np.isnan(visualizer.region_mask), 
                            min_values, np.nan)
    
    vmax = np.nanpercentile(np.abs(min_values), 95)
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im3 = ax3.pcolormesh(data_array.lon, data_array.lat, min_values,
                       cmap='RdBu_r', norm=norm,
                       transform=ccrs.PlateCarree())
    ax3.add_feature(cfeature.STATES, linewidth=0.5)
    ax3.add_feature(cfeature.COASTLINE)
    
    # Add shapefile outline
    if visualizer.shapefile is not None:
        try:
            for idx, row in visualizer.shapefile.iterrows():
                geom = row.geometry
                if geom.type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax3.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
                elif geom.type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax3.plot(x, y, 'k-', linewidth=2, transform=ccrs.PlateCarree())
        except:
            pass
    
    plt.colorbar(im3, ax=ax3, label='Minimum Value (cm)')
    ax3.set_title(f'Magnitude of Minimum {display_name}')
    
    # 4. Histogram of extreme years (with fixed ticks)
    ax4.remove()  # Remove map projection
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Count occurrences by year
    years = []
    valid = ~np.isnan(min_year)
    for y in min_year[valid]:
        years.append(int(y))
    
    if years:
        year_counts = pd.Series(years).value_counts().sort_index()
        
        ax4.bar(year_counts.index, year_counts.values, color='darkred', alpha=0.7)
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Number of Pixels with Annual Minimum')
        ax4.set_title(f'Distribution of {display_name} Minima by Year')
        ax4.grid(True, alpha=0.3)
        
        # Fix x-axis to show integer years
        year_min = min(year_counts.index)
        year_max = max(year_counts.index)
        year_ticks = list(range(year_min, year_max + 1, 2))  # Every 2 years
        ax4.set_xticks(year_ticks)
        ax4.set_xticklabels([str(y) for y in year_ticks])
        
        # Rotate x labels
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle(f'Timing and Magnitude of {display_name} Extremes', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{var_name}_extreme_timing.png"
    plt.savefig(visualizer.subdirs['extremes'] / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      💾 Saved: {filename}")


def main():
    """Main function."""
    print("🚀 GRACE Selective Figure Generator")
    print("="*50)
    
    # Initialize visualizer
    shapefile_path = "data/shapefiles/processed/mississippi_river_basin.shp"
    
    if not os.path.exists(shapefile_path):
        print("⚠️ Mississippi Basin shapefile not found!")
        print("📌 Run this first: python scripts/create_mississippi_basin_from_huc.py")
        shapefile_path = None
    
    print("📦 Initializing GRACE visualizer...")
    visualizer = AdvancedGRACEVisualizer(
        base_dir=".",
        shapefile_path=shapefile_path
    )
    
    # Get user choices
    choices = get_user_choices()
    
    if not choices:
        print("👋 No figures selected. Goodbye!")
        return
    
    # Execute selected figure generation
    execute_figure_generation(visualizer, choices)
    
    print("\n🎉 Selective figure generation complete!")


if __name__ == "__main__":
    main() 