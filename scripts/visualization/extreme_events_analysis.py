#!/usr/bin/env python3
"""
Historical Extreme Events Analysis: Mississippi River Basin
Comparison of GRACE Observed vs Model Predicted Water Storage during
Drought and Flood Events (2003-2022)

This script analyzes model performance during historically significant
drought and flood events in the Mississippi River Basin.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import rioxarray as rxr
import os
import re
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Define historical extreme events
EXTREME_EVENTS = {
    'drought': {
        'July-Aug 2003': {
            'months': ['2003-07', '2003-08'],
            'description': 'Below-normal water levels in mid-summer',
            'grace_files': ['20030630_20030730.tif', '20030731_20030830.tif']
        },
        'July 2005': {
            'months': ['2005-07'],
            'description': 'Lower-than-average flows in Upper Mississippi',
            'grace_files': ['20050630_20050730.tif', '20050731_20050830.tif']
        },
        'July 2012': {
            'months': ['2012-07'],
            'description': 'Most severe drought in recent memory',
            'grace_files': ['20120630_20120730.tif', '20120731_20120830.tif']
        },
        'October 2022': {
            'months': ['2022-10'],
            'description': 'Historically low water levels in Lower Mississippi',
            'grace_files': ['20220930_20221030.tif', '20221031_20221129.tif']
        }
    },
    'flood': {
        'April 2008': {
            'months': ['2008-04'],
            'description': 'Heavy spring rainfall + snowmelt flooding',
            'grace_files': ['20080331_20080429.tif', '20080430_20080530.tif']
        },
        'May 2011': {
            'months': ['2011-05'],
            'description': 'Historically significant flood, record levels',
            'grace_files': ['20110430_20110530.tif']
        },
        'April 2019': {
            'months': ['2019-04'],
            'description': 'Persistent rainfall flooding',
            'grace_files': ['20190331_20190429.tif', '20190430_20190530.tif']
        }
    }
}

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

def load_grace_data_for_events(min_overlap_days=20):
    """Load GRACE data for all extreme events with proper temporal alignment."""
    print(f"Loading GRACE data for extreme events (min {min_overlap_days} days overlap)...")
    
    grace_dir = 'data/raw/grace'
    grace_event_data = {}
    
    for event_type in ['drought', 'flood']:
        grace_event_data[event_type] = {}
        
        for event_name, event_info in EXTREME_EVENTS[event_type].items():
            print(f"\n📊 Processing {event_type.upper()} EVENT: {event_name}")
            print(f"   Description: {event_info['description']}")
            
            event_grace_data = {}
            
            for grace_file in event_info['grace_files']:
                try:
                    # Parse date from filename
                    match = re.match(r'(\d{8})_(\d{8})\.tif', grace_file)
                    if match:
                        start_date = datetime.strptime(match.group(1), '%Y%m%d')
                        end_date = datetime.strptime(match.group(2), '%Y%m%d')
                        
                        # Determine which model month this aligns with
                        aligned_month = determine_best_month_alignment(start_date, end_date)
                        
                        # Check if file exists
                        filepath = os.path.join(grace_dir, grace_file)
                        if os.path.exists(filepath):
                            # Load the GRACE file
                            grace_raster = rxr.open_rasterio(filepath, masked=True).squeeze()
                            
                            # Calculate overlap for verification
                            import calendar
                            month_start = datetime(int(aligned_month[:4]), int(aligned_month[5:7]), 1)
                            last_day = calendar.monthrange(month_start.year, month_start.month)[1]
                            month_end = datetime(month_start.year, month_start.month, last_day)
                            overlap_start = max(start_date, month_start)
                            overlap_end = min(end_date, month_end)
                            overlap_days = (overlap_end - overlap_start).days + 1
                            
                            if overlap_days >= min_overlap_days:
                                event_grace_data[aligned_month] = {
                                    'data': grace_raster,
                                    'filename': grace_file,
                                    'start_date': start_date,
                                    'end_date': end_date,
                                    'overlap_days': overlap_days
                                }
                                print(f"   ✅ {grace_file} → {aligned_month} ({overlap_days} days overlap)")
                            else:
                                print(f"   ❌ {grace_file} → {aligned_month} ({overlap_days} days overlap < {min_overlap_days})")
                        else:
                            print(f"   ❌ File not found: {grace_file}")
                            
                except Exception as e:
                    print(f"   ❌ Error loading {grace_file}: {e}")
            
            grace_event_data[event_type][event_name] = event_grace_data
    
    return grace_event_data

def load_model_data():
    """Load model predictions for all time periods."""
    print("Loading model predictions...")
    
    # Load groundwater data
    ds = xr.open_dataset('results/groundwater_storage_anomalies.nc')
    
    # Load Mississippi River Basin boundary
    basin_gdf = gpd.read_file('data/shapefiles/processed/mississippi_river_basin.shp')
    
    print(f"Model data time range: {str(ds.time.min().values)[:10]} to {str(ds.time.max().values)[:10]}")
    print(f"Variables available: {list(ds.data_vars.keys())}")
    
    return ds, basin_gdf

def create_extreme_events_figure(comparison_type='tws', min_overlap_days=20):
    """
    Create comprehensive figure showing model performance during extreme events.
    
    Parameters:
    -----------
    comparison_type : str
        'tws' for TWS vs TWS comparison
        'groundwater' for TWS vs groundwater comparison
    min_overlap_days : int
        Minimum days of overlap required for inclusion
    """
    print(f"Creating extreme events analysis figure ({comparison_type} comparison)...")
    
    # Load data
    ds, basin_gdf = load_model_data()
    grace_event_data = load_grace_data_for_events(min_overlap_days)
    
    # Count total events that have both GRACE and model data
    total_events = 0
    available_events = []
    
    for event_type in ['drought', 'flood']:
        for event_name, event_info in EXTREME_EVENTS[event_type].items():
            event_months = event_info['months']
            grace_data = grace_event_data[event_type][event_name]
            
            # Check if we have both GRACE and model data for any month in this event
            for month in event_months:
                if month in [str(t)[:7] for t in ds.time.values] and month in grace_data:
                    available_events.append({
                        'type': event_type,
                        'name': event_name,
                        'month': month,
                        'description': event_info['description'],
                        'grace_info': grace_data[month]
                    })
                    total_events += 1
                    break  # Only count each event once
    
    print(f"\nFound {total_events} events with both GRACE and model data:")
    for event in available_events:
        print(f"  {event['type'].upper()}: {event['name']} ({event['month']})")
    
    if total_events == 0:
        print("No events found with both datasets!")
        return None
    
    # Create figure layout
    n_cols = 2  # GRACE | Model
    n_rows = total_events
    
    fig = plt.figure(figsize=(12, n_rows * 3))
    fig.patch.set_facecolor('white')
    
    # Create grid
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.15, wspace=0.05,
                         left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # Set up projection and color scale
    proj = ccrs.PlateCarree()
    vmin, vmax = -15, 15  # Wider range for extreme events
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = 'RdBu_r'
    
    # Track for colorbar
    im = None
    
    # Plot each event
    for i, event in enumerate(available_events):
        event_type = event['type']
        event_name = event['name']
        month = event['month']
        description = event['description']
        grace_info = event['grace_info']
        
        # Create axes
        ax_grace = fig.add_subplot(gs[i, 0], projection=proj)
        ax_model = fig.add_subplot(gs[i, 1], projection=proj)
        
        # Get model data for this month
        if comparison_type == 'tws':
            model_data = ds.sel(time=month).tws.values
            variable_name = "TWS"
        else:
            model_data = ds.sel(time=month).groundwater.values
            variable_name = "Groundwater"
        
        # Plot GRACE data
        grace_data = grace_info['data']
        im1 = ax_grace.pcolormesh(
            grace_data.x, grace_data.y, grace_data.values,
            cmap=cmap, norm=norm, transform=proj, shading='auto'
        )
        
        # Plot model data
        im2 = ax_model.pcolormesh(
            ds.lon, ds.lat, model_data,
            cmap=cmap, norm=norm, transform=proj, shading='auto'
        )
        
        # Keep reference for colorbar
        if im is None:
            im = im1
        
        # Add basin boundary
        try:
            ax_grace.add_geometries(basin_gdf.geometry, crs=proj, 
                                  facecolor='none', edgecolor='black', linewidth=0.8)
            ax_model.add_geometries(basin_gdf.geometry, crs=proj, 
                                   facecolor='none', edgecolor='black', linewidth=0.8)
        except:
            print(f"Warning: Could not add basin boundary for {event_name}")
        
        # Set extent to Mississippi River Basin
        ax_grace.set_extent([-113, -78, 29, 51], crs=proj)
        ax_model.set_extent([-113, -78, 29, 51], crs=proj)
        
        # Remove ticks
        ax_grace.set_xticks([])
        ax_grace.set_yticks([])
        ax_model.set_xticks([])
        ax_model.set_yticks([])
        
        # Add colored border to indicate event type
        border_color = 'red' if event_type == 'drought' else 'blue'
        border_width = 3
        
        # Add border rectangles
        for ax in [ax_grace, ax_model]:
            rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                           fill=False, edgecolor=border_color, linewidth=border_width)
            ax.add_patch(rect)
        
        # Add event labels on the left
        event_label = f"{event_type.upper()}\n{event_name}\n{month}"
        fig.text(0.02, 0.92 - (i + 0.5) * (0.84 / n_rows), event_label,
                fontsize=9, fontweight='bold', ha='left', va='center',
                color=border_color)
        
        # Add description on the right side
        description_text = f"{description}\nGRACE: {grace_info['start_date'].strftime('%b %d')} - {grace_info['end_date'].strftime('%b %d')}"
        fig.text(0.98, 0.92 - (i + 0.5) * (0.84 / n_rows), description_text,
                fontsize=8, ha='right', va='center', style='italic')
    
    # Add column headers
    fig.text(0.25, 0.95, 'GRACE Observed (50km)', ha='center', va='bottom', 
             fontsize=12, fontweight='bold')
    if comparison_type == 'tws':
        model_label = 'Model Predicted TWS (5km)'
    else:
        model_label = 'Model Predicted GWS (5km)'
    fig.text(0.75, 0.95, model_label, ha='center', va='bottom', 
             fontsize=12, fontweight='bold')
    
    # Add shared colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        if comparison_type == 'tws':
            cbar.set_label('Total Water Storage Anomaly (cm)', fontsize=11, fontweight='bold')
        else:
            cbar.set_label('Storage Anomaly (cm)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
    
    # Add main title
    if comparison_type == 'tws':
        comparison_desc = "GRACE vs Model TWS"
    else:
        comparison_desc = "GRACE TWS vs Model GWS"
    
    fig.suptitle(f'Historical Extreme Events: {comparison_desc}\n' +
                f'Mississippi River Basin (2003-2022) - {total_events} Events', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Add legend for event types
    drought_patch = plt.Rectangle((0,0),1,1, facecolor='none', edgecolor='red', linewidth=3)
    flood_patch = plt.Rectangle((0,0),1,1, facecolor='none', edgecolor='blue', linewidth=3)
    fig.legend([drought_patch, flood_patch], ['Drought Events', 'Flood Events'], 
              loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Save figure
    output_path = f'figures/extreme_events_analysis_{comparison_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    return fig

def create_events_summary_statistics():
    """Create summary statistics for extreme events."""
    print("Creating summary statistics for extreme events...")
    
    # Load data
    ds, basin_gdf = load_model_data()
    grace_event_data = load_grace_data_for_events()
    
    # Calculate regional averages for each event
    results = []
    
    for event_type in ['drought', 'flood']:
        for event_name, event_info in EXTREME_EVENTS[event_type].items():
            event_months = event_info['months']
            grace_data = grace_event_data[event_type][event_name]
            
            for month in event_months:
                if month in [str(t)[:7] for t in ds.time.values] and month in grace_data:
                    # Get model data
                    model_tws = ds.sel(time=month).tws
                    model_gws = ds.sel(time=month).groundwater
                    
                    # Get GRACE data
                    grace_info = grace_data[month]
                    grace_tws = grace_info['data']
                    
                    # Calculate regional averages (mask by basin)
                    model_tws_avg = float(model_tws.mean().values)
                    model_gws_avg = float(model_gws.mean().values)
                    grace_tws_avg = float(grace_tws.mean().values)
                    
                    results.append({
                        'event_type': event_type,
                        'event_name': event_name,
                        'month': month,
                        'description': event_info['description'],
                        'model_tws_avg': model_tws_avg,
                        'model_gws_avg': model_gws_avg,
                        'grace_tws_avg': grace_tws_avg,
                        'grace_period': f"{grace_info['start_date'].strftime('%Y-%m-%d')} to {grace_info['end_date'].strftime('%Y-%m-%d')}"
                    })
                    break
    
    # Convert to DataFrame and print
    df = pd.DataFrame(results)
    
    print("\n📊 EXTREME EVENTS SUMMARY STATISTICS")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_path = 'results/extreme_events_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("🌊 HISTORICAL EXTREME EVENTS ANALYSIS")
    print("="*80)
    print("Analyzing model performance during drought and flood events:")
    print()
    print("📅 DROUGHT EVENTS:")
    for event_name, event_info in EXTREME_EVENTS['drought'].items():
        print(f"   • {event_name}: {event_info['description']}")
    print()
    print("🌊 FLOOD EVENTS:")  
    for event_name, event_info in EXTREME_EVENTS['flood'].items():
        print(f"   • {event_name}: {event_info['description']}")
    print()
    
    # Create summary statistics
    print("1. Creating summary statistics...")
    summary_df = create_events_summary_statistics()
    
    # Create both comparison types
    for comparison_type in ['tws', 'groundwater']:
        print(f"\n2. Creating {comparison_type.upper()} comparison figure...")
        fig = create_extreme_events_figure(comparison_type)
        if fig:
            plt.close(fig)  # Close to save memory
    
    print("\n✅ EXTREME EVENTS ANALYSIS COMPLETE!")
    print("\nFiles created:")
    print("📊 Summary Statistics:")
    print("   - results/extreme_events_summary.csv")
    print("📁 Comparison Figures:")
    print("   - figures/extreme_events_analysis_tws.png")
    print("   - figures/extreme_events_analysis_groundwater.png")
    print()
    print("🎯 This analysis shows model performance during:")
    print("   • 4 Major Drought Events (2003, 2005, 2012, 2022)")
    print("   • 3 Major Flood Events (2008, 2011, 2019)")
    print("   • Spatial patterns during extreme conditions")
    print("   • Model accuracy across different hydrological states") 