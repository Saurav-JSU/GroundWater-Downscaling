import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
import argparse

def visualize_monthly_groundwater(input_file="results/groundwater_storage_anomalies.nc", 
                                output_dir="figures/monthly_groundwater",
                                start_month=None, end_month=None, 
                                create_gif=True,
                                lower_percentile=10,
                                upper_percentile=90):
    """
    Visualize monthly groundwater storage anomalies with percentile-based color scaling.
    
    Parameters:
    -----------
    input_file : str
        Path to the NetCDF file containing groundwater storage anomalies
    output_dir : str
        Directory to save the monthly maps
    start_month : str, optional
        Start month in YYYY-MM format (e.g., "2003-01")
    end_month : str, optional
        End month in YYYY-MM format (e.g., "2022-12")
    create_gif : bool
        Whether to create an animated GIF of the monthly maps
    lower_percentile : int
        Lower percentile bound for color scaling (default: 10%)
    upper_percentile : int
        Upper percentile bound for color scaling (default: 90%)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load groundwater data
    print(f"Loading groundwater data from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Groundwater data file not found: {input_file}")
    
    ds = xr.open_dataset(input_file)
    
    # Print basic information about the dataset
    print("\nDataset Information:")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"Spatial dimensions: {len(ds.lat)}×{len(ds.lon)}")
    print(f"Lat range: {ds.lat.min().values} to {ds.lat.max().values}")
    print(f"Lon range: {ds.lon.min().values} to {ds.lon.max().values}")
    
    # Get global min/max for reference
    gw_min = ds.groundwater.min().values
    gw_max = ds.groundwater.max().values
    print(f"Groundwater anomaly range: {gw_min:.2f} to {gw_max:.2f} cm")
    
    # Get percentile-based bounds for better visualization
    gw_lower = np.nanpercentile(ds.groundwater.values, lower_percentile)
    gw_upper = np.nanpercentile(ds.groundwater.values, upper_percentile)
    print(f"Using {lower_percentile}th to {upper_percentile}th percentile range: {gw_lower:.2f} to {gw_upper:.2f} cm")
    
    # Set color mapping bounds based on percentiles
    if abs(gw_lower) > abs(gw_upper):
        abs_bound = abs(gw_lower)
    else:
        abs_bound = abs(gw_upper)
    
    # Filter time range if specified
    if start_month is not None:
        ds = ds.sel(time=slice(start_month, None))
    if end_month is not None:
        ds = ds.sel(time=slice(None, end_month))
    
    # Create a figure for each month
    print("\nGenerating monthly maps...")
    filenames = []
    
    for i, time in enumerate(tqdm(ds.time.values)):
        # Create a diverging colormap centered at zero with percentile-based bounds
        norm = TwoSlopeNorm(vmin=-abs_bound, vcenter=0, vmax=abs_bound)
        
        # Create a figure with state boundaries
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Get current month's data
        current_data = ds.groundwater.sel(time=time)
        
        # Plot groundwater anomaly
        im = ax.pcolormesh(ds.lon, ds.lat, current_data, 
                        cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree())
        
        # Add state boundaries and coastline
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Groundwater Storage Anomaly (cm)')
        
        # Add title with formatted date
        month_str = str(time).split('T')[0]  # Handle different time formats
        
        # Add percentile info to the title
        plt.title(f'Groundwater Storage Anomaly - {month_str}\n(Color scale: {lower_percentile}th to {upper_percentile}th percentile)')
        
        # Save figure
        filename = output_path / f'groundwater_{month_str}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        
        filenames.append(filename)
    
    # Create animated GIF if requested
    if create_gif and len(filenames) > 0:
        try:
            from PIL import Image
            import imageio
            
            print("\nCreating animated GIF...")
            gif_path = output_path / 'groundwater_animation.gif'
            
            # Read images and create GIF
            images = [imageio.imread(filename) for filename in filenames]
            imageio.mimsave(gif_path, images, duration=0.5)  # 0.5 seconds per frame
            
            print(f"Animation saved to {gif_path}")
        except ImportError:
            print("Could not create GIF: imageio and/or PIL packages not found.")
            print("Install them with: pip install imageio pillow")
    
    print(f"\nAll maps saved to {output_path}")
    
    # Also create a 4-panel summary figure showing different time periods
    create_summary_figure(ds, output_path, abs_bound, lower_percentile, upper_percentile)

def create_summary_figure(ds, output_path, abs_bound, lower_percentile, upper_percentile):
    """Create a 4-panel summary figure showing different time periods"""
    # Calculate time indices for equal spacing through the dataset
    times = ds.time.values
    n_times = len(times)
    
    if n_times < 4:
        print("Not enough time periods for summary figure")
        return
    
    # Get 4 time periods evenly spaced
    indices = [0, n_times // 3, 2 * n_times // 3, n_times - 1]
    selected_times = [times[i] for i in indices]
    
    # Create norm using the percentile-based bounds
    norm = TwoSlopeNorm(vmin=-abs_bound, vcenter=0, vmax=abs_bound)
    
    # Create figure with 4 panels
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), 
                           subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()
    
    for i, time in enumerate(selected_times):
        ax = axs[i]
        
        # Plot groundwater anomaly
        im = ax.pcolormesh(ds.lon, ds.lat, ds.groundwater.sel(time=time), 
                        cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree())
        
        # Add state boundaries and coastline
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE)
        
        # Add title with formatted date
        month_str = str(time).split('T')[0]
        ax.set_title(f'{month_str}')
    
    # Add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Groundwater Storage Anomaly (cm)')
    
    # Add overall title with percentile information
    fig.suptitle(f'Groundwater Storage Anomalies - Quarterly View\n(Color scale: {lower_percentile}th to {upper_percentile}th percentile)', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig(output_path / 'groundwater_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary figure saved to {output_path / 'groundwater_summary.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize monthly groundwater storage anomalies')
    parser.add_argument('--input', default='results/groundwater_storage_anomalies.nc', 
                        help='Path to input NetCDF file with groundwater data')
    parser.add_argument('--output', default='figures/monthly_groundwater',
                        help='Directory to save the output maps')
    parser.add_argument('--start', default=None, help='Start month (YYYY-MM format)')
    parser.add_argument('--end', default=None, help='End month (YYYY-MM format)')
    parser.add_argument('--no-gif', action='store_true', help='Skip creating animated GIF')
    parser.add_argument('--lower-percentile', type=int, default=10, 
                        help='Lower percentile bound for color scaling (default: 10)')
    parser.add_argument('--upper-percentile', type=int, default=90, 
                        help='Upper percentile bound for color scaling (default: 90)')
    
    args = parser.parse_args()
    
    visualize_monthly_groundwater(
        input_file=args.input,
        output_dir=args.output,
        start_month=args.start,
        end_month=args.end,
        create_gif=not args.no_gif,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile
    )