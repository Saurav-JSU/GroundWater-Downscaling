import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
import joblib
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

def visualize_grace_downscaling():
    """Generate visualizations comparing original and downscaled GRACE data."""
    # Create output directory
    output_dir = Path("visualizations/grace_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature stack and model
    print("Loading feature stack and model...")
    ds = xr.open_dataset("data/processed/feature_stack.nc")
    model = joblib.load("models/rf_model.joblib")
    
    # Load original GRACE data
    print("Loading original GRACE data...")
    grace_dir = Path("data/raw/grace")
    grace_files = sorted([f for f in os.listdir(grace_dir) if f.endswith(".tif")])
    
    # Extract dates from filenames
    grace_dates = []
    for filename in grace_files:
        # Pattern: 20030131_20030227.tif
        match = re.match(r'(\d{8})_(\d{8})\.tif', filename)
        if match:
            start_date = datetime.strptime(match.group(1), '%Y%m%d')
            # Use mid-month date for labeling
            grace_dates.append(start_date.strftime('%Y-%m'))
    
    # Process each month
    print("Generating visualizations...")
    for i, (date, filename) in enumerate(tqdm(zip(grace_dates, grace_files))):
        # 1. Load original GRACE
        grace_path = grace_dir / filename
        original_grace = rxr.open_rasterio(grace_path, masked=True).squeeze()
        
        # 2. Get corresponding feature data for this date
        try:
            month_features = ds.sel(time=date)
        except KeyError:
            print(f"No feature data for {date}, skipping")
            continue
            
        # 3. Prepare feature data for prediction
        # Get the feature values and reshape properly
        features_data = month_features.features.values  # Shape (num_features, lat, lon)
        
        # Check if static features exist and prepare them
        has_static = 'static_features' in ds
        if has_static:
            static_data = ds.static_features.values  # Shape (num_static_features, lat, lon)
            
            # Get dimensions
            n_features = features_data.shape[0]
            n_static = static_data.shape[0]
            spatial_shape = features_data.shape[1:]  # (lat, lon)
            
            # Create a combined feature array
            # First flatten the spatial dimensions
            features_flat = features_data.reshape(n_features, -1)  # (n_features, lat*lon)
            static_flat = static_data.reshape(n_static, -1)  # (n_static, lat*lon)
            
            # Combine along feature dimension
            combined = np.vstack([features_flat, static_flat])  # (n_features+n_static, lat*lon)
            
            # Transpose for sklearn API: (n_samples, n_features)
            X = combined.T  # (lat*lon, n_features+n_static)
        else:
            # Just use temporal features
            # Flatten spatial dimensions and transpose
            X = features_data.reshape(features_data.shape[0], -1).T  # (lat*lon, n_features)
        
        # 4. Make predictions
        y_pred = model.predict(X)
        
        # Reshape back to spatial dimensions
        downscaled = y_pred.reshape(original_grace.shape)
        
        # 5. Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Find data min/max for consistent colormap
        vmin = min(np.nanmin(original_grace), np.nanmin(downscaled))
        vmax = max(np.nanmax(original_grace), np.nanmax(downscaled))
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        # Original GRACE
        im1 = axs[0].imshow(original_grace, cmap='RdBu', norm=norm)
        axs[0].set_title(f"Original GRACE: {date}")
        axs[0].axis('off')
        
        # Downscaled 
        im2 = axs[1].imshow(downscaled, cmap='RdBu', norm=norm)
        axs[1].set_title(f"Downscaled: {date}")
        axs[1].axis('off')
        
        plt.colorbar(im1, ax=axs[0], label='TWS Anomaly (cm)')
        plt.colorbar(im2, ax=axs[1], label='TWS Anomaly (cm)')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"grace_comparison_{date}.png", dpi=150)
        plt.close()
    
    # Create summary visualization (first, middle, last month)
    print("Creating summary comparison...")
    dates_to_show = [grace_dates[0], grace_dates[len(grace_dates)//2], grace_dates[-1]]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, date in enumerate(dates_to_show):
        try:
            # Load images instead of recalculating
            img = plt.imread(output_dir / f"grace_comparison_{date}.png")
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(date)
        except FileNotFoundError:
            print(f"No visualization found for {date}")
    
    # Add overall title
    fig.suptitle("GRACE TWS: Original vs Downscaled (2003-2022)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "grace_summary_comparison.png", dpi=200)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_grace_downscaling()