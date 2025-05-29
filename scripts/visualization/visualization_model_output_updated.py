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

def create_lagged_features(features_data, time_indices, lag_months=[1, 3, 6]):
    """
    Create lagged features for model input with multiple time periods.
    
    Parameters:
    -----------
    features_data : numpy.ndarray
        Array of shape (time, features, lat, lon) containing temporal features
    time_indices : list
        List of time indices or dates corresponding to the time dimension
    lag_months : list
        List of lag periods to include
        
    Returns:
    --------
    numpy.ndarray
        Array with original and lagged features concatenated along feature dimension
    """
    print(f"Creating lagged features with lags: {lag_months}")
    
    # Get dimensions
    n_times, n_features, n_lat, n_lon = features_data.shape
    
    # Create a time dictionary for easier lookup
    time_dict = {}
    for i, t in enumerate(time_indices):
        # Convert to string in YYYY-MM format if needed
        if isinstance(t, (np.datetime64, pd.Timestamp, datetime)):
            t_str = pd.to_datetime(t).strftime('%Y-%m')
        else:
            # Try to parse t if it's a string already
            try:
                t_str = pd.to_datetime(t).strftime('%Y-%m')
            except:
                t_str = str(t)
        time_dict[t_str] = i
    
    # Initialize array to hold all features (original + lagged)
    n_lags = len(lag_months)
    total_features = n_features * (1 + n_lags)  # Original + lags
    X_with_lags = np.zeros((n_times, total_features, n_lat, n_lon))
    
    # Copy original features
    X_with_lags[:, :n_features, :, :] = features_data
    
    # Add lagged features
    feature_idx = n_features
    for lag in lag_months:
        print(f"  Adding lag {lag} features")
        
        for t_idx, t in enumerate(time_indices):
            # Get target date for lag
            t_date = pd.to_datetime(t)
            year, month = t_date.year, t_date.month
            
            # Calculate lagged date
            lag_month = month - lag
            lag_year = year
            while lag_month <= 0:
                lag_month += 12
                lag_year -= 1
            
            lagged_date = f"{lag_year:04d}-{lag_month:02d}"
            
            # If lagged date exists in our dataset, use it
            if lagged_date in time_dict:
                lag_idx = time_dict[lagged_date]
                X_with_lags[t_idx, feature_idx:feature_idx+n_features, :, :] = features_data[lag_idx, :, :, :]
            else:
                # Use zeros or previous time point if lagged date not available
                if t_idx >= lag:
                    X_with_lags[t_idx, feature_idx:feature_idx+n_features, :, :] = features_data[t_idx-lag, :, :, :]
                # Otherwise leave as zeros
        
        feature_idx += n_features
    
    return X_with_lags

def create_seasonal_features(time_indices, shape):
    """
    Create seasonal features using cyclical encoding of months.
    
    Parameters:
    -----------
    time_indices : list
        List of time indices or dates
    shape : tuple
        Spatial shape (lat, lon) for broadcasting
        
    Returns:
    --------
    numpy.ndarray
        Array with seasonal features of shape (time, 2, lat, lon)
    """
    n_times = len(time_indices)
    seasonal = np.zeros((n_times, 2, shape[0], shape[1]))
    
    for i, t in enumerate(time_indices):
        # Get month from time index
        if isinstance(t, (np.datetime64, pd.Timestamp, datetime)):
            month = pd.to_datetime(t).month
        else:
            # Try to parse t if it's a string
            try:
                month = pd.to_datetime(t).month
            except:
                # Default to January if can't parse
                month = 1
                print(f"Warning: Could not extract month from {t}, using default")
        
        # Create cyclical encoding
        sin_val = np.sin(2 * np.pi * month / 12)
        cos_val = np.cos(2 * np.pi * month / 12)
        
        # Broadcast to spatial dimensions
        seasonal[i, 0, :, :] = sin_val
        seasonal[i, 1, :, :] = cos_val
    
    return seasonal

def prepare_model_input(ds, time_index, lag_months=[1, 3, 6]):
    """
    Prepare enhanced input for RF model prediction.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing features
    time_index : str
        Time index to prepare features for
    lag_months : list
        List of lag periods to include
        
    Returns:
    --------
    numpy.ndarray
        Prepared input array ready for model prediction
    """
    # Extract features for target time
    try:
        target_features = ds.sel(time=time_index).features.values
        feature_shape = target_features.shape
        spatial_shape = feature_shape[1:]  # (lat, lon)
        
        # Prepare lagged features if possible
        lagged_features = []
        
        # Add current features
        all_features = [target_features]
        
        # Try to add lagged features
        for lag in lag_months:
            try:
                # Calculate lagged date
                target_date = pd.to_datetime(time_index)
                year, month = target_date.year, target_date.month
                
                lag_month = month - lag
                lag_year = year
                while lag_month <= 0:
                    lag_month += 12
                    lag_year -= 1
                
                lagged_date = f"{lag_year:04d}-{lag_month:02d}"
                
                # Try to get data for lagged date
                lagged_feature = ds.sel(time=lagged_date).features.values
                all_features.append(lagged_feature)
            except KeyError:
                # If lagged date not available, use zeros
                all_features.append(np.zeros_like(target_features))
        
        # Create seasonal features (sin/cos encoding of month)
        month = pd.to_datetime(time_index).month
        month_sin = np.sin(2 * np.pi * month / 12) * np.ones(spatial_shape)
        month_cos = np.cos(2 * np.pi * month / 12) * np.ones(spatial_shape)
        
        all_features.append(month_sin[np.newaxis, :, :])
        all_features.append(month_cos[np.newaxis, :, :])
        
        # Add static features if available
        if 'static_features' in ds:
            static_features = ds.static_features.values
            all_features.append(static_features)
        
        # Convert to single array and reshape for model input
        X = np.vstack(all_features)
        X_flat = X.reshape(X.shape[0], -1).T
        
        return X_flat
    
    except Exception as e:
        print(f"Error preparing input for {time_index}: {e}")
        return None

def visualize_grace_downscaling():
    """Generate visualizations comparing original and downscaled GRACE data using enhanced model."""
    # Create output directory
    output_dir = Path("visualizations/grace_comparison_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature stack and enhanced model
    print("Loading feature stack and enhanced model...")
    ds = xr.open_dataset("data/processed/feature_stack.nc")
    model = joblib.load("models/rf_model_enhanced.joblib")
    
    # Print model input information
    print(f"Model expects {model.n_features_in_} features as input")
    
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
    successful_vizs = 0
    
    for i, (date, filename) in enumerate(tqdm(zip(grace_dates, grace_files))):
        # 1. Load original GRACE
        grace_path = grace_dir / filename
        original_grace = rxr.open_rasterio(grace_path, masked=True).squeeze()
        
        # 2. Get corresponding feature data for this date
        try:
            # Prepare enhanced input with lagged features, seasonal features and static features
            X = prepare_model_input(ds, date)
            
            if X is None:
                print(f"  Skipping {date} due to input preparation error")
                continue
                
            # Check if feature count matches model expectations
            if X.shape[1] != model.n_features_in_:
                print(f"  Feature count mismatch for {date}: got {X.shape[1]}, expected {model.n_features_in_}")
                continue
            
            # 3. Make predictions
            y_pred = model.predict(X)
            
            # Reshape back to spatial dimensions
            downscaled = y_pred.reshape(original_grace.shape)
            
            # 4. Create visualization
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
            axs[1].set_title(f"Enhanced Model: {date}")
            axs[1].axis('off')
            
            plt.colorbar(im1, ax=axs[0], label='TWS Anomaly (cm)')
            plt.colorbar(im2, ax=axs[1], label='TWS Anomaly (cm)')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"grace_comparison_{date}.png", dpi=150)
            plt.close()
            
            successful_vizs += 1
            
            # Create a difference map every 10 months
            if i % 10 == 0:
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                # Original GRACE
                im1 = axs[0].imshow(original_grace, cmap='RdBu', norm=norm)
                axs[0].set_title(f"Original GRACE: {date}")
                axs[0].axis('off')
                
                # Downscaled 
                im2 = axs[1].imshow(downscaled, cmap='RdBu', norm=norm)
                axs[1].set_title(f"Enhanced Model: {date}")
                axs[1].axis('off')
                
                # Difference
                diff = downscaled - original_grace
                diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
                diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
                
                im3 = axs[2].imshow(diff, cmap='RdBu', norm=diff_norm)
                axs[2].set_title(f"Difference (Model - Original)")
                axs[2].axis('off')
                
                plt.colorbar(im1, ax=axs[0], label='TWS Anomaly (cm)')
                plt.colorbar(im2, ax=axs[1], label='TWS Anomaly (cm)')
                plt.colorbar(im3, ax=axs[2], label='Difference (cm)')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"grace_diff_{date}.png", dpi=200)
                plt.close()
                
        except Exception as e:
            print(f"Error processing {date}: {e}")
            continue
    
    print(f"Successfully created {successful_vizs} visualizations")
    
    # Create summary visualization (first, middle, last month)
    if successful_vizs > 0:
        try:
            print("Creating summary comparison...")
            # Get all image files
            viz_files = sorted(list(output_dir.glob("grace_comparison_*.png")))
            
            if len(viz_files) >= 3:
                # Select representative images (first, middle, last)
                selected_files = [viz_files[0], 
                                  viz_files[len(viz_files)//2], 
                                  viz_files[-1]]
                
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                for i, img_file in enumerate(selected_files):
                    img = plt.imread(img_file)
                    axs[i].imshow(img)
                    axs[i].axis('off')
                    
                    # Extract date from filename
                    date_match = re.search(r'(\d{4}-\d{2})\.png', str(img_file))
                    if date_match:
                        date = date_match.group(1)
                        axs[i].set_title(date)
                
                # Add overall title
                fig.suptitle("Enhanced Model: Original vs Downscaled GRACE TWS", fontsize=16)
                plt.tight_layout()
                plt.savefig(output_dir / "summary_comparison.png", dpi=200)
                plt.close()
        except Exception as e:
            print(f"Error creating summary: {e}")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_grace_downscaling()