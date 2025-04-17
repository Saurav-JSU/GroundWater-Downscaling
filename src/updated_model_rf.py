# Simplified model_rf.py - avoiding timestamp type issues
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import rioxarray as rxr
import re
from datetime import datetime
import pandas as pd
import joblib

def load_grace_tws(grace_dir):
    """Load all available GRACE TWS data and return with exact timestamps."""
    print(f"Loading all available GRACE data from {grace_dir}...")
    grace_files = sorted(os.listdir(grace_dir))
    grace_files = [f for f in grace_files if f.endswith('.tif')]
    
    if not grace_files:
        raise ValueError(f"No GRACE .tif files found in {grace_dir}")
    
    print(f"Found {len(grace_files)} GRACE files.")
    
    # Process each file and extract timestamp from filename
    grace_data = []
    grace_dates = []
    
    for grace_file in grace_files:
        try:
            # Extract date from filename (format like: 20030131_20030227.tif)
            match = re.match(r'(\d{8})_(\d{8})\.tif', grace_file)
            if match:
                # Use the start date (first 8 digits) to represent this month
                date_str = match.group(1)
                # Convert to YYYY-MM format for consistent comparison
                date = datetime.strptime(date_str, '%Y%m%d')
                grace_date = date.strftime('%Y-%m')
                
                # Load the file
                grace_path = os.path.join(grace_dir, grace_file)
                grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                
                grace_data.append(grace_raster.values)
                grace_dates.append(grace_date)
                print(f"✅ Loaded {grace_file} as {grace_date}")
            else:
                print(f"⚠️ Skipping {grace_file} - could not parse date format")
                
        except Exception as e:
            print(f"❌ Error loading {grace_file}: {e}")
    
    print(f"Successfully loaded {len(grace_data)} GRACE files")
    return np.stack(grace_data), grace_dates

def create_lagged_features(X_data, lag_months=[1, 3, 6]):
    """Create lagged features without relying on original dataset structure."""
    print(f"Creating lagged features with lags: {lag_months}...")
    
    # Get dimensions
    n_times, n_features, n_lat, n_lon = X_data.shape
    
    # Initialize array to hold all features (original + lagged)
    n_lags = len(lag_months)
    total_features = n_features * (1 + n_lags)  # Original + lags
    all_features = np.zeros((n_times, total_features, n_lat, n_lon))
    
    # Copy original features
    all_features[:, :n_features, :, :] = X_data
    
    # Generate feature names
    feature_names = [f"feat_{i}" for i in range(n_features)]
    
    # Add lagged features
    feature_idx = n_features
    for lag in lag_months:
        if lag >= n_times:
            print(f"⚠️ Skipping lag {lag} as it exceeds time dimension ({n_times})")
            continue
            
        print(f"Adding lag {lag} features")
        # Add current features from previous time steps
        all_features[lag:, feature_idx:feature_idx+n_features, :, :] = X_data[:-lag, :, :, :]
        
        # Add lagged feature names
        for i in range(n_features):
            feature_names.append(f"feat_{i}_lag{lag}")
            
        feature_idx += n_features
    
    print(f"✅ Created lagged features. New shape: {all_features.shape}")
    return all_features, feature_names

def main():
    """Main function to train enhanced RF model."""
    print("📦 Loading feature stack...")
    features_path = "data/processed/feature_stack.nc"
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    # Load the feature dataset
    try:
        ds = xr.open_dataset(features_path)
        print(f"✅ Loaded feature dataset with dimensions: {dict(ds.dims)}")
        
        # Extract all feature data at once to avoid timestamp selection issues
        feature_data = ds.features.values
        feature_times = ds.time.values
        
        # Convert time values to strings in YYYY-MM format for consistent comparison
        # This is critical to avoid timestamp type issues
        feature_dates = []
        for t in feature_times:
            # Handle different possible formats
            if isinstance(t, np.datetime64) or isinstance(t, pd.Timestamp):
                # Convert numpy/pandas datetime to string
                date_str = pd.Timestamp(t).strftime('%Y-%m')
            elif isinstance(t, str):
                # Try to parse the string and reformat to YYYY-MM
                try:
                    date_str = pd.to_datetime(t).strftime('%Y-%m')
                except:
                    # If parsing fails, use the string directly if it looks like YYYY-MM
                    if re.match(r'\d{4}-\d{2}', t):
                        date_str = t
                    else:
                        print(f"⚠️ Unrecognized time format: {t}, using as is")
                        date_str = t
            else:
                print(f"⚠️ Unrecognized time type: {type(t)}, trying string conversion")
                date_str = str(t)
                
            feature_dates.append(date_str)
            
        print(f"✅ Extracted {len(feature_dates)} feature time points")
        
        # Load static features if available
        static_data = None
        static_names = []
        if 'static_features' in ds:
            static_data = ds.static_features.values
            
            if hasattr(ds.static_feature, 'values'):
                try:
                    static_names = [str(f) for f in ds.static_feature.values]
                except:
                    static_names = [f"static_{i}" for i in range(static_data.shape[0])]
            else:
                static_names = [f"static_{i}" for i in range(static_data.shape[0])]
                
            print(f"✅ Loaded {len(static_names)} static features")
            
    except Exception as e:
        print(f"❌ Error loading feature dataset: {e}")
        raise
        
    # Load all GRACE data
    print("📥 Loading GRACE TWS labels...")
    grace_dir = "data/raw/grace"
    grace_data, grace_dates = load_grace_tws(grace_dir)
    
    # Find common dates between feature and GRACE datasets
    print("🔄 Finding common dates between feature and GRACE datasets...")
    
    # Create dictionaries for easy lookup
    feature_dict = {date: idx for idx, date in enumerate(feature_dates)}
    grace_dict = {date: idx for idx, date in enumerate(grace_dates)}
    
    # Find common dates
    common_dates = sorted(set(feature_dates).intersection(set(grace_dates)))
    print(f"✅ Found {len(common_dates)} common dates between datasets")
    
    if not common_dates:
        # Print some sample dates from each dataset to help diagnose
        print("⚠️ No common dates found! Sample dates from each dataset:")
        print(f"Feature dates (first 5): {feature_dates[:5]}")
        print(f"Feature dates (last 5): {feature_dates[-5:]}")
        print(f"GRACE dates (first 5): {grace_dates[:5]}")
        print(f"GRACE dates (last 5): {grace_dates[-5:]}")
        raise ValueError("No common dates found between feature and GRACE datasets!")
    
    # Extract feature and GRACE data for common dates
    common_feature_indices = [feature_dict[date] for date in common_dates]
    common_grace_indices = [grace_dict[date] for date in common_dates]
    
    # Subset both datasets to common dates
    X_temporal = feature_data[common_feature_indices]
    grace_tws = grace_data[common_grace_indices]
    
    print(f"✅ Extracted data for common dates. Feature shape: {X_temporal.shape}, GRACE shape: {grace_tws.shape}")
    
    # Create enhanced features
    print("🔄 Creating enhanced feature set...")
    
    # 1. Create lagged features
    X_with_lags, feature_names = create_lagged_features(X_temporal, lag_months=[1, 3, 6])
    print(f"✅ Added lagged features: {X_with_lags.shape}")
    
    # 2. Add seasonal (monthly) features using cyclical encoding
    months = np.array([pd.to_datetime(date).month for date in common_dates])
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    
    # Create seasonal features with spatial dimensions
    n_times, n_features, n_lat, n_lon = X_with_lags.shape
    seasonal = np.zeros((n_times, 2, n_lat, n_lon))
    
    # Broadcast seasonal values across all spatial dimensions
    for t in range(n_times):
        seasonal[t, 0, :, :] = month_sin[t]
        seasonal[t, 1, :, :] = month_cos[t]
    
    # Combine with lag features
    X_enhanced = np.concatenate([X_with_lags, seasonal], axis=1)
    feature_names = feature_names + ["month_sin", "month_cos"]
    print(f"✅ Added seasonal features. New shape: {X_enhanced.shape}")
    
    # 3. Add static features if available
    if static_data is not None:
        # Create a 4D array with static features repeated for each time step
        static_expanded = np.zeros((n_times, static_data.shape[0], n_lat, n_lon))
        
        for t in range(n_times):
            static_expanded[t, :, :, :] = static_data
            
        # Combine with existing features
        X_enhanced = np.concatenate([X_enhanced, static_expanded], axis=1)
        feature_names.extend(static_names)
        print(f"✅ Added static features. Final shape: {X_enhanced.shape}")
    
    # Reshape for model training: (time, feature, lat, lon) -> (samples, features)
    n_times, n_features, n_lat, n_lon = X_enhanced.shape
    X = X_enhanced.reshape(n_times, n_features, -1).transpose(0, 2, 1).reshape(-1, n_features)
    y = grace_tws.reshape(-1)
    
    print(f"✅ Reshaped data: X={X.shape}, y={y.shape}")
    
    # Filter out invalid data points (NaN values)
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    valid_ratio = X_valid.shape[0] / X.shape[0] * 100
    print(f"✅ Filtered out NaN values: {X_valid.shape[0]} valid samples ({valid_ratio:.1f}%)")
    
    if X_valid.shape[0] == 0:
        raise ValueError("No valid data points after filtering NaNs!")
    
    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train the Random Forest model with optimized parameters
    print("🔍 Training Random Forest model...")
    
    # Use optimized parameters for high-performance system
    model = RandomForestRegressor(
        n_estimators=200,           # Increase for better performance
        max_depth=25,               # Deeper trees for complex relationships
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',        # Standard for RF
        n_jobs=-1,                  # Use all available cores
        random_state=42,
        verbose=1                   # Show progress
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model_enhanced.joblib")
    print("✅ Enhanced model saved to models/rf_model_enhanced.joblib")
    
    # Save feature importances
    importances = model.feature_importances_
    feature_import_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    feature_import_df.to_csv("models/feature_importances.csv", index=False)
    
    # Plot feature importance (top 20 features)
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(12, 8))
    top_features = feature_import_df.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Features by Importance')
    plt.tight_layout()
    plt.savefig("figures/feature_importance_enhanced.png", dpi=300)
    print("✅ Feature importance plot saved")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual GRACE TWS')
    plt.ylabel('Predicted GRACE TWS')
    plt.title('Actual vs Predicted GRACE TWS')
    plt.tight_layout()
    plt.savefig("figures/actual_vs_predicted_enhanced.png", dpi=300)
    print("✅ Actual vs Predicted plot saved")
    
    # Additional diagnostics: save training results
    results = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_features': n_features,
        'n_samples': X_valid.shape[0],
        'valid_data_ratio': valid_ratio,
        'common_dates': len(common_dates),
        'model_params': model.get_params()
    }
    
    # Save as text file for easy viewing
    with open("models/training_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
            
    print("✅ Training diagnostics saved to models/training_results.txt")


if __name__ == "__main__":
    main()