# model_rf.py
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import rioxarray as rxr
import re
from datetime import datetime

def load_grace_tws(grace_dir, valid_months):
    """Load GRACE TWS data aligned with feature months."""
    grace_files = sorted(os.listdir(grace_dir))
    grace_arrays = []
    loaded_months = []
    
    for month in valid_months:
        month_dt = datetime.strptime(month, "%Y-%m")
        month_str = month_dt.strftime("%Y%m")
        
        # Find files with matching month pattern (e.g., 202209)
        matching_files = [f for f in grace_files if re.match(f"{month_str}\d\d_.*\.tif", f)]
        
        if not matching_files:
            print(f"⚠️ No GRACE file found for month {month}")
            continue
            
        # Try to load any matching file
        for grace_file in matching_files:
            grace_path = os.path.join(grace_dir, grace_file)
            try:
                grace_raster = rxr.open_rasterio(grace_path, masked=True).squeeze()
                grace_arrays.append(grace_raster.values)
                loaded_months.append(month)
                print(f"✅ Loaded {grace_file} for {month}")
                break
            except Exception as e:
                print(f"❌ Error with {grace_file}: {e}")
                continue
    
    if len(grace_arrays) == 0:
        raise ValueError("No GRACE data could be loaded")
        
    print(f"✅ Loaded {len(grace_arrays)} GRACE arrays")
    return np.stack(grace_arrays, axis=0), loaded_months

def main():
    print("📦 Loading feature stack...")
    features_path = "data/processed/feature_stack.nc"
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    # Load the feature dataset
    ds = xr.open_dataset(features_path)
    feature_months = ds.time.values
    print(f"✅ Loaded features with {len(feature_months)} time steps")
    
    # Load GRACE data without reprojection
    print("📥 Loading GRACE TWS labels...")
    grace_dir = "data/raw/grace"
    grace_tws, loaded_months = load_grace_tws(grace_dir, feature_months)
    
    # Adjust features to match loaded GRACE months
    if len(loaded_months) != len(feature_months):
        print(f"⚠️ Found {len(loaded_months)} GRACE vs {len(feature_months)} feature months")
        ds = ds.sel(time=loaded_months)
        print(f"✅ Adjusted features to match GRACE: {len(loaded_months)} time steps")
    
    # Reshape feature data for model training
    X_temporal = ds.features.values  # shape: (time, feature, lat, lon)
    n_times, n_features, n_lat, n_lon = X_temporal.shape
    
    # Add static features if available
    if 'static_features' in ds:
        X_static = ds.static_features.values
        n_static = X_static.shape[0]
        
        X_combined = np.zeros((n_times, n_features + n_static, n_lat, n_lon))
        X_combined[:, :n_features, :, :] = X_temporal
        
        for t in range(n_times):
            X_combined[t, n_features:, :, :] = X_static
            
        X_temporal = X_combined
        n_features += n_static
    
    # Reshape for model: (time, feature, lat, lon) -> (time*lat*lon, feature)
    X = X_temporal.reshape(n_times, n_features, -1).transpose(0, 2, 1).reshape(-1, n_features)
    y = grace_tws.reshape(-1)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Filter out invalid data points
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_valid = X[mask]
    y_valid = y[mask]
    
    print(f"Valid data points: {X_valid.shape[0]} out of {X.shape[0]}")
    
    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42
    )
    
    print("🔍 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
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
    joblib.dump(model, "models/rf_model.joblib")
    print("✅ Model saved to models/rf_model.joblib")
    
    importances = model.feature_importances_

    # Load feature names from xarray Dataset
    feature_names = list(ds.feature.values.astype(str))
    if 'static_feature' in ds.coords:
        feature_names.extend(list(ds.static_feature.values.astype(str)))

    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sorted_indices = np.argsort(importances)[::-1]
    plt.barh(range(n_features), importances[sorted_indices])
    plt.yticks(range(n_features), [feature_names[i] for i in sorted_indices])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/feature_importance.png", dpi=300)
    print("✅ Feature importance plot saved")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual GRACE TWS')
    plt.ylabel('Predicted GRACE TWS')
    plt.title('Actual vs Predicted GRACE TWS')
    plt.tight_layout()
    plt.savefig("figures/actual_vs_predicted.png", dpi=300)
    print("✅ Actual vs Predicted plot saved")

if __name__ == "__main__":
    main()