# src/groundwater_enhanced.py - Robust version with NaN handling for all model types
"""
Enhanced groundwater storage calculation with robust NaN handling for different model types.

This version handles:
- Neural Networks (require NaN preprocessing and scaling)
- Tree-based models (can handle NaN natively)
- Proper fallback strategies
"""

import os
import numpy as np
import xarray as xr
import joblib
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from pathlib import Path
import yaml
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class RobustModelLoader:
    """Handles loading and using different types of trained models with NaN handling."""
    
    def __init__(self, model_dir="models", config_path="src/config.yaml"):
        self.model_dir = Path(model_dir)
        self.config_path = config_path
        self.loaded_model = None
        self.model_info = None
        self.scaler = None
        self.imputer = None
        self.model_type = None
        
    def load_config(self):
        """Load pipeline configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            return {
                'pipeline': {
                    'groundwater_model': 'best',
                    'fallback_models': ['rf', 'xgb']
                }
            }
    
    def find_best_model(self):
        """Find the best available model based on comparison results."""
        comparison_file = self.model_dir / "model_comparison.csv"
        
        if comparison_file.exists():
            try:
                comparison = pd.read_csv(comparison_file)
                best_row = comparison.loc[comparison['test_r2'].idxmax()]
                best_model_name = best_row['model_name']
                
                print(f"🏆 Best model identified: {best_row['display_name']} "
                      f"(R² = {best_row['test_r2']:.4f})")
                
                return best_model_name
            except Exception as e:
                print(f"⚠️ Error reading model comparison: {e}")
        
        return None
    
    def get_model_type(self, model):
        """Determine model type for appropriate preprocessing."""
        model_class = type(model).__name__
        
        if 'MLP' in model_class or 'Neural' in model_class:
            return 'neural_network'
        elif 'SVR' in model_class or 'SVC' in model_class:
            return 'svm'
        elif any(name in model_class for name in ['Forest', 'XGB', 'LGB', 'CatBoost', 'Gradient']):
            return 'tree_based'
        else:
            return 'other'
    
    def load_model(self, model_name=None):
        """Load a specific model with appropriate preprocessing setup."""
        config = self.load_config()
        
        # Determine which model to load
        if model_name is None:
            preference = config.get('pipeline', {}).get('groundwater_model', 'best')
            
            if preference == 'best':
                model_name = self.find_best_model()
                if model_name is None:
                    print("⚠️ No model comparison found, trying fallback...")
                    model_name = 'rf'
            else:
                model_name = preference
        
        # Try to load the specified model
        model_path = self.model_dir / f"{model_name}_model.joblib"
        
        if not model_path.exists():
            print(f"⚠️ Model {model_name} not found at {model_path}")
            
            # Try fallback models
            fallback_models = config.get('pipeline', {}).get('fallback_models', ['rf'])
            for fallback in fallback_models:
                fallback_path = self.model_dir / f"{fallback}_model.joblib"
                if fallback_path.exists():
                    print(f"🔄 Using fallback model: {fallback}")
                    model_path = fallback_path
                    model_name = fallback
                    break
            else:
                # Try legacy filename
                legacy_path = self.model_dir / "rf_model_enhanced.joblib"
                if legacy_path.exists():
                    print("🔄 Using legacy model file")
                    model_path = legacy_path
                    model_name = 'rf_legacy'
                else:
                    print("❌ No valid model found!")
                    return False
        
        # Load the model
        try:
            print(f"📦 Loading model from {model_path}...")
            model_package = joblib.load(model_path)
            
            # Handle different save formats
            if isinstance(model_package, dict):
                self.loaded_model = model_package['model']
                self.model_info = model_package.get('config', {})
                self.scaler = model_package.get('scaler', None)
                
                model_display_name = self.model_info.get('name', model_name)
                print(f"✅ Loaded {model_display_name}")
                
                if 'metrics' in model_package:
                    metrics = model_package['metrics']
                    print(f"   Performance: R² = {metrics.get('test_r2', 'N/A'):.4f}")
                
            else:
                # Legacy format
                self.loaded_model = model_package
                self.model_info = {'name': model_name, 'needs_scaling': False}
                self.scaler = None
                print(f"✅ Loaded legacy model: {model_name}")
            
            # Determine model type and setup preprocessing
            self.model_type = self.get_model_type(self.loaded_model)
            print(f"   Model type: {self.model_type}")
            
            # Setup NaN handling for models that need it
            if self.model_type in ['neural_network', 'svm']:
                self.imputer = SimpleImputer(strategy='mean')
                print(f"   ✅ NaN imputer initialized for {self.model_type}")
            
            # Check if we have a scaler
            if self.scaler is not None:
                print(f"   ✅ Scaler loaded for data preprocessing")
            elif self.model_info.get('needs_scaling', False):
                print(f"   ⚠️ Model needs scaling but no scaler found!")
                # Create a simple scaler as fallback
                self.scaler = StandardScaler()
                print(f"   🔄 Created fallback scaler")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            return False
    
    def preprocess_input(self, X):
        """Preprocess input based on model requirements."""
        if X is None:
            return None
        
        X_processed = X.copy()
        
        # Handle NaN values for models that can't handle them
        if self.model_type in ['neural_network', 'svm'] and self.imputer is not None:
            # Check if we have NaN values
            if np.isnan(X_processed).any():
                print(f"   🔧 Imputing NaN values for {self.model_type}")
                
                # Fit imputer on the fly if needed
                if not hasattr(self.imputer, 'statistics_'):
                    # Remove completely NaN columns for fitting
                    non_nan_mask = ~np.isnan(X_processed).all(axis=0)
                    if non_nan_mask.any():
                        X_for_fit = X_processed[:, non_nan_mask]
                        self.imputer.fit(X_for_fit)
                        
                        # Apply to all columns
                        X_processed_temp = np.zeros_like(X_processed)
                        X_processed_temp[:, non_nan_mask] = self.imputer.transform(X_for_fit)
                        
                        # Fill remaining columns with mean of non-NaN columns
                        for i in range(X_processed.shape[1]):
                            if not non_nan_mask[i]:
                                X_processed_temp[:, i] = np.nanmean(X_processed_temp[:, non_nan_mask], axis=1)
                        
                        X_processed = X_processed_temp
                    else:
                        print(f"   ⚠️ All features are NaN, filling with zeros")
                        X_processed = np.zeros_like(X_processed)
                else:
                    X_processed = self.imputer.transform(X_processed)
        
        # Apply scaling if needed
        if self.scaler is not None and self.model_info.get('needs_scaling', False):
            print(f"   🔧 Applying scaling for {self.model_type}")
            X_processed = self.scaler.transform(X_processed)
        
        return X_processed
    
    def predict(self, X):
        """Make predictions with appropriate preprocessing."""
        if self.loaded_model is None:
            raise ValueError("No model loaded! Call load_model() first.")
        
        # Preprocess input
        X_processed = self.preprocess_input(X)
        
        if X_processed is None:
            raise ValueError("Input preprocessing failed!")
        
        # Check for remaining NaN values
        if np.isnan(X_processed).any():
            print(f"   ⚠️ Warning: {np.isnan(X_processed).sum()} NaN values remain after preprocessing")
            
            # For tree-based models, this might be OK, but for others we need to handle it
            if self.model_type in ['neural_network', 'svm']:
                print(f"   🔧 Replacing remaining NaN with zeros for {self.model_type}")
                X_processed = np.nan_to_num(X_processed, nan=0.0)
        
        # Make prediction
        try:
            predictions = self.loaded_model.predict(X_processed)
            return predictions
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            return None


def prepare_model_input(ds, time_index, lag_months=[1, 3, 6]):
    """Prepare enhanced input for model prediction with robust error handling."""
    try:
        target_features = ds.sel(time=time_index).features.values
        feature_shape = target_features.shape
        spatial_shape = feature_shape[1:]
        
        # Check for completely missing data
        if np.all(np.isnan(target_features)):
            print(f"   ⚠️ All features are NaN for {time_index}")
            return None
        
        # Add current features
        all_features = [target_features]
        
        # Add lagged features
        for lag in lag_months:
            try:
                target_date = pd.to_datetime(time_index)
                year, month = target_date.year, target_date.month
                
                lag_month = month - lag
                lag_year = year
                while lag_month <= 0:
                    lag_month += 12
                    lag_year -= 1
                
                lagged_date = f"{lag_year:04d}-{lag_month:02d}"
                lagged_feature = ds.sel(time=lagged_date).features.values
                all_features.append(lagged_feature)
            except KeyError:
                # Use zeros if lagged data not available
                all_features.append(np.zeros_like(target_features))
        
        # Create seasonal features
        month = pd.to_datetime(time_index).month
        month_sin = np.sin(2 * np.pi * month / 12) * np.ones(spatial_shape)
        month_cos = np.cos(2 * np.pi * month / 12) * np.ones(spatial_shape)
        
        all_features.append(month_sin[np.newaxis, :, :])
        all_features.append(month_cos[np.newaxis, :, :])
        
        # Add static features if available
        if 'static_features' in ds:
            static_features = ds.static_features.values
            all_features.append(static_features)
        
        # Convert to single array and reshape
        X = np.vstack(all_features)
        X_flat = X.reshape(X.shape[0], -1).T
        
        return X_flat
    
    except Exception as e:
        print(f"   ❌ Error preparing input for {time_index}: {e}")
        return None


def calculate_groundwater_storage(model_name=None):
    """Calculate groundwater storage with robust model handling."""
    os.makedirs("results", exist_ok=True)
    
    print("🚀 Enhanced Groundwater Storage Calculation with Robust NaN Handling")
    print("="*70)
    
    # Initialize robust model loader
    model_loader = RobustModelLoader()
    
    # Load the best available model
    if not model_loader.load_model(model_name):
        raise ValueError("Failed to load any valid model!")
    
    # Load data
    print("\n📦 Loading feature stack...")
    ds = xr.open_dataset("data/processed/feature_stack.nc")
    
    # Define GRACE reference period
    reference_start = "2004-01"
    reference_end = "2009-12"
    
    # Identify water storage components
    soil_vars = [v for v in ds.feature.values if 'SoilMoi' in str(v)]
    swe_vars = [v for v in ds.feature.values if 'SWE' in str(v)]
    
    print(f"Identified {len(soil_vars)} soil moisture variables")
    print(f"Identified {len(swe_vars)} snow water equivalent variables")
    
    # Extract component time series with unit conversion
    print("\nExtracting component time series...")
    time_indices = ds.time.values
    all_soil_moisture = []
    all_swe = []
    valid_times = []
    
    GLDAS_TO_CM = 0.1  # kg/m² to cm conversion
    
    for time_index in tqdm(time_indices):
        try:
            # Extract soil moisture
            soil_moisture_t = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            soil_found = False
            
            if soil_vars:
                for var in soil_vars:
                    try:
                        var_idx = np.where(ds.feature.values == var)[0][0]
                        soil_values = ds.sel(time=time_index).features.isel(feature=var_idx).values
                        
                        if not np.all(np.isnan(soil_values)):
                            max_val = np.nanmax(soil_values)
                            if max_val > 10:  # Likely kg/m²
                                soil_values = soil_values * GLDAS_TO_CM
                            
                            soil_moisture_t += np.nan_to_num(soil_values, nan=0.0)
                            soil_found = True
                    except:
                        pass
            
            # Extract snow water equivalent
            swe_t = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            swe_found = False
            
            if swe_vars:
                for var in swe_vars:
                    try:
                        var_idx = np.where(ds.feature.values == var)[0][0]
                        swe_values = ds.sel(time=time_index).features.isel(feature=var_idx).values
                        
                        if not np.all(np.isnan(swe_values)):
                            max_val = np.nanmax(swe_values)
                            if max_val > 10:  # Likely kg/m²
                                swe_values = swe_values * GLDAS_TO_CM
                            
                            swe_t += np.nan_to_num(swe_values, nan=0.0)
                            swe_found = True
                    except:
                        pass
            
            if soil_found or swe_found:
                all_soil_moisture.append(soil_moisture_t)
                all_swe.append(swe_t)
                valid_times.append(time_index)
            
        except Exception as e:
            print(f"Error extracting components for time {time_index}: {e}")
    
    if not valid_times:
        raise ValueError("No valid data found for any time period")
    
    # Calculate reference period means
    print("\nCalculating reference period means...")
    time_dates = pd.to_datetime(valid_times)
    ref_start_date = pd.to_datetime(reference_start)
    ref_end_date = pd.to_datetime(reference_end)
    ref_mask = (time_dates >= ref_start_date) & (time_dates <= ref_end_date)
    
    ref_count = np.sum(ref_mask)
    if ref_count == 0:
        print(f"⚠️ No data in reference period, using all data")
        ref_mask = np.ones(len(valid_times), dtype=bool)
        ref_count = len(valid_times)
    
    print(f"Using {ref_count} months for reference period")
    
    # Calculate reference means
    if ref_count > 0:
        soil_moisture_ref = np.stack([all_soil_moisture[i] for i, is_ref in enumerate(ref_mask) if is_ref])
        swe_ref = np.stack([all_swe[i] for i, is_ref in enumerate(ref_mask) if is_ref])
        
        soil_moisture_mean = np.nanmean(soil_moisture_ref, axis=0)
        swe_mean = np.nanmean(swe_ref, axis=0)
        
        soil_moisture_mean = np.nan_to_num(soil_moisture_mean, nan=0.0)
        swe_mean = np.nan_to_num(swe_mean, nan=0.0)
    else:
        soil_moisture_mean = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
        swe_mean = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
    
    # Process each time step
    print(f"\nCalculating groundwater anomalies using {model_loader.model_info.get('name', 'loaded model')}...")
    print(f"Model type: {model_loader.model_type}")
    
    gws_data = []
    tws_data = []
    soil_moisture_data = []
    swe_data = []
    times = []
    
    MAX_REASONABLE_TWS = 150
    MAX_REASONABLE_COMPONENT = 50
    
    successful_predictions = 0
    failed_predictions = 0
    
    for i, time_index in enumerate(tqdm(valid_times)):
        try:
            # Prepare model input
            X = prepare_model_input(ds, time_index, lag_months=[1, 3, 6])
            
            if X is None:
                failed_predictions += 1
                continue
            
            # Make predictions with robust preprocessing
            tws_pred = model_loader.predict(X)
            
            if tws_pred is None:
                failed_predictions += 1
                continue
            
            # Reshape prediction
            n_lat = ds.lat.shape[0]
            n_lon = ds.lon.shape[0]
            tws_spatial = tws_pred.reshape(n_lat, n_lon)
            
            # Cap predictions
            tws_spatial = np.clip(tws_spatial, -MAX_REASONABLE_TWS, MAX_REASONABLE_TWS)
            
            # Get components for this time step
            soil_moisture = all_soil_moisture[i]
            swe = all_swe[i]
            
            # Calculate anomalies
            soil_moisture_anomaly = soil_moisture - soil_moisture_mean
            swe_anomaly = swe - swe_mean
            
            soil_moisture_anomaly = np.clip(soil_moisture_anomaly, -MAX_REASONABLE_COMPONENT, MAX_REASONABLE_COMPONENT)
            swe_anomaly = np.clip(swe_anomaly, -MAX_REASONABLE_COMPONENT, MAX_REASONABLE_COMPONENT)
            
            # Calculate groundwater anomaly
            gws = tws_spatial - soil_moisture_anomaly - swe_anomaly
            gws = np.clip(gws, -MAX_REASONABLE_TWS, MAX_REASONABLE_TWS)
            
            # Store results
            gws_data.append(gws)
            tws_data.append(tws_spatial)
            soil_moisture_data.append(soil_moisture_anomaly)
            swe_data.append(swe_anomaly)
            times.append(time_index)
            
            successful_predictions += 1
            
        except Exception as e:
            print(f"Error processing time {time_index}: {e}")
            failed_predictions += 1
    
    if len(gws_data) == 0:
        raise ValueError("No time steps could be processed successfully")
    
    print(f"\n✅ Successfully processed {successful_predictions}/{len(valid_times)} time steps")
    print(f"   Failed: {failed_predictions} time steps")
    
    # Create dataset
    model_name_for_attr = model_loader.model_info.get('name', 'Unknown')
    
    gws_ds = xr.Dataset(
        data_vars={
            "groundwater": (["time", "lat", "lon"], np.stack(gws_data)),
            "tws": (["time", "lat", "lon"], np.stack(tws_data)),
            "soil_moisture_anomaly": (["time", "lat", "lon"], np.stack(soil_moisture_data)),
            "swe_anomaly": (["time", "lat", "lon"], np.stack(swe_data))
        },
        coords={
            "time": times,
            "lat": ds.lat,
            "lon": ds.lon
        },
        attrs={
            "reference_period": f"{reference_start} to {reference_end}",
            "description": f"Water storage anomalies relative to {reference_start}-{reference_end} mean",
            "processing_notes": f"Generated using {model_name_for_attr} with robust NaN handling",
            "model_info": model_name_for_attr,
            "model_type": model_loader.model_type,
            "units": "cm water equivalent",
            "creation_date": datetime.now().isoformat(),
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions
        }
    )
    
    # Add variable metadata
    gws_ds.groundwater.attrs.update({
        "long_name": "Groundwater Storage Anomaly",
        "units": "cm water equivalent",
        "reference_period": f"{reference_start} to {reference_end}",
        "model_used": model_name_for_attr
    })
    
    gws_ds.tws.attrs.update({
        "long_name": "Total Water Storage Anomaly",
        "units": "cm water equivalent",
        "reference_period": f"{reference_start} to {reference_end}",
        "model_used": model_name_for_attr
    })
    
    gws_ds.soil_moisture_anomaly.attrs.update({
        "long_name": "Soil Moisture Anomaly",
        "units": "cm water equivalent",
        "reference_period": f"{reference_start} to {reference_end}"
    })
    
    gws_ds.swe_anomaly.attrs.update({
        "long_name": "Snow Water Equivalent Anomaly",
        "units": "cm water equivalent",
        "reference_period": f"{reference_start} to {reference_end}"
    })
    
    # Print final statistics
    print("\nFinal statistics:")
    for var in ['groundwater', 'tws', 'soil_moisture_anomaly', 'swe_anomaly']:
        data = gws_ds[var].values
        valid = data[~np.isnan(data)]
        print(f"  {var}: [{valid.min():.2f}, {valid.max():.2f}] cm, "
              f"mean={valid.mean():.2f}, std={valid.std():.2f}")
    
    # Save results
    output_path = "results/groundwater_storage_anomalies.nc"
    gws_ds.to_netcdf(output_path)
    print(f"\n✅ Groundwater storage anomalies saved to {output_path}")
    print(f"   Model used: {model_name_for_attr} ({model_loader.model_type})")
    print(f"   Success rate: {successful_predictions}/{len(valid_times)} ({successful_predictions/len(valid_times)*100:.1f}%)")
    
    return gws_ds


if __name__ == "__main__":
    calculate_groundwater_storage()