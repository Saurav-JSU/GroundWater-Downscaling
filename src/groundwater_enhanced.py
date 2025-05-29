# src/groundwater_enhanced.py - FIXED VERSION with unit conversions
"""
Enhanced groundwater storage calculation from GRACE TWS and component separation.

This module includes:
- Proper unit conversions (GLDAS kg/m² to cm)
- Extreme value capping
- Enhanced model input preparation with lagged features
"""

import os
import numpy as np
import xarray as xr
import joblib
from tqdm import tqdm
import pandas as pd
from datetime import datetime


def prepare_model_input(ds, time_index, lag_months=[1, 3, 6]):
    """
    Prepare enhanced input for RF model prediction with lagged and seasonal features.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing features
    time_index : str or datetime
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


def calculate_groundwater_storage():
    """Calculate groundwater storage anomalies from TWS and other components using enhanced model"""
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Load data
    print("Loading feature stack and enhanced model...")
    ds = xr.open_dataset("data/processed/feature_stack.nc")
    model = joblib.load("models/rf_model_enhanced.joblib")
    print(f"Model expects {model.n_features_in_} features as input")
    
    # Define GRACE reference period (2004-2009)
    reference_start = "2004-01"
    reference_end = "2009-12"
    
    # Identify water storage components by name patterns
    soil_vars = [v for v in ds.feature.values if 'SoilMoi' in str(v)]
    swe_vars = [v for v in ds.feature.values if 'SWE' in str(v)]
    
    print(f"Identified {len(soil_vars)} soil moisture variables")
    print(f"Identified {len(swe_vars)} snow water equivalent variables")
    
    # Check GLDAS units by looking at sample values
    print("\nChecking GLDAS variable ranges (for unit detection):")
    for var in soil_vars[:2]:  # Check first two soil layers
        try:
            var_idx = np.where(ds.feature.values == var)[0][0]
            sample_values = ds.features.isel(feature=var_idx, time=0).values
            valid_samples = sample_values[~np.isnan(sample_values)]
            if len(valid_samples) > 0:
                print(f"  {var}: range [{valid_samples.min():.2f}, {valid_samples.max():.2f}]")
        except:
            pass
    
    # Create arrays to store results
    gws_data = []
    tws_data = []
    soil_moisture_data = []
    swe_data = []
    times = []
    
    # 1. First extract all time series data for each component WITH UNIT CONVERSION
    print("\nExtracting component time series with unit conversions...")
    time_indices = ds.time.values
    all_soil_moisture = []
    all_swe = []
    valid_times = []
    
    # UNIT CONVERSION FACTOR
    # GLDAS is in kg/m^2, convert to cm: 1 kg/m^2 = 0.1 cm
    GLDAS_TO_CM = 0.1
    
    for time_index in tqdm(time_indices):
        try:
            # Extract soil moisture for this time WITH UNIT CONVERSION
            soil_moisture_t = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            soil_found = False
            
            if soil_vars:
                for var in soil_vars:
                    try:
                        var_idx = np.where(ds.feature.values == var)[0][0]
                        soil_values = ds.sel(time=time_index).features.isel(feature=var_idx).values
                        
                        # APPLY UNIT CONVERSION HERE
                        # Check if values are likely in kg/m^2 (typically > 10)
                        if not np.all(np.isnan(soil_values)):
                            max_val = np.nanmax(soil_values)
                            if max_val > 10:  # Likely kg/m^2
                                soil_values = soil_values * GLDAS_TO_CM
                            
                            soil_moisture_t += np.nan_to_num(soil_values, nan=0.0)
                            soil_found = True
                    except Exception:
                        pass
            
            # Extract snow water equivalent WITH UNIT CONVERSION
            swe_t = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            swe_found = False
            
            if swe_vars:
                for var in swe_vars:
                    try:
                        var_idx = np.where(ds.feature.values == var)[0][0]
                        swe_values = ds.sel(time=time_index).features.isel(feature=var_idx).values
                        
                        # APPLY UNIT CONVERSION HERE
                        if not np.all(np.isnan(swe_values)):
                            max_val = np.nanmax(swe_values)
                            if max_val > 10:  # Likely kg/m^2
                                swe_values = swe_values * GLDAS_TO_CM
                            
                            swe_t += np.nan_to_num(swe_values, nan=0.0)
                            swe_found = True
                    except Exception:
                        pass
            
            # Only store if we found data
            if soil_found or swe_found:
                all_soil_moisture.append(soil_moisture_t)
                all_swe.append(swe_t)
                valid_times.append(time_index)
            
        except Exception as e:
            print(f"Error extracting components for time {time_index}: {e}")
    
    if not valid_times:
        raise ValueError("No valid data found for any time period")
    
    # 2. Calculate reference period means
    print("\nCalculating reference period means...")
    time_dates = pd.to_datetime(valid_times)
    ref_start_date = pd.to_datetime(reference_start)
    ref_end_date = pd.to_datetime(reference_end)
    ref_mask = (time_dates >= ref_start_date) & (time_dates <= ref_end_date)
    
    ref_count = np.sum(ref_mask)
    if ref_count == 0:
        print(f"⚠️ Warning: No data found in reference period {reference_start} to {reference_end}")
        print("Using all available data to calculate mean instead.")
        ref_mask = np.ones(len(valid_times), dtype=bool)
        ref_count = len(valid_times)
    
    print(f"Using {ref_count} months for reference period mean calculation")
    
    # Calculate reference means
    if ref_count > 0:
        soil_moisture_ref = np.stack([all_soil_moisture[i] for i, is_ref in enumerate(ref_mask) if is_ref])
        swe_ref = np.stack([all_swe[i] for i, is_ref in enumerate(ref_mask) if is_ref])
        
        soil_moisture_mean = np.nanmean(soil_moisture_ref, axis=0)
        swe_mean = np.nanmean(swe_ref, axis=0)
        
        # Replace any remaining NaNs with zeros
        soil_moisture_mean = np.nan_to_num(soil_moisture_mean, nan=0.0)
        swe_mean = np.nan_to_num(swe_mean, nan=0.0)
    else:
        soil_moisture_mean = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
        swe_mean = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
    
    print(f"Component magnitudes in reference period:")
    print(f"  Soil moisture mean: {np.mean(soil_moisture_mean):.2f} cm")
    print(f"  SWE mean: {np.mean(swe_mean):.2f} cm")
    
    # 3. Process each time step to calculate groundwater anomalies
    print("\nCalculating groundwater storage anomalies...")
    
    # Define reasonable bounds for capping
    MAX_REASONABLE_TWS = 150  # cm
    MAX_REASONABLE_COMPONENT = 50  # cm
    
    for i, time_index in enumerate(tqdm(valid_times)):
        try:
            # Prepare enhanced model input for TWS prediction
            X = prepare_model_input(ds, time_index, lag_months=[1, 3, 6])
            
            # Skip if input preparation failed
            if X is None:
                print(f"⚠️ Skipping {time_index}: Could not prepare model input")
                continue
            
            # Verify dimensions match what model expects
            if X.shape[1] != model.n_features_in_:
                print(f"⚠️ Feature count mismatch: have {X.shape[1]}, need {model.n_features_in_}")
                continue
                
            # Calculate predicted TWS anomaly
            tws_pred = model.predict(X)
            
            # Reshape prediction back to spatial grid
            n_lat = ds.lat.shape[0]
            n_lon = ds.lon.shape[0]
            tws_spatial = tws_pred.reshape(n_lat, n_lon)
            
            # CAP TWS PREDICTIONS
            tws_spatial = np.clip(tws_spatial, -MAX_REASONABLE_TWS, MAX_REASONABLE_TWS)
            
            # Get the soil moisture and SWE for this time step
            soil_moisture = all_soil_moisture[i]
            swe = all_swe[i]
            
            # Convert to anomalies relative to reference period
            soil_moisture_anomaly = soil_moisture - soil_moisture_mean
            swe_anomaly = swe - swe_mean
            
            # CAP COMPONENT ANOMALIES
            soil_moisture_anomaly = np.clip(soil_moisture_anomaly, -MAX_REASONABLE_COMPONENT, MAX_REASONABLE_COMPONENT)
            swe_anomaly = np.clip(swe_anomaly, -MAX_REASONABLE_COMPONENT, MAX_REASONABLE_COMPONENT)
            
            # Calculate groundwater anomaly: GWS_anomaly = TWS_anomaly - SM_anomaly - SWE_anomaly
            gws = tws_spatial - soil_moisture_anomaly - swe_anomaly
            
            # FINAL CAP ON GROUNDWATER
            gws = np.clip(gws, -MAX_REASONABLE_TWS, MAX_REASONABLE_TWS)
            
            # Store results
            gws_data.append(gws)
            tws_data.append(tws_spatial)
            soil_moisture_data.append(soil_moisture_anomaly)
            swe_data.append(swe_anomaly)
            times.append(time_index)
            
        except Exception as e:
            print(f"Error processing time {time_index}: {e}")
    
    if len(gws_data) == 0:
        raise ValueError("No time steps could be processed")
    
    # Create dataset with results
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
            "processing_notes": "Enhanced model with lagged features, unit-corrected (GLDAS kg/m² to cm), capped at ±150cm",
            "model_info": "Random Forest with lagged features and seasonal encoding",
            "units": "cm water equivalent"
        }
    )
    
    # Add metadata
    gws_ds.groundwater.attrs["long_name"] = "Groundwater Storage Anomaly"
    gws_ds.groundwater.attrs["units"] = "cm water equivalent"
    gws_ds.groundwater.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    gws_ds.tws.attrs["long_name"] = "Total Water Storage Anomaly"
    gws_ds.tws.attrs["units"] = "cm water equivalent"
    gws_ds.tws.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    gws_ds.soil_moisture_anomaly.attrs["long_name"] = "Soil Moisture Anomaly"
    gws_ds.soil_moisture_anomaly.attrs["units"] = "cm water equivalent"
    gws_ds.soil_moisture_anomaly.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    gws_ds.swe_anomaly.attrs["long_name"] = "Snow Water Equivalent Anomaly"
    gws_ds.swe_anomaly.attrs["units"] = "cm water equivalent"
    gws_ds.swe_anomaly.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    # Print final statistics
    print("\nFinal statistics:")
    for var in ['groundwater', 'tws', 'soil_moisture_anomaly', 'swe_anomaly']:
        data = gws_ds[var].values
        valid = data[~np.isnan(data)]
        print(f"  {var}: [{valid.min():.2f}, {valid.max():.2f}] cm, "
              f"mean={valid.mean():.2f}, std={valid.std():.2f}")
    
    # Save results - use standard filename for compatibility
    output_path = "results/groundwater_storage_anomalies.nc"
    gws_ds.to_netcdf(output_path)
    print(f"\n✅ Groundwater storage anomalies saved to {output_path}")
    print(f"   All anomalies are relative to the {reference_start} to {reference_end} reference period")
    
    return gws_ds


if __name__ == "__main__":
    calculate_groundwater_storage()