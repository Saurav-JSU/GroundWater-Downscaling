# src/groundwater_enhanced.py - Modified for the enhanced model
import os
import numpy as np
import xarray as xr
import joblib
from tqdm import tqdm
import rioxarray as rxr
from pathlib import Path
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
    et_vars = [v for v in ds.feature.values if 'Evap' in str(v) or 'aet' in str(v)]
    
    print(f"Identified {len(soil_vars)} soil moisture variables")
    print(f"Identified {len(swe_vars)} snow water equivalent variables")
    print(f"Identified {len(et_vars)} evapotranspiration variables")
    
    # Create arrays to store results
    gws_data = []
    tws_data = []
    soil_moisture_data = []
    swe_data = []
    times = []
    
    # 1. First extract all time series data for each component
    print("Extracting component time series...")
    time_indices = ds.time.values
    all_soil_moisture = []
    all_swe = []
    valid_times = []
    
    for time_index in tqdm(time_indices):
        try:
            # Extract soil moisture for this time
            soil_moisture_t = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            soil_found = False
            if soil_vars:
                for var in soil_vars:
                    try:
                        var_idx = np.where(ds.feature.values == var)[0][0]
                        soil_values = ds.sel(time=time_index).features.isel(feature=var_idx).values
                        
                        # Check for all NaN values
                        if not np.all(np.isnan(soil_values)):
                            soil_moisture_t += np.nan_to_num(soil_values, nan=0.0)
                            soil_found = True
                    except (IndexError, ValueError):
                        try:
                            soil_values = ds.sel(time=time_index).features.sel(feature=var, method='nearest').values
                            if not np.all(np.isnan(soil_values)):
                                soil_moisture_t += np.nan_to_num(soil_values, nan=0.0)
                                soil_found = True
                        except Exception as inner_e:
                            pass  # Continue to next variable
            
            # Extract snow water equivalent for this time
            swe_t = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
            swe_found = False
            if swe_vars:
                for var in swe_vars:
                    try:
                        var_idx = np.where(ds.feature.values == var)[0][0]
                        swe_values = ds.sel(time=time_index).features.isel(feature=var_idx).values
                        
                        # Check for all NaN values
                        if not np.all(np.isnan(swe_values)):
                            swe_t += np.nan_to_num(swe_values, nan=0.0)
                            swe_found = True
                    except (IndexError, ValueError):
                        try:
                            swe_values = ds.sel(time=time_index).features.sel(feature=var, method='nearest').values
                            if not np.all(np.isnan(swe_values)):
                                swe_t += np.nan_to_num(swe_values, nan=0.0)
                                swe_found = True
                        except Exception as inner_e:
                            pass  # Continue to next variable
            
            # Only store if we found data
            if soil_found or swe_found:
                all_soil_moisture.append(soil_moisture_t)
                all_swe.append(swe_t)
                valid_times.append(time_index)
            
        except Exception as e:
            print(f"Error extracting components for time {time_index}: {e}")
    
    if not valid_times:
        raise ValueError("No valid data found for any time period")
    
    # 2. Identify reference period indices
    # Convert time strings to pandas datetime for easier filtering
    time_dates = pd.to_datetime(valid_times)
    ref_start_date = pd.to_datetime(reference_start)
    ref_end_date = pd.to_datetime(reference_end)
    
    # Create boolean mask for reference period
    ref_mask = (time_dates >= ref_start_date) & (time_dates <= ref_end_date)
    
    ref_count = np.sum(ref_mask)
    if ref_count == 0:
        print(f"⚠️ Warning: No data found in reference period {reference_start} to {reference_end}")
        print("Using all available data to calculate mean instead.")
        ref_mask = np.ones(len(valid_times), dtype=bool)
        ref_count = len(valid_times)
    
    print(f"Using {ref_count} months for reference period mean calculation")
    
    # Debug: print first few reference period dates to verify
    ref_indices = np.where(ref_mask)[0]
    if len(ref_indices) > 0:
        print("First few reference period dates:")
        for i in range(min(5, len(ref_indices))):
            print(f"  {valid_times[ref_indices[i]]}")
    
    # 3. Calculate reference means for the components
    # Only use data from reference period (2004-2009)
    soil_moisture_ref = np.stack([all_soil_moisture[i] for i, is_ref in enumerate(ref_mask) if is_ref])
    swe_ref = np.stack([all_swe[i] for i, is_ref in enumerate(ref_mask) if is_ref])
    
    # Check for NaN values
    soil_nan_count = np.isnan(soil_moisture_ref).sum()
    soil_total_elements = soil_moisture_ref.size
    swe_nan_count = np.isnan(swe_ref).sum()
    swe_total_elements = swe_ref.size
    
    print(f"Soil moisture reference data shape: {soil_moisture_ref.shape}")
    print(f"Soil moisture NaN values: {soil_nan_count} out of {soil_total_elements} elements ({soil_nan_count/soil_total_elements*100:.2f}%)")
    
    print(f"SWE reference data shape: {swe_ref.shape}")
    print(f"SWE NaN values: {swe_nan_count} out of {swe_total_elements} elements ({swe_nan_count/swe_total_elements*100:.2f}%)")
    
    # Calculate means over reference period, handling NaNs explicitly
    if soil_moisture_ref.size > 0 and not np.all(np.isnan(soil_moisture_ref)):
        # Replace NaNs with zeros for calculation
        soil_moisture_clean = np.nan_to_num(soil_moisture_ref, nan=0.0)
        soil_moisture_mean = np.mean(soil_moisture_clean, axis=0)
        print("Soil moisture mean calculation successful")
    else:
        print("⚠️ Warning: No valid soil moisture data in reference period, using zeros")
        soil_moisture_mean = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
    
    if swe_ref.size > 0 and not np.all(np.isnan(swe_ref)):
        # Replace NaNs with zeros for calculation
        swe_clean = np.nan_to_num(swe_ref, nan=0.0)
        swe_mean = np.mean(swe_clean, axis=0)
        print("SWE mean calculation successful")
    else:
        print("⚠️ Warning: No valid SWE data in reference period, using zeros")
        swe_mean = np.zeros((ds.lat.shape[0], ds.lon.shape[0]))
    
    print(f"Calculated component means over reference period {reference_start} to {reference_end}")
    
    # 4. Now process each time step to calculate groundwater anomalies
    print("Calculating groundwater storage anomalies...")
    
    # Process only valid times
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
            
            # Get the soil moisture and SWE for this time step
            soil_moisture = all_soil_moisture[i]
            swe = all_swe[i]
            
            # Convert to anomalies relative to 2004-2009 reference period
            soil_moisture_anomaly = soil_moisture - soil_moisture_mean
            swe_anomaly = swe - swe_mean
            
            # Calculate groundwater anomaly: GWS_anomaly = TWS_anomaly - SM_anomaly - SWE_anomaly
            # TWS from model is already an anomaly relative to the same period
            gws = tws_spatial - soil_moisture_anomaly - swe_anomaly
            
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
            "processing_notes": "Enhanced model with lagged and seasonal features",
            "model_info": "Random Forest with lagged features and seasonal encoding"
        }
    )
    
    # Add metadata
    gws_ds.groundwater.attrs["long_name"] = "Groundwater Storage Anomaly (Enhanced Model)"
    gws_ds.groundwater.attrs["units"] = "cm water equivalent"
    gws_ds.groundwater.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    gws_ds.tws.attrs["long_name"] = "Total Water Storage Anomaly (Enhanced Model)"
    gws_ds.tws.attrs["units"] = "cm water equivalent"
    gws_ds.tws.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    gws_ds.soil_moisture_anomaly.attrs["long_name"] = "Soil Moisture Anomaly"
    gws_ds.soil_moisture_anomaly.attrs["units"] = "cm water equivalent"
    gws_ds.soil_moisture_anomaly.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    gws_ds.swe_anomaly.attrs["long_name"] = "Snow Water Equivalent Anomaly"
    gws_ds.swe_anomaly.attrs["units"] = "cm water equivalent"
    gws_ds.swe_anomaly.attrs["reference_period"] = f"{reference_start} to {reference_end}"
    
    # Save results
    output_path = "results/groundwater_storage_anomalies_enhanced.nc"
    gws_ds.to_netcdf(output_path)
    print(f"✅ Enhanced groundwater storage anomalies saved to {output_path}")
    print(f"   All anomalies are relative to the {reference_start} to {reference_end} reference period")
    
    return gws_ds

if __name__ == "__main__":
    calculate_groundwater_storage()