# GRACE Downscaling and Groundwater Storage Modeling

## Overview

This repository contains code for downscaling GRACE (Gravity Recovery and Climate Experiment) satellite data and estimating groundwater storage anomalies for the Mississippi River Basin. The project uses Random Forest machine learning to combine multiple hydrometeorological datasets at higher resolution with coarse-resolution GRACE data to produce improved groundwater storage estimates.

## Features

- Automated data acquisition from various sources (GRACE, GLDAS, CHIRPS, TerraClimate, etc.)
- Feature engineering with temporal and spatial variables
- Random Forest modeling with enhanced feature sets (including lagged features)
- Groundwater storage anomaly calculation using water balance approach
- Comprehensive visualization tools for spatial and temporal analysis
- Publication-quality figure generation

## Directory Structure

```
.
├── data/                  # Data directory
│   ├── raw/               # Original downloaded datasets
│   └── processed/         # Processed feature stacks
├── models/                # Trained machine learning models
├── results/               # Output NetCDF files with groundwater estimates
├── figures/               # Generated figures and visualizations
│   ├── monthly_groundwater/       # Monthly groundwater maps
│   └── publication/               # Publication-quality figures
├── visualizations/        # Model output visualizations
│   ├── grace_comparison/          # Original model GRACE comparisons
│   └── grace_comparison_enhanced/ # Enhanced model GRACE comparisons
├── scripts/               # Utility and visualization scripts
└── src/                   # Core source code
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/grace-downscaling.git
cd grace-downscaling
```

2. Create a conda environment:
```bash
conda create -n grace-downscaling python=3.10
conda activate grace-downscaling
```

3. Install dependencies:
```bash
pip install numpy pandas xarray rioxarray scikit-learn joblib matplotlib cartopy seaborn tqdm
pip install earthengine-api geemap
```

4. Authenticate with Google Earth Engine (only needed for data downloading):
```bash
earthengine authenticate
```

## Data Acquisition

To download the required datasets:

```bash
python src/data_loader.py --download all --region mississippi
```

This will download data from:
- GRACE (Total Water Storage)
- GLDAS (Soil Moisture, Snow Water Equivalent)
- CHIRPS (Precipitation)
- TerraClimate (Temperature, Precipitation, Evapotranspiration)
- MODIS (Land Cover)
- USGS (DEM, Groundwater Well Data)
- OpenLandMap (Soil Properties)

## Workflow

### 1. Check Data

First, verify the downloaded data:

```bash
python scripts/check_tif_inventory.py
```

### 2. Prepare Features

Create the feature stack for model training:

```bash
python src/features.py
```

Test feature stack integrity:

```bash
python src/test_features.py
```

### 3. Train Models

#### Original Model
```bash
python src/model_rf.py
```

#### Enhanced Model (with lagged & seasonal features)
```bash
python src/updated_model_rf.py
```

### 4. Calculate Groundwater Storage

#### Using Original Model
```bash
python src/groundwater.py
```

#### Using Enhanced Model
```bash
python src/groundwater_enhanced.py
```

### 5. Visualization

#### Model Performance Visualization

Original model:
```bash
python scripts/visualization_model_output.py
```

Enhanced model:
```bash
python scripts/visualization_model_output_updated.py
```

#### Groundwater Visualization

Original model:
```bash
python scripts/visualization_groundwater_monthly.py
```

Enhanced model:
```bash
python scripts/visualization_groundwater_monthly.py --input results/groundwater_storage_anomalies_enhanced.nc --output figures/monthly_groundwater_enhanced
```

#### Publication Figures
```bash
python scripts/publication_figures.py
```

## Model Enhancement

The repository includes two model versions:

1. **Base Model**: Standard Random Forest using concurrent features
2. **Enhanced Model**: Advanced Random Forest with:
   - Lagged features (1, 3, and 6 months)
   - Seasonal encoding (sine/cosine of month)
   - Optimized hyperparameters

The enhanced model shows improved performance:
- Higher R² (0.86 vs 0.82 on test data)
- Better capture of seasonal patterns
- Average 6% difference in groundwater estimates

## Visualizing Results

### Model Comparison
Compare original vs. enhanced model performance:

```bash
# Compare visualizations between:
visualizations/grace_comparison/          # Original model
visualizations/grace_comparison_enhanced/ # Enhanced model
```

### Groundwater Maps
View monthly groundwater maps:

```bash
# Compare visualizations between:
figures/monthly_groundwater/          # Original model
figures/monthly_groundwater_enhanced/ # Enhanced model
```

### Publication Figures
Publication-quality figures are available in:
```
figures/publication/
```

## NetCDF Format

The output groundwater storage files are in NetCDF format with the following variables:
- `groundwater`: Groundwater storage anomalies
- `tws`: Total water storage anomalies
- `soil_moisture_anomaly`: Soil moisture anomalies
- `swe_anomaly`: Snow water equivalent anomalies

All anomalies are relative to the 2004-2009 reference period and in units of cm water equivalent.

## Validation

To validate results against USGS well observations:

```bash
python src/validation.py
```

## Quality Control

To check the integrity of NetCDF files:

```bash
python scripts/check_netcdf.py results/groundwater_storage_anomalies.nc
```
