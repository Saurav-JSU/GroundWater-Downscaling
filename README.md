# GRACE Satellite Data Downscaling for Groundwater Monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Earth Engine](https://img.shields.io/badge/Google-Earth%20Engine-green.svg)](https://earthengine.google.com/)

A machine learning pipeline for downscaling GRACE satellite data from ~300km to ~25km resolution to monitor groundwater storage changes across the Mississippi River Basin (2003-2022).

![Groundwater Storage Anomalies](figures/publication/main_validation_figure.png)
*Example output: Groundwater storage anomalies and validation performance across the Mississippi River Basin*

## 🎯 Overview

This project addresses the critical need for high-resolution groundwater monitoring by:
- **Downscaling** coarse GRACE satellite observations using machine learning
- **Integrating** multiple satellite and model datasets for enhanced predictions
- **Decomposing** total water storage into groundwater components
- **Validating** results against 1,400+ USGS monitoring wells

### Key Results
- ✅ **Model Performance**: R² = 0.86 on test data
- ✅ **Spatial Resolution**: Enhanced from ~300km to ~25km
- ✅ **Temporal Coverage**: Monthly data from 2003-2022
- ✅ **Validation**: Correlation with wells r = 0.22 (point) to 0.35 (spatial average)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Sources](#-data-sources)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [Results](#-results)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## 🚀 Features

- **Automated Data Pipeline**: Downloads and processes satellite data from Google Earth Engine
- **Advanced ML Model**: Random Forest with temporal features (lags, seasonality)
- **Water Balance Decomposition**: Separates total water storage into components
- **Comprehensive Validation**: Multiple validation approaches for different scales
- **Production Ready**: Modular design, extensive logging, error handling
- **Visualization Suite**: Publication-quality figures and animations

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account (free for research)
- 16GB+ RAM recommended
- 50GB+ free disk space

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/grace-downscaling.git
cd grace-downscaling
```

2. **Create conda environment**
```bash
conda create -n grace-downscaling python=3.8
conda activate grace-downscaling
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Authenticate Earth Engine**
```bash
earthengine authenticate
```

5. **Configure study area** (optional)
Edit `src/config.yaml` to modify the study region or parameters.

## 🚀 Quick Start

### Run the complete pipeline:
```bash
python pipeline.py --steps all
```

This will:
1. Download all satellite data (~4-8 hours)
2. Create aligned feature stack (~30 minutes)
3. Train the Random Forest model (~20 minutes)
4. Calculate groundwater storage anomalies (~10 minutes)
5. Validate against USGS wells (~5 minutes)

### Run specific steps:
```bash
# Just train model and calculate groundwater
python pipeline.py --steps train,gws

# Skip download if data exists
python pipeline.py --steps all --skip-download

# Only validate results
python pipeline.py --steps validate
```

## 📊 Data Sources

| Dataset | Source | Resolution | Variables | Purpose |
|---------|--------|------------|-----------|---------|
| GRACE MASCON | NASA/JPL | ~300km | Total water storage | Target variable |
| GLDAS-2.1 | NASA | 0.25° | Soil moisture (4 layers), ET, SWE | Water components |
| CHIRPS | UCSB | 0.05° | Precipitation | Climate forcing |
| TerraClimate | U of Idaho | 4km | Temperature, ET, water deficit | Climate variables |
| MODIS | NASA | 500m | Land cover classification | Static features |
| SRTM | USGS | 30m | Elevation | Topography |
| USGS NWIS | USGS | Point | Groundwater levels | Validation |
| OpenLandMap | OpenGeoHub | 250m | Soil properties | Static features |

## 🔬 Methodology

### 1. Data Preprocessing
- Temporal alignment to GRACE availability
- Spatial resampling to common 0.25° grid
- Quality control and gap filling
- Unit standardization

### 2. Feature Engineering
```python
# Temporal features (per variable)
- Current month value
- Lagged values (1, 3, 6 months)
- 12-month rolling statistics

# Seasonal features
- Cyclical encoding (sin/cos of month)

# Static features
- Elevation, slope
- Land cover type
- Soil properties (texture, depth)
```

### 3. Machine Learning Model
```python
RandomForestRegressor(
    n_estimators=200,      # Number of trees
    max_depth=25,          # Tree depth
    min_samples_split=5,   # Min samples to split
    max_features='sqrt',   # Features per split
    n_jobs=-1             # Parallel processing
)
```

### 4. Water Balance Decomposition
```
GWS_anomaly = TWS_anomaly - SM_anomaly - SWE_anomaly

Where:
- GWS: Groundwater Storage
- TWS: Total Water Storage (from GRACE)
- SM: Soil Moisture (from GLDAS)
- SWE: Snow Water Equivalent (from GLDAS)
```

### 5. Validation Approach
- **Point validation**: Direct comparison with individual wells
- **Spatial averaging**: Wells averaged within 50km radius
- **Metrics**: Pearson correlation, RMSE, trend correlation
- **Scale analysis**: Performance vs. averaging radius

## 📁 Project Structure

```
grace-downscaling/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── pipeline.py                   # Main orchestration script
├── src/                         # Source code
│   ├── config.yaml              # Configuration file
│   ├── data_loader.py           # Earth Engine data download
│   ├── features.py              # Feature engineering
│   ├── updated_model_rf.py      # Random Forest model
│   ├── groundwater_enhanced.py  # Groundwater calculation
│   ├── utils.py                 # Utility functions
│   └── validation/              # Validation module
│       └── validate_groundwater.py
├── scripts/                     # Additional scripts
│   ├── utilities/               # Data checking, testing
│   ├── visualization/           # Plotting scripts
│   └── analysis/               # Analysis notebooks
├── data/                       # Data directory (git-ignored)
│   ├── raw/                    # Downloaded satellite data
│   └── processed/              # Processed features
├── models/                     # Trained models (git-ignored)
├── results/                    # Output data (git-ignored)
└── figures/                    # Generated plots (git-ignored)
```

## 📖 Usage Guide

### Basic Usage

```python
# Import the main modules
from src.features import create_features
from src.updated_model_rf import train_model
from src.groundwater_enhanced import calculate_groundwater_storage
from src.validation.validate_groundwater import GroundwaterValidator

# Or use the pipeline
import subprocess
subprocess.run(['python', 'pipeline.py', '--steps', 'all'])
```

### Advanced Configuration

Edit `src/config.yaml`:
```yaml
# Study region
region:
  name: "Mississippi River Basin"
  lat_min: 28.0
  lat_max: 49.0
  lon_min: -100.0
  lon_max: -82.0

# Model parameters
model:
  n_estimators: 200
  max_depth: 25
  
# Processing options
resolution: 0.25  # degrees
reference_period: ["2004-01", "2009-12"]
```

### Custom Region Analysis

```python
# Modify data_loader.py to add your region
REGIONS = {
    "mississippi": ee.Geometry.Rectangle([-100, 28, -82, 49]),
    "your_region": ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
}

# Run pipeline for your region
python pipeline.py --region your_region
```

## 📈 Results

### Model Performance
- **Training R²**: 0.956
- **Testing R²**: 0.865
- **RMSE**: 3.98 cm
- **Feature Importance**: 
  - Lagged features: 45%
  - Current features: 35%
  - Static features: 15%
  - Seasonal: 5%

### Validation Metrics
| Method | Mean Correlation | Coverage | Best Performance |
|--------|-----------------|----------|------------------|
| Point wells | 0.22 ± 0.22 | 1,396 wells | r > 0.5: 10% |
| Spatial 50km | 0.35 ± 0.18 | 187 grid cells | r > 0.5: 28% |
| Spatial 100km | 0.42 ± 0.15 | 89 grid cells | r > 0.5: 41% |

### Output Products

1. **Groundwater Storage Anomalies**
   - File: `results/groundwater_storage_anomalies_corrected.nc`
   - Resolution: 0.25° (~25km)
   - Frequency: Monthly
   - Period: 2003-2022
   - Units: cm water equivalent

2. **Validation Reports**
   - Point validation: `results/validation/point_validation_metrics.csv`
   - Spatial validation: `results/validation/spatial_avg_50km_metrics.csv`
   - Summary report: `results/validation/validation_report.txt`

## 🔧 API Reference

### Main Functions

```python
# Data download
from src.data_loader import export_grace, export_gldas, download_usgs_well_data

# Feature engineering
from src.features import create_feature_stack, load_grace_months

# Model training
from src.updated_model_rf import train_enhanced_model, create_lagged_features

# Groundwater calculation
from src.groundwater_enhanced import calculate_groundwater_storage

# Validation
from src.validation.validate_groundwater import GroundwaterValidator
validator = GroundwaterValidator()
validator.validate_point_to_point()
validator.validate_spatial_average(radius_km=50)
```

### Data Access

```python
import xarray as xr

# Load results
gws = xr.open_dataset('results/groundwater_storage_anomalies_corrected.nc')

# Access variables
groundwater = gws.groundwater  # Main output
tws = gws.tws                  # Total water storage
soil_moisture = gws.soil_moisture_anomaly
snow = gws.swe_anomaly

# Get time series at a location
lat, lon = 35.0, -90.0
timeseries = gws.groundwater.sel(lat=lat, lon=lon, method='nearest')
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/Saurav-JSU/GroundWater-Downscaling

# Create development environment
conda create -n grace-dev python=3.8
conda activate grace-dev
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

### Reporting Issues
Please use the [GitHub issue tracker](https://github.com/yourusername/grace-downscaling/issues) to report bugs or request features.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{grace_downscaling_2024,
  author = {Bhattarai, Saurav},
  title = {GRACE Satellite Data Downscaling for Groundwater Monitoring},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Saurav-JSU/GroundWater-Downscaling},
}
```

### Related Publications
- Manuscript in preparation: "Machine Learning Downscaling of GRACE Satellite Data for High-Resolution Groundwater Monitoring"
- Conference presentation: AGU Fall Meeting 2024

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA GRACE/GRACE-FO team for satellite data
- Google Earth Engine for data access platform  
- USGS for groundwater well observations
- Funding support from [Your Funding Source]

## 📞 Contact

- **Lead Developer**: Saurav Bhattarai
- **Email**: [saurav.bhattarai@students.jsums.edu]
- **Lab Website**: [bit.ly/jsu_water]
- **Issues**: [GitHub Issues](https://github.com/Saurav-JSU/GroundWater-Downscaling)

---

**Last Updated**: November 2024  
**Version**: 1.0.0  
**Status**: Active Development