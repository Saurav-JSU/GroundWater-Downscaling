# GRACE Satellite Data Downscaling for Groundwater Monitoring

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Earth Engine](https://img.shields.io/badge/Google-Earth%20Engine-green.svg)](https://earthengine.google.com/)

A comprehensive machine learning pipeline for downscaling GRACE satellite data to monitor groundwater storage changes at high spatial resolution across the Mississippi River Basin (2003-2022).

## 🎯 Project Objectives

- **Downscale GRACE data** from ~300km to ~25km resolution using Random Forest
- **Extract groundwater storage anomalies** through water balance decomposition
- **Monitor groundwater trends** and drought impacts over 20 years
- **Validate results** against USGS well observations
- **Generate publication-quality visualizations** for scientific analysis

## 📊 Key Features

- **Multi-source data integration**: GRACE, GLDAS, CHIRPS, TerraClimate, MODIS, USGS
- **Advanced feature engineering**: Lagged features, seasonal encoding, static variables
- **Enhanced machine learning**: Random Forest with 200+ trees and optimized parameters
- **Comprehensive validation**: Statistical metrics and spatial performance maps
- **Automated pipeline**: End-to-end processing with quality control
- **Rich visualizations**: Monthly animations, trend maps, drought analysis

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- Google Earth Engine account (for data download)
- At least 16GB RAM recommended
- 50+ GB free disk space

### Python Dependencies

```bash
# Create virtual environment
conda create -n grace-downscaling python=3.8
conda activate grace-downscaling

# Install core scientific packages
conda install -c conda-forge xarray rasterio rioxarray netcdf4 dask
conda install -c conda-forge scikit-learn matplotlib seaborn cartopy
conda install -c conda-forge tqdm pyyaml joblib pandas numpy

# Install Earth Engine packages
pip install earthengine-api geemap

# Install additional packages
pip install dataretrieval imageio pillow
```

### Earth Engine Authentication

```bash
# Authenticate Earth Engine (one-time setup)
earthengine authenticate
```

## 📁 Project Structure

```
grace-downscaling/
├── data/
│   ├── raw/                    # Downloaded satellite data
│   │   ├── grace/             # GRACE TWS data
│   │   ├── gldas/             # GLDAS land surface model
│   │   ├── chirps/            # CHIRPS precipitation
│   │   ├── terraclimate/      # TerraClimate variables
│   │   ├── modis_land_cover/  # MODIS land cover
│   │   ├── usgs_dem/          # USGS elevation data
│   │   ├── usgs_well_data/    # Groundwater well observations
│   │   └── openlandmap/       # Soil properties
│   └── processed/             # Processed feature stacks
├── models/                    # Trained ML models
├── results/                   # Analysis outputs
├── figures/                   # Generated plots and maps
├── src/                       # Source code
├── scripts/                   # Utility and visualization scripts
├── logs/                      # Processing logs
├── pipeline.py               # Main processing pipeline
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd grace-downscaling
pip install -r requirements.txt
```

### 2. Configure Study Area
Edit `src/config.yaml`:
```yaml
region:
  name: "Mississippi River Basin"
  lat_min: 28.0
  lat_max: 49.0
  lon_min: -100.0
  lon_max: -82.0

resolution: 0.25  # degrees (~25km)
target_crs: EPSG:4326
```

### 3. Download Data
```bash
# Download all datasets (takes several hours)
python src/data_loader.py --download all --region mississippi

# Or download specific datasets
python src/data_loader.py --download grace gldas chirps --region mississippi
```

### 4. Process Features
```bash
# Create aligned feature stack
python src/features.py

# Validate feature quality
python scripts/check_netcdf.py data/processed/feature_stack.nc
```

### 5. Train Model
```bash
# Train enhanced Random Forest model
python src/updated_model_rf.py
```

### 6. Run Complete Pipeline
```bash
# Execute full analysis pipeline
python pipeline.py --steps all

# Or run specific steps
python pipeline.py --steps gws,validate
```

## 📋 Detailed Workflow

### Data Download (`src/data_loader.py`)

Downloads multi-source satellite and model data:

- **GRACE MASCON**: Total water storage anomalies (NASA/JPL)
- **GLDAS-2.1**: Soil moisture (4 layers), evapotranspiration, snow water equivalent
- **CHIRPS**: Daily precipitation aggregated to monthly
- **TerraClimate**: Temperature, precipitation, actual ET, climatic water deficit
- **MODIS MCD12Q1**: Annual land cover classification
- **USGS SRTM**: 30m digital elevation model
- **USGS NWIS**: Groundwater well observations
- **OpenLandMap**: Soil texture (sand/clay content by depth)

### Feature Engineering (`src/features.py`)

Creates comprehensive feature stack:
- **Temporal alignment**: All datasets resampled to monthly, 0.25° resolution
- **Quality control**: NaN handling, outlier detection
- **Feature naming**: Consistent timestamp-based naming convention

### Enhanced Modeling (`src/updated_model_rf.py`)

Advanced Random Forest implementation:
- **Lagged features**: 1, 3, and 6-month lags to capture memory effects
- **Seasonal encoding**: Cyclical month representation (sin/cos)
- **Static features**: Elevation, land cover, soil properties
- **Optimized parameters**: 200 trees, depth 25, sqrt features
- **Cross-validation**: 80/20 train/test split with performance metrics

### Groundwater Extraction (`src/groundwater_enhanced.py`)

Water balance decomposition:
```
GWS_anomaly = TWS_anomaly - SM_anomaly - SWE_anomaly
```
- **Reference period**: 2004-2009 climatological mean
- **Component separation**: Soil moisture and snow water equivalent
- **Uncertainty handling**: NaN value treatment and error propagation

### Validation (`src/validation.py`)

Statistical validation against observations:
- **USGS well data**: Monthly groundwater level anomalies
- **Metrics**: Pearson correlation, RMSE, Nash-Sutcliffe efficiency
- **Spatial analysis**: Performance maps and regional statistics
- **Time series plots**: Visual comparison of predicted vs observed

## 📊 Outputs

### Primary Results
- `results/groundwater_storage_anomalies.nc`: Main groundwater analysis
- `results/well_validation_metrics.csv`: Validation statistics
- `models/rf_model_enhanced.joblib`: Trained ML model

### Visualizations
- **Monthly maps**: `figures/monthly_groundwater/`
- **Publication figures**: `figures/publication/`
- **Validation plots**: `figures/validation/`
- **Model diagnostics**: `figures/feature_importance_enhanced.png`

### Key Datasets Generated

| File | Description | Format |
|------|-------------|---------|
| `feature_stack.nc` | Aligned multi-source features | NetCDF |
| `groundwater_storage_anomalies.nc` | Primary results | NetCDF |
| `well_validation_metrics.csv` | Performance statistics | CSV |
| `rf_model_enhanced.joblib` | Trained model | Joblib |

## ⚙️ Configuration Options

### Regional Settings (`src/config.yaml`)
```yaml
# Study region bounds
region:
  lat_min: 28.0    # Southern boundary
  lat_max: 49.0    # Northern boundary  
  lon_min: -100.0  # Western boundary
  lon_max: -82.0   # Eastern boundary

# Processing parameters
resolution: 0.25           # Spatial resolution (degrees)
reference_file: data/raw/grace/20030131_20030227.tif
target_crs: EPSG:4326     # Coordinate system
```

### Model Parameters (`src/updated_model_rf.py`)
```python
RandomForestRegressor(
    n_estimators=200,      # Number of trees
    max_depth=25,          # Tree depth
    min_samples_split=5,   # Minimum samples for split
    max_features='sqrt',   # Features per tree
    n_jobs=-1             # Use all CPU cores
)
```

## 🔧 Utility Scripts

### Data Quality Control
```bash
# Check NetCDF file structure and quality
python scripts/check_netcdf.py data/processed/feature_stack.nc

# Validate timestamp alignment
python scripts/validate_feature_timestamps.py

# Inventory raw data files
python scripts/check_tif_inventory.py
```

### Visualization Tools
```bash
# Generate monthly groundwater maps
python scripts/visualization_groundwater_monthly.py

# Create model comparison plots
python scripts/visualization_model_output_updated.py

# Generate publication figures
python scripts/publication_figures.py
```

### Data Checking
```bash
# Comprehensive data inventory
python src/data_checker.py

# Test processed features
python src/test_features.py
```

## 📈 Performance Expectations

### Computational Requirements
- **Memory**: 16+ GB RAM recommended
- **Storage**: 50+ GB for full dataset
- **Processing time**: 
  - Data download: 4-8 hours
  - Feature processing: 30 minutes
  - Model training: 15-30 minutes
  - Full pipeline: 1-2 hours

### Expected Accuracy
- **Well validation correlation**: 0.6-0.8 (typical)
- **RMSE**: 2-4 cm water equivalent
- **Nash-Sutcliffe efficiency**: 0.4-0.7
- **Spatial coverage**: 90%+ valid pixels

## 🐛 Troubleshooting

### Common Issues

**Earth Engine Authentication**
```bash
# Re-authenticate if getting permission errors
earthengine authenticate
python -c "import ee; ee.Initialize()"
```

**Memory Errors**
```python
# Reduce data loading chunk size in config
chunk_size: 50  # Reduce from default 100
```

**Missing Data Files**
```bash
# Check data inventory
python src/data_checker.py
# Re-download specific datasets
python src/data_loader.py --download grace --region mississippi
```

**Feature Dimension Mismatch**
```bash
# Validate feature alignment
python scripts/validate_feature_timestamps.py
# Regenerate features if needed
python src/features.py
```

### Error Logs
Check `logs/pipeline_YYYYMMDD_HHMMSS.log` for detailed error messages.

## 📚 Scientific Background

### Methodology
This project implements the methodology described in recent literature on GRACE data downscaling:

1. **Multi-source data fusion**: Combining satellite observations with land surface models
2. **Machine learning downscaling**: Using auxiliary high-resolution datasets as predictors
3. **Water balance decomposition**: Separating total water storage into components
4. **Temporal feature engineering**: Incorporating memory effects and seasonality

### Key References
- Rateb et al. (2020): GRACE data downscaling using machine learning
- Seyoum & Milewski (2017): Monitoring groundwater storage changes
- Feng et al. (2018): Evaluation of groundwater storage changes from GRACE

## 🤝 Contributing

### Development Setup
```bash
# Fork repository and create branch
git checkout -b feature/new-analysis

# Install development dependencies
pip install black flake8 pytest

# Run tests
python -m pytest tests/
```

### Code Style
- Use Black for formatting: `black src/`
- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include unit tests for new features

### Submitting Changes
1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [Contact information]

## 🏆 Citation

If you use this code in your research, please cite:

```bibtex
@software{grace_downscaling_2024,
  title={GRACE Satellite Data Downscaling for Groundwater Monitoring},
  author={[Author Names]},
  year={2024},
  url={[Repository URL]},
  doi={[DOI if available]}
}
```

## 🔄 Version History

- **v1.0.0**: Initial release with basic downscaling
- **v1.1.0**: Added enhanced features (lagged, seasonal)
- **v1.2.0**: Improved validation and visualization
- **v1.3.0**: Complete pipeline automation

---

**Last Updated**: [Current Date]  
**Maintainer**: Saurav Bhattarai
**Status**: Active Development