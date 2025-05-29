# pipeline.py
"""
GRACE Satellite Data Downscaling Pipeline

This pipeline orchestrates the complete workflow for downscaling GRACE satellite
data to monitor groundwater storage changes at high spatial resolution.

Usage:
    python pipeline.py --steps all              # Run complete pipeline
    python pipeline.py --steps train,gws        # Run specific steps
    python pipeline.py --download-only          # Only download data
    
Available steps:
    - download: Download satellite data from Earth Engine
    - features: Create aligned feature stack
    - train: Train Random Forest model
    - gws: Calculate groundwater storage
    - validate: Validate against wells
    - all: Run all steps
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline modules
from data_loader import main as download_data
from features import main as create_features
from updated_model_rf import main as train_model
from groundwater_enhanced import calculate_groundwater_storage
from validation.validate_groundwater import main as validate_results


def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('grace-pipeline')


def check_requirements():
    """Check if all required directories and files exist."""
    required_dirs = ["data/raw", "data/processed", "models", "results", "figures"]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
    
    # Check for config file
    if not Path("src/config.yaml").exists():
        raise FileNotFoundError("Configuration file 'src/config.yaml' not found!")
    
    return True


def run_download_step(logger):
    """Download satellite data."""
    logger.info("STEP 1: Downloading satellite data...")
    
    try:
        # Set up arguments for data_loader
        sys.argv = ['data_loader.py', '--download', 'all', '--region', 'mississippi']
        download_data()
        logger.info("✅ Data download completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Data download failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_features_step(logger):
    """Create feature stack."""
    logger.info("STEP 2: Creating feature stack...")
    
    try:
        create_features()
        logger.info("✅ Feature creation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Feature creation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_train_step(logger):
    """Train Random Forest model."""
    logger.info("STEP 3: Training Random Forest model...")
    
    try:
        train_model()
        logger.info("✅ Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_groundwater_step(logger):
    """Calculate groundwater storage."""
    logger.info("STEP 4: Calculating groundwater storage...")
    
    try:
        gws_ds = calculate_groundwater_storage()
        logger.info("✅ Groundwater calculation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Groundwater calculation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def run_validation_step(logger):
    """Validate results against wells."""
    logger.info("STEP 5: Validating against well observations...")
    
    try:
        validate_results()
        logger.info("✅ Validation completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="GRACE Satellite Data Downscaling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run (download,features,train,gws,validate,all)'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download data and exit'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step even if included in steps'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    logger.info("Starting GRACE downscaling pipeline")
    
    try:
        check_requirements()
    except Exception as e:
        logger.error(f"Requirements check failed: {str(e)}")
        return 1
    
    # Parse steps
    if args.download_only:
        steps = ['download']
    elif args.steps.lower() == 'all':
        steps = ['download', 'features', 'train', 'gws', 'validate']
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]
    
    if args.skip_download and 'download' in steps:
        steps.remove('download')
        logger.info("Skipping download step as requested")
    
    logger.info(f"Pipeline will run steps: {steps}")
    
    # Execute steps
    step_functions = {
        'download': run_download_step,
        'features': run_features_step,
        'train': run_train_step,
        'gws': run_groundwater_step,
        'validate': run_validation_step
    }
    
    failed_steps = []
    
    for step in steps:
        if step not in step_functions:
            logger.warning(f"Unknown step '{step}', skipping...")
            continue
        
        success = step_functions[step](logger)
        
        if not success:
            failed_steps.append(step)
            logger.error(f"Step '{step}' failed, stopping pipeline")
            break
    
    # Summary
    if failed_steps:
        logger.error(f"Pipeline failed at step(s): {failed_steps}")
        return 1
    else:
        logger.info("Pipeline completed successfully!")
        
        # Print final outputs
        logger.info("\n" + "="*60)
        logger.info("PIPELINE OUTPUTS:")
        logger.info("  - Feature stack: data/processed/feature_stack.nc")
        logger.info("  - Trained model: models/rf_model_enhanced.joblib")
        logger.info("  - Groundwater data: results/groundwater_storage_anomalies_corrected.nc")
        logger.info("  - Validation metrics: results/validation/")
        logger.info("  - Figures: figures/")
        logger.info("="*60)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())