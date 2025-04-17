# pipeline.py
import os
import time
import argparse
from pathlib import Path
import logging

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("grace-pipeline")

def run_groundwater_calculation():
    """Run groundwater storage calculation"""
    logger.info("STEP 1: Calculating groundwater storage anomalies")
    from src.groundwater import calculate_groundwater_storage
    
    try:
        result = calculate_groundwater_storage()
        logger.info("Groundwater calculation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to calculate groundwater: {e}")
        return False

def run_validation():
    """Run validation against wells"""
    logger.info("STEP 2: Validating against well observations")
    from src.validation import validate_with_wells
    
    try:
        metrics = validate_with_wells()
        logger.info(f"Validation completed with average correlation: {metrics['correlation'].mean():.2f}")
        return True
    except Exception as e:
        logger.error(f"Failed to run validation: {e}")
        return False

def create_figures():
    """Create publication figures"""
    logger.info("STEP 3: Creating publication figures")
    from scripts.publication_figures import create_publication_figures
    
    try:
        create_publication_figures()
        logger.info("Publication figures created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create figures: {e}")
        return False

def check_prerequisites():
    """Check if required files and directories exist"""
    # Check for processed feature stack
    if not os.path.exists("data/processed/feature_stack.nc"):
        logger.error("Feature stack not found at data/processed/feature_stack.nc")
        logger.error("Run feature preprocessing first")
        return False
    
    # Check for trained model
    if not os.path.exists("models/rf_model.joblib"):
        logger.error("RF model not found at models/rf_model.joblib")
        logger.error("Train the model first using model_rf.py")
        return False
    
    # Check for GRACE data
    if not os.path.exists("data/raw/grace"):
        logger.error("GRACE data not found at data/raw/grace")
        logger.error("Download GRACE data first")
        return False
    
    return True

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="GRACE Downscaling Pipeline")
    parser.add_argument("--steps", type=str, default="all",
                       help="Comma-separated list of steps to run: gws,validate,figures,all")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting GRACE downscaling pipeline")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisite check failed. Exiting.")
        exit(1)
    
    # Determine which steps to run
    steps = args.steps.lower().split(",")
    run_all = "all" in steps
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run pipeline steps
    if run_all or "gws" in steps:
        if not run_groundwater_calculation():
            logger.error("Groundwater calculation failed. Exiting pipeline.")
            exit(1)
    
    if run_all or "validate" in steps:
        if not run_validation():
            logger.error("Validation failed. Continuing with caution.")
    
    if run_all or "figures" in steps:
        if not create_figures():
            logger.error("Figure creation failed. Pipeline completed with errors.")
            exit(1)
    
    logger.info("Pipeline completed successfully!")