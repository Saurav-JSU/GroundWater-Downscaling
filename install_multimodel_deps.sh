#!/bin/bash
# install_multimodel_deps.sh
# Installation script for additional ML libraries

echo "🚀 Installing Multi-Model Dependencies for GRACE Downscaling"
echo "============================================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if pip exists
if ! command_exists pip; then
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

echo "📦 Installing additional machine learning libraries..."

# Install XGBoost
echo "Installing XGBoost..."
pip install xgboost

# Install LightGBM
echo "Installing LightGBM..."
pip install lightgbm

# Install CatBoost
echo "Installing CatBoost..."
pip install catboost

# Install additional useful packages
echo "Installing additional packages..."
pip install optuna      # For hyperparameter tuning (future enhancement)
pip install shap        # For model interpretability (future enhancement)

echo ""
echo "✅ Installation complete!"
echo ""
echo "Available models after installation:"
echo "  ✅ Random Forest (always available)"
echo "  ✅ XGBoost"
echo "  ✅ LightGBM" 
echo "  ✅ CatBoost"
echo "  ✅ Neural Network (always available)"
echo "  ✅ Support Vector Regression (always available)"
echo "  ✅ Gradient Boosting (always available)"
echo ""
echo "To test your installation, run:"
echo "  python pipeline.py --list-models"
echo ""
echo "To run the multi-model pipeline:"
echo "  python pipeline.py --steps train --models rf,xgb,lgb,catb"