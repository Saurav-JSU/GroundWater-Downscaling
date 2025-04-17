import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import match_resolution
import re

def load_feature_stack(path):
    ds = xr.open_dataset(path)
    return ds if "features" not in ds else ds["features"]

def get_grace_months(grace_dir):
    files = sorted([f for f in os.listdir(grace_dir) if f.endswith(".tif")])
    months = [pd.to_datetime(f.split("_")[0], format="%Y%m%d").strftime("%Y-%m") for f in files]
    return months

def load_grace_stack(grace_dir, timestamps, shape):
    files = sorted([f for f in os.listdir(grace_dir) if f.endswith(".tif")])
    data = []
    used_indices = []
    for i, ts in enumerate(timestamps):
        match = [f for f in files if f.startswith(ts.replace("-", ""))]
        if match:
            f = match[0]
            da = xr.open_dataarray(os.path.join(grace_dir, f), engine="rasterio").squeeze()
            data.append(da.data)
            used_indices.append(i)
    if len(data) == 0:
        raise ValueError("No GRACE files matched the feature timestamps.")
    return np.stack(data), used_indices

def evaluate_predictions(Y_true, Y_pred):
    mask = ~np.isnan(Y_true)
    rmse = np.sqrt(mean_squared_error(Y_true[mask], Y_pred[mask]))
    r2 = r2_score(Y_true[mask], Y_pred[mask])
    return rmse, r2

def plot_timeseries(Y_true, Y_pred, timestamps, iy, ix):
    series_true = Y_true[:, iy, ix]
    series_pred = Y_pred[:, iy, ix]
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, series_true, label="GRACE (obs)", lw=2)
    plt.plot(timestamps, series_pred, label="RF Prediction", lw=2)
    plt.title(f"GWS Timeseries at pixel (y={iy}, x={ix})")
    plt.xlabel("Time")
    plt.ylabel("TWS Anomaly")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"results/time_series/pixel_y{iy}_x{ix}.png")
    plt.close()

def plot_spatial_map(pred, obs, timestamp, idx):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axs[0].imshow(obs, cmap="viridis")
    axs[0].set_title("GRACE")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(pred, cmap="viridis")
    axs[1].set_title("RF Prediction")
    fig.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(pred - obs, cmap="RdBu")
    axs[2].set_title("Residual (Pred - Obs)")
    fig.colorbar(im2, ax=axs[2])
    for ax in axs:
        ax.axis("off")
    fig.suptitle(f"{timestamp.strftime('%Y-%m')}")
    plt.tight_layout()
    plt.savefig(f"results/maps/spatial_{idx:03d}.png")
    plt.close()

def main():
    os.makedirs("results/time_series", exist_ok=True)
    os.makedirs("results/maps", exist_ok=True)

    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    feature_stack_path = config["output_path"]
    grace_dir = config["grace_dir"]
    model_path = config["rf_model_path"]

    print("\n📦 Loading feature stack...")
    X_da = load_feature_stack(feature_stack_path)

    feature_names = X_da.feature.values
    temporal_features = [f for f in feature_names if re.search(r"_\d{4}-\d{2}$", f)]
    static_features = [f for f in feature_names if f not in temporal_features]

    model = joblib.load(model_path)
    print("✅ Model expects:", model.n_features_in_)
    print("📦 Feature stack shape:", X_da.shape)
    print(f"📆 Temporal features: {len(temporal_features)}")
    print(f"🗺️ Static features: {len(static_features)}")

    timestamps = [pd.to_datetime(f.split("_")[-1]) for f in temporal_features]
    T = len(timestamps)
    H, W = X_da.sizes["y"], X_da.sizes["x"]

    temporal_data = X_da.sel(feature=temporal_features).data.reshape(T, H, W, -1).astype(np.float32)
    static_data = X_da.sel(feature=static_features).data.transpose(1, 2, 0).astype(np.float32)
    static_data = np.broadcast_to(static_data, (T, H, W, static_data.shape[-1]))
    X = np.concatenate([temporal_data, static_data], axis=-1)

    print("\n📅 Loading GRACE TWS labels...")
    Y_true, valid_indices = load_grace_stack(grace_dir, [t.strftime("%Y-%m") for t in timestamps], (T, H, W))
    timestamps = [timestamps[i] for i in valid_indices]

    X = X[valid_indices]
    print("\n📤 Loading trained model...")
    print("\n🤖 Predicting with Random Forest...")
    print("🔍 Input to model:")
    print("X shape:", X.shape)
    print("X_2d shape:", X.reshape(-1, X.shape[-1]).shape)
    print("Model expects:", model.n_features_in_)

    if X.shape[-1] != model.n_features_in_:
        print("❌ Feature mismatch. Using alternate reshaping.")
        X_all = X_da.transpose("y", "x", "feature").data.astype(np.float32)
        X = np.stack([X_all for _ in range(len(valid_indices))], axis=0)

    X_2d = X.reshape(-1, X.shape[-1])
    Y_pred_flat = model.predict(X_2d)
    Y_pred = Y_pred_flat.reshape(X.shape[0], H, W)

    print("\n📈 Calculating metrics...")
    rmse, r2 = evaluate_predictions(Y_true.ravel(), Y_pred.ravel())
    print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

    print("\n🖼 Generating sample spatial maps and time series...")
    for i in [10, 50, 100, 150, 200]:
        if i < len(timestamps):
            plot_spatial_map(Y_pred[i], Y_true[i], timestamps[i], i)

    for iy, ix in [(10, 20), (25, 30), (40, 35)]:
        plot_timeseries(Y_true, Y_pred, timestamps, iy, ix)

    print("\n✅ Evaluation complete. Outputs saved to results/ folder.")

if __name__ == "__main__":
    main()