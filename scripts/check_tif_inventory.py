import os
import rasterio
from glob import glob
import yaml

def get_tif_info(folder):
    print(f"\n📂 Checking folder: {folder}")
    tif_files = sorted(glob(os.path.join(folder, "*.tif")))
    print(f"  Total .tif files: {len(tif_files)}")

    for f in tif_files[:5]:  # Show only first 5 as sample
        try:
            with rasterio.open(f) as src:
                print(f"    🗂 {os.path.basename(f)} | Shape: {src.shape} | CRS: {src.crs} | Bounds: {src.bounds}")
        except Exception as e:
            print(f"    ⚠️ Failed to read {f}: {e}")

    if len(tif_files) > 5:
        print(f"    ... (Only first 5 files shown)")

def main():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)
    
    input_dirs = config["input_dirs"]
    for folder in input_dirs:
        get_tif_info(folder)

if __name__ == "__main__":
    main()
