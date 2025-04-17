import argparse
import os
import ee
import geemap
from datetime import datetime
from dataretrieval import nwis
import pandas as pd

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Study region: Mississippi River Basin (example bounding box)
REGIONS = {
    "mississippi": ee.Geometry.Rectangle([-100.0, 28.0, -82.0, 49.0])
}

# Output directories
RAW_DIR = "data/raw"
ALL_DATASETS = ["grace", "gldas", "chirps", "modis", "terraclimate", "dem", "usgs", "openlandmap"]

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    for sub in ["grace", "gldas", "chirps", "modis_land_cover", "terraclimate", "usgs_dem", "usgs_well_data", "openlandmap"]:
        os.makedirs(os.path.join(RAW_DIR, sub), exist_ok=True)

def export_grace(region):
    collection = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI") \
        .select("lwe_thickness") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)

    print("Exporting GRACE...")
    geemap.ee_export_image_collection(
        collection,
        out_dir=os.path.join(RAW_DIR, "grace"),
        scale=50000,
        region=region,
        file_per_band=False
    )

def export_gldas(region):
    print("Aggregating and exporting GLDAS monthly means...")
    variables = [
        "SoilMoi0_10cm_inst",
        "SoilMoi10_40cm_inst",
        "SoilMoi40_100cm_inst",
        "SoilMoi100_200cm_inst",
        "Evap_tavg",
        "SWE_inst"
    ]
    for var in variables:
        monthly = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H") \
            .filterDate("2003-01-01", "2022-12-31") \
            .filterBounds(region) \
            .select(var)

        def monthly_composite(date):
            start = ee.Date(date)
            end = start.advance(1, 'month')
            return monthly.filterDate(start, end).mean().set('system:time_start', start.millis())

        months = ee.List.sequence(0, 12 * 20 - 1).map(lambda i: ee.Date("2003-01-01").advance(i, 'month'))
        monthly_collection = ee.ImageCollection(months.map(monthly_composite))

        geemap.ee_export_image_collection(
            monthly_collection,
            out_dir=os.path.join(RAW_DIR, "gldas", var),
            scale=25000,
            region=region,
            file_per_band=False
        )

def export_chirps(region):
    print("Aggregating and exporting CHIRPS monthly totals...")
    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)

    def monthly_sum(date):
        start = ee.Date(date)
        end = start.advance(1, 'month')
        return collection.filterDate(start, end).sum().set('system:time_start', start.millis())

    months = ee.List.sequence(0, 12 * 20 - 1).map(lambda i: ee.Date("2003-01-01").advance(i, 'month'))
    monthly_collection = ee.ImageCollection(months.map(monthly_sum))

    geemap.ee_export_image_collection(
        monthly_collection,
        out_dir=os.path.join(RAW_DIR, "chirps"),
        scale=5000,
        region=region,
        file_per_band=False
    )

def export_modis_landcover(region):
    collection = ee.ImageCollection("MODIS/061/MCD12Q1") \
        .select("LC_Type1") \
        .filterDate("2000-01-01", "2022-12-31") \
        .filterBounds(region)

    print("Exporting MODIS Land Cover...")
    geemap.ee_export_image_collection(
        collection,
        out_dir=os.path.join(RAW_DIR, "modis_land_cover"),
        scale=2000,
        region=region,
        file_per_band=False
    )

def export_terraclimate(region):
    collection = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
        .filterDate("2003-01-01", "2022-12-31") \
        .filterBounds(region)

    bands = ["tmmx", "tmmn", "pr", "aet", "def"]

    for band in bands:
        print(f"Exporting TerraClimate: {band}")
        subset = collection.select(band)
        geemap.ee_export_image_collection(
            subset,
            out_dir=os.path.join(RAW_DIR, "terraclimate", band),
            scale=4000,
            region=region,
            file_per_band=False
        )

def export_usgs_dem(region):
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")

    print("Exporting USGS DEM...")
    geemap.ee_export_image(
        dem,
        filename=os.path.join(RAW_DIR, "usgs_dem", "srtm_dem.tif"),
        scale=750,
        region=region
    )

def download_usgs_well_data():
    print("Downloading USGS well data (monthly groundwater anomalies)...")

    states = ['MS', 'AR', 'LA', 'TN', 'MO', 'KY', 'IL', 'IN', 'OH', 'AL']
    all_data = []

    for state in states:
        print(f"Fetching sites from {state}...")
        try:
            info, _ = nwis.get_info(stateCd=state, siteType="GW", siteStatus="active")
            site_ids = info['site_no'].unique().tolist()
        except Exception as e:
            print(f"Failed to fetch sites for {state}: {e}")
            continue

        for site in site_ids:
            try:
                df, _ = nwis.get_gwlevels(site, start='2003-01-01', end='2022-12-31', datetime_index=False)
                if df.empty or 'lev_dt' not in df.columns or 'lev_va' not in df.columns:
                    continue
                df['datetime'] = pd.to_datetime(df['lev_dt'])
                df['depth_m'] = df['lev_va'] * 0.3048
                monthly = df.set_index('datetime')['depth_m'].resample('MS').agg(['mean', 'count'])
                monthly = monthly[monthly['count'] >= 2]['mean']
                anomaly = monthly - monthly.mean()
                all_data.append(anomaly.rename(site))
            except Exception as e:
                continue

    if not all_data:
        print("No valid well data found.")
        return

    print("Saving final USGS well anomaly table...")
    combined_df = pd.concat(all_data, axis=1)
    combined_df.index.name = 'Date'
    combined_df.to_csv(os.path.join(RAW_DIR, "usgs_well_data", "monthly_groundwater_anomalies.csv"))
    print("Saved: monthly_groundwater_anomalies.csv")

def download_usgs_well_data():
    print("Downloading USGS well data for Mississippi River Basin...")
    site_data = nwis.get_sites(stateCd='MS', parameterCd='72019', siteType='GW')
    site_ids = site_data['site_no'].tolist()
    data = nwis.get_record(sites=site_ids, service='dv', start='2003-01-01', end='2022-12-31', parameterCd='72019')
    data.to_csv(os.path.join(RAW_DIR, "usgs_well_data", "mississippi_wells.csv"))
    print("USGS well data saved.")

def export_openlandmap_soil(region):
    datasets = {
        'clay': 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02',
        'sand': 'OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02',
        'silt': 'OpenLandMap/SOL/SOL_SILT-WFRACTION_USDA-3A1A1A_M/v02'
    }
    depths = {
        '0cm': 'b0',
        '10cm': 'b10',
        '30cm': 'b30',
        '60cm': 'b60',
        '100cm': 'b100',
        '200cm': 'b200'
    }
    for prop, asset_id in datasets.items():
        for depth_label, band_name in depths.items():
            print(f"Exporting {prop} at {depth_label}")
            try:
                img = ee.Image(asset_id).select(band_name).clip(region).reproject(crs='EPSG:4326', scale=250)
                geemap.ee_export_image(
                    img,
                    filename=os.path.join(RAW_DIR, "openlandmap", f"{prop}_{depth_label}.tif"),
                    scale=750,
                    region=region.bounds()
                )
            except Exception as e:
                print(f"Failed to export {prop} at {depth_label}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for GRACE downscaling")
    parser.add_argument("--download", nargs="+", choices=ALL_DATASETS + ["all"], required=True, help="Datasets to download")
    parser.add_argument("--region", type=str, default="mississippi", help="Region name (default: mississippi)")
    args = parser.parse_args()

    region = REGIONS.get(args.region.lower())
    if not region:
        raise ValueError(f"Unknown region: {args.region}")

    ensure_dirs()

    datasets_to_download = ALL_DATASETS if "all" in args.download else args.download

    if "grace" in datasets_to_download:
        export_grace(region)
    if "gldas" in datasets_to_download:
        export_gldas(region)
    if "chirps" in datasets_to_download:
        export_chirps(region)
    if "modis" in datasets_to_download:
        export_modis_landcover(region)
    if "terraclimate" in datasets_to_download:
        export_terraclimate(region)
    if "dem" in datasets_to_download:
        export_usgs_dem(region)
    if "usgs" in datasets_to_download:
        download_usgs_well_data()
    if "openlandmap" in datasets_to_download:
        export_openlandmap_soil(region)

if __name__ == "__main__":
    main()
