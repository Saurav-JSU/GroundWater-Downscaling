#!/usr/bin/env python3
"""
Download Shapefiles for Mississippi River Basin Analysis
========================================================

This script downloads relevant shapefiles for the GRACE groundwater analysis:
1. Mississippi River Basin boundary
2. Major aquifer boundaries
3. State boundaries
4. HUC (Hydrologic Unit Code) watersheds
5. Sub-regional boundaries

Sources:
- USGS Water Resources
- EPA WATERS
- Natural Earth Data
- HydroSHEDS
"""

import os
import requests
import zipfile
import geopandas as gpd
from pathlib import Path
import numpy as np
from tqdm import tqdm

class ShapefileDownloader:
    """Download and prepare shapefiles for analysis."""
    
    def __init__(self, output_dir="data/shapefiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define shapefile sources
        self.sources = {
            'mississippi_basin': {
                'url': 'https://water.usgs.gov/GIS/dsdl/mrb_e2rf1_bas_mrb.zip',
                'name': 'Mississippi River Basin',
                'description': 'USGS Mississippi River Basin boundary'
            },
            
            'major_aquifers': {
                'url': 'https://water.usgs.gov/GIS/dsdl/aquifers_us.zip',
                'name': 'US Principal Aquifers',
                'description': 'USGS principal aquifers of the United States'
            },
            
            'huc2_basins': {
                'url': 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip',
                'name': 'HUC2 Watershed Boundaries',
                'description': 'USGS Watershed Boundary Dataset - Region level',
                'note': 'Large file (~2GB), contains all HUC levels'
            },
            
            'states': {
                'url': 'https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip',
                'name': 'US State Boundaries',
                'description': 'US Census Bureau state boundaries (2023)'
            },
            
            'us_rivers': {
                'url': 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_rivers_north_america.zip',
                'name': 'North American Rivers',
                'description': 'Natural Earth major rivers'
            }
        }
    
    def download_file(self, url, filename, description=""):
        """Download a file with progress bar."""
        filepath = self.output_dir / filename
        
        if filepath.exists():
            print(f"  ✅ Already exists: {filename}")
            return filepath
        
        print(f"  📥 Downloading: {description}")
        print(f"     URL: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"  ✅ Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            print(f"  ❌ Error downloading {filename}: {e}")
            return None
    
    def extract_zip(self, zip_path, extract_to=None):
        """Extract a zip file."""
        if extract_to is None:
            extract_to = zip_path.parent / zip_path.stem
        
        extract_to.mkdir(exist_ok=True)
        
        print(f"  📦 Extracting: {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        return extract_to
    
    def download_all_shapefiles(self):
        """Download all shapefiles."""
        print("🌍 DOWNLOADING SHAPEFILES FOR MISSISSIPPI RIVER BASIN ANALYSIS")
        print("="*60)
        
        downloaded = {}
        
        for key, info in self.sources.items():
            print(f"\n📍 {info['name']}")
            print(f"   {info['description']}")
            
            if 'note' in info:
                print(f"   ⚠️  Note: {info['note']}")
                response = input("   Download this file? (y/n): ")
                if response.lower() != 'y':
                    print("   ⏭️  Skipping...")
                    continue
            
            filename = f"{key}.zip"
            filepath = self.download_file(info['url'], filename, info['description'])
            
            if filepath:
                extract_dir = self.extract_zip(filepath)
                downloaded[key] = extract_dir
        
        return downloaded
    
    def create_mississippi_subregions(self):
        """Create subregion shapefiles for Mississippi River Basin."""
        print("\n🔧 CREATING MISSISSIPPI RIVER BASIN SUBREGIONS")
        print("="*60)
        
        # First, try to load Mississippi basin shapefile
        basin_dir = self.output_dir / 'mississippi_basin'
        basin_shapefile = None
        
        # Look for the main basin shapefile
        for file in basin_dir.glob('*.shp'):
            if 'mrb' in file.name.lower():
                basin_shapefile = file
                break
        
        if not basin_shapefile:
            print("  ❌ Mississippi basin shapefile not found")
            return
        
        print(f"  📂 Loading: {basin_shapefile}")
        basin_gdf = gpd.read_file(basin_shapefile)
        
        # Ensure CRS is WGS84
        if basin_gdf.crs != 'EPSG:4326':
            basin_gdf = basin_gdf.to_crs('EPSG:4326')
        
        # Define subregions by state groups
        subregions = {
            'upper_mississippi': {
                'states': ['Minnesota', 'Wisconsin', 'Iowa', 'Illinois', 'Missouri'],
                'description': 'Upper Mississippi River Basin'
            },
            'ohio_tennessee': {
                'states': ['Ohio', 'Indiana', 'Kentucky', 'Tennessee', 'West Virginia', 
                          'Pennsylvania', 'North Carolina', 'Virginia'],
                'description': 'Ohio-Tennessee River Basins'
            },
            'lower_mississippi': {
                'states': ['Arkansas', 'Louisiana', 'Mississippi', 'Alabama'],
                'description': 'Lower Mississippi River Basin'
            },
            'missouri': {
                'states': ['Montana', 'North Dakota', 'South Dakota', 'Nebraska', 
                          'Kansas', 'Wyoming', 'Colorado'],
                'description': 'Missouri River Basin'
            },
            'arkansas_red': {
                'states': ['Oklahoma', 'Texas', 'New Mexico', 'Arkansas'],
                'description': 'Arkansas-Red River Basins'
            }
        }
        
        # Load state boundaries
        states_dir = self.output_dir / 'states'
        state_shapefile = None
        
        for file in states_dir.glob('*.shp'):
            if 'state' in file.name.lower():
                state_shapefile = file
                break
        
        if state_shapefile:
            print(f"  📂 Loading states: {state_shapefile}")
            states_gdf = gpd.read_file(state_shapefile)
            
            if states_gdf.crs != 'EPSG:4326':
                states_gdf = states_gdf.to_crs('EPSG:4326')
            
            # Create subregion shapefiles
            subregions_dir = self.output_dir / 'mississippi_subregions'
            subregions_dir.mkdir(exist_ok=True)
            
            for region_name, region_info in subregions.items():
                print(f"\n  🔨 Creating: {region_info['description']}")
                
                # Select states
                region_states = states_gdf[states_gdf['NAME'].isin(region_info['states'])]
                
                if len(region_states) > 0:
                    # Merge states into single geometry
                    region_boundary = region_states.unary_union
                    
                    # Clip to Mississippi basin
                    basin_boundary = basin_gdf.unary_union
                    clipped_region = region_boundary.intersection(basin_boundary)
                    
                    # Create GeoDataFrame
                    region_gdf = gpd.GeoDataFrame(
                        {'name': [region_name], 'description': [region_info['description']]},
                        geometry=[clipped_region],
                        crs='EPSG:4326'
                    )
                    
                    # Save
                    output_path = subregions_dir / f'{region_name}.shp'
                    region_gdf.to_file(output_path)
                    print(f"    ✅ Saved: {output_path}")
        else:
            print("  ❌ State boundaries not found")
    
    def create_aquifer_regions(self):
        """Extract major aquifers in Mississippi River Basin."""
        print("\n🔧 EXTRACTING MAJOR AQUIFERS IN MISSISSIPPI BASIN")
        print("="*60)
        
        # Load aquifers
        aquifer_dir = self.output_dir / 'major_aquifers'
        aquifer_shapefile = None
        
        for file in aquifer_dir.glob('*.shp'):
            if 'aquifer' in file.name.lower():
                aquifer_shapefile = file
                break
        
        if not aquifer_shapefile:
            print("  ❌ Aquifer shapefile not found")
            return
        
        print(f"  📂 Loading: {aquifer_shapefile}")
        aquifers_gdf = gpd.read_file(aquifer_shapefile)
        
        # Load Mississippi basin for clipping
        basin_dir = self.output_dir / 'mississippi_basin'
        basin_shapefile = None
        
        for file in basin_dir.glob('*.shp'):
            if 'mrb' in file.name.lower():
                basin_shapefile = file
                break
        
        if basin_shapefile:
            basin_gdf = gpd.read_file(basin_shapefile)
            basin_boundary = basin_gdf.unary_union
            
            # Major aquifers to extract
            target_aquifers = [
                'Mississippi River Valley alluvial aquifer',
                'High Plains aquifer',
                'Mississippi embayment-Texas coastal uplands aquifer system',
                'Cambrian-Ordovician aquifer system',
                'Silurian-Devonian aquifers'
            ]
            
            aquifers_dir = self.output_dir / 'mississippi_aquifers'
            aquifers_dir.mkdir(exist_ok=True)
            
            # Extract each aquifer
            for aq_name in aquifers_gdf['AQ_NAME'].unique():
                if any(target in aq_name for target in target_aquifers):
                    print(f"\n  🔨 Extracting: {aq_name}")
                    
                    # Get aquifer geometry
                    aquifer = aquifers_gdf[aquifers_gdf['AQ_NAME'] == aq_name]
                    
                    # Clip to basin
                    aquifer_clipped = aquifer.clip(basin_boundary)
                    
                    if not aquifer_clipped.empty:
                        # Save
                        safe_name = aq_name.replace(' ', '_').replace('/', '_')
                        output_path = aquifers_dir / f'{safe_name}.shp'
                        aquifer_clipped.to_file(output_path)
                        print(f"    ✅ Saved: {output_path}")
    
    def print_direct_download_links(self):
        """Print direct links for manual download."""
        print("\n📎 DIRECT DOWNLOAD LINKS")
        print("="*60)
        print("\n1. MISSISSIPPI RIVER BASIN:")
        print("   • USGS: https://water.usgs.gov/GIS/dsdl/mrb_e2rf1_bas_mrb.zip")
        print("   • EPA WATERS: https://www.epa.gov/waterdata/waters-geospatial-data-downloads")
        
        print("\n2. MAJOR AQUIFERS:")
        print("   • USGS Principal Aquifers: https://water.usgs.gov/GIS/dsdl/aquifers_us.zip")
        print("   • High Plains Aquifer: https://water.usgs.gov/GIS/dsdl/hp_aqbase.zip")
        
        print("\n3. WATERSHED BOUNDARIES (HUC):")
        print("   • WBD National: https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip")
        print("   • HUC2 only: https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/")
        
        print("\n4. ADDITIONAL RESOURCES:")
        print("   • HydroSHEDS: https://www.hydrosheds.org/products/hydrobasins")
        print("   • Natural Earth: https://www.naturalearthdata.com/downloads/")
        print("   • USGS National Map: https://apps.nationalmap.gov/downloader/")
        
        print("\n5. GROUNDWATER SPECIFIC:")
        print("   • USGS Groundwater Atlas: https://water.usgs.gov/ogw/aquifer/atlas-query.html")
        print("   • State Geological Surveys: Various state-specific sites")
        
        print("\n6. PRE-MADE BASIN BOUNDARIES:")
        print("   • HydroBASINS Level 3 (North America): ")
        print("     https://www.hydrosheds.org/products/hydrobasins")
        
        print("\n💡 TIP: For the Mississippi River Basin, you can also use:")
        print("   • HUC 07 (Upper Mississippi)")
        print("   • HUC 08 (Lower Mississippi)")
        print("   • HUC 10 (Missouri)")
        print("   • HUC 11 (Arkansas-White-Red)")
        print("   • HUC 05 (Ohio)")
        print("   • HUC 06 (Tennessee)")


def main():
    """Main function to download and prepare shapefiles."""
    print("🗺️ SHAPEFILE DOWNLOAD TOOL FOR GRACE ANALYSIS")
    print("="*70)
    
    downloader = ShapefileDownloader()
    
    # Print direct links first
    downloader.print_direct_download_links()
    
    # Ask user if they want to download
    print("\n" + "="*70)
    response = input("\n📥 Do you want to automatically download shapefiles? (y/n): ")
    
    if response.lower() == 'y':
        # Download all shapefiles
        downloaded = downloader.download_all_shapefiles()
        
        # Create subregions if Mississippi basin was downloaded
        if 'mississippi_basin' in downloaded:
            downloader.create_mississippi_subregions()
        
        # Extract aquifers if both datasets were downloaded
        if 'mississippi_basin' in downloaded and 'major_aquifers' in downloaded:
            downloader.create_aquifer_regions()
        
        print("\n✅ DOWNLOAD COMPLETE!")
        print(f"📁 Shapefiles saved to: {downloader.output_dir}")
        
        # Print summary of what's available
        print("\n📋 AVAILABLE SHAPEFILES:")
        for subdir in downloader.output_dir.iterdir():
            if subdir.is_dir():
                shp_files = list(subdir.glob('*.shp'))
                if shp_files:
                    print(f"\n  {subdir.name}:")
                    for shp in shp_files:
                        print(f"    • {shp.name}")
    
    print("\n🔧 TO USE IN VISUALIZATION:")
    print("Update the visualization script with these paths:")
    print("\n```python")
    print("# Main basin")
    print(f"shapefile_path = 'data/shapefiles/mississippi_basin/[basin_shapefile].shp'")
    print("\n# Subregions")
    print("regional_shapefiles = {")
    print("    'Upper_Mississippi': 'data/shapefiles/mississippi_subregions/upper_mississippi.shp',")
    print("    'Missouri': 'data/shapefiles/mississippi_subregions/missouri.shp',")
    print("    'Ohio_Tennessee': 'data/shapefiles/mississippi_subregions/ohio_tennessee.shp',")
    print("    'Lower_Mississippi': 'data/shapefiles/mississippi_subregions/lower_mississippi.shp',")
    print("    'Arkansas_Red': 'data/shapefiles/mississippi_subregions/arkansas_red.shp'")
    print("}")
    print("```")


if __name__ == "__main__":
    main()