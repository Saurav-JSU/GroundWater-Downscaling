#!/usr/bin/env python3
"""
Create Mississippi River Basin from HUC Data
============================================

This script:
1. Creates Mississippi River Basin boundary from HUC regions
2. Processes multi-geometry shapefiles (states, aquifers)
3. Creates individual shapefiles for each state/aquifer
4. Generates subregions automatically
"""

import os
import geopandas as gpd
import fiona
import pandas as pd
from pathlib import Path
import numpy as np
from shapely.ops import unary_union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BasinCreator:
    """Create Mississippi River Basin and process shapefiles."""
    
    def __init__(self, shapefile_dir="data/shapefiles"):
        self.shapefile_dir = Path(shapefile_dir)
        self.output_dir = self.shapefile_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        # Mississippi River Basin HUC codes
        self.mississippi_huc2_codes = [
            '05',  # Ohio Region
            '06',  # Tennessee Region
            '07',  # Upper Mississippi Region
            '08',  # Lower Mississippi Region
            '10',  # Missouri Region
            '11',  # Arkansas-White-Red Region
        ]
        
        # Define subregions by HUC codes
        self.subregions_by_huc = {
            'upper_mississippi': ['07'],
            'ohio_tennessee': ['05', '06'],
            'lower_mississippi': ['08'],
            'missouri': ['10'],
            'arkansas_red': ['11']
        }
        
        # Define subregions by states (for alternative approach)
        self.subregions_by_states = {
            'upper_mississippi': [
                'Minnesota', 'Wisconsin', 'Iowa', 'Illinois', 'Missouri'
            ],
            'ohio_tennessee': [
                'Ohio', 'Indiana', 'Kentucky', 'Tennessee', 'West Virginia',
                'Pennsylvania', 'North Carolina', 'Virginia', 'Alabama', 'Georgia'
            ],
            'lower_mississippi': [
                'Arkansas', 'Louisiana', 'Mississippi', 'Tennessee', 'Missouri'
            ],
            'missouri': [
                'Montana', 'North Dakota', 'South Dakota', 'Nebraska',
                'Kansas', 'Wyoming', 'Colorado', 'Iowa', 'Missouri'
            ],
            'arkansas_red': [
                'Oklahoma', 'Texas', 'New Mexico', 'Arkansas', 'Kansas', 'Louisiana'
            ]
        }
    
    def find_huc_data(self):
        """Find HUC data in the downloaded files."""
        print("🔍 Looking for HUC data...")
        
        huc_dir = self.shapefile_dir / "huc2_basins"
        
        # Look for geodatabase
        gdb_files = list(huc_dir.glob("*.gdb"))
        if gdb_files:
            print(f"  ✅ Found geodatabase: {gdb_files[0]}")
            return gdb_files[0], 'gdb'
        
        # Look for shapefiles
        shp_files = list(huc_dir.glob("**/*.shp"))
        if shp_files:
            # Look for HUC2 or WBDHU2
            for shp in shp_files:
                if 'HU2' in shp.name or 'HUC2' in shp.name or 'WBDHU2' in shp.name:
                    print(f"  ✅ Found HUC2 shapefile: {shp}")
                    return shp, 'shp'
            
            # If no HUC2 specific, return first shapefile
            print(f"  ✅ Found shapefile: {shp_files[0]}")
            return shp_files[0], 'shp'
        
        print("  ❌ No HUC data found")
        return None, None
    
    def create_mississippi_basin_from_huc(self):
        """Create Mississippi River Basin from HUC boundaries."""
        print("\n🏞️ CREATING MISSISSIPPI RIVER BASIN FROM HUC DATA")
        print("="*60)
        
        huc_path, data_type = self.find_huc_data()
        
        if not huc_path:
            print("  ❌ No HUC data found to create basin")
            return None
        
        try:
            # Load HUC data
            if data_type == 'gdb':
                # List layers in geodatabase
                import fiona
                layers = fiona.listlayers(str(huc_path))
                print(f"  📋 Available layers: {layers}")
                
                # Look for HUC2 layer
                huc2_layer = None
                for layer in layers:
                    if 'HUC2' in layer or 'HU2' in layer or 'WBDHU2' in layer:
                        huc2_layer = layer
                        break
                
                if huc2_layer:
                    print(f"  📂 Loading layer: {huc2_layer}")
                    huc_gdf = gpd.read_file(str(huc_path), layer=huc2_layer)
                else:
                    # Use first layer
                    print(f"  📂 Loading layer: {layers[0]}")
                    huc_gdf = gpd.read_file(str(huc_path), layer=layers[0])
            else:
                # Load shapefile
                print(f"  📂 Loading shapefile...")
                huc_gdf = gpd.read_file(str(huc_path))
            
            print(f"  ✅ Loaded {len(huc_gdf)} HUC regions")
            print(f"  📋 Columns: {list(huc_gdf.columns)}")
            
            # Find HUC code column
            huc_col = None
            for col in huc_gdf.columns:
                if 'HUC' in col.upper() and ('2' in col or len(str(huc_gdf[col].iloc[0])) == 2):
                    huc_col = col
                    break
            
            if not huc_col:
                # Try to find by inspecting values
                for col in huc_gdf.columns:
                    if huc_gdf[col].dtype == 'object':
                        sample = str(huc_gdf[col].iloc[0])
                        if len(sample) == 2 and sample.isdigit():
                            huc_col = col
                            break
            
            if not huc_col:
                print("  ❌ Could not find HUC code column")
                print(f"  📋 Available columns: {list(huc_gdf.columns)}")
                return None
            
            print(f"  ✅ Using HUC column: {huc_col}")
            
            # Filter for Mississippi River Basin HUCs
            print(f"  🔍 Filtering for Mississippi HUCs: {self.mississippi_huc2_codes}")
            mississippi_hucs = huc_gdf[huc_gdf[huc_col].isin(self.mississippi_huc2_codes)]
            
            if len(mississippi_hucs) == 0:
                print("  ❌ No matching HUCs found")
                print(f"  📋 Available HUC codes: {sorted(huc_gdf[huc_col].unique())}")
                return None
            
            print(f"  ✅ Found {len(mississippi_hucs)} Mississippi River Basin HUCs")
            
            # Merge into single basin
            print("  🔧 Merging HUCs into single basin...")
            basin_geometry = unary_union(mississippi_hucs.geometry)
            
            # Create GeoDataFrame
            mississippi_basin = gpd.GeoDataFrame(
                {'name': ['Mississippi River Basin'],
                 'description': ['Merged from HUC regions 05, 06, 07, 08, 10, 11'],
                 'huc_codes': [','.join(self.mississippi_huc2_codes)]},
                geometry=[basin_geometry],
                crs=mississippi_hucs.crs
            )
            
            # Save basin
            output_path = self.output_dir / "mississippi_river_basin.shp"
            mississippi_basin.to_file(output_path)
            print(f"  💾 Saved: {output_path}")
            
            # Also create individual HUC regions
            huc_dir = self.output_dir / "huc_regions"
            huc_dir.mkdir(exist_ok=True)
            
            for _, huc in mississippi_hucs.iterrows():
                huc_code = str(huc[huc_col])
                huc_name = huc.get('NAME', f'HUC_{huc_code}')
                
                huc_gdf = gpd.GeoDataFrame(
                    {'huc_code': [huc_code], 'name': [huc_name]},
                    geometry=[huc.geometry],
                    crs=mississippi_hucs.crs
                )
                
                output_path = huc_dir / f"huc_{huc_code}.shp"
                huc_gdf.to_file(output_path)
                print(f"    💾 Saved HUC {huc_code}: {huc_name}")
            
            # Create subregions based on HUC groupings
            self.create_subregions_from_hucs(mississippi_hucs, huc_col)
            
            return mississippi_basin
            
        except Exception as e:
            print(f"  ❌ Error creating basin: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_subregions_from_hucs(self, huc_gdf, huc_col):
        """Create subregions by grouping HUCs."""
        print("\n  🔧 Creating subregions from HUCs...")
        
        subregions_dir = self.output_dir / "subregions_huc"
        subregions_dir.mkdir(exist_ok=True)
        
        for region_name, huc_codes in self.subregions_by_huc.items():
            region_hucs = huc_gdf[huc_gdf[huc_col].isin(huc_codes)]
            
            if len(region_hucs) > 0:
                # Merge geometries
                region_geometry = unary_union(region_hucs.geometry)
                
                region_gdf = gpd.GeoDataFrame(
                    {'name': [region_name],
                     'description': [f'HUC regions: {", ".join(huc_codes)}'],
                     'huc_codes': [','.join(huc_codes)]},
                    geometry=[region_geometry],
                    crs=huc_gdf.crs
                )
                
                output_path = subregions_dir / f"{region_name}.shp"
                region_gdf.to_file(output_path)
                print(f"    💾 Created subregion: {region_name}")
    
    def process_states_shapefile(self):
        """Process states shapefile - create individual state files."""
        print("\n🏛️ PROCESSING STATES SHAPEFILE")
        print("="*60)
        
        states_dir = self.shapefile_dir / "states"
        states_files = list(states_dir.glob("*.shp"))
        
        if not states_files:
            print("  ❌ No states shapefile found")
            return
        
        print(f"  📂 Loading: {states_files[0]}")
        states_gdf = gpd.read_file(states_files[0])
        
        # Create individual state shapefiles
        individual_states_dir = self.output_dir / "individual_states"
        individual_states_dir.mkdir(exist_ok=True)
        
        print(f"  ✅ Found {len(states_gdf)} states")
        
        # Get state name column
        name_col = None
        for col in ['NAME', 'STATE_NAME', 'STATENAME', 'Name']:
            if col in states_gdf.columns:
                name_col = col
                break
        
        if not name_col:
            print("  ❌ Could not find state name column")
            return
        
        # Save each state
        for _, state in tqdm(states_gdf.iterrows(), total=len(states_gdf), desc="Creating state files"):
            state_name = state[name_col]
            safe_name = state_name.replace(' ', '_')
            
            state_gdf = gpd.GeoDataFrame(
                [state],
                geometry='geometry',
                crs=states_gdf.crs
            )
            
            output_path = individual_states_dir / f"{safe_name}.shp"
            state_gdf.to_file(output_path)
        
        print(f"  ✅ Created {len(states_gdf)} individual state shapefiles")
        
        # Create subregions by state groups
        self.create_subregions_from_states(states_gdf, name_col)
    
    def create_subregions_from_states(self, states_gdf, name_col):
        """Create subregions by grouping states."""
        print("\n  🔧 Creating subregions from states...")
        
        subregions_dir = self.output_dir / "subregions_states"
        subregions_dir.mkdir(exist_ok=True)
        
        # Load Mississippi basin to clip states
        basin_path = self.output_dir / "mississippi_river_basin.shp"
        if basin_path.exists():
            basin_gdf = gpd.read_file(basin_path)
            basin_boundary = basin_gdf.unary_union
        else:
            basin_boundary = None
        
        for region_name, state_names in self.subregions_by_states.items():
            region_states = states_gdf[states_gdf[name_col].isin(state_names)]
            
            if len(region_states) > 0:
                # Merge geometries
                region_geometry = unary_union(region_states.geometry)
                
                # Clip to basin if available
                if basin_boundary is not None:
                    region_geometry = region_geometry.intersection(basin_boundary)
                
                region_gdf = gpd.GeoDataFrame(
                    {'name': [region_name],
                     'description': [f'States: {", ".join(state_names[:3])}...'],
                     'states': [','.join(state_names)]},
                    geometry=[region_geometry],
                    crs=states_gdf.crs
                )
                
                output_path = subregions_dir / f"{region_name}.shp"
                region_gdf.to_file(output_path)
                print(f"    💾 Created subregion: {region_name}")
    
    def process_aquifers_shapefile(self):
        """Process aquifers shapefile - create individual aquifer files."""
        print("\n💧 PROCESSING AQUIFERS SHAPEFILE")
        print("="*60)
        
        aquifers_dir = self.shapefile_dir / "major_aquifers"
        aquifer_files = list(aquifers_dir.glob("*.shp"))
        
        if not aquifer_files:
            print("  ❌ No aquifers shapefile found")
            return
        
        print(f"  📂 Loading: {aquifer_files[0]}")
        aquifers_gdf = gpd.read_file(aquifer_files[0])
        
        print(f"  ✅ Found {len(aquifers_gdf)} aquifer polygons")
        print(f"  📋 Columns: {list(aquifers_gdf.columns)}")
        
        # Find aquifer name column
        name_col = None
        for col in ['AQ_NAME', 'NAME', 'AQUIFER', 'Aquifer_Name']:
            if col in aquifers_gdf.columns:
                name_col = col
                break
        
        if not name_col:
            print("  ❌ Could not find aquifer name column")
            return
        
        # Get unique aquifer names
        unique_aquifers = aquifers_gdf[name_col].unique()
        print(f"  ✅ Found {len(unique_aquifers)} unique aquifers")
        
        # Create individual aquifer shapefiles
        individual_aquifers_dir = self.output_dir / "individual_aquifers"
        individual_aquifers_dir.mkdir(exist_ok=True)
        
        # Load Mississippi basin for clipping
        basin_path = self.output_dir / "mississippi_river_basin.shp"
        if basin_path.exists():
            basin_gdf = gpd.read_file(basin_path)
            basin_boundary = basin_gdf.unary_union
            print("  ✅ Will clip aquifers to Mississippi Basin")
        else:
            basin_boundary = None
        
        # Major aquifers to focus on
        major_aquifers = [
            'Mississippi River Valley alluvial aquifer',
            'High Plains aquifer',
            'Mississippi embayment',
            'Coastal lowlands aquifer system',
            'Surficial aquifer system'
        ]
        
        for aquifer_name in tqdm(unique_aquifers, desc="Processing aquifers"):
            # Get all polygons for this aquifer
            aquifer_polys = aquifers_gdf[aquifers_gdf[name_col] == aquifer_name]
            
            # Merge if multiple polygons
            if len(aquifer_polys) > 1:
                aquifer_geometry = unary_union(aquifer_polys.geometry)
                aquifer_gdf = gpd.GeoDataFrame(
                    {'name': [aquifer_name], 'area_km2': [aquifer_geometry.area / 1e6]},
                    geometry=[aquifer_geometry],
                    crs=aquifers_gdf.crs
                )
            else:
                aquifer_gdf = aquifer_polys.copy()
            
            # Save full aquifer
            safe_name = aquifer_name.replace(' ', '_').replace('/', '_').replace(',', '')
            output_path = individual_aquifers_dir / f"{safe_name}.shp"
            aquifer_gdf.to_file(output_path)
            
            # If major aquifer and basin available, create clipped version
            if basin_boundary is not None and any(maj in aquifer_name for maj in major_aquifers):
                try:
                    clipped = aquifer_gdf.clip(basin_boundary)
                    if not clipped.empty:
                        clipped_dir = self.output_dir / "aquifers_mississippi"
                        clipped_dir.mkdir(exist_ok=True)
                        
                        output_path = clipped_dir / f"{safe_name}_mississippi.shp"
                        clipped.to_file(output_path)
                        print(f"    💾 Created clipped aquifer: {safe_name}")
                except:
                    pass
        
        print(f"  ✅ Created individual aquifer shapefiles")
    
    def create_summary_shapefile(self):
        """Create a summary shapefile with all regions."""
        print("\n📊 CREATING SUMMARY SHAPEFILE")
        print("="*60)
        
        # Collect all created shapefiles
        all_regions = []
        
        # Add main basin
        basin_path = self.output_dir / "mississippi_river_basin.shp"
        if basin_path.exists():
            basin = gpd.read_file(basin_path)
            basin['type'] = 'basin'
            basin['scale'] = 'full'
            all_regions.append(basin)
        
        # Add HUC subregions
        subregions_huc = self.output_dir / "subregions_huc"
        if subregions_huc.exists():
            for shp in subregions_huc.glob("*.shp"):
                region = gpd.read_file(shp)
                region['type'] = 'subregion_huc'
                region['scale'] = 'regional'
                all_regions.append(region)
        
        # Add state subregions
        subregions_states = self.output_dir / "subregions_states"
        if subregions_states.exists():
            for shp in subregions_states.glob("*.shp"):
                region = gpd.read_file(shp)
                region['type'] = 'subregion_state'
                region['scale'] = 'regional'
                all_regions.append(region)
        
        if all_regions:
            # Combine all
            summary_gdf = gpd.GeoDataFrame(pd.concat(all_regions, ignore_index=True))
            
            # Save
            output_path = self.output_dir / "all_regions_summary.shp"
            summary_gdf.to_file(output_path)
            print(f"  💾 Created summary shapefile with {len(summary_gdf)} regions")
    
    def print_usage_instructions(self):
        """Print instructions for using the created shapefiles."""
        print("\n📋 USAGE INSTRUCTIONS")
        print("="*60)
        print("\n1. For the main visualization script, update paths:")
        print("\n```python")
        print("# Main basin")
        print(f"shapefile_path = '{self.output_dir}/mississippi_river_basin.shp'")
        print("\n# Option 1: HUC-based subregions")
        print("regional_shapefiles = {")
        for region in ['upper_mississippi', 'missouri', 'ohio_tennessee', 'lower_mississippi', 'arkansas_red']:
            print(f"    '{region}': '{self.output_dir}/subregions_huc/{region}.shp',")
        print("}")
        print("\n# Option 2: State-based subregions")
        print("regional_shapefiles = {")
        for region in ['upper_mississippi', 'missouri', 'ohio_tennessee', 'lower_mississippi', 'arkansas_red']:
            print(f"    '{region}': '{self.output_dir}/subregions_states/{region}.shp',")
        print("}")
        print("\n# Option 3: Individual states")
        print("regional_shapefiles = {")
        print("    'Minnesota': 'data/shapefiles/processed/individual_states/Minnesota.shp',")
        print("    'Iowa': 'data/shapefiles/processed/individual_states/Iowa.shp',")
        print("    'Missouri': 'data/shapefiles/processed/individual_states/Missouri.shp',")
        print("    # Add more states as needed")
        print("}")
        print("\n# Option 4: Process all states automatically")
        print("# The visualization script can be modified to loop through all state files")
        print("```")


def main():
    """Main function to create basin and process shapefiles."""
    print("🏞️ MISSISSIPPI RIVER BASIN CREATOR")
    print("="*70)
    
    creator = BasinCreator()
    
    # 1. Create Mississippi River Basin from HUC data
    basin = creator.create_mississippi_basin_from_huc()
    
    # 2. Process states shapefile
    creator.process_states_shapefile()
    
    # 3. Process aquifers shapefile
    creator.process_aquifers_shapefile()
    
    # 4. Create summary shapefile
    creator.create_summary_shapefile()
    
    # 5. Print usage instructions
    creator.print_usage_instructions()
    
    print("\n✅ PROCESSING COMPLETE!")
    print(f"📁 All processed shapefiles saved to: {creator.output_dir}")


if __name__ == "__main__":
    main()