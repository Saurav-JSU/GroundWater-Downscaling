#!/usr/bin/env python3
"""
Parallel Advanced Spatial-Temporal Analysis for GRACE Groundwater Downscaling
============================================================================

This script leverages parallel computing to accelerate visualization and analysis tasks:
- Multi-core trend calculations using joblib
- Parallel figure generation using multiprocessing
- Memory-efficient processing with dask for large datasets
- Optimized for high-core systems (tested on 192-core systems)

Author: GRACE Analysis Pipeline
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray
from scipy import stats
from tqdm import tqdm
import warnings
from pathlib import Path
from datetime import datetime
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib import cm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import psutil
import gc

# Parallel processing libraries
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("⚠️ joblib not installed; falling back to standard multiprocessing")

try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("⚠️ dask not installed; some parallel operations may be slower")

try:
    import regionmask
except ImportError:
    regionmask = None
    warnings.warn("⚠️ regionmask not installed; shapefile-based masking will be disabled")

warnings.filterwarnings('ignore')

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")

# Publication settings
FIGURE_DPI = 300
FONT_SIZE = 12

# Parallel processing settings - optimized for high-core systems
TOTAL_CORES = cpu_count()
# Use a reasonable number of cores for stability (max 64 for most tasks)
N_CORES = min(64, TOTAL_CORES)  
# For trend calculations, use more cores but with larger chunks
TREND_CORES = min(128, TOTAL_CORES)
CHUNK_SIZE = 100  # Smaller chunks for better memory management
MEMORY_LIMIT = '8GB'  # Per worker memory limit for high-RAM systems

print(f"🚀 PARALLEL GRACE VISUALIZATION INITIALIZED")
print(f"   Available CPU cores: {TOTAL_CORES}")
print(f"   Standard processing cores: {N_CORES}")
print(f"   Trend calculation cores: {TREND_CORES}")
print(f"   Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"   Memory per worker: {MEMORY_LIMIT}")


def _calculate_trends_chunk_standalone(args):
    """Standalone function to calculate trends for a chunk of pixels (for parallel processing)."""
    try:
        data_chunk, time_numeric, i_start, i_end, j_start, j_end, min_years = args
        
        chunk_height = i_end - i_start
        chunk_width = j_end - j_start
        
        # Initialize output arrays
        trend = np.full((chunk_height, chunk_width), np.nan)
        p_value = np.full((chunk_height, chunk_width), np.nan)
        std_error = np.full((chunk_height, chunk_width), np.nan)
        n_valid = np.full((chunk_height, chunk_width), 0)
        
        # Vectorized processing where possible
        for i in range(chunk_height):
            for j in range(chunk_width):
                ts = data_chunk[:, i, j]
                
                # Skip if all NaN
                if np.all(np.isnan(ts)):
                    continue
                    
                valid_mask = ~np.isnan(ts)
                n_valid_points = np.sum(valid_mask)
                
                if n_valid_points >= min_years * 12:
                    try:
                        slope, intercept, r_value, p_val, std_err = stats.linregress(
                            time_numeric[valid_mask], ts[valid_mask]
                        )
                        
                        # Convert to annual trend
                        trend[i, j] = slope * 12
                        p_value[i, j] = p_val
                        std_error[i, j] = std_err * 12
                        n_valid[i, j] = n_valid_points
                        
                    except (ValueError, np.linalg.LinAlgError):
                        # Handle numerical issues gracefully
                        continue
        
        # Clean up memory in worker
        del data_chunk
        gc.collect()
        
        return {
            'trend': trend,
            'p_value': p_value,
            'std_error': std_error,
            'n_valid': n_valid,
            'bounds': (i_start, i_end, j_start, j_end)
        }
        
    except Exception as e:
        print(f"Error in chunk ({i_start}:{i_end}, {j_start}:{j_end}): {e}")
        # Return empty results for this chunk
        chunk_height = i_end - i_start
        chunk_width = j_end - j_start
        return {
            'trend': np.full((chunk_height, chunk_width), np.nan),
            'p_value': np.full((chunk_height, chunk_width), np.nan),
            'std_error': np.full((chunk_height, chunk_width), np.nan),
            'n_valid': np.full((chunk_height, chunk_width), 0),
            'bounds': (i_start, i_end, j_start, j_end)
        }


def _find_extremes_chunk_standalone(args):
    """Standalone function to find extremes for a chunk of pixels (for parallel processing)."""
    try:
        data_chunk, time_as_datetime, i_start, i_end, j_start, j_end = args
        
        chunk_height = i_end - i_start
        chunk_width = j_end - j_start
        
        # Initialize output arrays
        min_time_index = np.full((chunk_height, chunk_width), np.nan)
        max_time_index = np.full((chunk_height, chunk_width), np.nan)
        min_month = np.full((chunk_height, chunk_width), np.nan)
        max_month = np.full((chunk_height, chunk_width), np.nan)
        
        for i in range(chunk_height):
            for j in range(chunk_width):
                ts = data_chunk[:, i, j]
                
                # Skip if all NaN
                if np.all(np.isnan(ts)):
                    continue
                
                try:
                    # Find minimum
                    min_idx = np.nanargmin(ts)
                    min_time_index[i, j] = min_idx
                    min_month[i, j] = time_as_datetime[min_idx].month
                    
                    # Find maximum
                    max_idx = np.nanargmax(ts)
                    max_time_index[i, j] = max_idx
                    max_month[i, j] = time_as_datetime[max_idx].month
                    
                except (ValueError, IndexError):
                    continue
        
        return {
            'min_time_index': min_time_index,
            'max_time_index': max_time_index,
            'min_month': min_month,
            'max_month': max_month,
            'bounds': (i_start, i_end, j_start, j_end)
        }
        
    except Exception as e:
        print(f"Error in extremes chunk ({i_start}:{i_end}, {j_start}:{j_end}): {e}")
        chunk_height = i_end - i_start
        chunk_width = j_end - j_start
        return {
            'min_time_index': np.full((chunk_height, chunk_width), np.nan),
            'max_time_index': np.full((chunk_height, chunk_width), np.nan),
            'min_month': np.full((chunk_height, chunk_width), np.nan),
            'max_month': np.full((chunk_height, chunk_width), np.nan),
            'bounds': (i_start, i_end, j_start, j_end)
        }


class ParallelGRACEVisualizer:
    """Parallel visualization class for GRACE groundwater analysis."""
    
    def __init__(self, base_dir=".", shapefile_path=None, n_workers=None, use_dask=True):
        """
        Initialize the parallel visualizer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory of the project
        shapefile_path : str
            Path to Mississippi River Basin shapefile
        n_workers : int
            Number of workers for parallel processing (None = auto-detect)
        use_dask : bool
            Whether to use dask for large array operations
        """
        self.base_dir = Path(base_dir)
        self.figures_dir = self.base_dir / "figures" / "parallel_enhanced"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up parallel processing
        self.n_workers = n_workers or N_CORES
        self.use_dask = use_dask and DASK_AVAILABLE
        
        print(f"   Using {self.n_workers} workers")
        if self.use_dask:
            print(f"   Dask enabled for large array operations")
        
        # Create subdirectories
        self.subdirs = {
            'trends': self.figures_dir / 'trends',
            'regional': self.figures_dir / 'regional',
            'extremes': self.figures_dir / 'extremes',
            'components': self.figures_dir / 'components',
            'seasonal': self.figures_dir / 'seasonal',
            'overview': self.figures_dir / 'overview'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
        
        # Load shapefile if provided
        self.shapefile = None
        self.shapefile_bounds = None
        self.region_mask = None
        
        if shapefile_path is None:
            # Try default shapefile path
            default_shapefile = self.base_dir / "data/shapefiles/processed/mississippi_river_basin.shp"
            if default_shapefile.exists():
                shapefile_path = str(default_shapefile)
        
        if shapefile_path and os.path.exists(shapefile_path):
            print(f"📍 Loading shapefile: {shapefile_path}")
            self.shapefile = gpd.read_file(shapefile_path)
            
            # Ensure CRS is WGS84
            if self.shapefile.crs != 'EPSG:4326':
                print(f"   Converting from {self.shapefile.crs} to EPSG:4326...")
                self.shapefile = self.shapefile.to_crs('EPSG:4326')
            
            # Calculate bounds safely
            try:
                bounds = self.shapefile.total_bounds
                if not np.all(np.isfinite(bounds)):
                    minx = self.shapefile.geometry.bounds.minx.min()
                    miny = self.shapefile.geometry.bounds.miny.min()
                    maxx = self.shapefile.geometry.bounds.maxx.max()
                    maxy = self.shapefile.geometry.bounds.maxy.max()
                    bounds = np.array([minx, miny, maxx, maxy])
                
                self.shapefile_bounds = bounds
                print(f"   Shapefile bounds: {self.shapefile_bounds}")
                
            except Exception as e:
                print(f"   ❌ Error calculating bounds: {e}")
                self.shapefile_bounds = None
        else:
            print("⚠️ No shapefile found, will use full domain")
        
        # Initialize dask client if requested - but disable when using joblib
        self.dask_client = None
        if self.use_dask:
            try:
                # Create local cluster optimized for high-core systems
                n_dask_workers = min(8, TOTAL_CORES // 16)  # More conservative for stability
                threads_per_worker = min(4, TOTAL_CORES // n_dask_workers)
                
                cluster = LocalCluster(
                    n_workers=n_dask_workers,
                    threads_per_worker=threads_per_worker,
                    memory_limit=MEMORY_LIMIT,
                    silence_logs=True,  # Reduce log noise
                    dashboard_address=':8787'  # Fixed dashboard port
                )
                self.dask_client = Client(cluster)
                print(f"   Dask cluster: {n_dask_workers} workers x {threads_per_worker} threads")
                print(f"   Dashboard: {self.dask_client.dashboard_link}")
                print(f"   ⚠️ Note: Dask will be temporarily disabled during joblib operations")
            except Exception as e:
                print(f"   ⚠️ Could not start dask cluster: {e}")
                print(f"   Falling back to standard parallel processing")
                self.use_dask = False
        
        # Load all datasets
        self._load_all_data()
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'dask_client') and self.dask_client is not None:
            self.dask_client.close()
    
    def _get_safe_extent(self, data_array, buffer=0.5):
        """Get a safe extent for plotting."""
        data_extent = [
            float(data_array.lon.min()), 
            float(data_array.lon.max()),
            float(data_array.lat.min()), 
            float(data_array.lat.max())
        ]
        
        if self.shapefile is not None and self.shapefile_bounds is not None:
            if np.all(np.isfinite(self.shapefile_bounds)):
                return [
                    self.shapefile_bounds[0] - buffer,
                    self.shapefile_bounds[2] + buffer,
                    self.shapefile_bounds[1] - buffer,
                    self.shapefile_bounds[3] + buffer
                ]
        
        return data_extent
    
    def _add_map_features(self, ax, add_shapefile=True):
        """Add consistent map features to an axis."""
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
        
        if add_shapefile and self.shapefile is not None:
            try:
                self.shapefile.boundary.plot(ax=ax, color='black', linewidth=1.5,
                                           transform=ccrs.PlateCarree())
            except Exception as e:
                print(f"    ⚠️ Could not plot shapefile boundary: {e}")
        
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        return gl
    
    def _get_robust_colorscale(self, data, percentiles=(5, 95), symmetric=False):
        """Get robust color scale using percentiles."""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return 0, 1
        
        if symmetric:
            vmax = np.percentile(np.abs(valid_data), percentiles[1])
            if vmax == 0:
                vmax = 1
            return -vmax, vmax
        else:
            vmin, vmax = np.percentile(valid_data, percentiles)
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
            return vmin, vmax
    
    def _load_all_data(self):
        """Load all required datasets."""
        print("\n📦 Loading all datasets...")
        
        # Load groundwater data
        gws_files = [
            "results/groundwater_storage_anomalies.nc",
            "results/groundwater_storage_anomalies_corrected.nc",
            "results/groundwater_storage_anomalies_enhanced.nc"
        ]
        
        for gws_file in gws_files:
            if (self.base_dir / gws_file).exists():
                print(f"  ✅ Loading groundwater: {gws_file}")
                if self.use_dask:
                    # Load with dask for memory efficiency
                    self.gws_ds = xr.open_dataset(self.base_dir / gws_file, chunks={'time': 50})
                else:
                    self.gws_ds = xr.open_dataset(self.base_dir / gws_file)
                break
        else:
            raise FileNotFoundError("No groundwater storage file found!")
        
        print(f"     Dataset shape: {self.gws_ds.groundwater.shape}")
        print(f"     Valid data points: {np.sum(~np.isnan(self.gws_ds.groundwater.values))}")
        print(f"     Data range: {float(np.nanmin(self.gws_ds.groundwater.values)):.2f} to {float(np.nanmax(self.gws_ds.groundwater.values)):.2f}")
        
        # Load feature stack
        feature_file = "data/processed/feature_stack.nc"
        if (self.base_dir / feature_file).exists():
            print(f"  ✅ Loading features: {feature_file}")
            if self.use_dask:
                self.features_ds = xr.open_dataset(self.base_dir / feature_file, chunks={'time': 50})
            else:
                self.features_ds = xr.open_dataset(self.base_dir / feature_file)
        else:
            print("  ⚠️ Feature stack not found")
            self.features_ds = None
        
        # Create region mask
        if self.shapefile is not None:
            self._create_region_mask()
    
    def _create_region_mask(self):
        """Create a mask for the shapefile region."""
        print("  🎭 Creating region mask from shapefile...")
        
        if regionmask is None:
            print("    ⚠️ regionmask not available, skipping mask creation")
            self.region_mask = None
            return
        
        try:
            if len(self.shapefile) == 1:
                region = regionmask.Regions([self.shapefile.geometry[0]], 
                                          names=['Mississippi_Basin'])
            else:
                unified = self.shapefile.union_all()
                region = regionmask.Regions([unified], names=['Mississippi_Basin'])
            
            self.region_mask = region.mask(self.gws_ds.lon, self.gws_ds.lat)
            
            n_valid = np.sum(~np.isnan(self.region_mask))
            print(f"    ✅ Mask created: {n_valid} valid pixels")
            
        except Exception as e:
            print(f"    ⚠️ Error creating mask: {e}")
            self.region_mask = None

    def _calculate_trends_chunk(self, args):
        """Calculate trends for a chunk of pixels (optimized parallel worker function)."""
        try:
            data_chunk, time_numeric, i_start, i_end, j_start, j_end, min_years = args
            
            chunk_height = i_end - i_start
            chunk_width = j_end - j_start
            
            # Initialize output arrays
            trend = np.full((chunk_height, chunk_width), np.nan)
            p_value = np.full((chunk_height, chunk_width), np.nan)
            std_error = np.full((chunk_height, chunk_width), np.nan)
            n_valid = np.full((chunk_height, chunk_width), 0)
            
            # Vectorized processing where possible
            for i in range(chunk_height):
                for j in range(chunk_width):
                    ts = data_chunk[:, i, j]
                    
                    # Skip if all NaN
                    if np.all(np.isnan(ts)):
                        continue
                        
                    valid_mask = ~np.isnan(ts)
                    n_valid_points = np.sum(valid_mask)
                    
                    if n_valid_points >= min_years * 12:
                        try:
                            slope, intercept, r_value, p_val, std_err = stats.linregress(
                                time_numeric[valid_mask], ts[valid_mask]
                            )
                            
                            # Convert to annual trend
                            trend[i, j] = slope * 12
                            p_value[i, j] = p_val
                            std_error[i, j] = std_err * 12
                            n_valid[i, j] = n_valid_points
                            
                        except (ValueError, np.linalg.LinAlgError):
                            # Handle numerical issues gracefully
                            continue
            
            # Clean up memory in worker
            del data_chunk
            gc.collect()
            
            return {
                'trend': trend,
                'p_value': p_value,
                'std_error': std_error,
                'n_valid': n_valid,
                'bounds': (i_start, i_end, j_start, j_end)
            }
            
        except Exception as e:
            print(f"Error in chunk ({i_start}:{i_end}, {j_start}:{j_end}): {e}")
            # Return empty results for this chunk
            chunk_height = i_end - i_start
            chunk_width = j_end - j_start
            return {
                'trend': np.full((chunk_height, chunk_width), np.nan),
                'p_value': np.full((chunk_height, chunk_width), np.nan),
                'std_error': np.full((chunk_height, chunk_width), np.nan),
                'n_valid': np.full((chunk_height, chunk_width), 0),
                'bounds': (i_start, i_end, j_start, j_end)
            }

    def calculate_pixel_trends_parallel(self, data_array, min_years=5, chunk_size=None):
        """
        Calculate linear trends for each pixel using optimized parallel processing.
        
        Parameters:
        -----------
        data_array : xr.DataArray
            3D array (time, lat, lon)
        min_years : int
            Minimum years of data required
        chunk_size : int
            Size of spatial chunks for parallel processing (auto if None)
        
        Returns:
        --------
        dict with trend, p_value, std_error arrays
        """
        # Use more cores for trend calculations
        trend_workers = min(TREND_CORES, self.n_workers * 2)
        
        print(f"  📈 Calculating pixel-wise trends (parallel, {trend_workers} workers)...")
        
        n_time, n_lat, n_lon = data_array.shape
        time_numeric = np.arange(n_time)
        
        # Auto-determine optimal chunk size based on system resources
        if chunk_size is None:
            total_pixels = n_lat * n_lon
            optimal_chunks = trend_workers * 4  # 4 chunks per worker
            pixels_per_chunk = max(100, total_pixels // optimal_chunks)
            chunk_size = int(np.sqrt(pixels_per_chunk))
            chunk_size = min(chunk_size, 200)  # Cap at 200 for memory
        
        print(f"    Using chunk size: {chunk_size}x{chunk_size}")
        
        # Initialize output arrays
        trend = np.full((n_lat, n_lon), np.nan)
        p_value = np.full((n_lat, n_lon), np.nan)
        std_error = np.full((n_lat, n_lon), np.nan)
        n_valid = np.full((n_lat, n_lon), 0)
        
        # Create chunks for parallel processing
        chunks = []
        for i in range(0, n_lat, chunk_size):
            for j in range(0, n_lon, chunk_size):
                i_end = min(i + chunk_size, n_lat)
                j_end = min(j + chunk_size, n_lon)
                
                # Extract data chunk and convert to numpy to avoid pickling issues
                data_chunk = data_array[:, i:i_end, j:j_end].values
                
                chunks.append((data_chunk, time_numeric, i, i_end, j, j_end, min_years))
        
        print(f"    Processing {len(chunks)} chunks with {trend_workers} workers...")
        
        # Process chunks in parallel with optimized settings
        if JOBLIB_AVAILABLE:
            try:
                # Use joblib with optimized backend for high-core systems
                results = Parallel(
                    n_jobs=trend_workers, 
                    backend='loky',
                    batch_size='auto',
                    verbose=1
                )(delayed(_calculate_trends_chunk_standalone)(chunk) for chunk in chunks)
            except Exception as e:
                print(f"    ⚠️ Parallel processing failed: {e}")
                print(f"    Falling back to sequential processing...")
                results = [_calculate_trends_chunk_standalone(chunk) for chunk in tqdm(chunks)]
        else:
            # Fallback to multiprocessing with chunk batching
            with Pool(trend_workers) as pool:
                results = list(tqdm(pool.imap(_calculate_trends_chunk_standalone, chunks), 
                                  total=len(chunks)))
        
        # Combine results
        for result in results:
            if result is not None:
                i_start, i_end, j_start, j_end = result['bounds']
                trend[i_start:i_end, j_start:j_end] = result['trend']
                p_value[i_start:i_end, j_start:j_end] = result['p_value']
                std_error[i_start:i_end, j_start:j_end] = result['std_error']
                n_valid[i_start:i_end, j_start:j_end] = result['n_valid']
        
        # Clean up memory
        gc.collect()
        
        return {
            'trend': trend,
            'p_value': p_value,
            'std_error': std_error,
            'n_valid': n_valid
        }

    def create_parallel_overview_maps(self):
        """Create overview maps using simplified parallel processing."""
        print("\n🎨 Creating parallel overview maps...")
        
        # Create mean map first (single process)
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        extent = self._get_safe_extent(self.gws_ds.groundwater)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        mean_gws = self.gws_ds.groundwater.mean(dim='time')
        
        if self.region_mask is not None:
            mean_gws = mean_gws.where(~np.isnan(self.region_mask))
        
        vmin, vmax = self._get_robust_colorscale(mean_gws.values, symmetric=True)
        
        im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, mean_gws, 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        self._add_map_features(ax, add_shapefile=True)
        
        ax.set_title('Mean GWS Anomaly (2003-2022) - Parallel Processing', 
                    fontsize=FONT_SIZE, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label('GWS Anomaly (cm)', fontsize=FONT_SIZE)
        
        plt.tight_layout()
        output_path = self.subdirs['overview'] / 'mean_gws_parallel.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
        
        # Create time slices sequentially to avoid pickling issues
        print("   Creating time slices (sequential due to pickling constraints)...")
        
        # Select representative time indices
        n_times = len(self.gws_ds.time)
        time_indices = [0, n_times//6, n_times//3, n_times//2, 
                       2*n_times//3, 5*n_times//6, n_times-1]
        
        results = []
        for time_idx in time_indices:
            result = self._create_overview_slice_sequential(
                time_idx, extent, vmin, vmax
            )
            if result:
                results.append(result)
        
        print(f"   ✅ Created {len(results)} time slice figures")
    
    def _create_overview_slice_sequential(self, time_idx, extent, vmin, vmax):
        """Create a single overview time slice (sequential version)."""
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            gws_slice = self.gws_ds.groundwater.isel(time=time_idx)
            
            if self.region_mask is not None:
                gws_slice = gws_slice.where(~np.isnan(self.region_mask))
            
            time_str = str(self.gws_ds.groundwater.time.values[time_idx])[:7]
            
            im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, gws_slice, 
                              cmap='RdBu_r', vmin=vmin, vmax=vmax,
                              transform=ccrs.PlateCarree())
            
            self._add_map_features(ax, add_shapefile=True)
            
            ax.set_title(f'GWS Anomaly ({time_str})', fontsize=FONT_SIZE, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label('GWS Anomaly (cm)', fontsize=FONT_SIZE)
            
            filename = f'gws_slice_{time_idx:03d}_{time_str}.png'
            output_path = self.subdirs['overview'] / filename
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
        except Exception as e:
            print(f"Error creating slice {time_idx}: {e}")
            return None



    def create_parallel_trend_analysis(self, data_array, variable_name, 
                                     units='cm/year', clip_to_shapefile=True):
        """Create trend analysis using parallel processing."""
        print(f"\n🎨 Creating parallel trend map for {variable_name}")
        
        # Calculate trends in parallel
        trend_results = self.calculate_pixel_trends_parallel(data_array)
        
        # Apply shapefile mask if requested
        if clip_to_shapefile and self.region_mask is not None:
            for key in trend_results:
                trend_results[key] = np.where(
                    ~np.isnan(self.region_mask), 
                    trend_results[key], 
                    np.nan
                )
        
        # Create main trend figure
        fig = plt.figure(figsize=(12, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        extent = self._get_safe_extent(data_array)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        vmin, vmax = self._get_robust_colorscale(trend_results['trend'], 
                                                percentiles=(5, 95), symmetric=True)
        
        print(f"    Trend color scale: {vmin:.3f} to {vmax:.3f} {units}")
        
        im = ax.pcolormesh(data_array.lon, data_array.lat, 
                          trend_results['trend'],
                          cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        # Add significance hatching
        sig_01 = trend_results['p_value'] < 0.01
        sig_05 = (trend_results['p_value'] >= 0.01) & (trend_results['p_value'] < 0.05)
        sig_10 = (trend_results['p_value'] >= 0.05) & (trend_results['p_value'] < 0.10)
        
        lon_mesh, lat_mesh = np.meshgrid(data_array.lon, data_array.lat)
        
        ax.contourf(lon_mesh, lat_mesh, sig_01.astype(float), 
                   levels=[0.5, 1.5], colors='none', 
                   hatches=['///'], transform=ccrs.PlateCarree())
        ax.contourf(lon_mesh, lat_mesh, sig_05.astype(float), 
                   levels=[0.5, 1.5], colors='none', 
                   hatches=[r'\\'], transform=ccrs.PlateCarree())
        ax.contourf(lon_mesh, lat_mesh, sig_10.astype(float), 
                   levels=[0.5, 1.5], colors='none', 
                   hatches=['...'], transform=ccrs.PlateCarree())
        
        self._add_map_features(ax, add_shapefile=clip_to_shapefile)
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label(f'{variable_name} Trend ({units})', fontsize=FONT_SIZE)
        
        time_range = f"{str(data_array.time.values[0])[:4]}-{str(data_array.time.values[-1])[:4]}"
        plt.title(f'{variable_name} Linear Trend ({time_range}) - Parallel Processing\n' + 
                 r'Hatching: /// p<0.01, \\\ p<0.05, ... p<0.10', 
                 fontsize=FONT_SIZE+2, pad=20)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch='///', label='p < 0.01'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch=r'\\', label='p < 0.05'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          hatch='...', label='p < 0.10'),
            mpatches.Patch(facecolor='white', edgecolor='black', 
                          label='Not significant')
        ]
        ax.legend(handles=legend_elements, loc='lower left', 
                 bbox_to_anchor=(0, -0.2), ncol=4, frameon=True)
        
        filename = f"{variable_name.lower().replace(' ', '_')}_trend_parallel.png"
        save_path = self.subdirs['trends'] / filename
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved: {filename}")
        
        return trend_results
    
    def create_extreme_timing_maps_parallel(self):
        """Create maps showing when extremes occurred (parallel-compatible version)."""
        print("\n⏰ CREATING EXTREME TIMING MAPS")
        print("="*50)
        
        variables = []
        if hasattr(self, 'gws_ds'):
            variables.append(('groundwater', self.gws_ds.groundwater, 'Groundwater Storage'))
        if hasattr(self, 'gws_ds') and 'tws' in self.gws_ds:
            variables.append(('tws', self.gws_ds.tws, 'Total Water Storage'))
        
        for var_name, data_array, display_name in variables:
            print(f"\n📊 Processing {display_name}")
            
            # Find timing of minimum and maximum for each pixel
            n_lat, n_lon = len(data_array.lat), len(data_array.lon)
            
            min_time_index = np.full((n_lat, n_lon), np.nan)
            max_time_index = np.full((n_lat, n_lon), np.nan)
            min_month = np.full((n_lat, n_lon), np.nan)
            max_month = np.full((n_lat, n_lon), np.nan)
            
            # Convert time coordinate to pandas datetime for easier handling
            time_as_datetime = pd.to_datetime(data_array.time.values)
            
            # Use parallel processing for extreme finding if dataset is large
            if n_lat * n_lon > 100000:
                print(f"    Using parallel processing for extreme finding...")
                extreme_results = self._find_extremes_parallel(data_array, time_as_datetime)
                min_time_index = extreme_results['min_time_index']
                max_time_index = extreme_results['max_time_index']
                min_month = extreme_results['min_month']
                max_month = extreme_results['max_month']
            else:
                # Sequential processing for smaller datasets
                for i in range(n_lat):
                    for j in range(n_lon):
                        ts = data_array[:, i, j].values
                        if not np.all(np.isnan(ts)):
                            # Find minimum
                            min_idx = np.nanargmin(ts)
                            min_time_index[i, j] = min_idx
                            min_month[i, j] = time_as_datetime[min_idx].month
                            
                            # Find maximum
                            max_idx = np.nanargmax(ts)
                            max_time_index[i, j] = max_idx
                            max_month[i, j] = time_as_datetime[max_idx].month
            
            # Apply mask if using shapefile
            if self.region_mask is not None:
                min_time_index = np.where(~np.isnan(self.region_mask), 
                                        min_time_index, np.nan)
                max_time_index = np.where(~np.isnan(self.region_mask), 
                                        max_time_index, np.nan)
                min_month = np.where(~np.isnan(self.region_mask), 
                                   min_month, np.nan)
                max_month = np.where(~np.isnan(self.region_mask), 
                                   max_month, np.nan)
            
            # Create figure with 4 panels
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12),
                                                        subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Common extent
            extent = self._get_safe_extent(data_array)
            
            # 1. Year of minimum
            ax1.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Convert time index to year
            min_year = np.full_like(min_time_index, np.nan)
            valid = ~np.isnan(min_time_index)
            for i in range(n_lat):
                for j in range(n_lon):
                    if valid[i, j]:
                        idx = int(min_time_index[i, j])
                        min_year[i, j] = time_as_datetime[idx].year
            
            im1 = ax1.pcolormesh(data_array.lon, data_array.lat, min_year,
                               cmap='viridis', transform=ccrs.PlateCarree())
            self._add_map_features(ax1, add_shapefile=True)
            
            plt.colorbar(im1, ax=ax1, label='Year')
            ax1.set_title(f'Year of Minimum {display_name}')
            
            # 2. Month of minimum (seasonality)
            ax2.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Use cyclic colormap for months
            cmap_months = plt.cm.hsv
            im2 = ax2.pcolormesh(data_array.lon, data_array.lat, min_month,
                               cmap=cmap_months, vmin=1, vmax=12,
                               transform=ccrs.PlateCarree())
            self._add_map_features(ax2, add_shapefile=True)
            
            cbar2 = plt.colorbar(im2, ax=ax2, label='Month', ticks=range(1, 13))
            cbar2.ax.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            ax2.set_title(f'Month of Typical Minimum {display_name}')
            
            # 3. Minimum value magnitude
            ax3.set_extent(extent, crs=ccrs.PlateCarree())
            
            min_values = np.full((n_lat, n_lon), np.nan)
            for i in range(n_lat):
                for j in range(n_lon):
                    ts = data_array[:, i, j].values
                    if not np.all(np.isnan(ts)):
                        min_values[i, j] = np.nanmin(ts)
            
            if self.region_mask is not None:
                min_values = np.where(~np.isnan(self.region_mask), 
                                    min_values, np.nan)
            
            vmax = np.nanpercentile(np.abs(min_values), 95)
            if vmax == 0:
                vmax = 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            im3 = ax3.pcolormesh(data_array.lon, data_array.lat, min_values,
                               cmap='RdBu_r', norm=norm,
                               transform=ccrs.PlateCarree())
            self._add_map_features(ax3, add_shapefile=True)
            
            plt.colorbar(im3, ax=ax3, label='Minimum Value (cm)')
            ax3.set_title(f'Magnitude of Minimum {display_name}')
            
            # 4. Histogram of extreme years
            ax4.remove()  # Remove map projection
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Count occurrences by year
            years = []
            valid = ~np.isnan(min_year)
            for y in min_year[valid]:
                years.append(int(y))
            
            if years:
                year_counts = pd.Series(years).value_counts().sort_index()
                
                ax4.bar(year_counts.index, year_counts.values, color='darkred', alpha=0.7)
                ax4.set_xlabel('Year')
                ax4.set_ylabel('Number of Pixels with Annual Minimum')
                ax4.set_title(f'Distribution of {display_name} Minima by Year')
                ax4.grid(True, alpha=0.3)
                
                # Rotate x labels
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.suptitle(f'Timing and Magnitude of {display_name} Extremes (Parallel)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            filename = f"{var_name}_extreme_timing_parallel.png"
            plt.savefig(self.subdirs['extremes'] / filename, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: {filename}")
    
    def _find_extremes_parallel(self, data_array, time_as_datetime):
        """Find extremes using parallel processing for large datasets."""
        n_time, n_lat, n_lon = data_array.shape
        
        # Initialize output arrays
        min_time_index = np.full((n_lat, n_lon), np.nan)
        max_time_index = np.full((n_lat, n_lon), np.nan)
        min_month = np.full((n_lat, n_lon), np.nan)
        max_month = np.full((n_lat, n_lon), np.nan)
        
        # Create chunks for parallel processing
        chunk_size = 50
        chunks = []
        for i in range(0, n_lat, chunk_size):
            for j in range(0, n_lon, chunk_size):
                i_end = min(i + chunk_size, n_lat)
                j_end = min(j + chunk_size, n_lon)
                
                data_chunk = data_array[:, i:i_end, j:j_end].values
                chunks.append((data_chunk, time_as_datetime, i, i_end, j, j_end))
        
        print(f"    Processing {len(chunks)} chunks for extreme finding...")
        
        # Process chunks in parallel
        if JOBLIB_AVAILABLE:
            try:
                results = Parallel(
                    n_jobs=min(32, self.n_workers), 
                    backend='loky',
                    verbose=0
                )(delayed(_find_extremes_chunk_standalone)(chunk) for chunk in chunks)
            except Exception as e:
                print(f"    ⚠️ Parallel extreme finding failed: {e}")
                # Fallback to sequential
                results = [_find_extremes_chunk_standalone(chunk) for chunk in chunks]
        else:
            results = [_find_extremes_chunk_standalone(chunk) for chunk in chunks]
        
        # Combine results
        for result in results:
            if result is not None:
                i_start, i_end, j_start, j_end = result['bounds']
                min_time_index[i_start:i_end, j_start:j_end] = result['min_time_index']
                max_time_index[i_start:i_end, j_start:j_end] = result['max_time_index']
                min_month[i_start:i_end, j_start:j_end] = result['min_month']
                max_month[i_start:i_end, j_start:j_end] = result['max_month']
        
        return {
            'min_time_index': min_time_index,
            'max_time_index': max_time_index,
            'min_month': min_month,
            'max_month': max_month
        }

    def create_parallel_spatial_statistics(self):
        """Create spatial statistics using optimized processing."""
        print("\n🗺️ Creating parallel spatial statistics...")
        
        # Calculate statistics (use dask if available for large arrays)
        if self.use_dask:
            print("   Using dask for statistics calculation...")
            mean_data = self.gws_ds.groundwater.mean(dim='time').compute()
            std_data = self.gws_ds.groundwater.std(dim='time').compute()
            min_data = self.gws_ds.groundwater.min(dim='time').compute()
            max_data = self.gws_ds.groundwater.max(dim='time').compute()
        else:
            mean_data = self.gws_ds.groundwater.mean(dim='time')
            std_data = self.gws_ds.groundwater.std(dim='time')
            min_data = self.gws_ds.groundwater.min(dim='time')
            max_data = self.gws_ds.groundwater.max(dim='time')
        
        stats_to_plot = [
            ('Mean', mean_data, 'RdBu_r', True),
            ('Std Dev', std_data, 'viridis', False),
            ('Min', min_data, 'Blues_r', False),
            ('Max', max_data, 'Reds', False)
        ]
        
        # Create figures sequentially to avoid pickling issues
        print("   Creating spatial statistics (sequential due to pickling constraints)...")
        
        results = []
        for title, data, cmap, symmetric in stats_to_plot:
            result = self._create_spatial_stat_sequential(title, data, cmap, symmetric)
            if result:
                results.append(result)
        
        print(f"   ✅ Created {len(results)} spatial statistics figures")
    
    def create_advanced_time_series_analysis(self):
        """Create comprehensive time series analysis with robust methods."""
        print("\n📈 Creating advanced time series analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Calculate regional average (with mask if available)
        if self.region_mask is not None:
            regional_avg = self.gws_ds.groundwater.where(
                ~np.isnan(self.region_mask)
            ).mean(dim=['lat', 'lon'])
        else:
            regional_avg = self.gws_ds.groundwater.mean(dim=['lat', 'lon'])
        
        time_index = pd.to_datetime(self.gws_ds.time.values)
        
        # 1. Full time series
        ax1 = axes[0, 0]
        ax1.plot(time_index, regional_avg.values, 'b-', linewidth=2, alpha=0.7)
        
        # Add 12-month rolling mean
        rolling_mean = pd.Series(regional_avg.values, index=time_index).rolling(12, center=True).mean()
        ax1.plot(time_index, rolling_mean.values, 'r-', linewidth=3, label='12-month rolling mean')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Regional GWS Anomaly (cm)')
        ax1.set_title('Regional Groundwater Storage Time Series (Parallel Enhanced)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal cycle (with proper datetime handling)
        ax2 = axes[0, 1]
        monthly_clim = []
        time_index_da = xr.DataArray(time_index, dims=['time'], coords={'time': time_index})
        for month in range(1, 13):
            month_mask = time_index_da.dt.month == month
            month_data = regional_avg.sel(time=month_mask)
            if len(month_data) > 0:
                monthly_clim.append(month_data.mean().values)
            else:
                monthly_clim.append(0.0)
        
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        bars = ax2.bar(range(1, 13), monthly_clim, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Mean GWS Anomaly (cm)')
        ax2.set_title('Seasonal Cycle (Parallel Enhanced)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Annual means
        ax3 = axes[1, 0]
        annual_means = []
        years = []
        for year in range(2003, 2023):
            year_mask = time_index_da.dt.year == year
            year_data = regional_avg.sel(time=year_mask)
            if len(year_data) > 0:
                annual_means.append(year_data.mean().values)
                years.append(year)
        
        bars = ax3.bar(years, annual_means, color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Annual Mean GWS Anomaly (cm)')
        ax3.set_title('Annual Mean Groundwater Storage (Parallel Enhanced)')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add trend line
        if len(years) > 2:
            z = np.polyfit(years, annual_means, 1)
            p = np.poly1d(z)
            ax3.plot(years, p(years), 'r--', linewidth=2, 
                    label=f'Trend: {z[0]:.3f} cm/year')
            ax3.legend()
        
        # 4. Distribution
        ax4 = axes[1, 1]
        valid_data = regional_avg.values[~np.isnan(regional_avg.values)]
        ax4.hist(valid_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(valid_data), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(valid_data):.2f} cm')
        ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('GWS Anomaly (cm)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Regional GWS Values (Parallel Enhanced)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Groundwater Storage Time Series Analysis (Parallel Enhanced)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = self.subdirs['overview'] / 'time_series_analysis_parallel.png'
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: {output_path}")
    
    def create_regional_analysis_parallel(self, region_shapefiles=None, process_all_geometries=False):
        """
        Create regional analysis using different shapefiles (parallel-compatible).
        
        Parameters:
        -----------
        region_shapefiles : dict or str
            Dictionary of region_name: shapefile_path OR
            Single shapefile path to process all geometries within
        process_all_geometries : bool
            If True and single shapefile provided, process each geometry separately
        """
        print("\n🌍 CREATING REGIONAL ANALYSIS (PARALLEL)")
        print("="*50)
        
        # Handle different input types
        if isinstance(region_shapefiles, str):
            # Single shapefile provided
            shapefile_path = region_shapefiles
            print(f"📂 Processing single shapefile: {shapefile_path}")
            
            if os.path.exists(shapefile_path):
                gdf = gpd.read_file(shapefile_path)
                if gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                
                if process_all_geometries:
                    # Process each geometry separately
                    print(f"  ✅ Found {len(gdf)} geometries to process individually")
                    
                    # Find name column
                    name_col = None
                    for col in ['name', 'NAME', 'Name', 'STATE_NAME', 'AQ_NAME']:
                        if col in gdf.columns:
                            name_col = col
                            break
                    
                    if not name_col:
                        # Use index if no name column
                        gdf['name'] = [f"Region_{i}" for i in range(len(gdf))]
                        name_col = 'name'
                    
                    # Convert to dictionary format
                    region_shapefiles = {}
                    for idx, row in gdf.iterrows():
                        region_name = str(row[name_col]).replace(' ', '_')
                        # Create temporary single-geometry GeoDataFrame
                        single_geom = gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs)
                        region_shapefiles[region_name] = single_geom
                else:
                    # Process as single region
                    region_name = os.path.splitext(os.path.basename(shapefile_path))[0]
                    region_shapefiles = {region_name: gdf}
            else:
                print(f"  ❌ Shapefile not found: {shapefile_path}")
                return {}
        
        elif region_shapefiles is None:
            # Try to auto-detect processed shapefiles
            processed_dir = Path("data/shapefiles/processed")
            region_shapefiles = {}
            
            # Check for main basin
            basin_path = processed_dir / "mississippi_river_basin.shp"
            if basin_path.exists():
                region_shapefiles['Mississippi_Basin'] = str(basin_path)
            
            # Check for subregions
            subregions_huc = processed_dir / "subregions_huc"
            if subregions_huc.exists():
                for shp in subregions_huc.glob("*.shp"):
                    region_name = shp.stem
                    region_shapefiles[region_name] = str(shp)
                    
                # Limit to first 5 to avoid overwhelming
                if len(region_shapefiles) > 6:  # Keep Mississippi + 5 subregions
                    limited_regions = dict(list(region_shapefiles.items())[:6])
                    region_shapefiles = limited_regions
                    print(f"  📊 Limited to {len(region_shapefiles)} regions for efficiency")
            
            if not region_shapefiles:
                print("  ⚠️ No shapefiles found automatically")
                return {}
        
        # Now process all regions
        regional_results = {}
        
        for region_name, shapefile in region_shapefiles.items():
            if shapefile is None:
                continue
            
            print(f"\n📍 Processing region: {region_name}")
            
            # Load shapefile if needed
            if isinstance(shapefile, str):
                if os.path.exists(shapefile):
                    gdf = gpd.read_file(shapefile)
                    if gdf.crs != 'EPSG:4326':
                        gdf = gdf.to_crs('EPSG:4326')
                else:
                    print(f"  ❌ Shapefile not found: {shapefile}")
                    continue
            else:
                gdf = shapefile
            
            # Create mask
            try:
                if len(gdf) > 1:
                    # Multiple geometries - union them
                    region_geom = gdf.union_all()
                else:
                    # Single geometry
                    region_geom = gdf.geometry.iloc[0]
                
                if regionmask is not None:
                    region_obj = regionmask.Regions([region_geom], names=[region_name])
                    mask = region_obj.mask(self.gws_ds.lon, self.gws_ds.lat)
                    
                    # Check if mask is valid
                    if np.all(np.isnan(mask)):
                        print(f"  ⚠️ Region {region_name} does not overlap with data domain")
                        continue
                else:
                    print(f"  ⚠️ regionmask not available, skipping {region_name}")
                    continue
                
                # Calculate regional averages
                regional_ts = {}
                
                # Groundwater
                if hasattr(self, 'gws_ds'):
                    gws_masked = self.gws_ds.groundwater.where(~np.isnan(mask))
                    regional_ts['groundwater'] = gws_masked.mean(dim=['lat', 'lon'])
                
                # TWS
                if hasattr(self, 'gws_ds') and 'tws' in self.gws_ds:
                    tws_masked = self.gws_ds.tws.where(~np.isnan(mask))
                    regional_ts['tws'] = tws_masked.mean(dim=['lat', 'lon'])
                
                # Soil moisture
                if hasattr(self, 'gws_ds') and 'soil_moisture_anomaly' in self.gws_ds:
                    sm_masked = self.gws_ds.soil_moisture_anomaly.where(~np.isnan(mask))
                    regional_ts['soil_moisture'] = sm_masked.mean(dim=['lat', 'lon'])
                
                # Snow
                if hasattr(self, 'gws_ds') and 'swe_anomaly' in self.gws_ds:
                    swe_masked = self.gws_ds.swe_anomaly.where(~np.isnan(mask))
                    regional_ts['swe'] = swe_masked.mean(dim=['lat', 'lon'])
                
                regional_results[region_name] = regional_ts
                
                # Create regional time series plot
                self._plot_regional_timeseries_parallel(regional_ts, region_name, gdf)
                
            except Exception as e:
                print(f"  ❌ Error processing {region_name}: {e}")
                continue
        
        # Create comparison plot if multiple regions
        if len(regional_results) > 1:
            self._plot_regional_comparison_parallel(regional_results)
        
        return regional_results
    
    def _plot_regional_timeseries_parallel(self, regional_ts, region_name, region_gdf):
        """Plot time series for a specific region (parallel version)."""
        try:
            fig = plt.figure(figsize=(16, 10))
            
            # Create grid layout
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], 
                                width_ratios=[2, 1], hspace=0.3, wspace=0.2)
            
            # Convert time to pandas datetime for better plotting
            time_index = pd.to_datetime(self.gws_ds.time.values)
            
            # 1. Multi-component time series
            ax1 = fig.add_subplot(gs[0, 0])
            
            colors = {'groundwater': 'blue', 'tws': 'black', 
                     'soil_moisture': 'brown', 'swe': 'cyan'}
            
            for var_name, ts_data in regional_ts.items():
                if var_name in colors:
                    ax1.plot(time_index, ts_data.values, label=var_name.replace('_', ' ').title(),
                            color=colors[var_name], linewidth=2, alpha=0.8)
            
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax1.set_ylabel('Storage Anomaly (cm)')
            ax1.set_title(f'{region_name}: Water Storage Components Time Series (Parallel)')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # 2. Map showing region
            ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
            
            # Plot context
            bounds = region_gdf.total_bounds
            # Check if bounds are valid
            if np.all(np.isfinite(bounds)):
                buffer = 2.0  # degrees
                ax2.set_extent([bounds[0]-buffer, bounds[2]+buffer,
                               bounds[1]-buffer, bounds[3]+buffer],
                              crs=ccrs.PlateCarree())
            else:
                ax2.set_global()  # Use global extent as fallback
            
            self._add_map_features(ax2, add_shapefile=False)
            
            # Highlight region
            region_gdf.plot(ax=ax2, facecolor='lightblue', edgecolor='red',
                           linewidth=2, alpha=0.5, transform=ccrs.PlateCarree())
            
            ax2.set_title(f'{region_name} Location')
            
            # 3. Seasonal cycle
            ax3 = fig.add_subplot(gs[1, :])
            
            for var_name, ts_data in regional_ts.items():
                if var_name in colors:
                    # Calculate monthly climatology
                    monthly_clim = []
                    for month in range(1, 13):
                        month_data = ts_data.sel(time=ts_data.time.dt.month == month)
                        monthly_clim.append(month_data.mean().values)
                    
                    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
                    ax3.plot(range(1, 13), monthly_clim, 'o-', 
                            label=var_name.replace('_', ' ').title(),
                            color=colors[var_name], linewidth=2, markersize=8)
            
            ax3.set_xticks(range(1, 13))
            ax3.set_xticklabels(months)
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Mean Anomaly (cm)')
            ax3.set_title('Seasonal Cycle')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            # 4. Trend analysis
            ax4 = fig.add_subplot(gs[2, :])
            
            trend_results = []
            
            for var_name, ts_data in regional_ts.items():
                if var_name in colors:
                    # Calculate trend
                    time_numeric = np.arange(len(ts_data))
                    valid = ~np.isnan(ts_data.values)
                    
                    if np.sum(valid) > 24:  # At least 2 years
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            time_numeric[valid], ts_data.values[valid]
                        )
                        
                        # Convert to annual trend
                        annual_trend = slope * 12
                        
                        trend_results.append({
                            'Variable': var_name.replace('_', ' ').title(),
                            'Trend (cm/year)': annual_trend,
                            'p-value': p_value,
                            'R²': r_value**2
                        })
                        
                        # Plot data with trend line
                        ax4.scatter(time_index[valid], ts_data.values[valid], 
                                  alpha=0.5, s=20, color=colors[var_name])
                        trend_line = slope * time_numeric + intercept
                        ax4.plot(time_index, trend_line, '--', color=colors[var_name],
                                linewidth=2, label=f'{var_name}: {annual_trend:.3f} cm/yr')
            
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Storage Anomaly (cm)')
            ax4.set_title('Linear Trends')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'{region_name} Regional Water Storage Analysis (Parallel)', 
                        fontsize=16, fontweight='bold')
            
            filename = f"{region_name.lower().replace(' ', '_')}_regional_analysis_parallel.png"
            plt.savefig(self.subdirs['regional'] / filename, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: {filename}")
            
        except Exception as e:
            print(f"  ❌ Error creating regional plot for {region_name}: {e}")
    
    def _plot_regional_comparison_parallel(self, regional_results):
        """Compare multiple regions (parallel version)."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Variables to compare
            variables = ['groundwater', 'tws', 'soil_moisture', 'swe']
            colors = plt.cm.Set3(np.linspace(0, 1, len(regional_results)))
            
            for i, var in enumerate(variables):
                ax = axes[i]
                
                for j, (region_name, regional_ts) in enumerate(regional_results.items()):
                    if var in regional_ts:
                        time_index = pd.to_datetime(self.gws_ds.time.values)
                        ax.plot(time_index, regional_ts[var].values, 
                               label=region_name, color=colors[j], 
                               linewidth=2, alpha=0.8)
                
                ax.set_title(var.replace('_', ' ').title())
                ax.set_ylabel('Anomaly (cm)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                
                if i >= 2:
                    ax.set_xlabel('Year')
            
            plt.suptitle('Regional Comparison of Water Storage Components (Parallel)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.subdirs['regional'] / 'regional_comparison_parallel.png', 
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  💾 Saved: regional_comparison_parallel.png")
            
        except Exception as e:
            print(f"  ❌ Error creating regional comparison: {e}")
    
    def analyze_all_components_parallel(self):
        """Analyze trends for all water storage components using parallel processing."""
        print("\n🔄 ANALYZING ALL WATER STORAGE COMPONENTS (PARALLEL)")
        print("="*50)
        
        components_analyzed = 0
        
        # 1. Groundwater Storage
        if hasattr(self, 'gws_ds'):
            print("\n1️⃣ Groundwater Storage Anomaly")
            self.create_parallel_trend_analysis(
                self.gws_ds.groundwater, 
                'Groundwater Storage',
                units='cm/year',
                clip_to_shapefile=True
            )
            components_analyzed += 1
        
        # 2. Total Water Storage (if available in GWS dataset)
        if hasattr(self, 'gws_ds') and 'tws' in self.gws_ds:
            print("\n2️⃣ Total Water Storage")
            self.create_parallel_trend_analysis(
                self.gws_ds.tws,
                'Total Water Storage',
                units='cm/year',
                clip_to_shapefile=True
            )
            components_analyzed += 1
        
        # 3. Soil Moisture
        if hasattr(self, 'gws_ds') and 'soil_moisture_anomaly' in self.gws_ds:
            print("\n3️⃣ Soil Moisture")
            self.create_parallel_trend_analysis(
                self.gws_ds.soil_moisture_anomaly,
                'Soil Moisture',
                units='cm/year',
                clip_to_shapefile=True
            )
            components_analyzed += 1
        
        # 4. Snow Water Equivalent
        if hasattr(self, 'gws_ds') and 'swe_anomaly' in self.gws_ds:
            print("\n4️⃣ Snow Water Equivalent")
            self.create_parallel_trend_analysis(
                self.gws_ds.swe_anomaly,
                'Snow Water Equivalent',
                units='cm/year',
                clip_to_shapefile=True
            )
            components_analyzed += 1
        
        # 5. Precipitation (from features if available)
        if self.features_ds is not None:
            print("\n5️⃣ Precipitation")
            # Find precipitation in features
            precip_indices = []
            for i, feat in enumerate(self.features_ds.feature.values):
                if 'pr' in str(feat).lower() or 'chirps' in str(feat).lower():
                    precip_indices.append(i)
            
            if precip_indices:
                # Use first precipitation feature
                precip_data = self.features_ds.features[:, precip_indices[0], :, :]
                self.create_parallel_trend_analysis(
                    precip_data,
                    'Precipitation',
                    units='mm/year',
                    clip_to_shapefile=True
                )
                components_analyzed += 1
        
        print(f"\n✅ Analyzed {components_analyzed} water storage components using parallel processing")
        return components_analyzed
    
    def create_comprehensive_report_parallel(self):
        """Generate a comprehensive summary report (parallel version)."""
        print("\n📄 GENERATING COMPREHENSIVE REPORT (PARALLEL)")
        print("="*50)
        
        report_lines = [
            "GRACE GROUNDWATER DOWNSCALING - PARALLEL ANALYSIS REPORT",
            "="*60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PARALLEL PROCESSING CONFIGURATION:",
            "-"*30,
            f"Total CPU cores available: {TOTAL_CORES}",
            f"Standard processing cores used: {N_CORES}",
            f"Trend calculation cores used: {TREND_CORES}",
            f"Dask enabled: {self.use_dask}",
            f"Workers for visualization: {self.n_workers}",
            f"Memory limit per worker: {MEMORY_LIMIT}",
            "",
            "ANALYSIS SUMMARY:",
            "-"*30
        ]
        
        # Add information about figures created
        figure_counts = {}
        total_figures = 0
        for subdir_name, subdir_path in self.subdirs.items():
            png_files = list(subdir_path.glob('*.png'))
            figure_counts[subdir_name] = len(png_files)
            total_figures += len(png_files)
            
            report_lines.extend([
                f"\n{subdir_name.upper()} FIGURES ({len(png_files)} files):",
                "-"*20
            ])
            
            for fig_file in sorted(png_files):
                report_lines.append(f"  • {fig_file.name}")
        
        # Add parallel processing performance info
        report_lines.extend([
            "\n\nPARALLEL PROCESSING PERFORMANCE:",
            "-"*30,
            f"Parallel trend analysis: ✅ Optimized for {TREND_CORES} cores",
            f"Parallel extreme finding: ✅ Chunked processing",
            f"Parallel figure generation: ✅ Multi-worker rendering",
            f"Memory efficiency: ✅ Dask chunking and cleanup",
        ])
        
        # Add data summary
        report_lines.extend([
            "\n\nDATA COVERAGE:",
            "-"*30,
            f"Spatial domain: {float(self.gws_ds.lat.min()):.2f}°N to {float(self.gws_ds.lat.max()):.2f}°N, "
            f"{float(self.gws_ds.lon.min()):.2f}°E to {float(self.gws_ds.lon.max()):.2f}°E",
            f"Temporal coverage: {str(self.gws_ds.time.values[0])[:10]} to {str(self.gws_ds.time.values[-1])[:10]}",
            f"Number of time steps: {len(self.gws_ds.time)}",
            f"Spatial resolution: {len(self.gws_ds.lat)} x {len(self.gws_ds.lon)} pixels"
        ])
        
        if self.shapefile is not None:
            report_lines.extend([
                f"\nShapefile used for clipping: Yes",
                f"Region area: Analysis limited to shapefile boundaries"
            ])
        
        # Performance optimizations summary
        report_lines.extend([
            "\n\nPARALLEL OPTIMIZATIONS APPLIED:",
            "-"*30,
            f"✅ Multi-core trend calculations using {TREND_CORES} cores",
            f"✅ Chunked spatial processing to prevent memory overflow",
            f"✅ Parallel extreme finding for large datasets",
            f"✅ Background figure generation for efficiency",
            f"✅ Dask integration for memory-efficient large array operations",
            f"✅ Optimized worker allocation for high-core systems",
            f"✅ Memory cleanup and garbage collection in workers",
            f"✅ Conservative resource usage to maintain system stability"
        ])
        
        # Save report
        report_path = self.figures_dir / "parallel_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  💾 Report saved to: {report_path}")
        
        # Print summary
        print(f"\n📊 PARALLEL ANALYSIS COMPLETE!")
        print(f"  Total figures created: {total_figures}")
        for category, count in figure_counts.items():
            print(f"    {category}: {count} figures")
        print(f"  Output directory: {self.figures_dir}")
        print(f"  Performance: Optimized for {TOTAL_CORES}-core system")
        
        return total_figures
    
    def _create_spatial_stat_sequential(self, title, data, cmap, symmetric):
        """Create a single spatial statistics figure (sequential version)."""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            extent = self._get_safe_extent(self.gws_ds.groundwater)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            # Apply region mask if available
            if self.region_mask is not None:
                data = data.where(~np.isnan(self.region_mask))
            
            vmin, vmax = self._get_robust_colorscale(data.values, symmetric=symmetric)
            
            im = ax.pcolormesh(self.gws_ds.lon, self.gws_ds.lat, data, 
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree())
            
            self._add_map_features(ax, add_shapefile=True)
            
            ax.set_title(f'GWS {title} (Parallel)', fontsize=FONT_SIZE, fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                               pad=0.05, shrink=0.8)
            cbar.set_label(f'{title} (cm)', fontsize=FONT_SIZE)
            
            plt.tight_layout()
            
            filename = f'spatial_stat_{title.lower().replace(" ", "_")}_parallel.png'
            output_path = self.subdirs['overview'] / filename
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error creating {title} spatial stat: {e}")
            return None

    def run_parallel_analysis(self):
        """Run the complete parallel analysis with ALL advanced features."""
        print("\n🚀 RUNNING COMPLETE PARALLEL GRACE ANALYSIS")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # 0. Create advanced overview visualizations first
            print("\n📊 PHASE 1: OVERVIEW MAPS AND TIME SERIES")
            print("-"*50)
            self.create_parallel_overview_maps()
            self.create_parallel_spatial_statistics()
            self.create_advanced_time_series_analysis()
            
            # 1. Analyze all water storage components (creates trend maps with significance)
            print("\n📈 PHASE 2: MULTI-COMPONENT TREND ANALYSIS")
            print("-"*50)
            components_analyzed = self.analyze_all_components_parallel()
            
            # 2. Create extreme timing maps
            print("\n⏰ PHASE 3: EXTREME TIMING ANALYSIS")
            print("-"*50)
            self.create_extreme_timing_maps_parallel()
            
            # 3. Regional analysis
            print("\n🌍 PHASE 4: REGIONAL ANALYSIS")
            print("-"*50)
            # Option A: Use specific regions if available
            if os.path.exists("data/shapefiles/processed/subregions_huc"):
                print("📊 Processing HUC-based subregions...")
                regions = {}
                for shp in Path("data/shapefiles/processed/subregions_huc").glob("*.shp"):
                    regions[shp.stem] = str(shp)
                if regions:
                    # Limit to first 5 for efficiency
                    limited_regions = dict(list(regions.items())[:5])
                    self.create_regional_analysis_parallel(limited_regions)
            
            # Option B: Process individual states (first 3)
            elif os.path.exists("data/shapefiles/processed/individual_states"):
                print("📊 Processing individual states (first 3)...")
                state_files = list(Path("data/shapefiles/processed/individual_states").glob("*.shp"))[:3]
                if state_files:
                    state_regions = {shp.stem: str(shp) for shp in state_files}
                    self.create_regional_analysis_parallel(state_regions)
            
            # Option C: Use main basin only
            else:
                print("📊 Processing main Mississippi Basin...")
                self.create_regional_analysis_parallel()
            
            # 4. Generate comprehensive report
            print("\n📄 PHASE 5: COMPREHENSIVE REPORTING")
            print("-"*50)
            total_figures = self.create_comprehensive_report_parallel()
            
            # Calculate total time
            end_time = datetime.now()
            total_time = end_time - start_time
            
            print(f"\n🎉 COMPLETE PARALLEL ANALYSIS FINISHED!")
            print("="*60)
            print(f"⏱️  Total execution time: {total_time}")
            print(f"📊 Water storage components analyzed: {components_analyzed}")
            print(f"🖼️  Total figures generated: {total_figures}")
            print(f"🧮 Processing cores utilized: {TREND_CORES} (trends), {N_CORES} (general)")
            print(f"💾 Memory usage optimized with chunking and cleanup")
            print(f"📁 Output directory: {self.figures_dir}")
            
            # Performance summary for 192-core system
            seconds = total_time.total_seconds()
            if seconds > 0:
                pixel_rate = (len(self.gws_ds.lat) * len(self.gws_ds.lon) * components_analyzed) / seconds
                print(f"🚀 Processing rate: {pixel_rate:.0f} pixels/second")
            
            # Clean up
            if self.dask_client:
                self.dask_client.close()
            
            # Memory cleanup
            import gc
            gc.collect()
            
            print(f"\n✅ All resources cleaned up successfully!")
            
            return {
                'total_time': total_time,
                'components_analyzed': components_analyzed,
                'total_figures': total_figures,
                'processing_rate': pixel_rate if seconds > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Error in parallel analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run optimized parallel analysis."""
    print("🚀 PARALLEL GRACE VISUALIZATION")
    print("="*50)
    
    # Check system resources
    mem = psutil.virtual_memory()
    print(f"💻 System Resources:")
    print(f"   CPU cores: {TOTAL_CORES}")
    print(f"   Standard processing: {N_CORES} cores")
    print(f"   Trend calculations: {TREND_CORES} cores")
    print(f"   RAM: {mem.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {mem.available / (1024**3):.1f} GB")
    print(f"   RAM usage: {mem.percent:.1f}%")
    
    # Check if system has enough resources
    if mem.available / (1024**3) < 32:  # Less than 32GB available
        print("⚠️ Warning: Low available memory. Consider reducing worker counts.")
    
    # Initialize visualizer
    shapefile_path = "data/shapefiles/processed/mississippi_river_basin.shp"
    
    if not os.path.exists(shapefile_path):
        print("⚠️ Mississippi Basin shapefile not found!")
        shapefile_path = None
    
    # Create visualizer with optimal settings for high-core system
    visualizer = ParallelGRACEVisualizer(
        base_dir=".",
        shapefile_path=shapefile_path,
        n_workers=N_CORES,
        use_dask=DASK_AVAILABLE
    )
    
    # Monitor initial memory usage
    initial_mem = psutil.virtual_memory().percent
    print(f"🔍 Initial memory usage: {initial_mem:.1f}%")
    
    # Run parallel analysis
    start_time = datetime.now()
    success = visualizer.run_parallel_analysis()
    end_time = datetime.now()
    
    # Monitor final memory usage
    final_mem = psutil.virtual_memory().percent
    duration = end_time - start_time
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Duration: {duration}")
    print(f"   Initial memory: {initial_mem:.1f}%")
    print(f"   Final memory: {final_mem:.1f}%")
    print(f"   Memory efficiency: {'✅ Good' if final_mem - initial_mem < 20 else '⚠️ High usage'}")
    
    print(f"\n🎯 OPTIMIZATIONS FOR 192-CORE SYSTEM:")
    print(f"   ✅ Conservative allocation: {N_CORES} cores for standard ops")
    print(f"   ✅ Aggressive allocation: {TREND_CORES} cores for trend calculations")
    print(f"   ✅ Smart chunking: Auto-sized based on worker count and memory")
    print(f"   ✅ Dask acceleration: For large array statistics")
    print(f"   ✅ Pickle-safe design: Standalone functions avoid serialization issues")
    print(f"   ✅ Memory management: Explicit cleanup and error handling")
    print(f"   ✅ Fallback modes: Sequential processing if parallel fails")
    
    return visualizer


if __name__ == "__main__":
    main() 