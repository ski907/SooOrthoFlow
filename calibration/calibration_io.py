"""
Calibration I/O utilities for CSV-based camera calibration storage.

This module provides functions to load and save camera calibrations in the new
CSV format, separating calibration parameters from orthorectification cache.
"""

import pandas as pd
import numpy as np
import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional


def load_camera_calibrations(csv_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load camera calibrations from CSV file.

    Parameters:
    -----------
    csv_file : str or Path
        Path to camera_calibrations_YYYYMMDD.csv

    Returns:
    --------
    dict : Camera calibrations in dictionary format
        Keys are camera_ids, values are calibration dicts with:
        - K: (3,3) intrinsic matrix
        - D: (4,1) distortion coefficients
        - rvec: (3,1) rotation vector
        - tvec: (3,1) translation vector
        - rms: float, reprojection error
        - n_gcps: int, number of GCPs used
        - image_size: tuple (width, height)
        - geotransform: dict with x_min, y_max, pixel_width, pixel_height
        - calibration_date: str, YYYYMMDD
        - recalibrated: bool
        - recalibration_mode: str
        - gcps_skipped: int
    """
    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    calibrations = {}

    for _, row in df.iterrows():
        camera_id = row['camera_id']

        # Reconstruct K matrix (3x3 camera intrinsic matrix)
        K = np.array([
            [row['fx'], row['skew'], row['cx']],
            [row['s'], row['fy'], row['cy']],
            [row['K_20'], row['K_21'], row['K_22']]
        ], dtype=np.float64)

        # Reconstruct D coefficients (4,1 fisheye distortion coefficients)
        D = np.array([
            [row['k1']],
            [row['k2']],
            [row['k3']],
            [row['k4']]
        ], dtype=np.float64)

        # Reconstruct rvec (3,1 rotation vector)
        rvec = np.array([
            [row['rvec_x']],
            [row['rvec_y']],
            [row['rvec_z']]
        ], dtype=np.float64)

        # Reconstruct tvec (3,1 translation vector)
        tvec = np.array([
            [row['tvec_x']],
            [row['tvec_y']],
            [row['tvec_z']]
        ], dtype=np.float64)

        # Geotransform dict
        geotransform = {
            'x_min': row['geotransform_x_min'],
            'y_max': row['geotransform_y_max'],
            'pixel_width': row['geotransform_pixel_width'],
            'pixel_height': row['geotransform_pixel_height'],
            'rotation_x': 0,
            'rotation_y': 0
        }

        # Image size tuple
        image_size = (int(row['image_width']), int(row['image_height']))

        # Parse GCP pixel coordinates from JSON if present
        gcp_pixel_coords = []
        if 'gcp_pixel_coords' in row and pd.notna(row['gcp_pixel_coords']) and row['gcp_pixel_coords']:
            try:
                gcp_pixel_coords = json.loads(row['gcp_pixel_coords'])
            except (json.JSONDecodeError, TypeError):
                gcp_pixel_coords = []

        # Build calibration dict
        calibrations[camera_id] = {
            'K': K,
            'D': D,
            'rvec': rvec,
            'tvec': tvec,
            'rms': row['rms'],
            'n_gcps': int(row['n_gcps']),
            'image_size': image_size,
            'geotransform': geotransform,
            'calibration_date': str(row['calibration_date']),
            'recalibrated': bool(row['recalibrated']),
            'recalibration_mode': str(row['recalibration_mode']) if pd.notna(row['recalibration_mode']) else '',
            'gcps_skipped': int(row['gcps_skipped']),
            'output_width': int(row['output_width']) if 'output_width' in row and pd.notna(row['output_width']) else None,
            'output_height': int(row['output_height']) if 'output_height' in row and pd.notna(row['output_height']) else None,
            'gcp_pixel_coords': gcp_pixel_coords,
            'local_origin_x': float(row['local_origin_x']) if 'local_origin_x' in row and pd.notna(row['local_origin_x']) else 0.0,
            'local_origin_y': float(row['local_origin_y']) if 'local_origin_y' in row and pd.notna(row['local_origin_y']) else 0.0,
            'local_origin_z': float(row['local_origin_z']) if 'local_origin_z' in row and pd.notna(row['local_origin_z']) else 0.0,
            'model_crs': str(row['model_crs']) if 'model_crs' in row and pd.notna(row['model_crs']) else 'EPSG:26919'
        }

    return calibrations


def save_camera_calibrations(calibrations: Dict[str, Dict[str, Any]], csv_file: str):
    """
    Save camera calibrations to CSV file.

    Parameters:
    -----------
    calibrations : dict
        Dictionary of camera calibrations (camera_id -> calib_dict)
    csv_file : str or Path
        Output path for camera_calibrations_YYYYMMDD.csv
    """
    csv_file = Path(csv_file)

    csv_data = []

    for camera_id, calib in calibrations.items():
        # Flatten matrices
        K = calib['K']
        D = calib['D'].flatten()
        rvec = calib['rvec'].flatten()
        tvec = calib['tvec'].flatten()

        geo = calib['geotransform']
        img_size = calib['image_size']

        # Serialize GCP pixel coordinates to JSON
        gcp_pixel_coords_json = ''
        if 'gcp_pixel_coords' in calib and calib['gcp_pixel_coords']:
            try:
                gcp_pixel_coords_json = json.dumps(calib['gcp_pixel_coords'])
            except (TypeError, ValueError):
                gcp_pixel_coords_json = ''

        row = {
            'camera_id': camera_id,
            'calibration_date': calib['calibration_date'],

            # K matrix (3x3 camera intrinsic matrix)
            'fx': K[0, 0], 'skew': K[0, 1], 'cx': K[0, 2],
            's': K[1, 0], 'fy': K[1, 1], 'cy': K[1, 2],
            'K_20': K[2, 0], 'K_21': K[2, 1], 'K_22': K[2, 2],

            # D coefficients (4, fisheye distortion)
            'k1': D[0], 'k2': D[1], 'k3': D[2], 'k4': D[3],

            # rvec (3, rotation vector)
            'rvec_x': rvec[0], 'rvec_y': rvec[1], 'rvec_z': rvec[2],

            # tvec (3, translation vector)
            'tvec_x': tvec[0], 'tvec_y': tvec[1], 'tvec_z': tvec[2],

            # Metrics
            'rms': calib['rms'],
            'n_gcps': calib['n_gcps'],

            # Image size
            'image_width': img_size[0],
            'image_height': img_size[1],

            # Geotransform
            'geotransform_x_min': geo['x_min'],
            'geotransform_y_max': geo['y_max'],
            'geotransform_pixel_width': geo['pixel_width'],
            'geotransform_pixel_height': geo['pixel_height'],

            # Recalibration info
            'recalibrated': calib.get('recalibrated', False),
            'recalibration_mode': calib.get('recalibration_mode', ''),
            'gcps_skipped': calib.get('gcps_skipped', 0),

            # Output dimensions (for cache regeneration)
            'output_width': calib.get('output_width', ''),
            'output_height': calib.get('output_height', ''),

            # GCP pixel coordinates (JSON)
            'gcp_pixel_coords': gcp_pixel_coords_json,

            # Coordinate system info
            'local_origin_x': calib.get('local_origin_x', 0.0),
            'local_origin_y': calib.get('local_origin_y', 0.0),
            'local_origin_z': calib.get('local_origin_z', 0.0),
            'model_crs': calib.get('model_crs', 'EPSG:26919'),
        }

        csv_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(csv_data)

    # Column order
    column_order = [
        'camera_id', 'calibration_date',
        'fx', 'skew', 'cx', 's', 'fy', 'cy', 'K_20', 'K_21', 'K_22',
        'k1', 'k2', 'k3', 'k4',
        'rvec_x', 'rvec_y', 'rvec_z',
        'tvec_x', 'tvec_y', 'tvec_z',
        'rms', 'n_gcps',
        'image_width', 'image_height',
        'geotransform_x_min', 'geotransform_y_max',
        'geotransform_pixel_width', 'geotransform_pixel_height',
        'output_width', 'output_height',
        'recalibrated', 'recalibration_mode', 'gcps_skipped',
        'gcp_pixel_coords',
        'local_origin_x', 'local_origin_y', 'local_origin_z',
        'model_crs'
    ]

    df = df[column_order]
    df.to_csv(csv_file, index=False, float_format='%.12f')


def update_camera_calibration(csv_file: str, camera_id: str, calibration_data: Dict[str, Any],
                               new_csv_file: Optional[str] = None):
    """
    Update a single camera's calibration in the CSV file.

    This is used during recalibration to update one camera's parameters.

    Parameters:
    -----------
    csv_file : str or Path
        Existing calibration CSV file
    camera_id : str
        Camera identifier to update
    calibration_data : dict
        New calibration data for this camera (same structure as load_camera_calibrations returns)
    new_csv_file : str or Path, optional
        Path for updated CSV. If None, overwrites existing file.
    """
    csv_file = Path(csv_file)

    # Load existing calibrations
    calibrations = load_camera_calibrations(csv_file)

    # Update or add camera
    calibrations[camera_id] = calibration_data

    # Save to new or existing file
    output_file = Path(new_csv_file) if new_csv_file else csv_file
    save_camera_calibrations(calibrations, output_file)


def find_most_recent_calibration_csv(calibration_dir: str = 'calibration') -> Optional[Path]:
    """
    Find the most recent camera_calibrations_YYYYMMDD.csv file.

    Parameters:
    -----------
    calibration_dir : str or Path
        Directory containing calibration files

    Returns:
    --------
    Path or None : Path to most recent CSV file, or None if not found
    """
    calibration_dir = Path(calibration_dir)

    csv_files = list(calibration_dir.glob('camera_calibrations_*.csv'))

    if not csv_files:
        return None

    # Sort by date in filename (YYYYMMDD)
    csv_files.sort(key=lambda p: p.stem.split('_')[-1], reverse=True)

    return csv_files[0]


def compute_cache_hash(K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                       geotransform: Dict[str, float], resolution: float) -> str:
    """
    Compute a hash of calibration parameters and resolution for cache naming.

    This ensures cache is invalidated when any calibration parameter or resolution changes.

    Parameters:
    -----------
    K : np.ndarray
        Camera intrinsic matrix (3x3)
    D : np.ndarray
        Distortion coefficients (4x1)
    rvec : np.ndarray
        Rotation vector (3x1)
    tvec : np.ndarray
        Translation vector (3x1)
    geotransform : dict
        Geotransform parameters (x_min, y_max, pixel_width, pixel_height)
    resolution : float
        Output resolution in meters/pixel

    Returns:
    --------
    str : 12-character hash string
    """
    # Combine all parameters into a single bytes object
    params = []
    params.extend(K.flatten().tolist())
    params.extend(D.flatten().tolist())
    params.extend(rvec.flatten().tolist())
    params.extend(tvec.flatten().tolist())
    params.extend([
        geotransform['x_min'],
        geotransform['y_max'],
        geotransform['pixel_width'],
        geotransform['pixel_height']
    ])
    params.append(resolution)

    # Convert to bytes and hash
    param_str = ','.join(f'{p:.12f}' for p in params)
    hash_obj = hashlib.sha256(param_str.encode('utf-8'))

    # Return first 12 characters of hex digest (sufficient for uniqueness)
    return hash_obj.hexdigest()[:12]


def load_ortho_cache(camera_id: str, K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                     geotransform: Dict[str, float], resolution: float, resolution_name: str = 'hires',
                     cache_dir: str = 'orthorectification/ortho_cache') -> Optional[Dict[str, Any]]:
    """
    Load orthorectification cache for a specific camera with hash-based validation.

    Parameters:
    -----------
    camera_id : str
        Camera identifier
    K, D, rvec, tvec : np.ndarray
        Calibration parameters for computing hash
    geotransform : dict
        Geotransform parameters for computing hash
    resolution : float
        Output resolution in meters/pixel
    resolution_name : str
        Resolution name for cache file suffix ('hires' or 'lowres')
    cache_dir : str or Path
        Directory containing cache files

    Returns:
    --------
    dict or None : Cache data with map_x, map_y, output_width, output_height, resolution, cache_hash
                   Returns None if cache file doesn't exist or is corrupted
    """
    cache_dir = Path(cache_dir)

    # Compute hash from current parameters
    cache_hash = compute_cache_hash(K, D, rvec, tvec, geotransform, resolution)
    cache_file = cache_dir / f"{camera_id}_ortho_cache_{cache_hash}_{resolution_name}.pkl"
    print(cache_file)

    if not cache_file.exists():
        return None

    # Check if file is empty or corrupted
    if cache_file.stat().st_size == 0:
        print(f"  WARNING: Cache file {cache_file.name} is empty, deleting...")
        cache_file.unlink()
        return None

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # Validate cached hash matches (extra paranoia check)
        if 'cache_hash' in cache_data and cache_data['cache_hash'] != cache_hash:
            print(f"  WARNING: Cache hash mismatch, regenerating...")
            cache_file.unlink()
            return None

        # Validate resolution matches (extra paranoia check)
        if 'resolution' in cache_data and abs(cache_data['resolution'] - resolution) > 1e-6:
            print(f"  WARNING: Cache resolution mismatch, regenerating...")
            cache_file.unlink()
            return None

        return cache_data
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"  WARNING: Cache file {cache_file.name} is corrupted ({e}), deleting...")
        cache_file.unlink()
        return None


def save_ortho_cache(camera_id: str, K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                     geotransform: Dict[str, float], resolution: float, resolution_name: str,
                     cache_data: Dict[str, Any], cache_dir: str = 'orthorectification/ortho_cache'):
    """
    Save orthorectification cache for a specific camera with hash-based naming.

    Thread-safe: Uses unique temp file names to avoid conflicts when
    multiple processes try to save the same cache simultaneously.

    Parameters:
    -----------
    camera_id : str
        Camera identifier
    K, D, rvec, tvec : np.ndarray
        Calibration parameters for computing hash
    geotransform : dict
        Geotransform parameters for computing hash
    resolution : float
        Output resolution in meters/pixel
    resolution_name : str
        Resolution name for cache file suffix ('hires' or 'lowres')
    cache_data : dict
        Cache data with keys: map_x, map_y, output_width, output_height
    cache_dir : str or Path
        Directory to save cache files
    """
    import os
    import time

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute hash and build cache filename
    cache_hash = compute_cache_hash(K, D, rvec, tvec, geotransform, resolution)
    cache_file = cache_dir / f"{camera_id}_ortho_cache_{cache_hash}_{resolution_name}.pkl"

    # Add metadata to cache_data for validation
    cache_data['cache_hash'] = cache_hash
    cache_data['resolution'] = resolution
    cache_data['resolution_name'] = resolution_name

    # Use process ID and timestamp to create unique temp file name
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    temp_file = cache_dir / f"{camera_id}_ortho_cache_{cache_hash}_{resolution_name}.pkl.tmp.{pid}.{timestamp}"

    # Write to temporary file first to avoid corruption
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # Only replace the actual cache file if write was successful
        if temp_file.exists() and temp_file.stat().st_size > 0:
            # Use atomic rename on Windows - if cache_file exists, another process beat us to it
            try:
                temp_file.replace(cache_file)
            except PermissionError:
                # Another process is writing or already wrote the file, that's OK
                # Just clean up our temp file
                if temp_file.exists():
                    temp_file.unlink()
        else:
            raise IOError("Temporary cache file is empty or was not created")
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass  # Ignore errors during cleanup
        # Don't raise if another process already created the file
        if not cache_file.exists():
            raise IOError(f"Failed to save cache for {camera_id}: {e}")


def delete_old_ortho_cache(camera_id: str, cache_dir: str = 'orthorectification/ortho_cache'):
    """
    Delete all ortho cache files for a specific camera.

    Used when a camera is recalibrated to clear outdated cache files.

    Parameters:
    -----------
    camera_id : str
        Camera identifier
    cache_dir : str or Path
        Directory containing cache files
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return

    # Find all cache files for this camera
    pattern = f"{camera_id}_ortho_cache_*.pkl"
    cache_files = list(cache_dir.glob(pattern))

    for cache_file in cache_files:
        cache_file.unlink()
        print(f"  Deleted old cache: {cache_file.name}")
