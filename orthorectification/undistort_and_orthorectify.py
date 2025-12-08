import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import rasterio
from rasterio.transform import rowcol
import argparse
import sys
import json
from scipy.optimize import least_squares

# Import new calibration I/O utilities
sys.path.append(str(Path(__file__).parent.parent / 'calibration'))
from calibration_io import (
    load_camera_calibrations,
    save_camera_calibrations,
    load_ortho_cache,
    save_ortho_cache,
    delete_old_ortho_cache
)


def fisheye_residuals(params, objpoints, imgpoints, image_size):
    """
    Compute reprojection residuals for fisheye camera.

    Args:
        params: [fx, fy, cx, cy, k1, k2, k3, k4, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z]
        objpoints: Nx3 object points
        imgpoints: Nx2 image points
        image_size: (width, height)

    Returns:
        residuals: Flattened reprojection errors (2N values)
    """
    # Unpack parameters
    fx, fy, cx, cy = params[0:4]
    k1, k2, k3, k4 = params[4:8]
    rvec = params[8:11].reshape(3, 1)
    tvec = params[11:14].reshape(3, 1)

    # Build K and D
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)

    # Project points
    try:
        projected, _ = cv2.fisheye.projectPoints(
            objpoints.reshape(-1, 1, 3).astype(np.float32),
            rvec.astype(np.float64),
            tvec.astype(np.float64),
            K, D
        )
        projected = projected.reshape(-1, 2)
    except cv2.error:
        # Return large residuals if projection fails
        return np.full(len(imgpoints) * 2, 1e6, dtype=np.float64)

    # Compute residuals
    residuals = (projected - imgpoints).flatten()

    return residuals


def get_initial_guess_opencv(objpoints, imgpoints, image_size):
    """Get rough initial guess using OpenCV without bounds."""
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_CHECK_COND +
             cv2.fisheye.CALIB_FIX_SKEW)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_size, K, D,
            flags=flags, criteria=criteria
        )
        return K, D, rvecs, tvecs
    except cv2.error as e:
        print(f"  Warning: OpenCV initial calibration failed: {e}")
        # Return default guess
        w, h = image_size
        K = np.array([[2000, 0, w/2], [0, 2000, h/2], [0, 0, 1]], dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        rvecs = [np.zeros((3, 1), dtype=np.float64)]
        tvecs = [np.zeros((3, 1), dtype=np.float64)]
        return K, D, rvecs, tvecs


def setup_optimization_params(K, D, rvec, tvec, bounds_config, image_size):
    """Setup parameter vector and bounds for optimization."""
    w, h = image_size

    # Initial parameter vector
    params_init = np.concatenate([
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],  # fx, fy, cx, cy
        D.flatten(),  # k1, k2, k3, k4
        rvec.flatten(),  # rvec_x, rvec_y, rvec_z
        tvec.flatten()   # tvec_x, tvec_y, tvec_z
    ])

    # Bounds (lower, upper)
    lower_bounds = [
        bounds_config['fx_min'],
        bounds_config['fy_min'],
        w * bounds_config['cx_min_ratio'],
        h * bounds_config['cy_min_ratio'],
        bounds_config['k1_min'],
        bounds_config['k2_min'],
        bounds_config['k3_min'],
        bounds_config['k4_min'],
        -np.pi, -np.pi, -np.pi,  # rvec bounds
        -np.inf, -np.inf, -np.inf  # tvec bounds (no limit)
    ]

    upper_bounds = [
        bounds_config['fx_max'],
        bounds_config['fy_max'],
        w * bounds_config['cx_max_ratio'],
        h * bounds_config['cy_max_ratio'],
        bounds_config['k1_max'],
        bounds_config['k2_max'],
        bounds_config['k3_max'],
        bounds_config['k4_max'],
        np.pi, np.pi, np.pi,  # rvec bounds
        np.inf, np.inf, np.inf  # tvec bounds
    ]

    return params_init, (lower_bounds, upper_bounds)


def unpack_params(params):
    """Unpack optimization parameters into K, D, rvec, tvec."""
    fx, fy, cx, cy = params[0:4]
    k1, k2, k3, k4 = params[4:8]
    rvec = params[8:11].reshape(3, 1)
    tvec = params[11:14].reshape(3, 1)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)

    return K, D, rvec, tvec


def calibrate_fisheye_with_bounds(objpoints, imgpoints, image_size, bounds_config,
                                   initial_K=None, initial_D=None):
    """
    Calibrate fisheye camera with parameter bounds using scipy optimization.

    Args:
        objpoints: List of object points (world coordinates)
        imgpoints: List of image points (pixel coordinates)
        image_size: (width, height)
        bounds_config: Dict with min/max for fx, fy, cx, cy, k1-k4
        initial_K: Optional initial intrinsic matrix
        initial_D: Optional initial distortion coefficients

    Returns:
        K, D, rvec, tvec, rms, final_params
    """
    # Step 1: Get initial guess (if not provided)
    if initial_K is None or initial_D is None:
        K_init, D_init, rvecs, tvecs = get_initial_guess_opencv(
            objpoints, imgpoints, image_size
        )
    else:
        K_init, D_init = initial_K, initial_D
        # Get initial pose estimate using solvePnP
        # First undistort points to normalized coordinates
        imgpoints_flat = imgpoints[0].reshape(-1, 1, 2).astype(np.float32)
        objpoints_flat = objpoints[0].reshape(-1, 1, 3).astype(np.float64)

        # Use pinhole model for initial pose
        fx_avg = (K_init[0, 0] + K_init[1, 1]) / 2
        K_pinhole = np.array([[fx_avg, 0, K_init[0, 2]],
                               [0, fx_avg, K_init[1, 2]],
                               [0, 0, 1]], dtype=np.float64)

        success, rvec_init, tvec_init = cv2.solvePnP(
            objpoints_flat, imgpoints_flat, K_pinhole, None
        )

        if success:
            rvecs = [rvec_init]
            tvecs = [tvec_init]
        else:
            rvecs = [np.zeros((3, 1), dtype=np.float64)]
            tvecs = [np.zeros((3, 1), dtype=np.float64)]

    # Step 2: Setup parameter vector and bounds
    params_init, bounds = setup_optimization_params(
        K_init, D_init, rvecs[0], tvecs[0], bounds_config, image_size
    )

    # Step 2.5: Clip initial guess to be within bounds
    # scipy least_squares requires initial guess to be within bounds
    lower_bounds, upper_bounds = bounds
    params_init_clipped = np.clip(params_init, lower_bounds, upper_bounds)

    # Log if we had to clip
    if not np.allclose(params_init, params_init_clipped):
        print("  Warning: Initial guess was outside bounds - clipping to valid range")
        print(f"    Original fx={params_init[0]:.1f}, clipped fx={params_init_clipped[0]:.1f}")
        print(f"    Original fy={params_init[1]:.1f}, clipped fy={params_init_clipped[1]:.1f}")

    # Step 3: Run bounded optimization
    print("  Running constrained optimization...")
    result = least_squares(
        fisheye_residuals,
        params_init_clipped,  # Use clipped values
        bounds=bounds,
        args=(objpoints[0], imgpoints[0].reshape(-1, 2), image_size),
        method='trf',  # Trust Region Reflective - handles bounds
        verbose=0,
        ftol=1e-6,
        xtol=1e-6,
        max_nfev=500
    )

    # Step 4: Extract results
    K, D, rvec, tvec = unpack_params(result.x)
    rms = np.sqrt(np.mean(result.fun**2))

    return K, D, rvec, tvec, rms, result.x


def validate_calibration_params(K, D, rms, image_size, validation_config, strict=True):
    """
    Validate calibration parameters are physically reasonable.

    Returns: (is_valid, warnings)
    """
    warnings = []
    is_valid = True

    # Focal lengths
    fx, fy = K[0, 0], K[1, 1]
    fx_min = validation_config.get('fx_min', 1500)
    fx_max = validation_config.get('fx_max', 3000)

    if not (fx_min <= fx <= fx_max):
        warnings.append(f"fx={fx:.1f} out of bounds [{fx_min}, {fx_max}]")
        is_valid = False
    if not (fx_min <= fy <= fx_max):
        warnings.append(f"fy={fy:.1f} out of bounds [{fx_min}, {fx_max}]")
        is_valid = False

    # Focal length ratio
    ratio = fx / fy
    ratio_min = validation_config.get('focal_ratio_min', 0.95)
    ratio_max = validation_config.get('focal_ratio_max', 1.05)
    if not (ratio_min <= ratio <= ratio_max):
        warnings.append(f"fx/fy={ratio:.3f} not near 1.0 (non-square pixels?)")
        if strict:
            is_valid = False

    # Principal point
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size
    if not (w * 0.3 <= cx <= w * 0.7):
        warnings.append(f"cx={cx:.1f} far from image center ({w/2:.1f})")
        is_valid = False
    if not (h * 0.3 <= cy <= h * 0.7):
        warnings.append(f"cy={cy:.1f} far from image center ({h/2:.1f})")
        is_valid = False

    # Distortion coefficients
    k1, k2, k3, k4 = D.flatten()
    if abs(k2) > 10:
        warnings.append(f"|k2|={abs(k2):.1f} > 10 (very large distortion)")
        is_valid = False
    if abs(k3) > 50:
        warnings.append(f"|k3|={abs(k3):.1f} > 50 (very large distortion)")
        is_valid = False
    if abs(k4) > 100:
        warnings.append(f"|k4|={abs(k4):.1f} > 100 (very large distortion)")
        is_valid = False

    # RMS error
    max_rms = validation_config.get('max_rms', 10.0)
    if rms > max_rms:
        warnings.append(f"RMS={rms:.1f}px > {max_rms} (poor fit)")
        is_valid = False

    return is_valid, warnings


def compute_average_intrinsics(valid_calibrations):
    """Compute average intrinsics from valid calibrations."""
    K_list = [calib['K'] for calib in valid_calibrations.values()]
    D_list = [calib['D'] for calib in valid_calibrations.values()]

    K_avg = np.mean(K_list, axis=0)
    D_avg = np.mean(D_list, axis=0)

    return K_avg, D_avg


def calibrate_fisheye_camera(gcp_data, image_path, camera_id):
    """
    Calibrate a single fisheye camera using GCP correspondences
    """
    # Filter GCPs for this camera
    #camera_gcps = gcp_data[gcp_data['image_name'].str.contains(camera_id)]
    camera_gcps = gcp_data[gcp_data['camera_name'].str.contains(camera_id)]
    
    # Extract 3D object points (X, Y, Z from LiDAR)
    object_points = camera_gcps[['X', 'Y', 'Z']].values.astype(np.float64)
    
    # Extract 2D image points (column, row)
    image_points = camera_gcps[['col_sample', 'row_sample']].values.astype(np.float64)
    
    # Reshape for OpenCV fisheye calibration
    objpoints = [object_points.reshape(1, -1, 3).astype(np.float64)]
    imgpoints = [image_points.reshape(1, -1, 2).astype(np.float64)]
    
    # Get image dimensions
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    image_size = (w, h)
    
    # Initialize camera matrix and distortion coefficients
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    
    # Calibration flags for fisheye
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
    )
    
    # Run calibration
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs=None,
        tvecs=None,
        flags=calibration_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print(f"{camera_id} - RMS reprojection error: {rms:.4f} pixels")
    
    return K, D, rvecs[0], tvecs[0], rms, image_size, camera_gcps


def undistort_fisheye(img, K, D, balance=0.0):
    """
    Simple undistortion without orthorectification
    For QC purposes
    """
    h, w = img.shape[:2]
    
    # Compute new camera matrix
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    
    # Create undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    
    # Undistort
    undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    
    return undistorted


def compute_camera_specific_bounds(camera_gcps, padding_meters=0.5):
    """
    Compute tight bounds for THIS camera's field of view
    """
    x_min, x_max = camera_gcps['X'].min(), camera_gcps['X'].max()
    y_min, y_max = camera_gcps['Y'].min(), camera_gcps['Y'].max()
    
    # Add padding in meters
    x_min -= padding_meters
    x_max += padding_meters
    y_min -= padding_meters
    y_max += padding_meters
    
    return x_min, x_max, y_min, y_max


def create_orthorectification_params(camera_gcps, resolution=0.005, padding_meters=0.5, local_origin=None):
    """
    Compute parameters for orthorectification for a SPECIFIC camera

    Parameters:
    - camera_gcps: DataFrame with X, Y, Z columns (in absolute coordinates)
    - resolution: meters per pixel in output (0.005 = 5mm)
    - padding_meters: extra space around GCP bounds
    - local_origin: numpy array [x, y, z] of local origin (to convert absolute coords to local)

    Returns:
    - width, height: output image dimensions
    - geotransform: dict with x_min, y_max, etc. in ABSOLUTE coordinates (for GeoTIFF)
    """
    # Compute bounds in absolute coordinates
    x_min_abs, x_max_abs, y_min_abs, y_max_abs = compute_camera_specific_bounds(camera_gcps, padding_meters)

    # If local origin provided, work in local coordinates for dimensions
    if local_origin is not None:
        # Convert bounds to local coordinates
        x_min_local = x_min_abs - local_origin[0]
        x_max_local = x_max_abs - local_origin[0]
        y_min_local = y_min_abs - local_origin[1]
        y_max_local = y_max_abs - local_origin[1]

        # Compute output image size from local coordinates
        width = int((x_max_local - x_min_local) / resolution)
        height = int((y_max_local - y_min_local) / resolution)
    else:
        # No local origin, use absolute coordinates directly
        width = int((x_max_abs - x_min_abs) / resolution)
        height = int((y_max_abs - y_min_abs) / resolution)

    print(f"  Output bounds (absolute): X=[{x_min_abs:.3f}, {x_max_abs:.3f}], Y=[{y_min_abs:.3f}, {y_max_abs:.3f}]")
    print(f"  Output size: {width} x {height} pixels @ {resolution*1000:.1f}mm/pixel")

    # Create world file parameters (geotransform) in ABSOLUTE coordinates for output GeoTIFF
    geotransform = {
        'x_min': x_min_abs,      # Absolute UTM for GeoTIFF
        'y_max': y_max_abs,      # Absolute UTM for GeoTIFF
        'pixel_width': resolution,
        'pixel_height': -resolution,
        'rotation_x': 0,
        'rotation_y': 0
    }

    return width, height, geotransform


def load_dem_from_tiff_old(dem_path, width, height, geotransform, nodata_value=None):
    """
    Load elevation data from LiDAR DEM TIFF and resample to output grid
    
    Parameters:
    - dem_path: path to DEM TIFF file
    - width, height: output grid dimensions
    - geotransform: output coordinate transformation parameters
    - nodata_value: value representing no data (optional, will use DEM's nodata if not specified)
    
    Returns:
    - dem_array: 2D array of Z values (height x width)
    """
    print(f"  Loading DEM from: {dem_path}")
    
    with rasterio.open(dem_path) as src:
        print(f"    DEM size: {src.width} x {src.height}")
        print(f"    DEM bounds: {src.bounds}")
        print(f"    DEM resolution: {src.res}")
        
        # Get nodata value
        if nodata_value is None:
            nodata_value = src.nodata
        
        # Read the DEM data
        dem_data = src.read(1)
        
        # Create output grid coordinates
        x_min = geotransform['x_min']
        y_max = geotransform['y_max']
        pixel_width = geotransform['pixel_width']
        pixel_height = geotransform['pixel_height']
        
        # Initialize output array
        dem_array = np.zeros((height, width), dtype=np.float32)
        
        # Sample DEM at each output pixel location
        for row in range(height):
            if row % 100 == 0:
                print(f"    Sampling DEM row {row}/{height}")
            
            for col in range(width):
                # Calculate world coordinates for this output pixel
                world_x = x_min + col * pixel_width
                world_y = y_max + row * pixel_height
                
                # Convert world coordinates to DEM pixel coordinates
                try:
                    dem_row, dem_col = rowcol(src.transform, world_x, world_y)
                    
                    # Check if within DEM bounds
                    if 0 <= dem_row < src.height and 0 <= dem_col < src.width:
                        z_value = dem_data[dem_row, dem_col]
                        
                        # Check for nodata
                        if nodata_value is not None and z_value == nodata_value:
                            # Use nearby valid value or default
                            dem_array[row, col] = np.nan
                        else:
                            dem_array[row, col] = z_value
                    else:
                        # Outside DEM bounds
                        dem_array[row, col] = np.nan
                        
                except Exception as e:
                    dem_array[row, col] = np.nan
        
        # Fill any NaN values with interpolation or mean
        if np.any(np.isnan(dem_array)):
            n_nan = np.sum(np.isnan(dem_array))
            print(f"    Found {n_nan} pixels outside DEM or with nodata")
            
            # Fill with mean of valid values
            valid_mean = np.nanmean(dem_array)
            dem_array[np.isnan(dem_array)] = valid_mean
            print(f"    Filled with mean Z = {valid_mean:.3f}")
        
        print(f"  DEM loaded: Z range = [{dem_array.min():.3f}, {dem_array.max():.3f}] meters")
        
        return dem_array

def load_dem_from_tiff(dem_path, width, height, geotransform, nodata_value=None):
    """
    Vectorized version using rasterio's built-in resampling (much faster).
    This is equivalent to load_dem_from_tiff_resampled but with better handling.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    
    print(f"  Loading DEM from: {dem_path} (VECTORIZED)")
    
    with rasterio.open(dem_path) as src:
        print(f"    DEM size: {src.width} x {src.height}")
        print(f"    DEM bounds: {src.bounds}")
        print(f"    DEM resolution: {src.res}")
        
        # Create output transform
        x_min = geotransform['x_min']
        y_max = geotransform['y_max']
        pixel_width = geotransform['pixel_width']
        pixel_height = abs(geotransform['pixel_height'])
        
        x_max = x_min + width * pixel_width
        y_min = y_max - height * pixel_height
        
        dst_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
        
        # Initialize output array
        dem_array = np.zeros((height, width), dtype=np.float32)
        
        # Reproject/resample using rasterio's fast C implementation
        print("    Resampling DEM (bilinear interpolation)...")
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )
        
        # Handle nodata values if specified
        if nodata_value is not None:
            mask = dem_array == nodata_value
            if np.any(mask):
                n_nodata = np.sum(mask)
                print(f"    Found {n_nodata} nodata pixels, filling with mean...")
                valid_mean = np.mean(dem_array[~mask])
                dem_array[mask] = valid_mean
        
        print(f"  DEM loaded: Z range = [{dem_array.min():.3f}, {dem_array.max():.3f}] meters")
        
        return dem_array

def load_dem_from_tiff_resampled(dem_path, width, height, geotransform):
    """
    Alternative method: Load and resample DEM using rasterio's built-in resampling
    This is faster for large DEMs but may be less accurate for small output grids
    """
    print(f"  Loading and resampling DEM from: {dem_path}")
    
    with rasterio.open(dem_path) as src:
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
        
        # Create output transform
        x_min = geotransform['x_min']
        y_max = geotransform['y_max']
        pixel_width = geotransform['pixel_width']
        pixel_height = abs(geotransform['pixel_height'])
        
        x_max = x_min + width * pixel_width
        y_min = y_max - height * pixel_height
        
        dst_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
        
        # Initialize output array
        dem_array = np.zeros((height, width), dtype=np.float32)
        
        # Reproject/resample
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,  # Assuming same CRS
            resampling=Resampling.bilinear
        )
        
        print(f"  DEM resampled: Z range = [{dem_array.min():.3f}, {dem_array.max():.3f}] meters")
        
        return dem_array


def create_ortho_lookup_tables_with_dem_old(K, D, rvec, tvec, width, height, 
                                        geotransform, dem_array):
    """
    Pre-compute mapping from output pixels to input pixels using DEM elevations
    This accounts for 3D relief in your model
    """
    print(f"  Creating DEM-based lookup tables for {width}x{height} output...")
    
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    x_min = geotransform['x_min']
    y_max = geotransform['y_max']
    pixel_width = geotransform['pixel_width']
    pixel_height = geotransform['pixel_height']
    
    R, _ = cv2.Rodrigues(rvec)
    
    for row in range(height):
        if row % 100 == 0:
            print(f"    Row {row}/{height}")
            
        for col in range(width):
            # Get world X, Y coordinates
            world_x = x_min + col * pixel_width
            world_y = y_max + row * pixel_height
            
            # Get elevation from DEM
            world_z = dem_array[row, col]
            
            # 3D point in world coordinates
            world_point = np.array([[world_x, world_y, world_z]], dtype=np.float64)
            
            # Transform to camera coordinates
            camera_point = R @ world_point.T + tvec
            camera_point_reshaped = camera_point.T.reshape(1, 1, 3)
            
            try:
                # Project to image with fisheye distortion
                image_point, _ = cv2.fisheye.projectPoints(
                    camera_point_reshaped, 
                    np.zeros((3, 1)), 
                    np.zeros((3, 1)), 
                    K, 
                    D
                )
                
                map_x[row, col] = image_point[0, 0, 0]
                map_y[row, col] = image_point[0, 0, 1]
                
            except cv2.error:
                # Point projects outside valid range
                map_x[row, col] = -1
                map_y[row, col] = -1
    
    return map_x, map_y

def create_ortho_lookup_tables_with_dem(K, D, rvec, tvec, width, height,
                                                    geotransform, dem_array, local_origin=None):
    """
    Vectorized version: Pre-compute mapping from output pixels to input pixels using DEM elevations.
    This is 10-50x faster than the pixel-by-pixel loop version.

    Parameters:
    - K: camera intrinsic matrix (3x3)
    - D: fisheye distortion coefficients (4x1)
    - rvec: rotation vector (3x1) - in LOCAL coordinates
    - tvec: translation vector (3x1) - in LOCAL coordinates
    - width, height: output image dimensions
    - geotransform: dict with x_min, y_max, pixel_width, pixel_height (in ABSOLUTE coordinates)
    - dem_array: 2D array of elevations (height x width)
    - local_origin: numpy array [x, y, z] of local origin (None if no transformation)

    Returns:
    - map_x, map_y: lookup tables for cv2.remap
    """
    print(f"  Creating DEM-based lookup tables for {width}x{height} output (VECTORIZED)...")

    # Extract geotransform parameters (these are in ABSOLUTE coordinates for GeoTIFF)
    x_min_abs = geotransform['x_min']
    y_max_abs = geotransform['y_max']
    pixel_width = geotransform['pixel_width']
    pixel_height = geotransform['pixel_height']

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Create coordinate grids for ALL pixels at once
    print("    Generating coordinate grids...")
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate world X, Y coordinates for all pixels in ABSOLUTE coordinates
    world_x_abs = x_min_abs + cols * pixel_width
    world_y_abs = y_max_abs + rows * pixel_height

    # Convert to LOCAL coordinates for projection (if local origin provided)
    if local_origin is not None:
        world_x = world_x_abs - local_origin[0]
        world_y = world_y_abs - local_origin[1]
        # DEM elevations are still absolute, but we need them relative to local origin Z
        world_z = dem_array - local_origin[2]
    else:
        world_x = world_x_abs
        world_y = world_y_abs
        world_z = dem_array  # Already in the right shape (height, width)
    
    # Stack into (height*width, 3) array of 3D world points
    print("    Stacking 3D world points...")
    world_points = np.stack([
        world_x.ravel(),
        world_y.ravel(),
        world_z.ravel()
    ], axis=1).astype(np.float64)
    
    # Transform ALL points to camera coordinates at once
    print("    Transforming to camera coordinates...")
    camera_points = (R @ world_points.T).T + tvec.T
    
    # Reshape for OpenCV fisheye projection: (N, 1, 3)
    camera_points_reshaped = camera_points.reshape(-1, 1, 3)
    
    # Project ALL points to image plane in a single call
    print("    Projecting points (this is the main computation)...")
    image_points, _ = cv2.fisheye.projectPoints(
        camera_points_reshaped,
        np.zeros((3, 1), dtype=np.float64),  # Zero rotation (already applied)
        np.zeros((3, 1), dtype=np.float64),  # Zero translation (already applied)
        K,
        D
    )
    
    # Reshape back to 2D lookup tables
    print("    Reshaping to lookup tables...")
    map_x = image_points[:, 0, 0].reshape(height, width).astype(np.float32)
    map_y = image_points[:, 0, 1].reshape(height, width).astype(np.float32)
    
    print(f"    Lookup tables created!")
    print(f"    Map X range: [{map_x.min():.1f}, {map_x.max():.1f}]")
    print(f"    Map Y range: [{map_y.min():.1f}, {map_y.max():.1f}]")
    
    return map_x, map_y

def orthorectify_with_lookup(img, map_x, map_y):
    """
    Fast orthorectification using pre-computed lookup tables
    """
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def save_geotiff(img, geotransform, output_path, crs='EPSG:26919'):
    """
    Save image as GeoTIFF with embedded georeferencing

    Parameters:
    - img: image array
    - geotransform: dict with x_min, y_max, pixel_width, pixel_height
    - output_path: path to save GeoTIFF
    - crs: coordinate reference system (e.g., 'EPSG:26919')
    """
    from rasterio.transform import from_bounds

    # Get image dimensions
    if len(img.shape) == 3:
        height, width, bands = img.shape
    else:
        height, width = img.shape
        bands = 1

    # Create rasterio transform from geotransform dict
    # The geotransform has: x_min, y_max, pixel_width (positive), pixel_height (negative)
    transform = from_bounds(
        west=geotransform['x_min'],
        south=geotransform['y_max'] + height * geotransform['pixel_height'],  # pixel_height is negative
        east=geotransform['x_min'] + width * geotransform['pixel_width'],
        north=geotransform['y_max'],
        width=width,
        height=height
    )

    # Prepare image data for rasterio (needs to be bands-first: [bands, height, width])
    if len(img.shape) == 3:
        # BGR to RGB and transpose to bands-first
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_data = np.transpose(img_rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    else:
        img_data = img[np.newaxis, :, :]  # (H, W) -> (1, H, W)

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=img_data.dtype,
        crs=crs,  # Use parameter instead of hardcoded
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(img_data)

    print(f"  Saved GeoTIFF: {output_path}")


# Main workflow
def calibrate_all_cameras(gcp_file, image_dir, dem_path, resolution=0.005,
                         padding_meters=0.5, output_dir='output',
                         save_undistorted=True, use_fast_resample=False,
                         calibration_date=None):
    """
    Calibrate all cameras using two-pass approach with constrained optimization.

    Parameters:
    - gcp_file: CSV file with GCP correspondences
    - image_dir: directory with camera images
    - dem_path: path to LiDAR DEM TIFF file
    - resolution: output resolution in meters per pixel (0.005 = 5mm)
    - padding_meters: extra space around GCP bounds
    - output_dir: where to save outputs
    - save_undistorted: whether to save simple undistorted images for QC
    - use_fast_resample: if True, use rasterio's resampling (faster but potentially less accurate)
    - calibration_date: Date stamp (YYYYMMDD) for calibration file. If None, auto-detected from image folder.
    """
    # Auto-detect calibration date from image folder if not provided
    if calibration_date is None:
        # Try to extract YYYYMMDD from folder name
        folder_name = Path(image_dir).name
        import re
        date_match = re.search(r'(\d{8})', folder_name)
        if date_match:
            calibration_date = date_match.group(1)
        else:
            # Default to current date
            from datetime import datetime
            calibration_date = datetime.now().strftime('%Y%m%d')
            print(f"Warning: Could not extract date from folder '{folder_name}', using today's date: {calibration_date}")

    print(f"Calibration date: {calibration_date}")

    # Setup logging to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    log_file = output_path / f'calibration_log_{calibration_date}.txt'

    import logging
    # Create logger
    logger = logging.getLogger('calibration')
    logger.setLevel(logging.DEBUG)
    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("="*80)
    logger.info("INITIAL CAMERA CALIBRATION - TWO-PASS CONSTRAINED OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"GCP file: {gcp_file}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"DEM file: {dem_path}")
    logger.info(f"Resolution: {resolution} m/pixel")
    logger.info(f"Padding: {padding_meters} m")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Calibration date: {calibration_date}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Load calibration bounds and coordinate system from master_control.json
    config_path = Path(__file__).parent.parent / 'master_control.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        bounds_config = config.get('calibration', {}).get('bounds', {})
        validation_config = config.get('calibration', {}).get('validation', {})
        # Merge bounds and validation for convenience
        validation_config.update({
            'fx_min': bounds_config.get('fx_min', 1500),
            'fx_max': bounds_config.get('fx_max', 3000)
        })

        # Load coordinate system configuration
        coord_config = config.get('coordinate_system', {})
        model_crs = coord_config.get('model_crs', 'EPSG:26919')
        local_origin = np.array([
            coord_config.get('local_origin_x', 0.0),
            coord_config.get('local_origin_y', 0.0),
            coord_config.get('local_origin_z', 0.0)
        ], dtype=np.float64)

        logger.info(f"Loaded calibration bounds from {config_path}")
        logger.debug(f"Bounds config: {bounds_config}")
        logger.debug(f"Validation config: {validation_config}")
        logger.info(f"Model CRS: {model_crs}")
        logger.info(f"Local origin: X={local_origin[0]:.1f}, Y={local_origin[1]:.1f}, Z={local_origin[2]:.1f}")
    else:
        logger.warning("master_control.json not found, using default bounds")
        bounds_config = {
            'fx_min': 1500, 'fx_max': 3000,
            'fy_min': 1500, 'fy_max': 3000,
            'cx_min_ratio': 0.3, 'cx_max_ratio': 0.7,
            'cy_min_ratio': 0.3, 'cy_max_ratio': 0.7,
            'k1_min': -2.0, 'k1_max': 2.0,
            'k2_min': -10.0, 'k2_max': 10.0,
            'k3_min': -50.0, 'k3_max': 50.0,
            'k4_min': -100.0, 'k4_max': 100.0
        }
        validation_config = {
            'max_rms': 10.0,
            'focal_ratio_min': 0.95,
            'focal_ratio_max': 1.05,
            'fx_min': 1500,
            'fx_max': 3000
        }
        # Default coordinate system (no local origin)
        model_crs = 'EPSG:26919'
        local_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Load GCP data
    logger.info(f"Loading GCP data from {gcp_file}")
    gcp_data = pd.read_csv(gcp_file)

    # Get unique camera IDs
    gcp_data['camera_id'] = gcp_data['camera_name']
    camera_ids = gcp_data['camera_id'].unique()
    logger.info(f"Found {len(camera_ids)} cameras in GCP file: {sorted(camera_ids)}")

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    if save_undistorted:
        undistorted_dir = output_path / 'undistorted'
        undistorted_dir.mkdir(exist_ok=True)
    ortho_dir = output_path / 'orthorectified'
    ortho_dir.mkdir(exist_ok=True)

    # PASS 1: Calibrate all cameras with constrained optimization
    logger.info("")
    logger.info("="*80)
    logger.info("PASS 1: Initial constrained calibration for all cameras")
    logger.info("="*80)

    calibration_results = {}
    valid_calibrations = {}
    failed_cameras = []

    # Track camera data for all cameras
    camera_data_map = {}  # Store image paths and GCPs for all cameras

    for camera_id in sorted(camera_ids):
        logger.info("")
        logger.info("="*80)
        logger.info(f"Processing {camera_id}")
        logger.info("="*80)

        # Find the image file for this camera using image_name from GCP file
        camera_gcp_rows = gcp_data[gcp_data['camera_id'] == camera_id]
        if len(camera_gcp_rows) == 0:
            logger.warning(f"No GCP data for {camera_id} - SKIPPING")
            continue

        image_name = camera_gcp_rows.iloc[0]['image_name']
        image_path = Path(image_dir) / image_name

        if not image_path.exists():
            logger.error(f"Image not found: {image_path} - SKIPPING {camera_id}")
            continue

        logger.info(f"Image: {image_path.name}")
        logger.info(f"Number of GCPs: {len(camera_gcp_rows)}")

        # Store camera data for later use
        camera_data_map[camera_id] = (image_path, camera_gcp_rows)

        try:
            # Prepare data for calibration
            # Load absolute GCP coordinates from CSV
            object_points_absolute = camera_gcp_rows[['X', 'Y', 'Z']].values.astype(np.float64)

            # Convert to local coordinates by subtracting origin
            object_points_local = object_points_absolute - local_origin

            logger.debug(f"  GCP absolute range: X=[{object_points_absolute[:,0].min():.1f}, {object_points_absolute[:,0].max():.1f}], Y=[{object_points_absolute[:,1].min():.1f}, {object_points_absolute[:,1].max():.1f}]")
            logger.debug(f"  GCP local range: X=[{object_points_local[:,0].min():.1f}, {object_points_local[:,0].max():.1f}], Y=[{object_points_local[:,1].min():.1f}, {object_points_local[:,1].max():.1f}]")

            # Use local coordinates for calibration
            image_points = camera_gcp_rows[['col_sample', 'row_sample']].values.astype(np.float64)
            objpoints = [object_points_local.reshape(1, -1, 3).astype(np.float64)]
            imgpoints = [image_points.reshape(1, -1, 2).astype(np.float64)]

            # Get image dimensions
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            image_size = (w, h)
            n_gcps = len(camera_gcp_rows)
            logger.debug(f"Image size: {w}x{h}")

            # Run constrained calibration
            logger.info("Running constrained optimization with scipy...")
            K, D, rvec, tvec, rms, _ = calibrate_fisheye_with_bounds(
                objpoints, imgpoints, image_size, bounds_config
            )

            logger.info(f"RMS reprojection error: {rms:.4f} pixels")
            logger.info(f"  Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
            logger.info(f"  Distortion: k1={D[0,0]:.4f}, k2={D[1,0]:.4f}, k3={D[2,0]:.4f}, k4={D[3,0]:.4f}")

            # Validate calibration parameters
            is_valid, warnings = validate_calibration_params(
                K, D, rms, image_size, validation_config
            )

            if is_valid:
                logger.info("✓ Calibration PASSED validation")
                valid_calibrations[camera_id] = {'K': K, 'D': D}
            else:
                logger.warning(f"✗ Calibration FAILED validation:")
                for warning in warnings:
                    logger.warning(f"    - {warning}")
                failed_cameras.append(camera_id)

            # Store initial calibration (may be replaced in pass 2)
            # NOTE: tvec is in LOCAL coordinates (relative to local_origin)
            calibration_results[camera_id] = {
                'K': K,
                'D': D,
                'rvec': rvec,
                'tvec': tvec,  # Translation in LOCAL coordinates
                'rms': rms,
                'image_size': image_size,
                'n_gcps': n_gcps,
                'calibration_date': calibration_date,
                'recalibrated': False,
                'recalibration_mode': '',
                'gcps_skipped': 0,
                'gcp_pixel_coords': [],
                'local_origin_x': local_origin[0],
                'local_origin_y': local_origin[1],
                'local_origin_z': local_origin[2],
                'model_crs': model_crs
            }

        except Exception as e:
            logger.error(f"Error calibrating {camera_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_cameras.append(camera_id)
            continue

    # PASS 2: Re-calibrate failed cameras using average intrinsics
    if failed_cameras and valid_calibrations:
        logger.info("")
        logger.info("="*80)
        logger.info(f"PASS 2: Re-calibrating {len(failed_cameras)} failed cameras with average intrinsics")
        logger.info("="*80)

        # Compute average intrinsics from valid calibrations
        K_avg, D_avg = compute_average_intrinsics(valid_calibrations)
        logger.info(f"Average intrinsics from {len(valid_calibrations)} valid cameras:")
        logger.info(f"  fx_avg={K_avg[0,0]:.1f}, fy_avg={K_avg[1,1]:.1f}")
        logger.info(f"  k1_avg={D_avg[0,0]:.4f}, k2_avg={D_avg[1,0]:.4f}")

        for camera_id in failed_cameras:
            if camera_id not in camera_data_map:
                logger.warning(f"Camera {camera_id} not in camera_data_map - skipping")
                continue

            logger.info(f"\nRe-calibrating {camera_id} with average initial guess...")
            image_path, camera_gcp_rows = camera_data_map[camera_id]

            try:
                # Prepare data
                # Load absolute GCP coordinates and convert to local
                object_points_absolute = camera_gcp_rows[['X', 'Y', 'Z']].values.astype(np.float64)
                object_points_local = object_points_absolute - local_origin

                image_points = camera_gcp_rows[['col_sample', 'row_sample']].values.astype(np.float64)
                objpoints = [object_points_local.reshape(1, -1, 3).astype(np.float64)]
                imgpoints = [image_points.reshape(1, -1, 2).astype(np.float64)]

                img = cv2.imread(str(image_path))
                h, w = img.shape[:2]
                image_size = (w, h)
                n_gcps = len(camera_gcp_rows)

                # Re-run calibration with average as initial guess
                K, D, rvec, tvec, rms, _ = calibrate_fisheye_with_bounds(
                    objpoints, imgpoints, image_size, bounds_config,
                    initial_K=K_avg, initial_D=D_avg
                )

                logger.info(f"  RMS={rms:.4f}px, fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

                # Validate again
                is_valid, warnings = validate_calibration_params(
                    K, D, rms, image_size, validation_config
                )

                if is_valid:
                    logger.info("  ✓ Success after re-calibration")
                    # Update calibration results
                    calibration_results[camera_id].update({
                        'K': K,
                        'D': D,
                        'rvec': rvec,
                        'tvec': tvec,
                        'rms': rms
                    })
                else:
                    logger.warning("  ✗ Still failed validation - keeping original")
                    for warning in warnings:
                        logger.warning(f"    - {warning}")

            except Exception as e:
                logger.error(f"  Error during re-calibration: {e}")
                import traceback
                logger.error(traceback.format_exc())
    elif failed_cameras and not valid_calibrations:
        logger.warning("")
        logger.warning("ALL cameras failed validation - cannot compute average intrinsics!")
        logger.warning("Proceeding with original calibrations (may be unreasonable)")
    elif not failed_cameras:
        logger.info("")
        logger.info("All cameras passed validation - skipping PASS 2")

    # Now generate orthorectified outputs for all cameras
    logger.info("")
    logger.info("="*80)
    logger.info("Generating orthorectified outputs")
    logger.info("="*80)

    for camera_id in sorted(calibration_results.keys()):
        if camera_id not in camera_data_map:
            logger.warning(f"{camera_id} not in camera_data_map - skipping output generation")
            continue

        logger.info("")
        logger.info(f"Processing outputs for {camera_id}...")
        image_path, camera_gcp_rows = camera_data_map[camera_id]
        calib = calibration_results[camera_id]

        try:
            # Load image
            img = cv2.imread(str(image_path))

            # Save undistorted image for QC
            if save_undistorted:
                logger.info("  Undistorting for QC...")
                undistorted = undistort_fisheye(img, calib['K'], calib['D'], balance=0.0)
                undist_path = undistorted_dir / f"{camera_id}_undistorted.tif"
                cv2.imwrite(str(undist_path), undistorted)
                logger.info(f"  Saved undistorted: {undist_path.name}")

            # Compute camera-specific orthorectification parameters
            logger.info("  Computing orthorectification parameters...")
            width, height, geotransform = create_orthorectification_params(
                camera_gcp_rows, resolution, padding_meters, local_origin=local_origin
            )
            logger.debug(f"  Output size: {width}x{height}")

            # Load DEM for this camera's view
            logger.info("  Loading DEM...")
            if use_fast_resample:
                dem_array = load_dem_from_tiff_resampled(
                    dem_path, width, height, geotransform
                )
            else:
                dem_array = load_dem_from_tiff(
                    dem_path, width, height, geotransform
                )

            # Create lookup tables using DEM
            logger.info("  Creating lookup tables with DEM...")
            map_x, map_y = create_ortho_lookup_tables_with_dem(
                calib['K'], calib['D'], calib['rvec'], calib['tvec'],
                width, height, geotransform, dem_array, local_origin=local_origin
            )

            # Orthorectify
            logger.info("  Orthorectifying...")
            ortho_img = orthorectify_with_lookup(img, map_x, map_y)

            # Save orthorectified image as GeoTIFF
            ortho_path = ortho_dir / f"{camera_id}_ortho.tif"
            save_geotiff(ortho_img, geotransform, ortho_path, crs=model_crs)
            logger.info(f"  Saved orthorectified image: {ortho_path.name}")

            # Update calibration results with output parameters
            calib['geotransform'] = geotransform
            calib['output_width'] = width
            calib['output_height'] = height

            # Save ortho cache separately (default to hires for calibration)
            cache_dir = output_path.parent / 'orthorectification' / 'ortho_cache'
            cache_data = {
                'map_x': map_x,
                'map_y': map_y,
                'output_width': width,
                'output_height': height
            }
            save_ortho_cache(
                camera_id,
                calib['K'], calib['D'], calib['rvec'], calib['tvec'],
                geotransform, resolution, 'hires',
                cache_data, cache_dir=str(cache_dir)
            )
            logger.info(f"  Saved ortho cache (hires, {resolution}m/pixel)")

        except Exception as e:
            logger.error(f"Error generating outputs for {camera_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Save calibration data as CSV
    csv_file = output_path / f'camera_calibrations_{calibration_date}.csv'
    save_camera_calibrations(calibration_results, csv_file)
    logger.info("")
    logger.info(f"Saved calibration CSV: {csv_file}")

    logger.info("")
    logger.info("="*80)
    logger.info("CALIBRATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total cameras processed: {len(calibration_results)}")
    logger.info(f"Cameras passed validation: {len(valid_calibrations)}")
    logger.info(f"Cameras failed validation: {len(failed_cameras)}")
    logger.info("")

    for cam_id, results in sorted(calibration_results.items()):
        logger.info(f"{cam_id}:")
        logger.info(f"  RMS error: {results['rms']:.4f} px")
        logger.info(f"  GCPs: {results['n_gcps']}")
        logger.info(f"  fx={results['K'][0,0]:.1f}, fy={results['K'][1,1]:.1f}")
        logger.info(f"  k2={results['D'][1,0]:.4f}, k3={results['D'][2,0]:.4f}, k4={results['D'][3,0]:.4f}")
        if cam_id in failed_cameras:
            logger.info(f"  Status: ⚠ FAILED VALIDATION")
        else:
            logger.info(f"  Status: ✓ PASSED")

    logger.info("")
    logger.info("="*80)
    logger.info("CALIBRATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Calibration CSV: {csv_file}")
    logger.info(f"Orthorectified images: {ortho_dir}")
    if save_undistorted:
        logger.info(f"Undistorted images: {undistorted_dir}")

    # Close logging handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    return calibration_results


# Fast processing of new images using saved calibration
def process_new_images_fast(new_image_dir, calibration_file, output_dir='new_ortho',
                            save_undistorted=True, dem_path=None, cache_dir='orthorectification/ortho_cache',
                            resolution=None, resolution_name='hires'):
    """
    Quickly process new images using pre-computed calibration and cached ortho lookup tables.

    This function now uses the new CSV + cache format. It will:
    1. Load calibration parameters from CSV
    2. Check for ortho cache files (map_x, map_y)
    3. Regenerate cache if missing or outdated
    4. Process images using cached lookup tables

    Parameters:
    - new_image_dir: directory with images to process
    - calibration_file: path to camera_calibrations_YYYYMMDD.csv
    - output_dir: where to save outputs
    - save_undistorted: whether to save undistorted images for QC
    - dem_path: path to DEM file (required if cache needs regeneration)
    - cache_dir: directory containing ortho cache files
    - resolution: output resolution in meters/pixel (if None, uses geotransform from calibration)
    - resolution_name: resolution name for cache file suffix ('hires' or 'lowres')
    """
    print("Loading calibration from CSV...")
    calibrations = load_camera_calibrations(calibration_file)

    # PRE-GENERATE any missing cache files before parallel processing
    # This prevents multiple processes from regenerating the same cache
    if dem_path:
        print(f"\nChecking ortho cache files ({resolution_name}: {resolution}m/pixel)...")
        missing_caches = []
        for camera_id, calib in calibrations.items():
            # Use specified resolution or fall back to calibration geotransform
            res = resolution if resolution is not None else abs(calib['geotransform']['pixel_width'])

            # Create geotransform for this resolution
            calib_res = abs(calib['geotransform']['pixel_width'])
            if abs(res - calib_res) > 1e-6:
                # Resolution differs - update geotransform
                geotransform = calib['geotransform'].copy()
                geotransform['pixel_width'] = res
                geotransform['pixel_height'] = -res
            else:
                # Use calibration geotransform as-is
                geotransform = calib['geotransform']

            cache_data = load_ortho_cache(
                camera_id,
                calib['K'], calib['D'], calib['rvec'], calib['tvec'],
                geotransform, res, resolution_name,
                cache_dir=cache_dir
            )
            if cache_data is None:
                missing_caches.append((camera_id, calib, res, geotransform))

        if missing_caches:
            print(f"Found {len(missing_caches)} cameras with missing cache files")
            print("Generating cache files (this may take a few minutes)...\n")

            for camera_id, calib, res, geotransform in missing_caches:
                print(f"  Generating cache for {camera_id} at {res}m/pixel...")

                # If resolution differs from calibration, need to recalculate output dimensions
                calib_res = abs(calib['geotransform']['pixel_width'])
                if abs(res - calib_res) > 1e-6:
                    # Resolution changed - recalculate output dimensions
                    # Scale factor: how much smaller/larger is new resolution
                    scale_factor = calib_res / res
                    width = int(calib['output_width'] * scale_factor)
                    height = int(calib['output_height'] * scale_factor)
                    print(f"    Adjusted dimensions: {width}x{height} (from {calib['output_width']}x{calib['output_height']})")
                else:
                    # Using calibration resolution
                    width = calib['output_width']
                    height = calib['output_height']

                # Load DEM
                dem_array = load_dem_from_tiff(dem_path, width, height, geotransform)

                # Extract local origin from calibration
                local_origin = np.array([
                    calib.get('local_origin_x', 0.0),
                    calib.get('local_origin_y', 0.0),
                    calib.get('local_origin_z', 0.0)
                ], dtype=np.float64)

                # Create lookup tables
                map_x, map_y = create_ortho_lookup_tables_with_dem(
                    calib['K'], calib['D'], calib['rvec'], calib['tvec'],
                    width, height, geotransform, dem_array, local_origin=local_origin
                )

                # Save cache
                cache_data = {
                    'map_x': map_x,
                    'map_y': map_y,
                    'output_width': width,
                    'output_height': height
                }
                save_ortho_cache(
                    camera_id,
                    calib['K'], calib['D'], calib['rvec'], calib['tvec'],
                    geotransform, res, resolution_name,
                    cache_data, cache_dir=cache_dir
                )
                print(f"    OK: Cache saved for {camera_id}")

            print(f"\nCache generation complete! All {len(missing_caches)} cache files created.\n")
        else:
            print("  All cache files present\n")

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    if save_undistorted:
        undistorted_dir = output_path / 'undistorted'
        undistorted_dir.mkdir(exist_ok=True)
    ortho_dir = output_path / 'orthorectified'
    ortho_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(Path(new_image_dir).glob('*.tif*'))
    print(f"Found {len(image_files)} images to process")
    print(f"Available calibrations: {', '.join(calibrations.keys())}\n")

    processed_count = 0
    skipped_count = 0

    for img_path in image_files:
        # Identify camera from filename
        camera_id = None
        for cam_id in calibrations.keys():
            if cam_id in str(img_path.name):
                camera_id = cam_id
                break

        if camera_id is None:
            print(f"WARNING: SKIPPING {img_path.name} - no matching calibration")
            print(f"  Available calibrations: {list(calibrations.keys())}")
            skipped_count += 1
            continue
        
        print(f"Processing {img_path.name} with {camera_id} calibration")

        # Get calibration for this camera
        calib = calibrations[camera_id]

        # Use specified resolution or fall back to calibration geotransform
        res = resolution if resolution is not None else abs(calib['geotransform']['pixel_width'])

        # Create geotransform for this resolution
        calib_res = abs(calib['geotransform']['pixel_width'])
        if abs(res - calib_res) > 1e-6:
            # Resolution differs - update geotransform
            geotransform = calib['geotransform'].copy()
            geotransform['pixel_width'] = res
            geotransform['pixel_height'] = -res
        else:
            # Use calibration geotransform as-is
            geotransform = calib['geotransform']

        # Load ortho cache (should exist after pre-generation)
        cache_data = load_ortho_cache(
            camera_id,
            calib['K'], calib['D'], calib['rvec'], calib['tvec'],
            geotransform, res, resolution_name,
            cache_dir=cache_dir
        )

        if cache_data is None:
            print(f"  ERROR: Cache missing for {camera_id} despite pre-generation. Skipping.")
            skipped_count += 1
            continue

        # Load image
        img = cv2.imread(str(img_path))

        # Save undistorted for QC
        if save_undistorted:
            undistorted = undistort_fisheye(img, calib['K'], calib['D'], balance=0.0)
            undist_path = undistorted_dir / f"{img_path.stem}_undistorted.tif"
            cv2.imwrite(str(undist_path), undistorted)
            print(f"  Undistorted: {undist_path.name}")

        # Orthorectify (FAST! Uses pre-computed lookup tables from cache)
        ortho_img = orthorectify_with_lookup(img, cache_data['map_x'], cache_data['map_y'])

        # Save as GeoTIFF with correct geotransform for this resolution
        # Include resolution in filename for clarity
        # Format: 2.5mm -> 2_5mm, 10mm -> 10mm (preserves decimal precision)
        res_mm_value = res * 1000
        if res_mm_value % 1 == 0:
            # Integer value (e.g., 10.0 -> "10mm")
            res_str = f"{int(res_mm_value)}mm"
        else:
            # Decimal value (e.g., 2.5 -> "2_5mm")
            res_str = f"{res_mm_value:.10g}".replace('.', '_') + "mm"
        ortho_path = ortho_dir / f"{img_path.stem}_ortho_{res_str}.tif"
        save_geotiff(ortho_img, geotransform, ortho_path)
        print(f"  Orthorectified: {ortho_path.name}\n")
        processed_count += 1

    # Print summary
    print("="*60)
    print(f"Processing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print("="*60)

    if processed_count == 0:
        print("\nERROR: No images were processed!")
        print("Possible causes:")
        print("  1. Camera names in images don't match calibration file")
        print("  2. Wrong calibration file specified")
        print("  3. Calibration file uses different camera naming")
        import sys
        sys.exit(1)


# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fisheye camera calibration and orthorectification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial calibration (one-time, slow):
  python undistort_and_orthorectify.py calibrate -g GCP_merged.csv -i images/ -d dem.tif -o output/
  
  # Process new images (fast):
  python undistort_and_orthorectify.py process -i new_images/ -c output/camera_calibrations.pkl -o new_ortho/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Calibrate command
    calib_parser = subparsers.add_parser('calibrate', help='Calibrate cameras (one-time)')
    calib_parser.add_argument('-g', '--gcp-file', required=True, help='GCP CSV file')
    calib_parser.add_argument('-i', '--image-dir', required=True, help='Directory with calibration images')
    calib_parser.add_argument('-d', '--dem', required=True, help='DEM TIFF file')
    calib_parser.add_argument('-o', '--output', default='output', help='Output directory (default: output)')
    calib_parser.add_argument('-r', '--resolution', type=float, default=0.005, 
                            help='Output resolution in m/pixel (default: 0.005)')
    calib_parser.add_argument('-p', '--padding', type=float, default=0.5,
                            help='Padding around GCPs in meters (default: 0.5)')
    calib_parser.add_argument('--no-undistorted', action='store_true',
                            help='Skip saving undistorted images')
    calib_parser.add_argument('--fast-resample', action='store_true',
                            help='Use fast DEM resampling (less accurate)')
    
    # Process command
    proc_parser = subparsers.add_parser('process', help='Process new images using saved calibration')
    proc_parser.add_argument('-i', '--image-dir', required=True, help='Directory with new images')
    proc_parser.add_argument('-c', '--calibration', default='output/camera_calibrations.pkl',
                           help='Calibration file (default: output/camera_calibrations.pkl)')
    proc_parser.add_argument('-o', '--output', default='new_ortho',
                           help='Output directory (default: new_ortho)')
    proc_parser.add_argument('-d', '--dem', help='DEM TIFF file (required for cache regeneration if cache missing)')
    proc_parser.add_argument('--cache-dir', default='orthorectification/ortho_cache',
                           help='Ortho cache directory (default: orthorectification/ortho_cache)')
    proc_parser.add_argument('--no-undistorted', action='store_true',
                           help='Skip saving undistorted images (faster)')
    proc_parser.add_argument('-r', '--resolution', type=float, default=None,
                           help='Output resolution in m/pixel (default: use calibration resolution)')
    proc_parser.add_argument('--resolution-name', choices=['hires', 'lowres'], default='hires',
                           help='Resolution name for cache files (default: hires)')
    
    args = parser.parse_args()
    
    if args.command == 'calibrate':
        calibrate_all_cameras(
            gcp_file=args.gcp_file,
            image_dir=args.image_dir,
            dem_path=args.dem,
            resolution=args.resolution,
            padding_meters=args.padding,
            output_dir=args.output,
            save_undistorted=not args.no_undistorted,
            use_fast_resample=args.fast_resample
        )
    
    elif args.command == 'process':
        process_new_images_fast(
            new_image_dir=args.image_dir,
            calibration_file=args.calibration,
            output_dir=args.output,
            save_undistorted=not args.no_undistorted,
            dem_path=args.dem,
            cache_dir=args.cache_dir,
            resolution=args.resolution,
            resolution_name=args.resolution_name
        )
    
    else:
        parser.print_help()

# python undistort_and_orthorectify.py calibrate -g inputs/GCPs_csepick.csv -i inputs/IR_concurrent_with_lidar -d inputs/lidar_DSM_filled_cropped.tif

    # gcp_file = './GCP_merged.csv'
    # image_dir = r'C:\Users\RDCRLCSE\Documents\FileCloud\My Files\Soo Locks\Ice Management Model Project\technical\GIS\stitch_tests\images'