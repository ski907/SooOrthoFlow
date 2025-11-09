import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import rasterio
from rasterio.transform import rowcol
import argparse

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


def create_orthorectification_params(camera_gcps, resolution=0.005, padding_meters=0.5):
    """
    Compute parameters for orthorectification for a SPECIFIC camera
    resolution: meters per pixel in output (0.005 = 5mm)
    """
    x_min, x_max, y_min, y_max = compute_camera_specific_bounds(camera_gcps, padding_meters)
    
    # Compute output image size
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    
    print(f"  Output bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"  Output size: {width} x {height} pixels @ {resolution*1000:.1f}mm/pixel")
    
    # Create world file parameters (geotransform)
    geotransform = {
        'x_min': x_min,
        'y_max': y_max,
        'pixel_width': resolution,
        'pixel_height': -resolution,
        'rotation_x': 0,
        'rotation_y': 0
    }
    
    return width, height, geotransform


def load_dem_from_tiff(dem_path, width, height, geotransform, nodata_value=None):
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


def create_ortho_lookup_tables_with_dem(K, D, rvec, tvec, width, height, 
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


def orthorectify_with_lookup(img, map_x, map_y):
    """
    Fast orthorectification using pre-computed lookup tables
    """
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def save_with_worldfile(img, geotransform, output_path):
    """
    Save image with world file (no CRS needed for arbitrary coordinates)
    """
    cv2.imwrite(str(output_path), img)
    
    # Save world file (.tfw)
    world_file = Path(str(output_path).rsplit('.', 1)[0] + '.tfw')
    with open(world_file, 'w') as f:
        f.write(f"{geotransform['pixel_width']}\n")
        f.write(f"{geotransform['rotation_x']}\n")
        f.write(f"{geotransform['rotation_y']}\n")
        f.write(f"{geotransform['pixel_height']}\n")
        f.write(f"{geotransform['x_min']}\n")
        f.write(f"{geotransform['y_max']}\n")
    
    print(f"  Saved: {output_path}")
    print(f"  World file: {world_file}")


# Main workflow
def calibrate_all_cameras(gcp_file, image_dir, dem_path, resolution=0.005, 
                         padding_meters=0.5, output_dir='output', 
                         save_undistorted=True, use_fast_resample=False):
    """
    Calibrate all cameras and create orthorectified outputs using actual DEM
    
    Parameters:
    - gcp_file: CSV file with GCP correspondences
    - image_dir: directory with camera images
    - dem_path: path to LiDAR DEM TIFF file
    - resolution: output resolution in meters per pixel (0.005 = 5mm)
    - padding_meters: extra space around GCP bounds
    - output_dir: where to save outputs
    - save_undistorted: whether to save simple undistorted images for QC
    - use_fast_resample: if True, use rasterio's resampling (faster but potentially less accurate)
    """
    # Load GCP data
    gcp_data = pd.read_csv(gcp_file)
    
    # Get unique camera IDs
    #gcp_data['camera_id'] = gcp_data['image_name'].str.extract(r'(ch\d+)')
    gcp_data['camera_id'] = gcp_data['camera_name']
    camera_ids = gcp_data['camera_id'].unique()
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    if save_undistorted:
        undistorted_dir = output_path / 'undistorted'
        undistorted_dir.mkdir(exist_ok=True)
    ortho_dir = output_path / 'orthorectified'
    ortho_dir.mkdir(exist_ok=True)
    
    calibration_results = {}
    
    for camera_id in sorted(camera_ids):
        print(f"\n{'='*60}")
        print(f"Processing {camera_id}")
        print(f"{'='*60}")
        
        # Find the image file for this camera
        image_files = list(Path(image_dir).glob(f"*{camera_id}*.tif*"))
        if not image_files:
            print(f"Warning: No image found for {camera_id}")
            continue
            
        image_path = image_files[0]
        print(f"Image: {image_path.name}")
        
        try:
            # Calibrate
            K, D, rvec, tvec, rms, image_size, camera_gcps = calibrate_fisheye_camera(
                gcp_data, image_path, camera_id
            )
            
            n_gcps = len(camera_gcps)
            print(f"Using {n_gcps} GCPs")
            
            # Load original image
            img = cv2.imread(str(image_path))
            
            # Save undistorted image for QC
            if save_undistorted:
                print("\nUndistorting for QC...")
                undistorted = undistort_fisheye(img, K, D, balance=0.0)
                undist_path = undistorted_dir / f"{camera_id}_undistorted.tif"
                cv2.imwrite(str(undist_path), undistorted)
                print(f"  Saved undistorted: {undist_path}")
            
            # Compute camera-specific orthorectification parameters
            print("\nComputing orthorectification parameters...")
            
            width, height, geotransform = create_orthorectification_params(
                camera_gcps, resolution, padding_meters
            )
            
            # Load DEM for this camera's view
            print("\nLoading DEM...")
            if use_fast_resample:
                dem_array = load_dem_from_tiff_resampled(
                    dem_path, width, height, geotransform
                )
            else:
                dem_array = load_dem_from_tiff(
                    dem_path, width, height, geotransform
                )
            
            # Create lookup tables using DEM
            print("\nCreating lookup tables with DEM...")
            map_x, map_y = create_ortho_lookup_tables_with_dem(
                K, D, rvec, tvec, width, height, geotransform, dem_array
            )
            
            # Orthorectify
            print("\nOrthorectifying...")
            ortho_img = orthorectify_with_lookup(img, map_x, map_y)
            
            # Save orthorectified image with world file
            ortho_path = ortho_dir / f"{camera_id}_ortho.tif"
            save_with_worldfile(ortho_img, geotransform, ortho_path)

            cam_calib_file = output_path / f'{camera_id}_calibration.pkl'
            cam_calibration_results = {
                'K': K,
                'D': D,
                'rvec': rvec,
                'tvec': tvec,
                'rms': rms,
                'image_size': image_size,
                'n_gcps': n_gcps,
                'geotransform': geotransform,
                'dem_array': dem_array,
                'map_x': map_x,
                'map_y': map_y,
                'output_width': width,
                'output_height': height
            }
            with open(cam_calib_file, 'wb') as f:
                pickle.dump(cam_calibration_results, f)
            print(f"\nSaved calibration data: {cam_calib_file}")

            
            # Store results for future use
            calibration_results[camera_id] = {
                'K': K,
                'D': D,
                'rvec': rvec,
                'tvec': tvec,
                'rms': rms,
                'image_size': image_size,
                'n_gcps': n_gcps,
                'geotransform': geotransform,
                'dem_array': dem_array,
                'map_x': map_x,
                'map_y': map_y,
                'output_width': width,
                'output_height': height
            }
            
        except Exception as e:
            print(f"Error processing {camera_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save calibration data
    calib_file = output_path / 'camera_calibrations.pkl'
    with open(calib_file, 'wb') as f:
        pickle.dump(calibration_results, f)
    print(f"\nSaved calibration data: {calib_file}")
    
    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    for cam_id, results in calibration_results.items():
        print(f"{cam_id}:")
        print(f"  RMS error: {results['rms']:.4f} px")
        print(f"  GCPs: {results['n_gcps']}")
        print(f"  Output: {results['output_width']}x{results['output_height']} pixels")
        print(f"  Resolution: {results['geotransform']['pixel_width']*1000:.1f} mm/pixel")
        print(f"  DEM range: Z=[{results['dem_array'].min():.3f}, {results['dem_array'].max():.3f}]")
    
    return calibration_results


# Fast processing of new images using saved calibration
def process_new_images_fast(new_image_dir, calibration_file, output_dir='new_ortho', 
                            save_undistorted=True):
    """
    Quickly process new images using pre-computed calibration and DEM
    """
    print("Loading calibration...")
    with open(calibration_file, 'rb') as f:
        calibrations = pickle.load(f)
    
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
    print(f"Found {len(image_files)} images to process\n")
    
    for img_path in image_files:
        # Identify camera from filename
        camera_id = None
        for cam_id in calibrations.keys():
            if cam_id in str(img_path):
                camera_id = cam_id
                break
        
        if camera_id is None:
            print(f"Skipping {img_path.name} - no matching calibration")
            continue
        
        print(f"Processing {img_path.name} with {camera_id} calibration")
        
        # Get calibration for this camera
        calib = calibrations[camera_id]
        
        # Load image
        img = cv2.imread(str(img_path))
        
        # Save undistorted for QC
        if save_undistorted:
            undistorted = undistort_fisheye(img, calib['K'], calib['D'], balance=0.0)
            undist_path = undistorted_dir / f"{img_path.stem}_undistorted.tif"
            cv2.imwrite(str(undist_path), undistorted)
            print(f"  Undistorted: {undist_path.name}")
        
        # Orthorectify (FAST! Uses pre-computed lookup tables)
        ortho_img = orthorectify_with_lookup(img, calib['map_x'], calib['map_y'])
        
        # Save with world file
        ortho_path = ortho_dir / f"{img_path.stem}_ortho.tif"
        save_with_worldfile(ortho_img, calib['geotransform'], ortho_path)
        print(f"  Orthorectified: {ortho_path.name}\n")


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
    proc_parser.add_argument('--no-undistorted', action='store_true',
                           help='Skip saving undistorted images (faster)')
    
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
            save_undistorted=not args.no_undistorted
        )
    
    else:
        parser.print_help()

# python undistort_and_orthorectify.py calibrate -g inputs/GCPs_csepick.csv -i inputs/IR_concurrent_with_lidar -d inputs/lidar_DSM_filled_cropped.tif

    # gcp_file = './GCP_merged.csv'
    # image_dir = r'C:\Users\RDCRLCSE\Documents\FileCloud\My Files\Soo Locks\Ice Management Model Project\technical\GIS\stitch_tests\images'