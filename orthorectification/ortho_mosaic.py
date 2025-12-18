import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import argparse
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rasterio_mask
from rasterio.features import rasterize as rasterio_rasterize
from rasterio.transform import from_bounds
import fiona
import json
import os

def read_world_file(world_file_path):
    """
    Read affine transformation parameters from a world file.

    World file format (6 lines):
    Line 1: A - pixel size in x direction (x-scale)
    Line 2: D - rotation term for row
    Line 3: B - rotation term for column
    Line 4: E - pixel size in y direction (y-scale, typically negative)
    Line 5: C - x-coordinate of center of upper-left pixel
    Line 6: F - y-coordinate of center of upper-left pixel

    The affine transform matrix is:
    | A  D  C |
    | B  E  F |
    | 0  0  1 |

    This transforms from pixel coordinates to world coordinates.

    Returns:
        Affine transformation matrix as rasterio Affine object
    """
    from rasterio.transform import Affine

    with open(world_file_path, 'r') as f:
        params = [float(line.strip()) for line in f.readlines()[:6]]

    print(f"  World file parameters:")
    print(f"    Line 1 (A - x pixel size): {params[0]}")
    print(f"    Line 2 (D - row rotation): {params[1]}")
    print(f"    Line 3 (B - col rotation): {params[2]}")
    print(f"    Line 4 (E - y pixel size): {params[3]}")
    print(f"    Line 5 (C - x origin): {params[4]}")
    print(f"    Line 6 (F - y origin): {params[5]}")

    # Create affine transform: Affine(a, b, c, d, e, f)
    # Maps to matrix: [a b c]
    #                 [d e f]
    # World file order is: A, D, B, E, C, F
    # So: a=A, b=D, c=C, d=B, e=E, f=F
    transform = Affine(params[0], params[1], params[4],
                      params[2], params[3], params[5])

    print(f"  Created transform: {transform}")

    return transform


def read_geotiff_transform(tif_path):
    """Read geotransform from GeoTIFF metadata"""
    with rasterio.open(tif_path) as src:
        transform = src.transform

        # Convert rasterio transform to our geotransform dict format
        return {
            'pixel_width': transform.a,     # pixel width (x resolution)
            'rotation_y': transform.b,       # rotation (usually 0)
            'rotation_x': transform.d,       # rotation (usually 0)
            'pixel_height': transform.e,     # pixel height (y resolution, negative)
            'x_min': transform.c,            # x coordinate of upper-left corner
            'y_max': transform.f             # y coordinate of upper-left corner
        }


def get_image_bounds(img_shape, geotransform):
    """Calculate world coordinate bounds of an image"""
    height, width = img_shape[:2]
    
    x_min = geotransform['x_min']
    y_max = geotransform['y_max']
    x_max = x_min + width * geotransform['pixel_width']
    y_min = y_max + height * geotransform['pixel_height']
    
    return x_min, x_max, y_min, y_max


def compute_mosaic_bounds(ortho_dir, resolution_filter=None):
    """
    Compute the overall bounds needed for the mosaic

    Args:
        ortho_dir: Directory containing orthorectified images
        resolution_filter: Optional resolution suffix to filter files (e.g., "2mm", "10mm")
    """
    print("Computing mosaic bounds...")

    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []

    # Look for ortho files (with or without resolution suffix)
    tif_files = list(Path(ortho_dir).glob('*_ortho*.tif'))

    # Filter by resolution if specified
    if resolution_filter:
        tif_files = [f for f in tif_files if f.stem.endswith(f"_{resolution_filter}")]
        print(f"Filtering for resolution: {resolution_filter}")

    if not tif_files:
        raise ValueError(f"No *_ortho*.tif files found in {ortho_dir}" +
                        (f" with resolution {resolution_filter}" if resolution_filter else ""))

    print(f"Found {len(tif_files)} orthorectified GeoTIFF images")

    for tif_path in tif_files:
        # Read image and geotransform from GeoTIFF
        img = cv2.imread(str(tif_path))
        geotransform = read_geotiff_transform(tif_path)
        x_min, x_max, y_min, y_max = get_image_bounds(img.shape, geotransform)

        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)

    return min(x_mins), max(x_maxs), min(y_mins), max(y_maxs)


def generate_zone_map_raster(shapefile_path, mosaic_bounds, resolution, crs='EPSG:26919'):
    """
    Generate a rasterized zone map from shapefile and save to disk for caching.

    Args:
        shapefile_path: Path to the zone map shapefile
        mosaic_bounds: Tuple of (x_min, x_max, y_min, y_max) in model coordinates
        resolution: Pixel resolution in model coordinate units
        crs: CRS for the output raster (default: 'EPSG:26919')

    Returns:
        Tuple of (zone_array, camera_id_to_name, transform)
    """
    shapefile_path = Path(shapefile_path)

    # Include resolution in filename to avoid conflicts
    # Format: 2.5mm -> 2_5mm, 10mm -> 10mm (preserves decimal precision)
    res_mm_value = resolution * 1000
    if res_mm_value % 1 == 0:
        # Integer value (e.g., 10.0 -> "10mm")
        res_str = f"{int(res_mm_value)}mm"
    else:
        # Decimal value (e.g., 2.5 -> "2_5mm")
        res_str = f"{res_mm_value:.10g}".replace('.', '_') + "mm"
    output_tif = shapefile_path.parent / f"{shapefile_path.stem}_{res_str}.tif"
    output_json = shapefile_path.parent / f"{shapefile_path.stem}_{res_str}_lookup.json"

    print(f"\nGenerating zone map raster from {shapefile_path.name} at {res_str}/pixel")
    print(f"  Output: {output_tif}")

    # Read shapefile
    with fiona.open(shapefile_path, 'r') as shp:
        shapes_list = list(shp)
        print(f"  Found {len(shapes_list)} camera zones")

    # Extract unique cameras and create ID mapping
    # Sort by priority (lower = higher priority, will overwrite in rasterization)
    shapes_sorted = sorted(shapes_list, key=lambda f: f['properties']['priority'])

    camera_id_to_name = {}
    camera_name_to_id = {}

    for i, feat in enumerate(shapes_sorted):
        camera_name = feat['properties']['camera_nam']
        camera_id = i + 1  # Start from 1, 0 = no zone
        camera_id_to_name[camera_id] = camera_name
        camera_name_to_id[camera_name] = camera_id
        print(f"    Zone {camera_id}: {camera_name} (priority={feat['properties']['priority']})")

    # Calculate raster dimensions
    x_min, x_max, y_min, y_max = mosaic_bounds
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / abs(resolution))

    print(f"  Raster dimensions: {width} x {height} pixels")
    print(f"  Resolution: {resolution*1000:.2f} mm/pixel")

    # Create transform
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    # Initialize zone array
    zone_array = np.zeros((height, width), dtype=np.uint8)

    # Rasterize each zone in priority order (low priority first, high priority overwrites)
    for feat in shapes_sorted:
        camera_id = camera_name_to_id[feat['properties']['camera_nam']]
        shapes_to_burn = [(feat['geometry'], camera_id)]

        rasterio_rasterize(
            shapes_to_burn,
            out=zone_array,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8
        )

    # Count pixels per zone
    unique, counts = np.unique(zone_array[zone_array > 0], return_counts=True)
    print(f"  Zone pixel counts:")
    for zone_id, count in zip(unique, counts):
        print(f"    Zone {zone_id} ({camera_id_to_name[zone_id]}): {count} pixels")

    # Save as GeoTIFF
    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(zone_array, 1)
        dst.set_band_description(1, 'Camera zone ID (0=no zone)')

    # Save lookup JSON
    with open(output_json, 'w') as f:
        json.dump(camera_id_to_name, f, indent=2)

    print(f"  Saved zone map: {output_tif}")
    print(f"  Saved lookup: {output_json}")

    return zone_array, camera_id_to_name, transform


def load_zone_map_raster(shapefile_path, mosaic_bounds, resolution, crs='EPSG:26919'):
    """
    Load cached zone map raster, or generate if missing or stale.

    Args:
        shapefile_path: Path to the zone map shapefile
        mosaic_bounds: Tuple of (x_min, x_max, y_min, y_max) in model coordinates
        resolution: Pixel resolution in model coordinate units
        crs: CRS for the raster (default: 'EPSG:26919')

    Returns:
        Tuple of (zone_array, camera_id_to_name, transform)
    """
    shapefile_path = Path(shapefile_path)

    # Include resolution in filename to avoid conflicts
    # Format: 2.5mm -> 2_5mm, 10mm -> 10mm (preserves decimal precision)
    res_mm_value = resolution * 1000
    if res_mm_value % 1 == 0:
        # Integer value (e.g., 10.0 -> "10mm")
        res_str = f"{int(res_mm_value)}mm"
    else:
        # Decimal value (e.g., 2.5 -> "2_5mm")
        res_str = f"{res_mm_value:.10g}".replace('.', '_') + "mm"
    output_tif = shapefile_path.parent / f"{shapefile_path.stem}_{res_str}.tif"
    output_json = shapefile_path.parent / f"{shapefile_path.stem}_{res_str}_lookup.json"

    # Check if cache exists and is up-to-date
    regenerate = False

    if not output_tif.exists() or not output_json.exists():
        print(f"Zone map cache not found for {res_str} resolution, generating...")
        regenerate = True
    else:
        # Check if shapefile is newer than cached TIF
        shp_mtime = os.path.getmtime(shapefile_path)
        tif_mtime = os.path.getmtime(output_tif)

        if shp_mtime > tif_mtime:
            print(f"Zone map cache is stale (shapefile modified), regenerating...")
            regenerate = True
        else:
            # Check if bounds match - zone map must cover the mosaic bounds
            with rasterio.open(output_tif) as src:
                cached_transform = src.transform
                cached_height, cached_width = src.height, src.width

                # Calculate cached bounds
                cached_x_min = cached_transform.c
                cached_y_max = cached_transform.f
                cached_x_max = cached_x_min + cached_width * cached_transform.a
                cached_y_min = cached_y_max + cached_height * cached_transform.e

                # Check if bounds are approximately equal (not just if mosaic fits within cache)
                # This detects coordinate shifts from recalibration, not just size changes
                x_min, x_max, y_min, y_max = mosaic_bounds
                tolerance = resolution * 2.0  # 2-pixel tolerance for floating-point errors

                bounds_match = (
                    abs(x_min - cached_x_min) < tolerance and
                    abs(x_max - cached_x_max) < tolerance and
                    abs(y_min - cached_y_min) < tolerance and
                    abs(y_max - cached_y_max) < tolerance
                )

                if not bounds_match:
                    print(f"Zone map cache bounds don't match mosaic bounds:")
                    print(f"  Cached: X=[{cached_x_min:.3f}, {cached_x_max:.3f}], Y=[{cached_y_min:.3f}, {cached_y_max:.3f}]")
                    print(f"  Mosaic:  X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
                    print(f"  Difference: ΔX_min={x_min-cached_x_min:.3f}m, ΔX_max={x_max-cached_x_max:.3f}m, " +
                          f"ΔY_min={y_min-cached_y_min:.3f}m, ΔY_max={y_max-cached_y_max:.3f}m")
                    print(f"  Regenerating zone map...")
                    regenerate = True
                else:
                    print(f"Loading cached zone map from {output_tif.name}")

    if regenerate:
        return generate_zone_map_raster(shapefile_path, mosaic_bounds, resolution, crs)

    # Load cached files
    with rasterio.open(output_tif) as src:
        zone_array = src.read(1)
        transform = src.transform

    with open(output_json, 'r') as f:
        camera_id_to_name = json.load(f)
        # Convert string keys back to integers
        camera_id_to_name = {int(k): v for k, v in camera_id_to_name.items()}

    print(f"  Loaded {len(camera_id_to_name)} camera zones")

    return zone_array, camera_id_to_name, transform


def compute_image_quality_map(img, method='gradient'):
    """
    Compute a quality/cost map for each pixel
    Lower values = better places to put seams
    
    Methods:
    - 'gradient': Use gradient magnitude (high gradient = edges, avoid seams there)
    - 'variance': Use local variance
    - 'combined': Combination of multiple metrics
    """
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if method == 'gradient':
        # Compute gradient magnitude (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1
        cost_map = gradient / (gradient.max() + 1e-10)
        
    elif method == 'variance':
        # Local variance (high variance = texture, avoid seams)
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = mean_sq - mean**2
        
        cost_map = variance / (variance.max() + 1e-10)
        
    elif method == 'combined':
        # Combine gradient and variance
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient_norm = gradient / (gradient.max() + 1e-10)
        
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = mean_sq - mean**2
        variance_norm = variance / (variance.max() + 1e-10)
        
        cost_map = 0.5 * gradient_norm + 0.5 * variance_norm
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return cost_map


def create_seam_carved_mosaic(ortho_dir, output_path, resolution=None,
                              seam_method='gradient', save_seam_map=False,
                              world_file_transform=None,
                              world_transform_resampling='bilinear',
                              world_transform_threads=None,
                              world_transform_memory_mb=512,
                              crs='EPSG:26919',
                              output_crs='EPSG:26917',
                              clip_shapefile=None,
                              keep_intermediate=False,
                              save_downscaled=False,
                              downscaled_resolution=0.25,
                              compress_output=True):
    """
    Create mosaic using seam carving - finds optimal non-blended boundaries

    Strategy:
    1. Process images left-to-right (sorted by X position)
    2. For each new image, find optimal seam in overlap region
    3. Use hard cut at seam (no blending)

    Args:
        world_file_transform: Optional path to world file for coordinate transformation
        clip_shapefile: Optional path to shapefile for clipping the mosaic in model coordinates
        keep_intermediate: If True, keep model-space clipped mosaic; if False, delete after transformation (default: False)
    """

    # Compute overall bounds
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir)
    
    # Get resolution
    tif_files = sorted(Path(ortho_dir).glob('*_ortho*.tif'))
    first_geotransform = read_geotiff_transform(tif_files[0])

    if resolution is None:
        resolution = first_geotransform['pixel_width']

    print(f"\nMosaic resolution: {resolution*1000:.2f} mm/pixel")

    # Compute mosaic dimensions
    mosaic_width = int((x_max - x_min) / resolution)
    mosaic_height = int((y_max - y_min) / abs(resolution))

    print(f"Mosaic size: {mosaic_width} x {mosaic_height} pixels")

    # Initialize mosaic and mask
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
    seam_map = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)  # For visualization

    # Sort images by X position (left to right)
    image_data = []
    for tif_path in tif_files:
        geotransform = read_geotiff_transform(tif_path)
        image_data.append((geotransform['x_min'], tif_path))

    image_data.sort(key=lambda x: x[0])  # Sort by x_min

    print(f"\nProcessing {len(image_data)} GeoTIFF images left-to-right with seam carving...")

    for i, (_, tif_path) in enumerate(image_data, 1):
        print(f"\n[{i}/{len(image_data)}] Processing {tif_path.name}")

        # Load image and geotransform
        img = cv2.imread(str(tif_path))
        geotransform = read_geotiff_transform(tif_path)
        
        # Get image bounds
        img_x_min, img_x_max, img_y_min, img_y_max = get_image_bounds(img.shape, geotransform)
        
        # Convert to mosaic pixel coordinates
        mosaic_col_start = int((img_x_min - x_min) / resolution)
        mosaic_row_start = int((y_max - img_y_max) / abs(resolution))
        mosaic_col_end = mosaic_col_start + img.shape[1]
        mosaic_row_end = mosaic_row_start + img.shape[0]
        
        # Clip to mosaic bounds
        mosaic_col_start = max(0, mosaic_col_start)
        mosaic_row_start = max(0, mosaic_row_start)
        mosaic_col_end = min(mosaic_width, mosaic_col_end)
        mosaic_row_end = min(mosaic_height, mosaic_row_end)
        
        # Calculate corresponding image region
        img_col_start = max(0, -int((img_x_min - x_min) / resolution))
        img_row_start = max(0, -int((y_max - img_y_max) / abs(resolution)))
        img_col_end = img_col_start + (mosaic_col_end - mosaic_col_start)
        img_row_end = img_row_start + (mosaic_row_end - mosaic_row_start)
        
        # Extract the region
        img_region = img[img_row_start:img_row_end, img_col_start:img_col_end]
        
        # Create mask for valid pixels (non-black)
        valid_mask = np.any(img_region > 0, axis=2).astype(np.uint8)
        
        # Get existing mosaic region
        existing_region = mosaic[mosaic_row_start:mosaic_row_end, 
                                mosaic_col_start:mosaic_col_end]
        existing_mask = mosaic_mask[mosaic_row_start:mosaic_row_end,
                                   mosaic_col_start:mosaic_col_end]
        
        # Find overlap region
        overlap_mask = (existing_mask > 0) & (valid_mask > 0)
        
        if np.any(overlap_mask):
            print(f"  Found overlap - computing optimal seam")
            
            # Compute cost maps for both images in overlap
            cost_new = compute_image_quality_map(img_region, method=seam_method)
            cost_existing = compute_image_quality_map(existing_region, method=seam_method)
            
            # Combined cost (prefer low-gradient areas)
            combined_cost = (cost_new + cost_existing) / 2.0
            
            # Only consider overlap region
            combined_cost[~overlap_mask] = 0
            
            # Find vertical seam using dynamic programming
            seam_mask = find_optimal_seam_vertical(combined_cost, overlap_mask)
            
            # Apply seam: existing image on left, new image on right
            use_new = seam_mask & overlap_mask
            use_existing = overlap_mask & ~seam_mask
            
            # Copy new image where it should be used
            for c in range(3):
                existing_region[:, :, c][use_new] = img_region[:, :, c][use_new]
            
            # Mark seam in seam map
            seam_boundary = find_seam_boundary(seam_mask)
            seam_map[mosaic_row_start:mosaic_row_end,
                    mosaic_col_start:mosaic_col_end][seam_boundary] = 255
            
            print(f"  Seam carved through overlap region")
        
        # Add non-overlapping regions
        non_overlap = valid_mask & (existing_mask == 0)
        
        for c in range(3):
            existing_region[:, :, c][non_overlap] = img_region[:, :, c][non_overlap]
        
        # Update masks
        mosaic_mask[mosaic_row_start:mosaic_row_end,
                   mosaic_col_start:mosaic_col_end] = np.maximum(
            existing_mask, valid_mask
        )
        
        mosaic[mosaic_row_start:mosaic_row_end,
               mosaic_col_start:mosaic_col_end] = existing_region
        
        print(f"  Added to mosaic")
    
    # Save mosaic as GeoTIFF
    print(f"\nSaving mosaic as GeoTIFF to {output_path}")

    from rasterio.transform import from_bounds

    # Convert BGR to RGB
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    mosaic_data = np.transpose(mosaic_rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    # Create geotransform (in model coordinates)
    transform = from_bounds(
        west=x_min,
        south=y_min,
        east=x_max,
        north=y_max,
        width=mosaic_width,
        height=mosaic_height
    )

    # Determine final output path
    if world_file_transform:
        # Save model coords to original location
        temp_path = output_path
        # Save world coords to 'world_coords' subdirectory
        output_dir = Path(output_path).parent
        world_coords_dir = output_dir / 'world_coords'
        world_coords_dir.mkdir(exist_ok=True)
        final_path = world_coords_dir / Path(output_path).name
    else:
        temp_path = output_path
        final_path = output_path

    # Write GeoTIFF in model coordinates
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=mosaic_height,
        width=mosaic_width,
        count=3,
        dtype=mosaic_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mosaic_data)

    print(f"Saved mosaic GeoTIFF: {temp_path}")

    # Clip to shapefile if requested (in model coordinates)
    if clip_shapefile:
        clipped_path = Path(str(temp_path).rsplit('.', 1)[0] + '_clipped.tif')
        clip_raster_to_shapefile(temp_path, clipped_path, clip_shapefile)
        # Always delete the unclipped version (we don't need it)
        Path(temp_path).unlink()
        print(f"  Deleted unclipped mosaic: {temp_path}")
        # Replace temp_path with clipped version for subsequent operations
        temp_path = clipped_path
        # Also update final_path if no world transform (they're the same)
        if not world_file_transform:
            final_path = clipped_path

    # Apply coordinate transformation if requested
    if world_file_transform:
        apply_coordinate_transform(
            temp_path, final_path, world_file_transform,
            resampling_method=world_transform_resampling,
            num_threads=world_transform_threads,
            warp_mem_limit_mb=world_transform_memory_mb,
            output_crs=output_crs
        )
        # Delete the model coordinate clipped version after transformation unless keeping intermediates
        if not keep_intermediate:
            Path(temp_path).unlink()
            print(f"  Deleted model space clipped mosaic: {temp_path}")

        # Create downscaled version if requested (only for world-transformed outputs)
        if save_downscaled and downscaled_resolution:
            downscaled_path = Path(str(final_path).rsplit('.', 1)[0] + f'_downscaled_{int(downscaled_resolution*100)}cm.tif')
            create_downscaled_mosaic(final_path, downscaled_path, downscaled_resolution)

    # Apply LZW compression to final output if requested
    if compress_output:
        print(f"\nApplying LZW compression to final mosaic...")
        compress_geotiff(final_path)
        print(f"  Compressed: {final_path}")

        # Also compress the model-space clipped version if it was kept
        if world_file_transform and keep_intermediate and Path(temp_path).exists():
            print(f"  Compressing intermediate mosaic...")
            compress_geotiff(temp_path)
            print(f"  Compressed: {temp_path}")

    # Save seam map
    if save_seam_map:
        seam_path = Path(str(output_path).rsplit('.', 1)[0] + '_seams.tif')
        cv2.imwrite(str(seam_path), seam_map)
        print(f"Saved seam map: {seam_path}")

    print("\nDone!")
    return mosaic


def find_optimal_seam_vertical(cost_map, overlap_mask):
    """
    Find optimal vertical seam through overlap using dynamic programming
    Returns mask where True = use new image, False = use existing
    """
    height, width = cost_map.shape
    
    # Find left and right bounds of overlap
    overlap_cols = np.any(overlap_mask, axis=0)
    if not np.any(overlap_cols):
        return overlap_mask
    
    left_col = np.argmax(overlap_cols)
    right_col = width - np.argmax(overlap_cols[::-1]) - 1
    
    # Initialize DP table
    dp = np.full((height, width), np.inf)
    
    # Initialize first column of overlap
    for row in range(height):
        if overlap_mask[row, left_col]:
            dp[row, left_col] = cost_map[row, left_col]
    
    # Fill DP table
    for col in range(left_col + 1, right_col + 1):
        for row in range(height):
            if not overlap_mask[row, col]:
                continue
            
            # Check three possible previous positions
            candidates = []
            for prev_row in [row - 1, row, row + 1]:
                if 0 <= prev_row < height and dp[prev_row, col - 1] != np.inf:
                    candidates.append(dp[prev_row, col - 1])
            
            if candidates:
                dp[row, col] = min(candidates) + cost_map[row, col]
    
    # Backtrack to find seam
    seam_cols = np.zeros(height, dtype=int)
    
    # Find minimum in last column
    valid_rows = [r for r in range(height) if dp[r, right_col] != np.inf]
    if not valid_rows:
        # Fallback: split down middle
        middle_col = (left_col + right_col) // 2
        seam_mask = np.zeros_like(overlap_mask)
        seam_mask[:, middle_col:] = True
        return seam_mask
    
    current_row = min(valid_rows, key=lambda r: dp[r, right_col])
    seam_cols[current_row] = right_col
    
    # Backtrack
    for col in range(right_col - 1, left_col - 1, -1):
        # Find best previous row
        best_prev_row = current_row
        best_cost = np.inf
        
        for prev_row in [current_row - 1, current_row, current_row + 1]:
            if 0 <= prev_row < height and dp[prev_row, col] < best_cost:
                best_cost = dp[prev_row, col]
                best_prev_row = prev_row
        
        current_row = best_prev_row
        seam_cols[current_row] = col
    
    # Create mask: everything right of seam uses new image
    seam_mask = np.zeros_like(overlap_mask)
    for row in range(height):
        if overlap_mask[row, left_col]:
            # Find seam column for this row (interpolate if needed)
            seam_col = seam_cols[row]
            seam_mask[row, seam_col:] = True
    
    return seam_mask


def find_seam_boundary(seam_mask):
    """Find the boundary pixels of the seam for visualization"""
    # Dilate and subtract to find edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(seam_mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - seam_mask.astype(np.uint8)
    return boundary > 0


def clip_raster_to_shapefile(input_tif, output_tif, shapefile_path):
    """
    Clip a raster to the boundaries of a shapefile.

    Args:
        input_tif: Path to input GeoTIFF
        output_tif: Path to output clipped GeoTIFF
        shapefile_path: Path to shapefile for clipping
    """
    print(f"\nClipping raster to shapefile: {shapefile_path}")

    # Read the shapefile
    with fiona.open(shapefile_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        print(f"  Found {len(shapes)} polygon(s) in shapefile")

    # Clip the raster
    with rasterio.open(input_tif) as src:
        print(f"  Input raster CRS: {src.crs}")
        print(f"  Input raster bounds: {src.bounds}")

        out_image, out_transform = rasterio_mask(src, shapes, crop=True, all_touched=False)
        out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })

        print(f"  Clipped dimensions: {out_image.shape[2]} x {out_image.shape[1]}")

        # Write clipped raster
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"  Clipped raster saved to: {output_tif}")


def compress_geotiff(input_tif, output_tif=None):
    """
    Compress a GeoTIFF file using LZW compression.

    Args:
        input_tif: Path to input GeoTIFF
        output_tif: Path to output compressed GeoTIFF (if None, overwrites input)
    """
    if output_tif is None:
        output_tif = input_tif
        temp_file = str(input_tif).replace('.tif', '_temp.tif')
    else:
        temp_file = None

    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        meta.update(compress='lzw')

        # Write to temporary file or final output
        write_path = temp_file if temp_file else output_tif
        with rasterio.open(write_path, 'w', **meta) as dst:
            dst.write(src.read())

    # If we used a temp file, replace the original
    if temp_file:
        Path(input_tif).unlink()
        Path(temp_file).rename(input_tif)


def create_downscaled_mosaic(input_tif, output_tif, target_resolution):
    """
    Create a downscaled version of the mosaic at a coarser resolution.

    Args:
        input_tif: Path to input high-resolution GeoTIFF
        output_tif: Path to output downscaled GeoTIFF
        target_resolution: Target pixel size in the same units as the input (e.g., 0.25 for 25cm)
    """
    print(f"\nCreating downscaled mosaic at {target_resolution}m/pixel resolution")

    with rasterio.open(input_tif) as src:
        # Calculate scale factor
        src_resolution = abs(src.transform.a)  # Assume square pixels
        scale_factor = target_resolution / src_resolution

        if scale_factor <= 1.0:
            print(f"  Warning: Target resolution ({target_resolution}m) is finer than source ({src_resolution}m)")
            print(f"  Skipping downscaling")
            return

        # Calculate new dimensions
        new_width = int(src.width / scale_factor)
        new_height = int(src.height / scale_factor)

        print(f"  Source: {src.width}x{src.height} @ {src_resolution:.4f}m/pixel")
        print(f"  Target: {new_width}x{new_height} @ {target_resolution:.4f}m/pixel")
        print(f"  Scale factor: {scale_factor:.2f}x")

        # Calculate new transform
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Read and resample data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            'width': new_width,
            'height': new_height,
            'transform': new_transform
        })

        # Write downscaled raster
        with rasterio.open(output_tif, 'w', **out_meta) as dst:
            dst.write(data)

    print(f"  Downscaled mosaic saved to: {output_tif}")


def crop_nodata_padding(input_tif, output_tif):
    """
    Crop black/nodata padding from a raster by finding the minimum bounding box
    of valid (non-zero) data.

    Args:
        input_tif: Path to input GeoTIFF with padding
        output_tif: Path to output cropped GeoTIFF
    """
    print(f"\nCropping nodata padding from {input_tif}")

    with rasterio.open(input_tif) as src:
        # Read the data
        data = src.read()

        # Find pixels where any band has non-zero values
        valid_mask = np.any(data > 0, axis=0)

        if not np.any(valid_mask):
            print("  Warning: No valid data found, keeping original")
            return

        # Find bounding box of valid data
        rows, cols = np.where(valid_mask)
        row_min, row_max = rows.min(), rows.max() + 1
        col_min, col_max = cols.min(), cols.max() + 1

        print(f"  Original dimensions: {src.width} x {src.height}")
        print(f"  Valid data region: rows {row_min}-{row_max}, cols {col_min}-{col_max}")
        print(f"  Cropped dimensions: {col_max - col_min} x {row_max - row_min}")

        # Crop the data
        cropped_data = data[:, row_min:row_max, col_min:col_max]

        # Calculate new transform
        new_transform = src.transform * rasterio.Affine.translation(col_min, row_min)

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": row_max - row_min,
            "width": col_max - col_min,
            "transform": new_transform
        })

        # Write cropped data
        with rasterio.open(output_tif, "w", **out_meta) as dest:
            dest.write(cropped_data)

    print(f"  Cropped raster saved to: {output_tif}")


def apply_coordinate_transform(input_tif, output_tif, world_file_path,
                               resampling_method='bilinear',
                               num_threads=None,
                               warp_mem_limit_mb=512,
                               output_crs='EPSG:26917',
                               crop_padding=True):
    """
    Apply coordinate transformation from world file to reproject the mosaic.

    IMPORTANT: The world file is assumed to be created from the original reference image
    in QGIS, mapping FROM pixel coordinates of that reference TO world coordinates.
    We apply this transform directly to warp the mosaic image.

    Args:
        input_tif: Path to input GeoTIFF (in model coordinates)
        output_tif: Path to output GeoTIFF (in world coordinates)
        world_file_path: Path to world file with transformation parameters
        resampling_method: Resampling algorithm - 'nearest', 'bilinear', 'cubic', 'lanczos'
                          (default: 'bilinear'). Use 'nearest' for faster processing.
        num_threads: Number of threads for parallel processing (default: auto-detect CPU cores)
        warp_mem_limit_mb: Warp memory limit in MB (default: 512). Increase for better performance
                          on systems with more RAM.
        output_crs: CRS for the output GeoTIFF (default: 'EPSG:26917')
    """
    import multiprocessing
    import math
    from rasterio.transform import from_bounds

    print(f"\nApplying coordinate transformation from {world_file_path}")

    # Set threading - auto-detect CPU cores if not specified
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    # Map resampling methods
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos,
    }
    resampling_algo = resampling_map.get(resampling_method, Resampling.bilinear)

    # Read world file transform (pixel coords -> world coords)
    # This was created in QGIS by georeferencing the original image
    world_file_transform = read_world_file(world_file_path)

    with rasterio.open(input_tif) as src:
        src_crs = src.crs
        src_height, src_width = src.height, src.width
        num_bands = src.count
        src_transform = src.transform

        # Detect the model-space resolution from the source mosaic
        # The source transform tells us the model-space pixel size
        model_pixel_size = abs(src_transform.a)  # Assume square pixels

        # The world file was created by georeferencing a 2.5mm model-space mosaic
        # It maps: pixel coordinates -> world coordinates (UTM)
        # For a different resolution mosaic, we need to compose transforms:
        # 1. Source transform: pixel coords -> model coords (at current resolution)
        # 2. World file transform: pixel coords -> world coords (at reference resolution)

        # The key insight: the world file gives us a direct pixel->world mapping
        # We just need to use the SOURCE transform to tell reproject() where each
        # pixel maps in model space, then it will use that with the world file

        # Actually, we should use the source's model-space transform directly
        # and let the world file handle the pixel->world mapping
        # But the world file expects pixels at 2.5mm resolution

        # Solution: Compose the transforms properly
        # src pixel -> model coords (via src_transform)
        # model coords -> reference pixels (inverse of reference model transform)
        # reference pixels -> world coords (via world file)

        from rasterio.transform import Affine

        reference_resolution = 0.0025  # 2.5mm
        reference_model_transform = Affine(reference_resolution, 0, src_transform.c,
                                           0, -reference_resolution, src_transform.f)

        # Compose: src_transform -> model coords -> reference pixel coords -> world coords
        # src_transform takes us to model coords
        # ~reference_model_transform takes us from model coords to reference pixels
        # world_file_transform takes us from reference pixels to world coords
        pixel_to_world = world_file_transform * ~reference_model_transform * src_transform

        print(f"  Source image dimensions: {src_width} x {src_height}, {num_bands} bands")
        print(f"  Model-space resolution: {model_pixel_size*1000:.2f}mm/pixel")
        print(f"  Using {num_threads} threads, {warp_mem_limit_mb}MB warp memory")
        print(f"  Resampling method: {resampling_method}")
        print(f"  Source transform (model coords): {src_transform}")
        print(f"  Composed pixel-to-world transform: {pixel_to_world}")

        # Calculate the corners in pixel coordinates
        pixel_corners = [
            (0, 0),                    # top-left
            (src_width, 0),            # top-right
            (src_width, src_height),   # bottom-right
            (0, src_height)            # bottom-left
        ]

        # Transform corners to world coordinates using the world file transform
        world_corners = [pixel_to_world * (px, py) for px, py in pixel_corners]
        for (px, py), (wx, wy) in zip(pixel_corners, world_corners):
            print(f"    Pixel ({px}, {py}) -> World ({wx:.2f}, {wy:.2f})")

        # Calculate world bounds
        xs = [c[0] for c in world_corners]
        ys = [c[1] for c in world_corners]
        world_bounds = (min(xs), min(ys), max(xs), max(ys))

        print(f"  World bounds: {world_bounds}")

        # Calculate output dimensions to maintain pixel resolution
        # Optimized: Check for rotation first
        if abs(pixel_to_world.b) < 1e-6 and abs(pixel_to_world.d) < 1e-6:
            # No rotation, use direct values (faster)
            pixel_size_x = abs(pixel_to_world.a)
            pixel_size_y = abs(pixel_to_world.e)
        else:
            # Rotated, use full calculation
            pixel_size_x = math.sqrt(pixel_to_world.a**2 + pixel_to_world.d**2)
            pixel_size_y = math.sqrt(pixel_to_world.b**2 + pixel_to_world.e**2)

        print(f"  Pixel resolution in world coords: {pixel_size_x:.6f} x {pixel_size_y:.6f}")

        out_width = int(round((world_bounds[2] - world_bounds[0]) / pixel_size_x))
        out_height = int(round((world_bounds[3] - world_bounds[1]) / pixel_size_y))

        print(f"  Output dimensions: {out_width} x {out_height}")

        # Create the destination transform for the output image
        dst_transform = from_bounds(
            world_bounds[0], world_bounds[1],
            world_bounds[2], world_bounds[3],
            out_width, out_height
        )

        print(f"  Destination transform: {dst_transform}")

        # Load source data (optimized: read all bands at once)
        src_data = src.read()

        # Create output array
        dst_data = np.zeros((num_bands, out_height, out_width), dtype=src_data.dtype)

        # Reproject with optimizations
        print(f"  Reprojecting {num_bands} bands...")
        warp_mem_bytes = warp_mem_limit_mb * 1024 * 1024

        for band_idx in range(num_bands):
            if band_idx % 2 == 0 or num_bands <= 4:
                print(f"    Band {band_idx + 1}/{num_bands}...")

            reproject(
                source=src_data[band_idx],
                destination=dst_data[band_idx],
                src_transform=pixel_to_world,  # Use world file transform directly
                src_crs=output_crs,
                dst_transform=dst_transform,
                dst_crs=output_crs,
                resampling=resampling_algo,
                num_threads=num_threads,
                warp_mem_limit=warp_mem_bytes
            )

        # Write output with optimized settings
        out_profile = src.profile.copy()
        out_profile.update({
            'transform': dst_transform,
            'width': out_width,
            'height': out_height,
            'crs': output_crs,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        })

        # Write to temporary file if cropping is requested
        temp_output = output_tif if not crop_padding else str(output_tif).replace('.tif', '_temp.tif')

        with rasterio.open(temp_output, 'w', **out_profile) as dst:
            dst.write(dst_data)

    print(f"Transformed mosaic saved to: {temp_output}")

    # Crop padding if requested
    if crop_padding:
        crop_nodata_padding(temp_output, output_tif)
        # Delete the temporary uncropped file
        Path(temp_output).unlink()
        print(f"  Deleted temporary uncropped file: {temp_output}")

    print(f"  Successfully transformed to world coordinates")


def create_mosaic_simple_priority(ortho_dir, output_path, resolution=None,
                                  priority='center', world_file_transform=None,
                                  world_transform_resampling='bilinear',
                                  world_transform_threads=None,
                                  world_transform_memory_mb=512,
                                  crs='EPSG:26919',
                                  output_crs='EPSG:26917',
                                  clip_shapefile=None,
                                  keep_intermediate=False,
                                  save_downscaled=False,
                                  downscaled_resolution=0.25,
                                  compress_output=True,
                                  resolution_filter=None):
    """
    Simple approach: Assign priority to each image, last one wins

    Priority options:
    - 'center': Prefer images based on distance to the centroid of their valid orthorectified pixels
    - 'order': Just use the order of images (e.g., left to right)

    Args:
        world_file_transform: Optional path to world file for coordinate transformation
        clip_shapefile: Optional path to shapefile for clipping the mosaic in model coordinates
        keep_intermediate: If True, keep model-space clipped mosaic; if False, delete after transformation (default: False)
        resolution_filter: Optional resolution suffix to filter files (e.g., "2mm", "10mm")
    """

    # Compute overall bounds (filtering by resolution if specified)
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir, resolution_filter)

    tif_files = sorted(Path(ortho_dir).glob('*_ortho*.tif'))
    # Filter by resolution if specified
    if resolution_filter:
        tif_files = [f for f in tif_files if f.stem.endswith(f"_{resolution_filter}")]
    first_geotransform = read_geotiff_transform(tif_files[0])

    if resolution is None:
        resolution = first_geotransform['pixel_width']

    mosaic_width = int((x_max - x_min) / resolution)
    mosaic_height = int((y_max - y_min) / abs(resolution))

    print(f"Mosaic size: {mosaic_width} x {mosaic_height} pixels")

    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    # For center method, we need to track distance to nearest centroid
    if priority == 'center':
        distance_map = np.full((mosaic_height, mosaic_width), np.inf, dtype=np.float32)
        source_map = np.full((mosaic_height, mosaic_width), -1, dtype=np.int32)
    else:
        priority_map = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)

    print(f"\nCreating mosaic with '{priority}' priority")

    # For center method, collect all image data first to compute global priorities
    if priority == 'center':
        image_data_list = []

        for i, tif_path in enumerate(tif_files, 1):
            print(f"[{i}/{len(tif_files)}] Loading {tif_path.name}")

            img = cv2.imread(str(tif_path))
            geotransform = read_geotiff_transform(tif_path)

            img_x_min, img_x_max, img_y_min, img_y_max = get_image_bounds(img.shape, geotransform)

            mosaic_col_start = max(0, int((img_x_min - x_min) / resolution))
            mosaic_row_start = max(0, int((y_max - img_y_max) / abs(resolution)))
            mosaic_col_end = min(mosaic_width, mosaic_col_start + img.shape[1])
            mosaic_row_end = min(mosaic_height, mosaic_row_start + img.shape[0])

            img_col_start = max(0, -int((img_x_min - x_min) / resolution))
            img_row_start = max(0, -int((y_max - img_y_max) / abs(resolution)))
            img_col_end = img_col_start + (mosaic_col_end - mosaic_col_start)
            img_row_end = img_row_start + (mosaic_row_end - mosaic_row_start)

            img_region = img[img_row_start:img_row_end, img_col_start:img_col_end]

            # Find valid (non-black) pixels
            valid_mask = np.any(img_region > 0, axis=2)

            if np.any(valid_mask):
                # Compute centroid of valid pixels in mosaic coordinates
                valid_rows, valid_cols = np.where(valid_mask)
                centroid_row = mosaic_row_start + np.mean(valid_rows)
                centroid_col = mosaic_col_start + np.mean(valid_cols)

                print(f"  Centroid at mosaic coords: ({centroid_col:.1f}, {centroid_row:.1f})")

                image_data_list.append({
                    'idx': i - 1,
                    'img_region': img_region,
                    'valid_mask': valid_mask,
                    'mosaic_row_start': mosaic_row_start,
                    'mosaic_row_end': mosaic_row_end,
                    'mosaic_col_start': mosaic_col_start,
                    'mosaic_col_end': mosaic_col_end,
                    'centroid_row': centroid_row,
                    'centroid_col': centroid_col,
                    'name': tif_path.name
                })

        # Now assign pixels based on which centroid is closest
        print("\nAssigning pixels based on nearest centroid...")
        for img_data in image_data_list:
            print(f"Processing {img_data['name']}")

            # Get the region in the mosaic for this image
            mosaic_region = mosaic[img_data['mosaic_row_start']:img_data['mosaic_row_end'],
                                  img_data['mosaic_col_start']:img_data['mosaic_col_end']]
            distance_region = distance_map[img_data['mosaic_row_start']:img_data['mosaic_row_end'],
                                          img_data['mosaic_col_start']:img_data['mosaic_col_end']]
            source_region = source_map[img_data['mosaic_row_start']:img_data['mosaic_row_end'],
                                      img_data['mosaic_col_start']:img_data['mosaic_col_end']]

            # Compute distance from each pixel to this image's centroid
            h, w = img_data['img_region'].shape[:2]
            y_coords, x_coords = np.ogrid[:h, :w]

            # Distances in mosaic coordinates
            mosaic_y_coords = img_data['mosaic_row_start'] + y_coords
            mosaic_x_coords = img_data['mosaic_col_start'] + x_coords

            dist_to_centroid = np.sqrt(
                (mosaic_x_coords - img_data['centroid_col'])**2 +
                (mosaic_y_coords - img_data['centroid_row'])**2
            )

            # Only update where this pixel is valid AND closer to this centroid
            valid_mask = img_data['valid_mask']
            update_mask = valid_mask & (dist_to_centroid < distance_region)

            # Update mosaic where this image is closer
            for c in range(3):
                mosaic_region[:, :, c][update_mask] = img_data['img_region'][:, :, c][update_mask]

            distance_region[update_mask] = dist_to_centroid[update_mask]
            source_region[update_mask] = img_data['idx']

            # Update the maps
            mosaic[img_data['mosaic_row_start']:img_data['mosaic_row_end'],
                  img_data['mosaic_col_start']:img_data['mosaic_col_end']] = mosaic_region
            distance_map[img_data['mosaic_row_start']:img_data['mosaic_row_end'],
                        img_data['mosaic_col_start']:img_data['mosaic_col_end']] = distance_region
            source_map[img_data['mosaic_row_start']:img_data['mosaic_row_end'],
                      img_data['mosaic_col_start']:img_data['mosaic_col_end']] = source_region

    else:  # 'order' priority
        for i, tif_path in enumerate(tif_files, 1):
            print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")

            img = cv2.imread(str(tif_path))
            geotransform = read_geotiff_transform(tif_path)

            img_x_min, img_x_max, img_y_min, img_y_max = get_image_bounds(img.shape, geotransform)

            mosaic_col_start = max(0, int((img_x_min - x_min) / resolution))
            mosaic_row_start = max(0, int((y_max - img_y_max) / abs(resolution)))
            mosaic_col_end = min(mosaic_width, mosaic_col_start + img.shape[1])
            mosaic_row_end = min(mosaic_height, mosaic_row_start + img.shape[0])

            img_col_start = max(0, -int((img_x_min - x_min) / resolution))
            img_row_start = max(0, -int((y_max - img_y_max) / abs(resolution)))
            img_col_end = img_col_start + (mosaic_col_end - mosaic_col_start)
            img_row_end = img_row_start + (mosaic_row_end - mosaic_row_start)

            img_region = img[img_row_start:img_row_end, img_col_start:img_col_end]

            # Simple order-based priority
            img_priority = np.ones((img_region.shape[0], img_region.shape[1])) * i

            # Valid pixels
            valid_mask = np.any(img_region > 0, axis=2)
            img_priority = img_priority * valid_mask

            # Update where this image has higher priority
            mosaic_region = mosaic[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end]
            priority_region = priority_map[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end]

            update_mask = (img_priority > priority_region) & valid_mask

            for c in range(3):
                mosaic_region[:, :, c][update_mask] = img_region[:, :, c][update_mask]

            priority_region[update_mask] = img_priority[update_mask]

            mosaic[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end] = mosaic_region
            priority_map[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end] = priority_region
    
    # Save as GeoTIFF
    print(f"\nSaving mosaic as GeoTIFF to {output_path}")

    from rasterio.transform import from_bounds

    # Convert BGR to RGB
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    mosaic_data = np.transpose(mosaic_rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    # Create geotransform (in model coordinates)
    transform = from_bounds(
        west=x_min,
        south=y_min,
        east=x_max,
        north=y_max,
        width=mosaic_width,
        height=mosaic_height
    )

    # Determine final output path
    if world_file_transform:
        # Save model coords to original location
        temp_path = output_path
        # Save world coords to 'world_coords' subdirectory
        output_dir = Path(output_path).parent
        world_coords_dir = output_dir / 'world_coords'
        world_coords_dir.mkdir(exist_ok=True)
        final_path = world_coords_dir / Path(output_path).name
    else:
        temp_path = output_path
        final_path = output_path

    # Write GeoTIFF in model coordinates
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=mosaic_height,
        width=mosaic_width,
        count=3,
        dtype=mosaic_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mosaic_data)

    print(f"Saved mosaic GeoTIFF: {temp_path}")

    # Clip to shapefile if requested (in model coordinates)
    if clip_shapefile:
        clipped_path = Path(str(temp_path).rsplit('.', 1)[0] + '_clipped.tif')
        clip_raster_to_shapefile(temp_path, clipped_path, clip_shapefile)
        # Always delete the unclipped version (we don't need it)
        Path(temp_path).unlink()
        print(f"  Deleted unclipped mosaic: {temp_path}")
        # Replace temp_path with clipped version for subsequent operations
        temp_path = clipped_path
        # Also update final_path if no world transform (they're the same)
        if not world_file_transform:
            final_path = clipped_path

    # Apply coordinate transformation if requested
    if world_file_transform:
        apply_coordinate_transform(
            temp_path, final_path, world_file_transform,
            resampling_method=world_transform_resampling,
            num_threads=world_transform_threads,
            warp_mem_limit_mb=world_transform_memory_mb,
            output_crs=output_crs
        )
        # Delete the model coordinate clipped version after transformation unless keeping intermediates
        if not keep_intermediate:
            Path(temp_path).unlink()
            print(f"  Deleted model space clipped mosaic: {temp_path}")

        # Create downscaled version if requested (only for world-transformed outputs)
        if save_downscaled and downscaled_resolution:
            downscaled_path = Path(str(final_path).rsplit('.', 1)[0] + f'_downscaled_{int(downscaled_resolution*100)}cm.tif')
            create_downscaled_mosaic(final_path, downscaled_path, downscaled_resolution)

    # Apply LZW compression to final output if requested
    if compress_output:
        print(f"\nApplying LZW compression to final mosaic...")
        compress_geotiff(final_path)
        print(f"  Compressed: {final_path}")

        # Also compress the model-space clipped version if it was kept
        if world_file_transform and keep_intermediate and Path(temp_path).exists():
            print(f"  Compressing intermediate mosaic...")
            compress_geotiff(temp_path)
            print(f"  Compressed: {temp_path}")

    return mosaic


def create_mosaic_zone_map(ortho_dir, output_path, zone_map_shapefile, resolution=None,
                           world_file_transform=None,
                           world_transform_resampling='bilinear',
                           world_transform_threads=None,
                           world_transform_memory_mb=512,
                           crs='EPSG:26919',
                           output_crs='EPSG:26917',
                           clip_shapefile=None,
                           keep_intermediate=False,
                           save_downscaled=False,
                           downscaled_resolution=0.25,
                           compress_output=True,
                           resolution_filter=None):
    """
    Create mosaic using spatial zone map from shapefile.
    Each zone exclusively uses its assigned camera (strict mode).

    Args:
        zone_map_shapefile: Path to shapefile defining camera zones
        world_file_transform: Optional path to world file for coordinate transformation
        clip_shapefile: Optional path to shapefile for clipping the mosaic in model coordinates
        keep_intermediate: If True, keep model-space clipped mosaic; if False, delete after transformation
        resolution_filter: Optional resolution suffix to filter files (e.g., "2mm", "10mm")
    """

    # Compute overall bounds (filtering by resolution if specified)
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir, resolution_filter)

    tif_files = sorted(Path(ortho_dir).glob('*_ortho*.tif'))
    # Filter by resolution if specified
    if resolution_filter:
        tif_files = [f for f in tif_files if f.stem.endswith(f"_{resolution_filter}")]
    first_geotransform = read_geotiff_transform(tif_files[0])

    if resolution is None:
        resolution = first_geotransform['pixel_width']

    mosaic_width = int((x_max - x_min) / resolution)
    mosaic_height = int((y_max - y_min) / abs(resolution))

    print(f"\nMosaic size: {mosaic_width} x {mosaic_height} pixels")
    print(f"Mosaic resolution: {resolution*1000:.2f} mm/pixel")

    # Load or generate zone map
    zone_array, camera_id_to_name, zone_transform = load_zone_map_raster(
        zone_map_shapefile,
        (x_min, x_max, y_min, y_max),
        resolution,
        crs
    )

    # Create inverse lookup: camera_name -> camera_id
    camera_name_to_id = {v: k for k, v in camera_id_to_name.items()}

    # Initialize mosaic
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    coverage_map = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)  # Track which pixels were filled

    print(f"\nCreating mosaic using zone map (strict mode)")

    # Process each ortho file
    cameras_found = set()
    cameras_not_in_map = set()

    for i, tif_path in enumerate(tif_files, 1):
        print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")

        # Extract camera name from filename
        # Expected formats:
        #   "NVR1_N910A6_ch1_main_ortho.tif" (legacy)
        #   "NVR1_N910A6_ch1_main_ortho_25mm.tif" (with resolution)
        ortho_stem = tif_path.stem  # "NVR1_N910A6_ch1_main_ortho_25mm"
        # Remove "_ortho" and everything after it (including resolution suffix)
        if '_ortho' in ortho_stem:
            camera_name = ortho_stem.split('_ortho')[0]  # "NVR1_N910A6_ch1_main"
        else:
            camera_name = ortho_stem  # fallback if no _ortho found

        # Look up camera ID in zone map
        camera_id = camera_name_to_id.get(camera_name, 0)

        if camera_id == 0:
            if camera_name not in cameras_not_in_map:
                print(f"  Warning: Camera '{camera_name}' not found in zone map, skipping")
                cameras_not_in_map.add(camera_name)
            continue

        cameras_found.add(camera_name)

        # Load image and geotransform
        img = cv2.imread(str(tif_path))
        geotransform = read_geotiff_transform(tif_path)

        # Get image bounds
        img_x_min, img_x_max, img_y_min, img_y_max = get_image_bounds(img.shape, geotransform)

        # Convert to mosaic pixel coordinates
        mosaic_col_start = max(0, int((img_x_min - x_min) / resolution))
        mosaic_row_start = max(0, int((y_max - img_y_max) / abs(resolution)))
        mosaic_col_end = min(mosaic_width, mosaic_col_start + img.shape[1])
        mosaic_row_end = min(mosaic_height, mosaic_row_start + img.shape[0])

        # Calculate corresponding image region
        img_col_start = max(0, -int((img_x_min - x_min) / resolution))
        img_row_start = max(0, -int((y_max - img_y_max) / abs(resolution)))
        img_col_end = img_col_start + (mosaic_col_end - mosaic_col_start)
        img_row_end = img_row_start + (mosaic_row_end - mosaic_row_start)

        # Extract the region
        img_region = img[img_row_start:img_row_end, img_col_start:img_col_end]

        # Find valid (non-black) pixels
        valid_mask = np.any(img_region > 0, axis=2)

        # Get zone map region
        zone_region = zone_array[mosaic_row_start:mosaic_row_end,
                                mosaic_col_start:mosaic_col_end]

        # STRICT MODE: Only update where zone matches this camera AND pixel is valid
        zone_match = (zone_region == camera_id)
        update_mask = valid_mask & zone_match

        pixels_updated = np.sum(update_mask)

        if pixels_updated > 0:
            # Update mosaic
            mosaic_region = mosaic[mosaic_row_start:mosaic_row_end,
                                  mosaic_col_start:mosaic_col_end]
            coverage_region = coverage_map[mosaic_row_start:mosaic_row_end,
                                          mosaic_col_start:mosaic_col_end]

            for c in range(3):
                mosaic_region[:, :, c][update_mask] = img_region[:, :, c][update_mask]

            coverage_region[update_mask] = 1

            mosaic[mosaic_row_start:mosaic_row_end,
                  mosaic_col_start:mosaic_col_end] = mosaic_region
            coverage_map[mosaic_row_start:mosaic_row_end,
                        mosaic_col_start:mosaic_col_end] = coverage_region

            print(f"  + Added {pixels_updated} pixels from zone {camera_id} ({camera_name})")
        else:
            print(f"  No pixels in this camera's zone")

    # Report on missing cameras
    all_cameras_in_map = set(camera_id_to_name.values())
    missing_cameras = all_cameras_in_map - cameras_found

    if missing_cameras:
        print(f"\n  Warning: {len(missing_cameras)} camera(s) in zone map had no ortho files:")
        for cam in sorted(missing_cameras):
            print(f"    - {cam}")

    if cameras_not_in_map:
        print(f"\n  Info: {len(cameras_not_in_map)} ortho file(s) not in zone map (ignored):")
        for cam in sorted(cameras_not_in_map):
            print(f"    - {cam}")

    # Report coverage
    total_pixels = mosaic_height * mosaic_width
    covered_pixels = np.sum(coverage_map)
    coverage_pct = 100 * covered_pixels / total_pixels
    print(f"\n  Mosaic coverage: {covered_pixels}/{total_pixels} pixels ({coverage_pct:.1f}%)")

    # Save as GeoTIFF
    print(f"\nSaving mosaic as GeoTIFF to {output_path}")

    from rasterio.transform import from_bounds

    # Convert BGR to RGB
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    mosaic_data = np.transpose(mosaic_rgb, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    # Create geotransform (in model coordinates)
    transform = from_bounds(
        west=x_min,
        south=y_min,
        east=x_max,
        north=y_max,
        width=mosaic_width,
        height=mosaic_height
    )

    # Determine final output path
    if world_file_transform:
        # Save model coords to original location
        temp_path = output_path
        # Save world coords to 'world_coords' subdirectory
        output_dir = Path(output_path).parent
        world_coords_dir = output_dir / 'world_coords'
        world_coords_dir.mkdir(exist_ok=True)
        final_path = world_coords_dir / Path(output_path).name
    else:
        temp_path = output_path
        final_path = output_path

    # Write GeoTIFF in model coordinates
    with rasterio.open(
        temp_path,
        'w',
        driver='GTiff',
        height=mosaic_height,
        width=mosaic_width,
        count=3,
        dtype=mosaic_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mosaic_data)

    print(f"Saved mosaic GeoTIFF: {temp_path}")

    # Clip to shapefile if requested (in model coordinates)
    if clip_shapefile:
        clipped_path = Path(str(temp_path).rsplit('.', 1)[0] + '_clipped.tif')
        clip_raster_to_shapefile(temp_path, clipped_path, clip_shapefile)
        # Always delete the unclipped version (we don't need it)
        Path(temp_path).unlink()
        print(f"  Deleted unclipped mosaic: {temp_path}")
        # Replace temp_path with clipped version for subsequent operations
        temp_path = clipped_path
        # Also update final_path if no world transform (they're the same)
        if not world_file_transform:
            final_path = clipped_path

    # Apply coordinate transformation if requested
    if world_file_transform:
        apply_coordinate_transform(
            temp_path, final_path, world_file_transform,
            resampling_method=world_transform_resampling,
            num_threads=world_transform_threads,
            warp_mem_limit_mb=world_transform_memory_mb,
            output_crs=output_crs
        )
        # Delete the model coordinate clipped version after transformation unless keeping intermediates
        if not keep_intermediate:
            Path(temp_path).unlink()
            print(f"  Deleted model space clipped mosaic: {temp_path}")

        # Create downscaled version if requested (only for world-transformed outputs)
        if save_downscaled and downscaled_resolution:
            downscaled_path = Path(str(final_path).rsplit('.', 1)[0] + f'_downscaled_{int(downscaled_resolution*100)}cm.tif')
            create_downscaled_mosaic(final_path, downscaled_path, downscaled_resolution)

    # Apply LZW compression to final output if requested
    if compress_output:
        print(f"\nApplying LZW compression to final mosaic...")
        compress_geotiff(final_path)
        print(f"  Compressed: {final_path}")

        # Also compress the model-space clipped version if it was kept
        if world_file_transform and keep_intermediate and Path(temp_path).exists():
            print(f"  Compressing intermediate mosaic...")
            compress_geotiff(temp_path)
            print(f"  Compressed: {temp_path}")

    print("\nDone!")
    return mosaic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create mosaic from orthorectified images using seam carving (no blending)'
    )
    
    parser.add_argument(
        'ortho_dir',
        help='Directory containing *_ortho.tif GeoTIFF files'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='mosaic.tif',
        help='Output mosaic path (default: mosaic.tif)'
    )
    
    parser.add_argument(
        '-m', '--method',
        choices=['seam', 'center', 'order', 'zone_map'],
        default='center',
        help='Mosaic method: seam=seam carving, center=prefer image centers, order=left-to-right, zone_map=use spatial zone assignments (default: center)'
    )
    
    parser.add_argument(
        '-s', '--seam-quality',
        choices=['gradient', 'variance', 'combined'],
        default='gradient',
        help='Quality metric for seam finding (default: gradient)'
    )
    
    parser.add_argument(
        '-r', '--resolution',
        type=float,
        default=None,
        help='Output resolution in meters/pixel (default: use input resolution)'
    )
    
    parser.add_argument(
        '--save-seams',
        action='store_true',
        help='Save seam visualization map'
    )

    parser.add_argument(
        '--world-file',
        type=str,
        default=None,
        help='Path to world file for coordinate transformation (optional)'
    )

    parser.add_argument(
        '--world-resampling',
        type=str,
        choices=['nearest', 'bilinear', 'cubic', 'lanczos'],
        default='bilinear',
        help='Resampling method for world coordinate transformation (default: bilinear). Use "nearest" for faster processing.'
    )

    parser.add_argument(
        '--world-threads',
        type=int,
        default=None,
        help='Number of threads for world coordinate transformation (default: auto-detect CPU cores)'
    )

    parser.add_argument(
        '--world-memory',
        type=int,
        default=512,
        help='Warp memory limit in MB for world coordinate transformation (default: 512)'
    )

    parser.add_argument(
        '--clip-shapefile',
        type=str,
        default=None,
        help='Path to shapefile for clipping the mosaic in model coordinates (optional)'
    )

    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep model-space clipped mosaic after transformation (useful for debugging)'
    )

    parser.add_argument(
        '--save-downscaled',
        action='store_true',
        help='Save a downscaled version of the world-coordinate mosaic'
    )

    parser.add_argument(
        '--downscaled-resolution',
        type=float,
        default=0.25,
        help='Resolution for downscaled mosaic in meters/pixel (default: 0.25 = 25cm/pixel)'
    )

    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Disable LZW compression (results in larger files but faster processing)'
    )

    parser.add_argument(
        '--zone-map-shapefile',
        type=str,
        default=None,
        help='Path to shapefile for zone_map method (required if method=zone_map)'
    )

    args = parser.parse_args()

    # Validate zone_map requirements
    if args.method == 'zone_map' and not args.zone_map_shapefile:
        parser.error("--zone-map-shapefile is required when using zone_map method")

    # Extract resolution filter from output filename if present
    # e.g., "mosaic_20251203_141530_2_5mm.tif" -> resolution_filter = "2_5mm"
    # e.g., "mosaic_20251203_141530_10mm.tif" -> resolution_filter = "10mm"
    resolution_filter = None
    output_stem = Path(args.output).stem
    if '_' in output_stem:
        # Check if last part matches pattern like "2mm", "10mm", "2_5mm"
        last_part = output_stem.split('_')[-1]
        if last_part.endswith('mm'):
            # Check if it's a simple number (e.g., "10mm")
            if last_part[:-2].isdigit():
                resolution_filter = last_part
                print(f"Detected resolution filter from output filename: {resolution_filter}")
            else:
                # Check if it's a decimal format (e.g., "5mm" from "2_5mm")
                # Need to look at last two parts
                parts = output_stem.split('_')
                if len(parts) >= 2:
                    last_two = '_'.join(parts[-2:])
                    # Match pattern like "2_5mm"
                    if last_two.endswith('mm') and '_' in last_two:
                        before_mm = last_two[:-2]
                        parts_check = before_mm.split('_')
                        if len(parts_check) == 2 and all(p.isdigit() for p in parts_check):
                            resolution_filter = last_two
                            print(f"Detected resolution filter from output filename: {resolution_filter}")

    if args.method == 'seam':
        create_seam_carved_mosaic(
            args.ortho_dir,
            args.output,
            resolution=args.resolution,
            seam_method=args.seam_quality,
            save_seam_map=args.save_seams,
            world_file_transform=args.world_file,
            world_transform_resampling=args.world_resampling,
            world_transform_threads=args.world_threads,
            world_transform_memory_mb=args.world_memory,
            clip_shapefile=args.clip_shapefile,
            keep_intermediate=args.keep_intermediate,
            save_downscaled=args.save_downscaled,
            downscaled_resolution=args.downscaled_resolution,
            compress_output=not args.no_compress
        )
    elif args.method == 'zone_map':
        create_mosaic_zone_map(
            args.ortho_dir,
            args.output,
            zone_map_shapefile=args.zone_map_shapefile,
            resolution=args.resolution,
            world_file_transform=args.world_file,
            world_transform_resampling=args.world_resampling,
            world_transform_threads=args.world_threads,
            world_transform_memory_mb=args.world_memory,
            clip_shapefile=args.clip_shapefile,
            keep_intermediate=args.keep_intermediate,
            save_downscaled=args.save_downscaled,
            downscaled_resolution=args.downscaled_resolution,
            compress_output=not args.no_compress,
            resolution_filter=resolution_filter
        )
    else:
        create_mosaic_simple_priority(
            args.ortho_dir,
            args.output,
            resolution=args.resolution,
            priority=args.method,
            world_file_transform=args.world_file,
            world_transform_resampling=args.world_resampling,
            world_transform_threads=args.world_threads,
            world_transform_memory_mb=args.world_memory,
            clip_shapefile=args.clip_shapefile,
            keep_intermediate=args.keep_intermediate,
            save_downscaled=args.save_downscaled,
            downscaled_resolution=args.downscaled_resolution,
            compress_output=not args.no_compress,
            resolution_filter=resolution_filter
        )

#python ortho_mosaic.py output/orthorectified -o cse_all16IR_mosaic.tif -m center