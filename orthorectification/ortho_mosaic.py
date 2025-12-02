import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import argparse
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

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


def compute_mosaic_bounds(ortho_dir):
    """Compute the overall bounds needed for the mosaic"""
    print("Computing mosaic bounds...")

    x_mins, x_maxs, y_mins, y_maxs = [], [], [], []

    tif_files = list(Path(ortho_dir).glob('*_ortho.tif'))

    if not tif_files:
        raise ValueError(f"No *_ortho.tif files found in {ortho_dir}")

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
                              output_crs='EPSG:26917'):
    """
    Create mosaic using seam carving - finds optimal non-blended boundaries

    Strategy:
    1. Process images left-to-right (sorted by X position)
    2. For each new image, find optimal seam in overlap region
    3. Use hard cut at seam (no blending)

    Args:
        world_file_transform: Optional path to world file for coordinate transformation
    """
    
    # Compute overall bounds
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir)
    
    # Get resolution
    tif_files = sorted(Path(ortho_dir).glob('*_ortho.tif'))
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
        crs=crs,  # Use parameter instead of hardcoded
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(mosaic_data)

    print(f"Saved mosaic GeoTIFF: {temp_path}")

    # Apply coordinate transformation if requested
    if world_file_transform:
        apply_coordinate_transform(
            temp_path, final_path, world_file_transform,
            resampling_method=world_transform_resampling,
            num_threads=world_transform_threads,
            warp_mem_limit_mb=world_transform_memory_mb,
            output_crs=output_crs
        )
        # Optionally remove temp file
        # Path(temp_path).unlink()

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


def apply_coordinate_transform(input_tif, output_tif, world_file_path,
                               resampling_method='bilinear',
                               num_threads=None,
                               warp_mem_limit_mb=512,
                               output_crs='EPSG:26917'):
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
    pixel_to_world = read_world_file(world_file_path)

    with rasterio.open(input_tif) as src:
        src_crs = src.crs
        src_height, src_width = src.height, src.width
        num_bands = src.count

        print(f"  Source image dimensions: {src_width} x {src_height}, {num_bands} bands")
        print(f"  Using {num_threads} threads, {warp_mem_limit_mb}MB warp memory")
        print(f"  Resampling method: {resampling_method}")
        print(f"  Pixel-to-world transform from world file:")
        print(f"    {pixel_to_world}")

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

        with rasterio.open(output_tif, 'w', **out_profile) as dst:
            dst.write(dst_data)

    print(f"Transformed mosaic saved to: {output_tif}")
    print(f"  Successfully transformed to world coordinates")


def create_mosaic_simple_priority(ortho_dir, output_path, resolution=None,
                                  priority='center', world_file_transform=None,
                                  world_transform_resampling='bilinear',
                                  world_transform_threads=None,
                                  world_transform_memory_mb=512,
                                  crs='EPSG:26919',
                                  output_crs='EPSG:26917'):
    """
    Simple approach: Assign priority to each image, last one wins

    Priority options:
    - 'center': Prefer images where content is near image center (less distortion)
    - 'order': Just use the order of images (e.g., left to right)

    Args:
        world_file_transform: Optional path to world file for coordinate transformation
    """
    
    # Compute overall bounds
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir)

    tif_files = sorted(Path(ortho_dir).glob('*_ortho.tif'))
    first_geotransform = read_geotiff_transform(tif_files[0])

    if resolution is None:
        resolution = first_geotransform['pixel_width']

    mosaic_width = int((x_max - x_min) / resolution)
    mosaic_height = int((y_max - y_min) / abs(resolution))
    
    print(f"Mosaic size: {mosaic_width} x {mosaic_height} pixels")
    
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    priority_map = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)
    
    print(f"\nCreating mosaic with '{priority}' priority")
    
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
        
        # Compute priority for this image
        if priority == 'center':
            # Higher priority for pixels near image center
            h, w = img_region.shape[:2]
            y_coords, x_coords = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            img_priority = 1.0 - (dist_from_center / max_dist)
        else:  # 'order'
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
        crs=crs,  # Use parameter instead of hardcoded
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(mosaic_data)

    print(f"Saved mosaic GeoTIFF: {temp_path}")

    # Apply coordinate transformation if requested
    if world_file_transform:
        apply_coordinate_transform(
            temp_path, final_path, world_file_transform,
            resampling_method=world_transform_resampling,
            num_threads=world_transform_threads,
            warp_mem_limit_mb=world_transform_memory_mb,
            output_crs=output_crs
        )
        # Optionally remove temp file
        # Path(temp_path).unlink()

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
        choices=['seam', 'center', 'order'],
        default='seam',
        help='Mosaic method: seam=seam carving, center=prefer image centers, order=left-to-right (default: seam)'
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

    args = parser.parse_args()

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
            world_transform_memory_mb=args.world_memory
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
            world_transform_memory_mb=args.world_memory
        )

#python ortho_mosaic.py output/orthorectified -o cse_all16IR_mosaic.tif -m center