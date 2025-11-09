import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import argparse

def read_worldfile(tfw_path):
    """Read world file and return geotransform parameters"""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()
    
    return {
        'pixel_width': float(lines[0].strip()),
        'rotation_y': float(lines[1].strip()),
        'rotation_x': float(lines[2].strip()),
        'pixel_height': float(lines[3].strip()),
        'x_min': float(lines[4].strip()),
        'y_max': float(lines[5].strip())
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
    
    print(f"Found {len(tif_files)} orthorectified images")
    
    for tif_path in tif_files:
        tfw_path = tif_path.with_suffix('.tfw')
        
        if not tfw_path.exists():
            continue
        
        img = cv2.imread(str(tif_path))
        geotransform = read_worldfile(tfw_path)
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
                              seam_method='gradient', save_seam_map=False):
    """
    VECTORIZED: Create mosaic using seam carving - finds optimal non-blended boundaries
    
    Strategy:
    1. Process images left-to-right (sorted by X position)
    2. For each new image, find optimal seam in overlap region
    3. Use hard cut at seam (no blending)
    """
    
    # Compute overall bounds
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir)
    
    # Get resolution
    tif_files = sorted(Path(ortho_dir).glob('*_ortho.tif'))
    first_tfw = tif_files[0].with_suffix('.tfw')
    first_geotransform = read_worldfile(first_tfw)
    
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
        tfw_path = tif_path.with_suffix('.tfw')
        if not tfw_path.exists():
            continue
        geotransform = read_worldfile(tfw_path)
        image_data.append((geotransform['x_min'], tif_path, tfw_path))
    
    image_data.sort(key=lambda x: x[0])  # Sort by x_min
    
    print(f"\nProcessing {len(image_data)} images left-to-right with seam carving (VECTORIZED)...")
    
    for i, (_, tif_path, tfw_path) in enumerate(image_data, 1):
        print(f"\n[{i}/{len(image_data)}] Processing {tif_path.name}")
        
        # Load image and geotransform
        img = cv2.imread(str(tif_path))
        geotransform = read_worldfile(tfw_path)
        
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
            print(f"  Found overlap - computing optimal seam (VECTORIZED)")
            
            # Compute cost maps for both images in overlap
            cost_new = compute_image_quality_map(img_region, method=seam_method)
            cost_existing = compute_image_quality_map(existing_region, method=seam_method)
            
            # Combined cost (prefer low-gradient areas)
            combined_cost = (cost_new + cost_existing) / 2.0
            
            # Only consider overlap region
            combined_cost[~overlap_mask] = 0
            
            # Find vertical seam using VECTORIZED dynamic programming
            seam_mask = find_optimal_seam_vertical_vectorized(combined_cost, overlap_mask)
            
            # Apply seam: existing image on left, new image on right
            use_new = seam_mask & overlap_mask
            use_existing = overlap_mask & ~seam_mask
            
            # VECTORIZED: Copy new image where it should be used (all channels at once)
            existing_region[use_new] = img_region[use_new]
            
            # Mark seam in seam map
            seam_boundary = find_seam_boundary(seam_mask)
            seam_map[mosaic_row_start:mosaic_row_end,
                    mosaic_col_start:mosaic_col_end][seam_boundary] = 255
            
            print(f"  Seam carved through overlap region")
        
        # Add non-overlapping regions (VECTORIZED)
        non_overlap = valid_mask & (existing_mask == 0)
        existing_region[non_overlap] = img_region[non_overlap]
        
        # Update masks
        mosaic_mask[mosaic_row_start:mosaic_row_end,
                   mosaic_col_start:mosaic_col_end] = np.maximum(
            existing_mask, valid_mask
        )
        
        mosaic[mosaic_row_start:mosaic_row_end,
               mosaic_col_start:mosaic_col_end] = existing_region
        
        print(f"  Added to mosaic")
    
    # Save mosaic
    print(f"\nSaving mosaic to {output_path}")
    cv2.imwrite(str(output_path), mosaic)
    
    # Save world file
    tfw_output = Path(str(output_path).rsplit('.', 1)[0] + '.tfw')
    with open(tfw_output, 'w') as f:
        f.write(f"{resolution}\n0\n0\n{-resolution}\n{x_min}\n{y_max}\n")
    
    print(f"Saved world file: {tfw_output}")
    
    # Save seam map
    if save_seam_map:
        seam_path = Path(str(output_path).rsplit('.', 1)[0] + '_seams.tif')
        cv2.imwrite(str(seam_path), seam_map)
        print(f"Saved seam map: {seam_path}")
    
    print("\nDone!")
    return mosaic


def find_optimal_seam_vertical_vectorized(cost_map, overlap_mask):
    """
    VECTORIZED: Find optimal vertical seam through overlap using dynamic programming
    Returns mask where True = use new image, False = use existing
    
    This version uses NumPy array operations instead of nested Python loops.
    Expected speedup: 5-10x over the original version
    """
    height, width = cost_map.shape
    
    # Find left and right bounds of overlap
    overlap_cols = np.any(overlap_mask, axis=0)
    if not np.any(overlap_cols):
        return overlap_mask
    
    left_col = np.argmax(overlap_cols)
    right_col = width - np.argmax(overlap_cols[::-1]) - 1
    
    # Initialize DP table
    dp = np.full((height, width), np.inf, dtype=np.float64)
    parent = np.full((height, width), -1, dtype=np.int32)
    
    # Initialize first column of overlap
    first_col_mask = overlap_mask[:, left_col]
    dp[first_col_mask, left_col] = cost_map[first_col_mask, left_col]
    
    # VECTORIZED: Fill DP table column by column
    for col in range(left_col + 1, right_col + 1):
        current_mask = overlap_mask[:, col]
        if not np.any(current_mask):
            continue
        
        # Get previous column values
        prev_col = col - 1
        prev_values = dp[:, prev_col]
        
        # For each row in current column, find minimum from 3 neighbors
        # We'll compute costs for all three possible transitions at once
        
        # Create arrays for the three possible previous rows
        rows = np.arange(height)
        
        # Prepare candidate costs from three neighbors (top, middle, bottom)
        # Shape: (3, height) for the three transition options
        candidates = np.full((3, height), np.inf, dtype=np.float64)
        candidate_rows = np.full((3, height), -1, dtype=np.int32)
        
        # From same row (middle)
        valid = (prev_values != np.inf)
        candidates[0, valid] = prev_values[valid]
        candidate_rows[0, :] = rows
        
        # From row above (top)
        valid_top = (rows > 0) & (prev_values[np.maximum(rows - 1, 0)] != np.inf)
        candidates[1, valid_top] = prev_values[np.maximum(rows[valid_top] - 1, 0)]
        candidate_rows[1, :] = np.maximum(rows - 1, 0)
        
        # From row below (bottom)
        valid_bottom = (rows < height - 1) & (prev_values[np.minimum(rows + 1, height - 1)] != np.inf)
        candidates[2, valid_bottom] = prev_values[np.minimum(rows[valid_bottom] + 1, height - 1)]
        candidate_rows[2, :] = np.minimum(rows + 1, height - 1)
        
        # Find minimum candidate for each row
        min_indices = np.argmin(candidates, axis=0)
        min_costs = candidates[min_indices, rows]
        best_prev_rows = candidate_rows[min_indices, rows]
        
        # Update DP table only where current column has overlap
        valid_updates = current_mask & (min_costs != np.inf)
        dp[valid_updates, col] = min_costs[valid_updates] + cost_map[valid_updates, col]
        parent[valid_updates, col] = best_prev_rows[valid_updates]
    
    # VECTORIZED: Backtrack to find seam
    seam_cols = np.zeros(height, dtype=np.int32)
    
    # Find minimum in last column
    last_col_values = dp[:, right_col]
    valid_rows = overlap_mask[:, right_col] & (last_col_values != np.inf)
    
    if not np.any(valid_rows):
        # Fallback: split down middle
        middle_col = (left_col + right_col) // 2
        seam_mask = np.zeros_like(overlap_mask, dtype=bool)
        seam_mask[:, middle_col:] = True
        return seam_mask
    
    # Start from row with minimum cost
    current_row = np.argmin(np.where(valid_rows, last_col_values, np.inf))
    seam_cols[current_row] = right_col
    
    # Backtrack through parent pointers
    for col in range(right_col - 1, left_col - 1, -1):
        if parent[current_row, col + 1] != -1:
            current_row = parent[current_row, col + 1]
        seam_cols[current_row] = col
    
    # VECTORIZED: Create seam mask
    # Everything right of seam uses new image
    seam_mask = np.zeros_like(overlap_mask, dtype=bool)
    
    # Create column indices array
    col_indices = np.arange(width)
    
    # For each row, mark columns >= seam_col as True
    for row in range(height):
        if overlap_mask[row, left_col]:
            seam_col = seam_cols[row]
            seam_mask[row, col_indices >= seam_col] = True
    
    return seam_mask


def find_seam_boundary(seam_mask):
    """Find the boundary pixels of the seam for visualization"""
    # Dilate and subtract to find edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(seam_mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - seam_mask.astype(np.uint8)
    return boundary > 0


def create_mosaic_simple_priority(ortho_dir, output_path, resolution=None,
                                  priority='center'):
    """
    VECTORIZED: Simple approach: Assign priority to each image, last one wins
    
    Priority options:
    - 'center': Prefer images where content is near image center (less distortion)
    - 'order': Just use the order of images (e.g., left to right)
    """
    
    # Compute overall bounds
    x_min, x_max, y_min, y_max = compute_mosaic_bounds(ortho_dir)
    
    tif_files = sorted(Path(ortho_dir).glob('*_ortho.tif'))
    first_tfw = tif_files[0].with_suffix('.tfw')
    first_geotransform = read_worldfile(first_tfw)
    
    if resolution is None:
        resolution = first_geotransform['pixel_width']
    
    mosaic_width = int((x_max - x_min) / resolution)
    mosaic_height = int((y_max - y_min) / abs(resolution))
    
    print(f"Mosaic size: {mosaic_width} x {mosaic_height} pixels")
    
    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)
    priority_map = np.zeros((mosaic_height, mosaic_width), dtype=np.float32)
    
    print(f"\nCreating mosaic with '{priority}' priority (VECTORIZED)")
    
    for i, tif_path in enumerate(tif_files, 1):
        tfw_path = tif_path.with_suffix('.tfw')
        if not tfw_path.exists():
            continue
        
        print(f"[{i}/{len(tif_files)}] Processing {tif_path.name}")
        
        img = cv2.imread(str(tif_path))
        geotransform = read_worldfile(tfw_path)
        
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
            # VECTORIZED: Higher priority for pixels near image center
            h, w = img_region.shape[:2]
            y_coords, x_coords = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            img_priority = 1.0 - (dist_from_center / max_dist)
        else:  # 'order'
            img_priority = np.full((img_region.shape[0], img_region.shape[1]), i, dtype=np.float32)
        
        # Valid pixels
        valid_mask = np.any(img_region > 0, axis=2)
        img_priority = img_priority * valid_mask
        
        # Update where this image has higher priority (VECTORIZED)
        mosaic_region = mosaic[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end]
        priority_region = priority_map[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end]
        
        update_mask = (img_priority > priority_region) & valid_mask
        
        # VECTORIZED: Update all channels at once
        mosaic_region[update_mask] = img_region[update_mask]
        priority_region[update_mask] = img_priority[update_mask]
        
        mosaic[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end] = mosaic_region
        priority_map[mosaic_row_start:mosaic_row_end, mosaic_col_start:mosaic_col_end] = priority_region
    
    # Save
    cv2.imwrite(str(output_path), mosaic)
    
    tfw_output = Path(str(output_path).rsplit('.', 1)[0] + '.tfw')
    with open(tfw_output, 'w') as f:
        f.write(f"{resolution}\n0\n0\n{-resolution}\n{x_min}\n{y_max}\n")
    
    print(f"\nSaved: {output_path}")
    print(f"Saved world file: {tfw_output}")
    
    return mosaic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create mosaic from orthorectified images using seam carving (no blending) - VECTORIZED VERSION'
    )
    
    parser.add_argument(
        'ortho_dir',
        help='Directory containing *_ortho.tif and *.tfw files'
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
    
    args = parser.parse_args()
    
    if args.method == 'seam':
        create_seam_carved_mosaic(
            args.ortho_dir,
            args.output,
            resolution=args.resolution,
            seam_method=args.seam_quality,
            save_seam_map=args.save_seams
        )
    else:
        create_mosaic_simple_priority(
            args.ortho_dir,
            args.output,
            resolution=args.resolution,
            priority=args.method
        )

# Example usage:
# python ortho_mosaic_vectorized.py output/orthorectified -o mosaic.tif -m seam