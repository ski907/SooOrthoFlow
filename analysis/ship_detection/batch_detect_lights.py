import cv2
import numpy as np
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Error: rasterio required. Install with: pip install rasterio")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Error: geopandas required. Install with: pip install geopandas")
    sys.exit(1)


def load_and_resample_mask(mask_path, target_transform, target_crs, target_shape):
    """
    Load and resample mask once to match target grid.

    Parameters:
    -----------
    mask_path : str
        Path to mask GeoTIFF
    target_transform : affine.Affine
        Target geotransform
    target_crs : CRS
        Target CRS
    target_shape : tuple
        (height, width) of target grid

    Returns:
    --------
    np.ndarray : Binary mask (1 = valid, 0 = invalid)
    """
    with rasterio.open(mask_path) as mask_src:
        mask_data = mask_src.read(1)

        valid_mask = np.zeros(target_shape, dtype=np.float32)

        reproject(
            source=mask_data,
            destination=valid_mask,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        # Convert to binary
        valid_mask = (valid_mask >= 0.9).astype(np.uint8)

    return valid_mask


def detect_boat_lights_fast(image_path, valid_mask=None, min_red_area=2000,
                             min_light_area=150, max_light_area=500):
    """
    Fast detection of boat lights without debug output.

    Parameters:
    -----------
    image_path : str
        Path to input image
    valid_mask : np.ndarray, optional
        Pre-computed binary mask
    min_red_area : int
        Minimum area for red filter regions
    min_light_area : int
        Minimum area for light detection
    max_light_area : int
        Maximum area for light detection

    Returns:
    --------
    list of tuples : [(pixel_x, pixel_y, world_x, world_y), ...]
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    # Normalize
    if img.max() > 255:
        img = (img / img.max() * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Strong red filter
    strong_red_mask = (img_rgb[:, :, 0] > 150) & \
                      (img_rgb[:, :, 0] > img_rgb[:, :, 1] * 1.5) & \
                      (img_rgb[:, :, 0] > img_rgb[:, :, 2] * 1.5)

    # Apply spatial mask
    if valid_mask is not None:
        strong_red_mask = strong_red_mask & (valid_mask == 1)

    # Dilate (reduced iterations for speed)
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(strong_red_mask.astype(np.uint8), kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_red_area]

    # Detect lights
    detected_lights = []

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)

        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_rgb.shape[1], x + w + padding)
        y2 = min(img_rgb.shape[0], y + h + padding)

        roi_g = img_rgb[y1:y2, x1:x2, 1]
        roi_b = img_rgb[y1:y2, x1:x2, 2]

        # Smaller blur kernel for speed
        roi_g_blur = cv2.GaussianBlur(roi_g, (3, 3), 1)
        roi_b_blur = cv2.GaussianBlur(roi_b, (3, 3), 1)

        thresh_g = roi_g_blur > 180
        thresh_b = roi_b_blur > 180
        roi_thresh = (thresh_g & thresh_b).astype(np.uint8) * 255

        contours_light, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_light:
            area = cv2.contourArea(cnt)

            if min_light_area <= area <= max_light_area:
                (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(cnt)
                cx = int(cx_roi) + x1
                cy = int(cy_roi) + y1

                mask_temp = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_temp, [cnt], -1, 1, -1, offset=(x1, y1))

                if np.any(mask_temp == 1):
                    green_mean = np.mean(img_rgb[:, :, 1][mask_temp == 1])
                    blue_mean = np.mean(img_rgb[:, :, 2][mask_temp == 1])

                    if blue_mean > 200 and green_mean > 200:
                        detected_lights.append((cx, cy))

    # Non-maximum suppression
    if detected_lights:
        final_lights = []
        min_distance = 50

        for cx, cy in detected_lights:
            is_duplicate = False
            for final_cx, final_cy in final_lights:
                distance = np.sqrt((cx - final_cx)**2 + (cy - final_cy)**2)
                if distance < min_distance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_lights.append((cx, cy))

        detected_lights = final_lights

    # Get world coordinates
    results = []
    with rasterio.open(image_path) as src:
        transform = src.transform
        for cx, cy in detected_lights:
            world_x, world_y = transform * (cx, cy)
            results.append((cx, cy, world_x, world_y))

    return results


def process_single_image(img_path, mask_path):
    """
    Process a single image (for parallel processing).

    Returns:
    --------
    list of dict : Detection results
    """
    # Load mask if provided
    valid_mask = None
    if mask_path:
        with rasterio.open(img_path) as src:
            valid_mask = load_and_resample_mask(
                mask_path,
                src.transform,
                src.crs,
                (src.height, src.width)
            )

    # Detect lights
    detections = detect_boat_lights_fast(img_path, valid_mask)

    results = []
    if detections:
        for px, py, wx, wy in detections:
            results.append({
                'image': Path(img_path).name,
                'pixel_x': px,
                'pixel_y': py,
                'world_x': wx,
                'world_y': wy,
                'geometry': Point(wx, wy)
            })

    return Path(img_path).name, len(detections), results


def batch_detect_and_export(image_paths, mask_path, output_shapefile, output_csv=None, n_workers=None):
    """
    Process multiple images in parallel and export detections to shapefile.

    Parameters:
    -----------
    image_paths : list of str
        Paths to input images
    mask_path : str
        Path to mask (will be resampled for each image if needed)
    output_shapefile : str
        Path to output shapefile
    output_csv : str, optional
        Path to output CSV
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"Processing {len(image_paths)} images using {n_workers} workers...\n")

    all_detections = []

    # Process images in parallel
    process_func = partial(process_single_image, mask_path=mask_path)

    with Pool(n_workers) as pool:
        results = pool.map(process_func, image_paths)

    # Collect results
    for img_name, n_detections, detections in results:
        print(f"{img_name}: {n_detections} lights")
        all_detections.extend(detections)

    if not all_detections:
        print("No lights detected in any image")
        return

    # Get CRS from first image
    with rasterio.open(image_paths[0]) as src:
        crs = src.crs

    # Create GeoDataFrame with points
    gdf_points = gpd.GeoDataFrame(all_detections, crs=crs)

    # Create linestrings - one per image, connecting all points
    linestrings = []
    for img_name in gdf_points['image'].unique():
        img_points = gdf_points[gdf_points['image'] == img_name]
        if len(img_points) >= 2:
            coords = [(row['world_x'], row['world_y']) for _, row in img_points.iterrows()]
            linestrings.append({
                'image': img_name,
                'n_lights': len(coords),
                'geometry': LineString(coords)
            })

    # Export points shapefile
    points_shapefile = output_shapefile.replace('.shp', '_points.shp')
    gdf_points.to_file(points_shapefile)
    print(f"\nSaved {len(gdf_points)} points to {points_shapefile}")

    # Export lines shapefile
    if linestrings:
        gdf_lines = gpd.GeoDataFrame(linestrings, crs=crs)
        lines_shapefile = output_shapefile.replace('.shp', '_lines.shp')
        gdf_lines.to_file(lines_shapefile)
        print(f"Saved {len(gdf_lines)} lines to {lines_shapefile}")

    # Export CSV if requested
    if output_csv:
        gdf_points.drop(columns='geometry').to_csv(output_csv, index=False)
        print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    # Configuration
    mosaic_folder = "./test_mosaics"  # Folder containing mosaic TIFs
    mask_path = "area_to_review_ref.tif"  # Set to None to search entire images
    output_shapefile = "detected_boat_lights.shp"
    output_csv = "detected_boat_lights.csv"
    n_workers = None  # None = auto (CPU count - 1), or set specific number

    # Find all TIF files in folder
    mosaic_dir = Path(mosaic_folder)
    image_paths = sorted(mosaic_dir.glob("*.tif"))

    # Exclude mask file from processing
    if mask_path:
        mask_file = Path(mask_path).name
        image_paths = [p for p in image_paths if p.name != mask_file]

    print(f"Found {len(image_paths)} mosaic(s) in {mosaic_folder}")
    for p in image_paths:
        print(f"  - {p.name}")

    if not image_paths:
        print("No TIF files found!")
        sys.exit(1)

    # Convert to strings
    image_paths = [str(p) for p in image_paths]

    # Run batch detection
    batch_detect_and_export(image_paths, mask_path, output_shapefile, output_csv, n_workers)
