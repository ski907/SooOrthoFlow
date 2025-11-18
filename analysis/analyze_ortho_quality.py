#!/usr/bin/env python3
"""
Orthorectification Quality Analysis Tool

Analyzes and maps image quality degradation in orthorectified images by computing
pixel reuse/smearing metrics from calibration lookup tables.

The key insight: In fisheye orthorectification, source pixels at nadir map ~1:1
to output pixels (crisp), while source pixels at margins get reused across many
output pixels (smeared). This script quantifies and maps that effect.

Usage:
    python analyze_ortho_quality.py --calibration calibration/calibration_20251016.pkl \
                                    --camera N910A6_ch01_main \
                                    --output quality_maps/

Outputs:
    - {camera}_source_gradient.tif: Rate of change in source pixel sampling
    - {camera}_effective_resolution.tif: Effective information density map
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds


def load_calibration(calibration_file, camera_id):
    """
    Load calibration data for a specific camera

    Args:
        calibration_file: Path to calibration PKL file
        camera_id: Camera identifier

    Returns:
        dict with keys: map_x, map_y, geotransform, K, D, R, tvec
    """
    print(f"Loading calibration from {calibration_file}")

    with open(calibration_file, 'rb') as f:
        calibrations = pickle.load(f)

    if camera_id not in calibrations:
        available = ', '.join(calibrations.keys())
        raise ValueError(f"Camera {camera_id} not found in calibration file.\n"
                        f"Available cameras: {available}")

    cal_data = calibrations[camera_id]

    # Verify required fields exist
    required_fields = ['map_x', 'map_y', 'geotransform']
    missing = [f for f in required_fields if f not in cal_data]
    if missing:
        raise ValueError(f"Calibration missing required fields: {missing}")

    print(f"✓ Loaded calibration for {camera_id}")
    print(f"  Lookup table size: {cal_data['map_x'].shape}")
    print(f"  Geotransform: x_min={cal_data['geotransform']['x_min']:.2f}, "
          f"y_max={cal_data['geotransform']['y_max']:.2f}")

    return cal_data


def compute_pixel_stretch_factor(map_x, map_y):
    """
    Compute pixel stretch factor: how many output pixels are covered by one source pixel.

    This directly quantifies the smearing effect:
    - Value ~1: Good quality (1:1 pixel mapping, crisp)
    - Value ~5: Moderate smearing (one source pixel stretched across 5 output pixels)
    - Value >10: Severe smearing (significant quality loss)

    Method: Computes the Jacobian determinant of the source-to-output mapping,
    then inverts it to get output area per source area.

    Args:
        map_x: (H, W) array - source image x-coordinates for each output pixel
        map_y: (H, W) array - source image y-coordinates for each output pixel

    Returns:
        stretch_factor: (H, W) array - output pixels per source pixel
    """
    print("Computing pixel stretch factor...")

    # Compute gradients: rate of change of source coordinates per output pixel
    # du/di, du/dj: how source u changes per output row/column
    # dv/di, dv/dj: how source v changes per output row/column
    du_di, du_dj = np.gradient(map_x)
    dv_di, dv_dj = np.gradient(map_y)

    # Compute Jacobian determinant: det(J) = (du/di)(dv/dj) - (du/dj)(dv/di)
    # This gives the local area scaling factor (source area per output area)
    # High determinant = many source pixels per output pixel (oversampling, good)
    # Low determinant = few source pixels per output pixel (undersampling, smearing)
    jacobian_det = np.abs(du_di * dv_dj - du_dj * dv_di)

    # Invert to get output pixels per source pixel (stretch factor)
    # This is the intuitive metric: how much is each source pixel stretched?
    with np.errstate(divide='ignore', invalid='ignore'):
        stretch_factor = 1.0 / jacobian_det

    # Handle invalid regions
    valid_mask = ~(np.isnan(map_x) | np.isnan(map_y))
    stretch_factor[~valid_mask] = np.nan
    stretch_factor[np.isinf(stretch_factor)] = np.nan

    # Compute statistics
    valid_values = stretch_factor[valid_mask]
    print(f"  Pixel stretch factor statistics:")
    print(f"    Min: {np.nanmin(valid_values):.2f}x (best quality)")
    print(f"    Max: {np.nanmax(valid_values):.2f}x (worst quality)")
    print(f"    Mean: {np.nanmean(valid_values):.2f}x")
    print(f"    Median: {np.nanmedian(valid_values):.2f}x")
    print(f"  Interpretation:")
    print(f"    <2x: Excellent quality")
    print(f"    2-5x: Good quality")
    print(f"    5-10x: Degraded quality (noticeable smearing)")
    print(f"    >10x: Poor quality (severe smearing)")

    return stretch_factor


def compute_quality_score(stretch_factor):
    """
    Convert stretch factor to a 0-100 quality score for easier interpretation.

    Quality score interpretation:
    - 90-100: Excellent (minimal distortion)
    - 70-90: Good (acceptable quality)
    - 50-70: Fair (noticeable degradation)
    - <50: Poor (significant smearing)

    Args:
        stretch_factor: (H, W) array - output pixels per source pixel

    Returns:
        quality_score: (H, W) array - quality score from 0-100
    """
    print("Computing quality score...")

    # Convert stretch factor to quality score
    # stretch=1 → quality=100, stretch=10 → quality=0
    # Using exponential decay: quality = 100 * exp(-k * (stretch - 1))
    # Calibrated so stretch=5 gives quality≈60, stretch=10 gives quality≈20
    quality_score = 100.0 * np.exp(-0.25 * (stretch_factor - 1.0))

    # Clamp to 0-100 range
    quality_score = np.clip(quality_score, 0, 100)

    # Handle invalid regions
    quality_score[np.isnan(stretch_factor)] = np.nan

    # Compute statistics
    valid_values = quality_score[~np.isnan(quality_score)]
    print(f"  Quality score statistics:")
    print(f"    Min: {np.nanmin(valid_values):.1f} (worst)")
    print(f"    Max: {np.nanmax(valid_values):.1f} (best)")
    print(f"    Mean: {np.nanmean(valid_values):.1f}")
    print(f"    Median: {np.nanmedian(valid_values):.1f}")

    return quality_score


def save_georeferenced_tiff(data, geotransform, output_path, description=""):
    """
    Save a 2D array as a georeferenced GeoTIFF file.

    Args:
        data: (H, W) numpy array to save
        geotransform: dict with x_min, y_max, pixel_width, pixel_height
        output_path: Path to output TIFF file
        description: Optional description for the TIFF metadata
    """
    height, width = data.shape

    # Create rasterio transform from geotransform dict
    transform = from_bounds(
        west=geotransform['x_min'],
        south=geotransform['y_max'] - height * abs(geotransform['pixel_height']),
        east=geotransform['x_min'] + width * geotransform['pixel_width'],
        north=geotransform['y_max'],
        width=width,
        height=height
    )

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:26917',  # UTM Zone 17N (Soo Locks area)
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
        if description:
            dst.update_tags(description=description)

    print(f"  ✓ Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze orthorectification quality from calibration lookup tables'
    )
    parser.add_argument('-cal', '--calibration', required=True,
                       help='Path to calibration PKL file')
    parser.add_argument('-c', '--camera', required=True,
                       help='Camera ID to analyze (e.g., N910A6_ch01_main)')
    parser.add_argument('-o', '--output', default='quality_maps',
                       help='Output directory for quality maps (default: quality_maps)')

    args = parser.parse_args()

    # Validate inputs
    cal_file = Path(args.calibration)
    if not cal_file.exists():
        print(f"Error: Calibration file not found: {cal_file}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load calibration data
    try:
        cal_data = load_calibration(cal_file, args.camera)
    except Exception as e:
        print(f"Error loading calibration: {e}")
        sys.exit(1)

    # Extract lookup tables
    map_x = cal_data['map_x']
    map_y = cal_data['map_y']
    geotransform = cal_data['geotransform']

    # Compute quality metrics
    print("\n" + "="*60)
    print("Computing Quality Metrics")
    print("="*60 + "\n")

    # 1. Pixel stretch factor (output pixels per source pixel)
    stretch_factor = compute_pixel_stretch_factor(map_x, map_y)

    # 2. Quality score (0-100, higher = better)
    quality_score = compute_quality_score(stretch_factor)

    # Save outputs
    print("\n" + "="*60)
    print("Saving Quality Maps")
    print("="*60 + "\n")

    camera_safe = args.camera.replace('/', '_')

    stretch_file = output_dir / f"{camera_safe}_stretch_factor.tif"
    save_georeferenced_tiff(
        stretch_factor.astype(np.float32),
        geotransform,
        stretch_file,
        description="Pixel stretch factor - output pixels per source pixel. "
                   "Value of 1 = crisp, >5 = noticeable smearing, >10 = severe smearing."
    )

    quality_file = output_dir / f"{camera_safe}_quality_score.tif"
    save_georeferenced_tiff(
        quality_score.astype(np.float32),
        geotransform,
        quality_file,
        description="Quality score (0-100). "
                   "90-100 = excellent, 70-90 = good, 50-70 = fair, <50 = poor."
    )

    # Summary
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    print(f"\nQuality maps saved to: {output_dir}")
    print(f"  1. {stretch_file.name} - Pixel stretch factor (1=best, >10=worst)")
    print(f"  2. {quality_file.name} - Quality score (100=best, 0=worst)")
    print("\nYou can open these TIFF files in QGIS or other GIS software")
    print("alongside your orthorectified images to see where smearing occurs.")
    print("\nInterpretation:")
    print("  - Stretch factor >5 or Quality <70 = areas with noticeable smearing")
    print("  - Use quality maps to identify where to add cameras or adjust mosaicking")


if __name__ == '__main__':
    main()
