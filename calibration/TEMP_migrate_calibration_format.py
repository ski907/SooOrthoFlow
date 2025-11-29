"""
One-time migration script to convert legacy pickle calibration format to new CSV + cache format.

This script:
1. Reads the existing camera_calibrations_YYYYMMDD.pkl file
2. Extracts camera calibration parameters into a CSV file
3. Creates individual ortho cache files for each camera
4. Preserves all calibration data in the new format

DELETE THIS FILE AFTER SUCCESSFUL MIGRATION!
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def migrate_calibration_file(pickle_file, output_csv=None, cache_dir=None, calibration_date=None):
    """
    Migrate pickle calibration file to new CSV + cache format.

    Parameters:
    -----------
    pickle_file : str or Path
        Path to the existing .pkl calibration file
    output_csv : str or Path, optional
        Path for output CSV (default: same name as pickle but .csv)
    cache_dir : str or Path, optional
        Directory for ortho cache files (default: ../orthorectification/ortho_cache/)
    calibration_date : str, optional
        Date stamp to use (YYYYMMDD). If None, extracted from filename.
    """
    pickle_file = Path(pickle_file)

    # Extract date from filename if not provided
    if calibration_date is None:
        # Expect format: camera_calibrations_YYYYMMDD.pkl
        parts = pickle_file.stem.split('_')
        if len(parts) >= 3 and len(parts[-1]) == 8:
            calibration_date = parts[-1]
        else:
            raise ValueError(f"Cannot extract date from filename: {pickle_file.name}. "
                           f"Please provide calibration_date parameter.")

    # Set default output paths
    if output_csv is None:
        output_csv = pickle_file.parent / f"camera_calibrations_{calibration_date}.csv"
    else:
        output_csv = Path(output_csv)

    if cache_dir is None:
        cache_dir = pickle_file.parent.parent / 'orthorectification' / 'ortho_cache'
    else:
        cache_dir = Path(cache_dir)

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"="*70)
    print(f"MIGRATION SCRIPT: Legacy Pickle -> CSV + Cache")
    print(f"="*70)
    print(f"Input:  {pickle_file}")
    print(f"Output CSV: {output_csv}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Date stamp: {calibration_date}")
    print(f"="*70)
    print()

    # Load pickle file
    print("Loading pickle file...")
    with open(pickle_file, 'rb') as f:
        calibrations = pickle.load(f)

    print(f"Found {len(calibrations)} cameras: {', '.join(calibrations.keys())}\n")

    # Prepare CSV data
    csv_data = []

    for camera_id, calib in calibrations.items():
        print(f"Processing {camera_id}...")

        # Extract calibration parameters (flatten matrices)
        K = calib['K']
        D = calib['D'].flatten()  # (4,1) → (4,)
        rvec = calib['rvec'].flatten()  # (3,1) → (3,)
        tvec = calib['tvec'].flatten()  # (3,1) → (3,)

        geo = calib['geotransform']
        img_size = calib['image_size']

        # Build row for CSV
        row = {
            'camera_id': camera_id,
            'calibration_date': calibration_date,

            # Intrinsic matrix K (3x3 camera intrinsic matrix)
            'fx': K[0, 0], 'skew': K[0, 1], 'cx': K[0, 2],
            's': K[1, 0], 'fy': K[1, 1], 'cy': K[1, 2],
            'K_20': K[2, 0], 'K_21': K[2, 1], 'K_22': K[2, 2],

            # Distortion coefficients D (4, fisheye distortion)
            'k1': D[0], 'k2': D[1], 'k3': D[2], 'k4': D[3],

            # Rotation vector (3,)
            'rvec_x': rvec[0], 'rvec_y': rvec[1], 'rvec_z': rvec[2],

            # Translation vector (3,)
            'tvec_x': tvec[0], 'tvec_y': tvec[1], 'tvec_z': tvec[2],

            # Quality metrics
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

            # Output dimensions (for cache regeneration)
            'output_width': calib['output_width'],
            'output_height': calib['output_height'],

            # Recalibration info (may not exist in old files)
            'recalibrated': calib.get('recalibrated', False),
            'recalibration_mode': calib.get('recalibration_mode', ''),
            'gcps_skipped': calib.get('gcps_skipped', 0),

            # GCP pixel coordinates (will be populated on next recalibration)
            'gcp_pixel_coords': '',
        }

        csv_data.append(row)

        # Create ortho cache file for this camera
        cache_file = cache_dir / f"{camera_id}_ortho_cache_{calibration_date}.pkl"

        cache_data = {
            'map_x': calib['map_x'],
            'map_y': calib['map_y'],
            'output_width': calib['output_width'],
            'output_height': calib['output_height'],
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # Get cache file size
        size_mb = cache_file.stat().st_size / (1024**2)
        print(f"  OK Created cache: {cache_file.name} ({size_mb:.1f} MB)")

    # Create DataFrame and save CSV
    print(f"\nSaving CSV with {len(csv_data)} cameras...")
    df = pd.DataFrame(csv_data)

    # Reorder columns for better readability in Excel
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
        'gcp_pixel_coords'
    ]

    df = df[column_order]
    df.to_csv(output_csv, index=False, float_format='%.12f')

    csv_size_kb = output_csv.stat().st_size / 1024
    print(f"  OK CSV saved: {output_csv.name} ({csv_size_kb:.1f} KB)")

    # Summary
    print(f"\n{'='*70}")
    print(f"MIGRATION COMPLETE!")
    print(f"{'='*70}")
    print(f"CSV file:     {output_csv}")
    print(f"             ({len(csv_data)} cameras, {csv_size_kb:.1f} KB)")
    print(f"Cache files:  {cache_dir}")
    print(f"             ({len(csv_data)} files)")
    print(f"\nNext steps:")
    print(f"  1. Verify CSV opens correctly in Excel")
    print(f"  2. Check a few cache files can be loaded")
    print(f"  3. Test pipeline with new format")
    print(f"  4. DELETE THIS SCRIPT (TEMP_migrate_calibration_format.py)")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Migrate legacy pickle calibration to CSV + cache format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python TEMP_migrate_calibration_format.py calibration/camera_calibrations_20251119.pkl

This will create:
  - calibration/camera_calibrations_20251119.csv
  - orthorectification/ortho_cache/NVR1_N910A6_ch1_main_ortho_cache_20251119.pkl
  - orthorectification/ortho_cache/NVR1_N910A6_ch2_main_ortho_cache_20251119.pkl
  - ... (one cache file per camera)
"""
    )

    parser.add_argument('pickle_file', type=str,
                       help='Path to existing .pkl calibration file')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Output CSV path (default: same as pickle but .csv)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Cache directory (default: ../orthorectification/ortho_cache/)')
    parser.add_argument('--date', type=str, default=None,
                       help='Date stamp YYYYMMDD (default: extracted from filename)')

    args = parser.parse_args()

    migrate_calibration_file(
        pickle_file=args.pickle_file,
        output_csv=args.output_csv,
        cache_dir=args.cache_dir,
        calibration_date=args.date
    )
