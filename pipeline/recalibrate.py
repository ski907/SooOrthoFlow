#!/usr/bin/env python3
"""Wrapper for periodic recalibration"""
import subprocess
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime

# Module paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
RECALIBRATE_CAMERA = ROOT_DIR / 'calibration' / 'recalibrate_camera.py'


def recalibrate_single_camera(camera_name, image_path, master_config):
    """Recalibrate a single camera and update the calibration file"""
    
    print(f"\nRecalibrating camera: {camera_name}")
    print(f"Using image: {image_path}")
    
    # Add orthorectification directory to Python path so recalibrate_camera.py can import
    import sys
    ortho_dir = str(ROOT_DIR / 'orthorectification')
    if ortho_dir not in sys.path:
        sys.path.insert(0, ortho_dir)
    
    # Convert relative paths to absolute
    gcp_file = ROOT_DIR / master_config['paths']['gcp_file']
    dsm_file = ROOT_DIR / master_config['paths']['dsm_file']
    cal_file = ROOT_DIR / master_config['paths']['calibration_file']
    
    # Map new camera name to original GCP camera name
    # Pattern: {folder}_N910A6_ch{X}_main -> N910A6_ch{global_num}_main
    # folder1 = cameras 1-8, folder2 = cameras 9-16, etc.
    gcp_camera_name = camera_name
    
    import re
    # Match pattern like: dav1_N910A6_ch3_main or dvi2_N910A6_ch7_main
    match = re.match(r'([a-z]+)(\d+)_N910A6_ch(\d+)_main', camera_name)
    
    if match:
        prefix = match.group(1)  # e.g., 'dav' or 'dvi'
        folder_num = int(match.group(2))  # e.g., 1 or 2
        ch_num = int(match.group(3))  # e.g., 3 or 7
        
        # Calculate global camera number
        # folder1 ch1 = cam 1, folder1 ch8 = cam 8
        # folder2 ch1 = cam 9, folder2 ch8 = cam 16
        global_cam = (folder_num - 1) * 8 + ch_num
        
        gcp_camera_name = f'N910A6_ch{global_cam:02d}_main'
        print(f"Mapping {camera_name} -> {gcp_camera_name} (folder {folder_num}, ch {ch_num} = global camera {global_cam})")
    
    # Get resolution from master config
    resolution = master_config['processing'].get('ortho_resolution', 0.005)
    padding = master_config['processing'].get('ortho_padding', 0.5)

    cmd = [
        'python', str(RECALIBRATE_CAMERA),
        '-i', str(image_path),
        '-g', str(gcp_file),
        '-c', gcp_camera_name,  # Use GCP camera name
        '-d', str(dsm_file),
        '-cal', str(cal_file),
        '-r', str(resolution),
        '-p', str(padding)
    ]
    
    # Set PYTHONPATH environment variable
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = ortho_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    print(f"\nRunning recalibration...\n")
    result = subprocess.run(cmd, env=env, cwd=str(ROOT_DIR))
    
    if result.returncode == 0:
        print(f"\n✓ Recalibration successful for {gcp_camera_name}")

        # Now update the calibration file with the FULL camera name (new format)
        if gcp_camera_name != camera_name:
            # Extract date from timestamp folder path to find the correct dated calibration file
            import re
            image_path_obj = Path(image_path)
            date = None

            # Check parent directory names for date (e.g., frames/20251016_103100/)
            for parent in image_path_obj.parents:
                date_match = re.search(r'(\d{8})', parent.name)
                if date_match:
                    date = date_match.group(1)
                    break

            # Fall back to today's date if not found
            if date is None:
                date = datetime.now().strftime('%Y%m%d')

            cal_path = Path(cal_file)
            dated_cal_file = cal_path.parent / f"{cal_path.stem}_{date}.pkl"

            if dated_cal_file.exists():
                print(f"Updating calibration key from {gcp_camera_name} to {camera_name}")
                with open(dated_cal_file, 'rb') as f:
                    calibrations = pickle.load(f)

                if gcp_camera_name in calibrations:
                    calibrations[camera_name] = calibrations.pop(gcp_camera_name)

                    with open(dated_cal_file, 'wb') as f:
                        pickle.dump(calibrations, f)
                    print(f"✓ Calibration key updated to {camera_name} in {dated_cal_file}")
            else:
                print(f"⚠ Warning: Could not find dated calibration file {dated_cal_file}")
    else:
        print(f"\n✗ Recalibration failed for {camera_name}")

    return result.returncode == 0


def recalibrate_from_test(test_id, timestamp, master_config):
    """Recalibrate all cameras from a specific test timestamp"""
    
    output_base = master_config['paths']['output_base']
    frames_dir = Path(output_base) / test_id / 'frames' / timestamp
    
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return False
    
    # Get all image files
    image_files = list(frames_dir.glob('*.tiff')) + list(frames_dir.glob('*.tif'))
    
    if not image_files:
        print(f"Error: No images found in {frames_dir}")
        return False
    
    print(f"\nFound {len(image_files)} images to recalibrate")
    
    success_count = 0
    fail_count = 0
    
    for img_file in sorted(image_files):
        # Extract camera name from filename
        camera_name = img_file.stem.replace('.tiff', '').replace('.tif', '')
        
        if recalibrate_single_camera(camera_name, img_file, master_config):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Recalibration complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"{'='*60}")
    
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(description='Recalibrate cameras using recent images')
    parser.add_argument('--camera', help='Single camera to recalibrate (e.g., dav1_N910A6_ch3_main)')
    parser.add_argument('--image', help='Image file for single camera recalibration')
    parser.add_argument('--test-id', help='Test ID to recalibrate from (e.g., test_20251016_Nav1)')
    parser.add_argument('--timestamp', help='Timestamp folder to use (e.g., 20251016_103100)')
    
    args = parser.parse_args()
    
    # Load master config
    config_path = ROOT_DIR / 'master_control.json'
    if not config_path.exists():
        print("Error: master_control.json not found in root directory")
        return
    
    # Read and fix Windows backslashes
    with open(config_path, 'r') as f:
        content = f.read()
    
    import re
    content = re.sub(r':\\', r':/', content)
    content = content.replace('\\\\', '/')
    content = content.replace('\\', '/')
    
    config = json.loads(content)
    
    # Mode 1: Single camera recalibration
    if args.camera and args.image:
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            return
        
        recalibrate_single_camera(args.camera, args.image, config)
    
    # Mode 2: Recalibrate all from a test
    elif args.test_id and args.timestamp:
        recalibrate_from_test(args.test_id, args.timestamp, config)
    
    # Mode 3: Interactive - ask for path
    else:
        print("Recalibration Options:")
        print("1. Recalibrate single camera")
        print("2. Recalibrate all cameras from a test")
        choice = input("\nChoice (1 or 2): ").strip()
        
        if choice == '1':
            # List available tests
            output_base = Path(config['paths']['output_base'])
            tests = sorted([d.name for d in output_base.iterdir() if d.is_dir()])

            if not tests:
                print("No tests found in data directory")
                return

            print("\nAvailable tests:")
            for i, test in enumerate(tests, 1):
                print(f"  {i}. {test}")

            test_choice = input(f"\nSelect test (1-{len(tests)}): ").strip()
            try:
                test_id = tests[int(test_choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice")
                return

            # List timestamps
            frames_dir = output_base / test_id / 'frames'
            if not frames_dir.exists():
                print(f"No frames found for {test_id}")
                return

            timestamps = sorted([d.name for d in frames_dir.iterdir() if d.is_dir()])
            if not timestamps:
                print(f"No timestamps found for {test_id}")
                return

            print(f"\nAvailable timestamps for {test_id}:")
            for i, ts in enumerate(timestamps, 1):
                print(f"  {i}. {ts}")

            ts_choice = input(f"\nSelect timestamp (1-{len(timestamps)}): ").strip()
            try:
                timestamp = timestamps[int(ts_choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice")
                return

            # Loop for recalibrating multiple cameras
            while True:
                # List cameras
                ts_dir = frames_dir / timestamp
                images = sorted(ts_dir.glob('*.tif*'))
                cameras = [img.stem for img in images]

                print(f"\nAvailable cameras in {test_id}/{timestamp}:")
                for i, cam in enumerate(cameras, 1):
                    print(f"  {i}. {cam}")

                cam_choice = input(f"\nSelect camera (1-{len(cameras)}, or 'q' to quit): ").strip()

                if cam_choice.lower() == 'q':
                    print("Exiting recalibration")
                    break

                try:
                    camera_idx = int(cam_choice) - 1
                    camera = cameras[camera_idx]
                    image = images[camera_idx]
                except (ValueError, IndexError):
                    print("Invalid choice, try again")
                    continue

                # Recalibrate the selected camera
                recalibrate_single_camera(camera, image, config)

                # Ask if user wants to recalibrate another
                another = input("\nRecalibrate another camera? (y/n): ").strip().lower()
                if another != 'y':
                    print("Done with recalibration")
                    break
        
        elif choice == '2':
            # List available tests
            output_base = Path(config['paths']['output_base'])
            tests = sorted([d.name for d in output_base.iterdir() if d.is_dir()])
            
            if not tests:
                print("No tests found in data directory")
                return
            
            print("\nAvailable tests:")
            for i, test in enumerate(tests, 1):
                print(f"  {i}. {test}")
            
            test_choice = input(f"\nSelect test (1-{len(tests)}): ").strip()
            try:
                test_id = tests[int(test_choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice")
                return
            
            # List timestamps
            frames_dir = output_base / test_id / 'frames'
            timestamps = sorted([d.name for d in frames_dir.iterdir() if d.is_dir()])
            
            print(f"\nAvailable timestamps for {test_id}:")
            for i, ts in enumerate(timestamps, 1):
                print(f"  {i}. {ts}")
            
            ts_choice = input(f"\nSelect timestamp (1-{len(timestamps)}): ").strip()
            try:
                timestamp = timestamps[int(ts_choice) - 1]
            except (ValueError, IndexError):
                print("Invalid choice")
                return
            
            recalibrate_from_test(test_id, timestamp, config)
        
        else:
            print("Invalid choice")


if __name__ == '__main__':
    main()