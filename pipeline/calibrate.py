#!/usr/bin/env python3
"""Simple wrapper for initial camera calibration"""
import subprocess
import json
from pathlib import Path

# Module paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
ORTHORECTIFY = ROOT_DIR / 'orthorectification' / 'undistort_and_orthorectify.py'


def calibrate():
    """Run initial calibration using master_control.json settings"""
    
    # Load master config
    config_path = ROOT_DIR / 'master_control.json'
    if not config_path.exists():
        print("Error: master_control.json not found in root directory")
        print("This is needed for calibration paths (GCP file, DSM, etc.)")
        return
    
    with open(config_path) as f:
        content = f.read()

    import re
    content = re.sub(r':\\', r':/', content)
    content = content.replace('\\\\', '/')
    content = content.replace('\\', '/')
    
    config = json.loads(content)
    
    # Get calibration image directory
    calib_img_dir = input("Enter path to calibration images: ").strip()
    if not Path(calib_img_dir).exists():
        print(f"Error: {calib_img_dir} not found")
        return
    
    # Run calibration
    cmd = [
        'python', str(ORTHORECTIFY), 'calibrate',
        '-g', config['paths']['gcp_file'],
        '-i', calib_img_dir,
        '-d', config['paths']['dsm_file'],
        '-o', str(Path(config['paths']['calibration_file']).parent),
        '-r', str(config['processing']['ortho_resolution']),
        '-p', str(config['processing']['ortho_padding'])
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    
    print(f"\nCalibration saved to: {config['paths']['calibration_file']}")


if __name__ == '__main__':
    calibrate()
