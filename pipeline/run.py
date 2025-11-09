#!/usr/bin/env python3
"""
Pipeline orchestrator for thermal image processing
Reads master_control.json, generates time_config.json, runs processing
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import shutil
import argparse

# Module paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
FRAME_EXTRACTOR = ROOT_DIR / 'frame_extraction' / 'frame_extractor_time_optimized.py'
ORTHORECTIFY = ROOT_DIR / 'orthorectification' / 'undistort_and_orthorectify.py'
MOSAIC = ROOT_DIR / 'orthorectification' / 'ortho_mosaic.py'


def load_master_config(config_path='master_control.json'):
    """Load and validate master config"""
    if not Path(config_path).exists():
        print(f"Error: {config_path} not found")
        print("Create one from master_control.template.json")
        sys.exit(1)
    
    # Read file as text and fix backslashes before JSON parsing
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace single backslashes with forward slashes in string values
    # This handles Windows paths pasted directly
    import re
    # Find all string values and replace backslashes
    content = re.sub(r':\\', r':/', content)  # C:\ -> C:/
    content = content.replace('\\\\', '/')     # \\ -> /
    content = content.replace('\\', '/')       # \ -> /
    
    config = json.loads(content)
    
    # Extract test_id from video_folder path (last folder name)
    video_folder = Path(config['video_folder'])
    config['test_id'] = video_folder.name
    
    return config


def generate_time_config(master_config):
    """Generate time_config.json for frame extractor"""
    test_id = master_config['test_id']
    video_dir = master_config['video_folder']
    output_dir = Path(master_config['paths']['output_base']) / test_id / 'frames'
    
    time_config = {
        "video_directory": video_dir,
        "mode": "time_range",
        "time_range": {
            "start": master_config['start_time'],
            "end": master_config['end_time'],
            "interval": master_config['interval']
        },
        "output_directory": str(output_dir),
        "output_format": master_config['processing']['output_format'],
        "recursive": master_config['processing']['recursive'],
        "filename_pattern": master_config['processing']['filename_pattern']
    }
    
    with open('time_config.json', 'w') as f:
        json.dump(time_config, f, indent=4)
    
    return time_config


def setup_test_folder(master_config):
    """Create test folder structure and save configs"""
    test_dir = Path(master_config['paths']['output_base']) / master_config['test_id']
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save master config copy
    shutil.copy('master_control.json', test_dir / 'master_control.json')
    
    # Save generated time config
    shutil.copy('time_config.json', test_dir / 'time_config.json')
    
    # Create log file
    log_file = test_dir / 'processing_log.txt'
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Processing Log for {master_config['test_id']}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
    
    return test_dir, log_file


def log(log_file, message):
    """Write to log and print"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')


def run_extraction(master_config, log_file, show_output=True):
    """Run frame extraction"""
    log(log_file, "Starting frame extraction...")
    
    import os
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    cmd = ['python', str(FRAME_EXTRACTOR), 'time_config.json']
    
    if show_output:
        # Show live progress in console (no detailed logging)
        result = subprocess.run(cmd, env=env)
    else:
        # Log everything but no console output
        with open(log_file, 'a', encoding='utf-8') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    
    if result.returncode != 0:
        log(log_file, f"ERROR: Frame extraction failed (exit code {result.returncode})")
        return False
    
    log(log_file, "Frame extraction complete")
    return True


def run_orthorectification(master_config, log_file):
    """Run orthorectification on all timestamp folders"""
    log(log_file, "Starting orthorectification...")
    
    test_dir = Path(master_config['paths']['output_base']) / master_config['test_id']
    frames_dir = test_dir / 'frames'
    ortho_base = test_dir / 'orthos'
    ortho_base.mkdir(parents=True, exist_ok=True)  # Create base orthos directory
    calib_file = master_config['paths']['calibration_file']
    
    # Process each timestamp folder
    timestamp_folders = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    
    for ts_folder in timestamp_folders:
        log(log_file, f"  Processing {ts_folder.name}...")
        
        output_dir = ortho_base / ts_folder.name
        output_dir.mkdir(parents=True, exist_ok=True)  # Create timestamp output directory
        
        cmd = [
            'python', str(ORTHORECTIFY), 'process',
            '-i', str(ts_folder),
            '-c', calib_file,
            '-o', str(output_dir),
            '--no-undistorted'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            log(log_file, f"  ERROR processing {ts_folder.name}: {result.stderr}")
            continue
        
        # Show what happened
        if result.stdout:
            log(log_file, f"  Output: {result.stdout.strip()}")
        if result.stderr:
            log(log_file, f"  Warnings: {result.stderr.strip()}")
    
    log(log_file, f"Orthorectification complete ({len(timestamp_folders)} timestamps)")
    return True


def run_mosaicking(master_config, log_file):
    """Create mosaics for each timestamp"""
    log(log_file, "Starting mosaicking...")
    
    test_dir = Path(master_config['paths']['output_base']) / master_config['test_id']
    ortho_base = test_dir / 'orthos'
    mosaic_dir = test_dir / 'mosaics'
    mosaic_dir.mkdir(exist_ok=True)
    
    method = master_config['processing']['mosaic_method']
    
    # Process each timestamp
    timestamp_folders = sorted([d for d in ortho_base.iterdir() if d.is_dir()])
    
    for ts_folder in timestamp_folders:
        log(log_file, f"  Mosaicking {ts_folder.name}...")
        
        # Find orthorectified subfolder
        ortho_folder = ts_folder / 'orthorectified'
        if not ortho_folder.exists():
            log(log_file, f"  WARNING: No orthorectified folder in {ts_folder.name}")
            continue
        
        output_file = mosaic_dir / f"mosaic_{ts_folder.name}.tif"
        
        cmd = [
            'python', str(MOSAIC),
            str(ortho_folder),
            '-o', str(output_file),
            '-m', method
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            log(log_file, f"  ERROR mosaicking {ts_folder.name}: {result.stderr}")
            continue
    
    log(log_file, f"Mosaicking complete ({len(timestamp_folders)} mosaics)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run thermal image processing pipeline')
    parser.add_argument('--config', default='master_control.json', help='Master config file')
    parser.add_argument('--extract-only', action='store_true', help='Only extract frames')
    parser.add_argument('--process-only', action='store_true', help='Only orthorectify and mosaic')
    parser.add_argument('--mosaic-only', action='store_true', help='Only create mosaics')
    
    args = parser.parse_args()
    
    # Load config and generate time_config
    master_config = load_master_config(args.config)
    
    if not args.process_only and not args.mosaic_only:
        generate_time_config(master_config)
    
    # Setup test folder
    test_dir, log_file = setup_test_folder(master_config)
    
    # Run pipeline steps
    start_time = datetime.now()
    
    if args.mosaic_only:
        run_mosaicking(master_config, log_file)
    elif args.extract_only:
        run_extraction(master_config, log_file)
    elif args.process_only:
        run_orthorectification(master_config, log_file)
        run_mosaicking(master_config, log_file)
    else:
        # Full pipeline
        if run_extraction(master_config, log_file):
            if run_orthorectification(master_config, log_file):
                run_mosaicking(master_config, log_file)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    log(log_file, f"\nPipeline completed in {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {test_dir}")


if __name__ == '__main__':
    main()