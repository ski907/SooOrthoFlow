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
from multiprocessing import Pool, cpu_count

# Module paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
FRAME_EXTRACTOR = ROOT_DIR / 'frame_extraction' / 'frame_extractor_time_optimized.py'
ORTHORECTIFY = ROOT_DIR / 'orthorectification' / 'undistort_and_orthorectify.py'
MOSAIC = ROOT_DIR / 'orthorectification' / 'ortho_mosaic.py'
LIGHT_DETECTION = ROOT_DIR / 'analysis' / 'ship_detection' / 'batch_detect_lights.py'


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

    # Check time mode (default to "interval" for backward compatibility)
    time_mode = master_config.get('time_mode', 'interval')

    time_config = {
        "video_directory": video_dir,
        "mode": "time_range",
        "output_directory": str(output_dir),
        "output_format": master_config['processing']['output_format'],
        "recursive": master_config['processing']['recursive'],
        "filename_pattern": master_config['processing']['filename_pattern']
    }

    # Configure time range based on mode
    if time_mode == 'endpoints':
        # Extract only start and end frames (2 frames total)
        time_config["time_range"] = {
            "start": master_config['start_time'],
            "end": master_config['end_time'],
            "interval": None  # No interval, just endpoints
        }
        time_config["endpoints_only"] = True
    else:
        # Standard interval-based extraction
        time_config["time_range"] = {
            "start": master_config['start_time'],
            "end": master_config['end_time'],
            "interval": master_config['interval']
        }

    # Add camera time offsets if specified
    if 'camera_time_offsets' in master_config:
        time_config['camera_time_offsets'] = master_config['camera_time_offsets']

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

    # Print with encoding error handling for Windows console
    try:
        print(log_msg, flush=True)  # Force immediate flush
    except UnicodeEncodeError:
        # Replace Unicode symbols with ASCII for console
        safe_msg = log_msg.encode('ascii', errors='replace').decode('ascii')
        print(safe_msg, flush=True)  # Force immediate flush

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')
        f.flush()  # Force file flush too


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


def _process_single_ortho(args):
    """Worker function for parallel orthorectification"""
    ts_folder, ortho_base, calib_file, dem_file, orthorectify_script, resolution, resolution_name = args

    output_dir = ortho_base / ts_folder.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python', str(orthorectify_script), 'process',
        '-i', str(ts_folder),
        '-c', calib_file,
        '-o', str(output_dir),
        '--no-undistorted'
    ]

    # Add DEM path if provided (for cache regeneration)
    if dem_file:
        cmd.extend(['-d', dem_file])

    # Add resolution parameters if specified
    if resolution is not None:
        cmd.extend(['-r', str(resolution)])
    if resolution_name is not None:
        cmd.extend(['--resolution-name', resolution_name])

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

    return {
        'folder': ts_folder.name,
        'success': result.returncode == 0,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'resolution_name': resolution_name
    }


def run_orthorectification(master_config, log_file, n_jobs=None):
    """Run orthorectification on all timestamp folders in PARALLEL with multi-resolution support"""
    if n_jobs is None:
        n_jobs = cpu_count()

    log(log_file, f"Starting orthorectification (using {n_jobs} cores)...")

    test_dir = Path(master_config['paths']['output_base']) / master_config['test_id']
    frames_dir = test_dir / 'frames'
    ortho_base = test_dir / 'orthos'
    ortho_base.mkdir(parents=True, exist_ok=True)
    calib_file = master_config['paths']['calibration_file']
    dem_file = master_config['paths'].get('dsm_file')  # Optional, for cache regeneration

    # Get multi-resolution settings
    resolutions = master_config.get('resolutions', {'hires': 0.0025, 'lowres': 0.01})
    multi_res_config = master_config.get('multi_resolution', {})
    use_interval = multi_res_config.get('use_interval', True)
    interval_resolution = multi_res_config.get('interval_resolution', 'lowres')
    use_first_last = multi_res_config.get('use_first_last', False)
    first_last_resolution = multi_res_config.get('first_last_resolution', 'hires')

    # Get all timestamp folders
    timestamp_folders = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    # Determine which timestamps get which resolutions
    process_args = []
    for idx, ts_folder in enumerate(timestamp_folders):
        is_first = (idx == 0)
        is_last = (idx == len(timestamp_folders) - 1)

        # Determine resolution(s) for this timestamp
        resolutions_to_process = []

        # Add interval resolution if enabled and not first/last, OR if first/last are disabled
        if use_interval and not (is_first or is_last):
            resolutions_to_process.append((interval_resolution, resolutions[interval_resolution]))

        # Add first/last resolution if enabled and this is first or last frame
        if use_first_last and (is_first or is_last):
            resolutions_to_process.append((first_last_resolution, resolutions[first_last_resolution]))

        # If both are disabled for a timestamp, skip it
        # If first/last is disabled but interval is enabled, process first/last with interval settings
        if not resolutions_to_process:
            if use_interval:
                resolutions_to_process.append((interval_resolution, resolutions[interval_resolution]))

        # Create process args for each resolution
        for res_name, res_value in resolutions_to_process:
            process_args.append((
                ts_folder, ortho_base, calib_file, dem_file, ORTHORECTIFY,
                res_value, res_name
            ))

    log(log_file, f"Processing {len(timestamp_folders)} timestamps ({len(process_args)} total jobs)...")
    if use_interval:
        log(log_file, f"  Using {interval_resolution} ({resolutions[interval_resolution]}m) for intervals")
    if use_first_last:
        log(log_file, f"  Using {first_last_resolution} ({resolutions[first_last_resolution]}m) for first and last frames")

    # Process in parallel
    with Pool(n_jobs) as pool:
        results = pool.map(_process_single_ortho, process_args)
    
    # Log results and verify files were created
    success_count = 0
    for result in results:
        res_label = f" [{result['resolution_name']}]" if 'resolution_name' in result else ""
        if result['success']:
            # Verify output files actually exist
            ts_output_dir = ortho_base / result['folder'] / 'orthorectified'
            ortho_files = list(ts_output_dir.glob('*_ortho*.tif')) if ts_output_dir.exists() else []

            if ortho_files:
                log(log_file, f"  ✓ {result['folder']}{res_label} ({len(ortho_files)} files)")
                success_count += 1
            else:
                log(log_file, f"  ✗ {result['folder']}{res_label}: No output files created!")
                log(log_file, f"     STDOUT: {result['stdout'][:200]}")
                log(log_file, f"     STDERR: {result['stderr'][:200]}")
        else:
            log(log_file, f"  ✗ {result['folder']}{res_label}: {result['stderr']}")

    log(log_file, f"Orthorectification complete ({success_count}/{len(process_args)} jobs succeeded)")

    if success_count == 0:
        log(log_file, "\n✗ ERROR: No orthorectified images were created!")
        log(log_file, "Check that camera names in frames match calibration file")
        return False

    return True


def _process_single_mosaic(args):
    """Worker function for parallel mosaicking"""
    ts_folder, mosaic_dir, method, mosaic_script, world_file, transform_params, clip_shapefile, keep_intermediate, save_downscaled, downscaled_resolution, compress_mosaics, zone_map_shapefile = args

    # Find orthorectified subfolder
    ortho_folder = ts_folder / 'orthorectified'
    if not ortho_folder.exists():
        return {
            'folder': ts_folder.name,
            'success': False,
            'error': 'No orthorectified folder found'
        }

    # Detect ALL unique resolutions in the folder
    ortho_files = list(ortho_folder.glob('*_ortho*.tif'))
    if not ortho_files:
        return {
            'folder': ts_folder.name,
            'success': False,
            'error': 'No ortho files found'
        }

    # Extract all unique resolution suffixes
    resolutions = set()
    for f in ortho_files:
        if '_ortho_' in f.stem:
            res_part = f.stem.split('_ortho_')[-1]  # e.g., "25mm" or "2mm"
            resolutions.add(res_part)
        else:
            resolutions.add("2.5mm")  # legacy files without suffix

    # Process each resolution separately
    all_results = []
    for res_str in sorted(resolutions):
        output_file = mosaic_dir / f"mosaic_{ts_folder.name}_{res_str}.tif"

        cmd = [
            'python', str(mosaic_script),
            str(ortho_folder),
            '-o', str(output_file),
            '-m', method
        ]

        # Add world file transformation if specified
        if world_file:
            cmd.extend(['--world-file', str(world_file)])
            # Add optional transformation parameters
            if transform_params.get('resampling'):
                cmd.extend(['--world-resampling', transform_params['resampling']])
            if transform_params.get('threads'):
                cmd.extend(['--world-threads', str(transform_params['threads'])])
            if transform_params.get('memory_mb'):
                cmd.extend(['--world-memory', str(transform_params['memory_mb'])])

        # Add shapefile clipping if specified
        if clip_shapefile:
            cmd.extend(['--clip-shapefile', str(clip_shapefile)])

        # Add keep-intermediate flag if specified
        if keep_intermediate:
            cmd.append('--keep-intermediate')

        # Add downscaled mosaic options if specified
        if save_downscaled:
            cmd.append('--save-downscaled')
            cmd.extend(['--downscaled-resolution', str(downscaled_resolution)])

        # Add compression flag if disabled
        if not compress_mosaics:
            cmd.append('--no-compress')

        # Add zone map shapefile if using zone_map method
        if zone_map_shapefile:
            cmd.extend(['--zone-map-shapefile', str(zone_map_shapefile)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        all_results.append({
            'folder': f"{ts_folder.name}_{res_str}",
            'success': result.returncode == 0,
            'stderr': result.stderr if result.returncode != 0 else None
        })

    # Return combined result - success if all resolutions succeeded
    return {
        'folder': ts_folder.name,
        'success': all(r['success'] for r in all_results),
        'stderr': '\n'.join(r['stderr'] for r in all_results if r['stderr']) if not all(r['success'] for r in all_results) else None,
        'resolutions': list(resolutions)
    }


def run_mosaicking(master_config, log_file, n_jobs=None):
    """Create mosaics for each timestamp in PARALLEL"""
    if n_jobs is None:
        n_jobs = cpu_count()

    log(log_file, f"Starting mosaicking (using {n_jobs} cores)...")

    test_dir = Path(master_config['paths']['output_base']) / master_config['test_id']
    ortho_base = test_dir / 'orthos'
    mosaic_dir = test_dir / 'mosaics'
    mosaic_dir.mkdir(exist_ok=True)

    method = master_config['processing']['mosaic_method']

    # Check if world file transformation is requested
    apply_transform = master_config['processing'].get('apply_world_transform', False)
    world_file = None
    transform_params = {}

    if apply_transform:
        world_file_path = master_config['processing'].get('world_file_path', 'orthorectification/model_to_world.wld')
        world_file = ROOT_DIR / world_file_path
        if not world_file.exists():
            log(log_file, f"  Warning: World file not found: {world_file}")
            log(log_file, f"  Proceeding without coordinate transformation")
            world_file = None
        else:
            log(log_file, f"  Using world file for coordinate transformation: {world_file}")

            # Get optional transformation parameters
            transform_params = {
                'resampling': master_config['processing'].get('world_transform_resampling', 'bilinear'),
                'threads': master_config['processing'].get('world_transform_threads', None),
                'memory_mb': master_config['processing'].get('world_transform_memory_mb', 512)
            }
            log(log_file, f"  Transform settings: resampling={transform_params['resampling']}, "
                         f"threads={transform_params['threads'] or 'auto'}, "
                         f"memory={transform_params['memory_mb']}MB")

    # Check if shapefile clipping is requested
    clip_shapefile = None
    clip_shapefile_path = master_config['processing'].get('clip_shapefile', None)
    if clip_shapefile_path:
        clip_shapefile = ROOT_DIR / clip_shapefile_path
        if not clip_shapefile.exists():
            log(log_file, f"  Warning: Clip shapefile not found: {clip_shapefile}")
            log(log_file, f"  Proceeding without clipping")
            clip_shapefile = None
        else:
            log(log_file, f"  Using shapefile for clipping: {clip_shapefile}")

    # Check if intermediate mosaics should be kept
    keep_intermediate = master_config['processing'].get('keep_intermediate_mosaics', False)
    if keep_intermediate:
        log(log_file, f"  Keeping model-space clipped mosaics (for debugging/inspection)")
    else:
        log(log_file, f"  Deleting model-space clipped mosaics after transformation")

    # Check if downscaled mosaics should be saved
    save_downscaled = master_config['processing'].get('save_downscaled_mosaic', False)
    downscaled_resolution = master_config['processing'].get('downscaled_resolution', 0.25)
    if save_downscaled:
        log(log_file, f"  Saving downscaled mosaics at {downscaled_resolution}m/pixel ({int(downscaled_resolution*100)}cm/pixel)")

    # Check if compression should be applied
    compress_mosaics = master_config['processing'].get('compress_mosaics', True)
    if compress_mosaics:
        log(log_file, f"  Using LZW compression for mosaics")
    else:
        log(log_file, f"  Saving mosaics without compression (larger files)")

    # Check if zone map is being used
    zone_map_shapefile = None
    if method == 'zone_map':
        zone_map_path = master_config['processing'].get('zone_map_shapefile',
                                                          'orthorectification/camera_zone_map/camera_zone_map.shp')
        zone_map_shapefile = ROOT_DIR / zone_map_path
        if not zone_map_shapefile.exists():
            log(log_file, f"  ERROR: Zone map shapefile not found: {zone_map_shapefile}")
            return False
        log(log_file, f"  Using zone map for spatial ordering: {zone_map_shapefile}")

    # Get all timestamp folders
    timestamp_folders = sorted([d for d in ortho_base.iterdir() if d.is_dir()])

    log(log_file, f"Creating {len(timestamp_folders)} mosaics in parallel...")

    # Prepare arguments for parallel processing
    mosaic_args = [
        (ts_folder, mosaic_dir, method, MOSAIC, world_file, transform_params, clip_shapefile, keep_intermediate, save_downscaled, downscaled_resolution, compress_mosaics, zone_map_shapefile)
        for ts_folder in timestamp_folders
    ]

    # Process in parallel (with special handling for zone_map to avoid race condition)
    if method == 'zone_map' and len(mosaic_args) > 1:
        # Generate zone map cache with first mosaic to avoid parallel race condition
        log(log_file, f"  Processing first mosaic to generate zone map cache...")
        first_result = _process_single_mosaic(mosaic_args[0])
        results = [first_result]

        # Now process remaining mosaics in parallel
        log(log_file, f"  Processing remaining {len(mosaic_args)-1} mosaics in parallel...")
        with Pool(n_jobs) as pool:
            remaining_results = pool.map(_process_single_mosaic, mosaic_args[1:])
        results.extend(remaining_results)
    else:
        # Normal parallel processing for other methods or single mosaic
        with Pool(n_jobs) as pool:
            results = pool.map(_process_single_mosaic, mosaic_args)

    # Log results
    success_count = 0
    for result in results:
        if result['success']:
            log(log_file, f"  ✓ {result['folder']}")
            success_count += 1
        else:
            log(log_file, f"  ✗ {result['folder']}: {result.get('error', result.get('stderr', 'Unknown error'))}")

    log(log_file, f"Mosaicking complete ({success_count}/{len(timestamp_folders)} succeeded)")
    return True


def run_light_detection(master_config, log_file):
    """Detect boat lights in mosaics (optional post-processing)"""
    log(log_file, "Starting light detection...")

    test_dir = Path(master_config['paths']['output_base']) / master_config['test_id']
    mosaic_dir = test_dir / 'mosaics'
    light_dir = test_dir / 'light_detections'
    light_dir.mkdir(exist_ok=True)

    # Find all mosaic files
    mosaic_files = sorted(mosaic_dir.glob('mosaic_*.tif'))

    if not mosaic_files:
        log(log_file, "  No mosaics found to process")
        return True

    log(log_file, f"  Found {len(mosaic_files)} mosaics to process")

    # Get mask path if specified
    mask_path = master_config['processing'].get('light_detection_mask', None)
    if mask_path:
        mask_path_obj = Path(mask_path)
        if not mask_path_obj.is_absolute():
            # Make relative paths relative to ROOT_DIR
            mask_path_obj = ROOT_DIR / mask_path
        if not mask_path_obj.exists():
            log(log_file, f"  Warning: Mask file not found: {mask_path_obj}")
            mask_path = None
        else:
            mask_path = str(mask_path_obj)

    # Prepare output paths
    output_shapefile = str(light_dir / 'detected_lights.shp')
    output_csv = str(light_dir / 'detected_lights.csv')

    # Convert mosaic paths to strings
    image_paths = [str(p) for p in mosaic_files]

    # Import and run light detection function
    try:
        sys.path.insert(0, str(ROOT_DIR / 'analysis' / 'ship_detection'))
        from batch_detect_lights import batch_detect_and_export

        # Run the detection
        batch_detect_and_export(
            image_paths=image_paths,
            mask_path=mask_path,
            output_shapefile=output_shapefile,
            output_csv=output_csv,
            n_workers=None  # Auto-detect CPU count
        )

        log(log_file, "Light detection complete")
        return True

    except Exception as e:
        log(log_file, f"  ERROR: Light detection failed: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log(log_file, f"  {line}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run thermal image processing pipeline')
    parser.add_argument('--config', default='master_control.json', help='Master config file')
    parser.add_argument('--extract-only', action='store_true', help='Only extract frames')
    parser.add_argument('--process-only', action='store_true', help='Only orthorectify and mosaic')
    parser.add_argument('--mosaic-only', action='store_true', help='Only create mosaics')
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of parallel jobs (default: all cores)')
    
    args = parser.parse_args()
    
    # Load config and generate time_config
    master_config = load_master_config(args.config)
    
    if not args.process_only and not args.mosaic_only:
        generate_time_config(master_config)
    
    # Setup test folder
    test_dir, log_file = setup_test_folder(master_config)
    
    # Run pipeline steps
    start_time = datetime.now()

    # Check if light detection is enabled
    run_lights = master_config['processing'].get('run_light_detection', False)

    if args.mosaic_only:
        run_mosaicking(master_config, log_file, n_jobs=args.jobs)
        if run_lights:
            run_light_detection(master_config, log_file)
    elif args.extract_only:
        run_extraction(master_config, log_file)
    elif args.process_only:
        run_orthorectification(master_config, log_file, n_jobs=args.jobs)
        run_mosaicking(master_config, log_file, n_jobs=args.jobs)
        if run_lights:
            run_light_detection(master_config, log_file)
    else:
        # Full pipeline
        if run_extraction(master_config, log_file):
            if run_orthorectification(master_config, log_file, n_jobs=args.jobs):
                if run_mosaicking(master_config, log_file, n_jobs=args.jobs):
                    if run_lights:
                        run_light_detection(master_config, log_file)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    log(log_file, f"\nPipeline completed in {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {test_dir}")


if __name__ == '__main__':
    main()