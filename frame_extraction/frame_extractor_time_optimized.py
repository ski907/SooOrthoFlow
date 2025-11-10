#!/usr/bin/env python3
"""
Video Frame Extractor with Clock Time Support

Extracts high-quality still frames from video files at specified timestamps
using a JSON configuration file. Supports single files, batch processing,
and extraction at specific clock times across multiple cameras.
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
import cv2
import glob
from collections import defaultdict
from multiprocessing import Pool, cpu_count


# Supported video file extensions
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.dav'}


def parse_filename(filename, pattern="CAMERA_DATETIME_DATETIME"):
    """
    Parse filename to extract camera name, start time, and end time.
    """
    if pattern == "CAMERA_DATETIME_DATETIME":
        stem = Path(filename).stem
        stem = re.sub(r'\(\d+\)$', '', stem)
        parts = stem.split('_')
        
        if len(parts) < 3:
            return None
            
        datetime_parts = []
        camera_parts = []
        
        for part in parts:
            if len(part) == 14 and part.isdigit():
                datetime_parts.append(part)
            else:
                if len(datetime_parts) < 2:
                    camera_parts.append(part)
        
        if len(datetime_parts) < 2:
            return None
            
        camera_name = '_'.join(camera_parts)
        start_time_str = datetime_parts[0]
        end_time_str = datetime_parts[1]
        
        try:
            start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")
            end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M%S")
            
            return {
                'camera': camera_name,
                'start_time': start_time,
                'end_time': end_time,
                'filename': filename
            }
        except ValueError:
            return None
    
    return None


def parse_timestamp(timestamp_str):
    """Parse timestamp string to seconds."""
    try:
        if ':' in timestamp_str:
            parts = timestamp_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            return float(timestamp_str)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def parse_clock_time(clock_time_str):
    """Parse clock time string to datetime object."""
    if len(clock_time_str) == 14 and clock_time_str.isdigit():
        try:
            return datetime.strptime(clock_time_str, "%Y%m%d%H%M%S")
        except ValueError:
            pass
    
    try:
        return datetime.strptime(clock_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    
    raise ValueError(f"Invalid clock time format: {clock_time_str}")


def parse_interval(interval_str):
    """Parse interval string to seconds."""
    pattern = r'^([\d.]+)(s|sec|second|seconds|m|min|minute|minutes|h|hr|hour|hours)$'
    match = re.match(pattern, interval_str.lower().strip())
    
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}")
    
    value = float(match.group(1))
    unit = match.group(2)
    
    if unit in ['s', 'sec', 'second', 'seconds']:
        return value
    elif unit in ['m', 'min', 'minute', 'minutes']:
        return value * 60
    elif unit in ['h', 'hr', 'hour', 'hours']:
        return value * 3600
    
    raise ValueError(f"Unknown time unit: {unit}")


def generate_clock_times_from_range(start_str, end_str, interval_str):
    """Generate a list of clock times from start to end at specified intervals."""
    start_time = parse_clock_time(start_str)
    end_time = parse_clock_time(end_str)
    interval_seconds = parse_interval(interval_str)
    
    if start_time >= end_time:
        raise ValueError(f"Start time must be before end time")
    
    if interval_seconds <= 0:
        raise ValueError(f"Interval must be positive")
    
    clock_times = []
    current_time = start_time
    
    while current_time <= end_time:
        clock_times.append(current_time)
        current_time += timedelta(seconds=interval_seconds)
    
    return clock_times


def find_video_files(directory, recursive=False):
    """Find all video files in a directory."""
    video_files = []
    search_pattern = "**/*" if recursive else "*"
    
    for ext in VIDEO_EXTENSIONS:
        pattern = os.path.join(directory, search_pattern + ext)
        files = glob.glob(pattern, recursive=recursive)
        video_files.extend(files)
        
        pattern = os.path.join(directory, search_pattern + ext.upper())
        files = glob.glob(pattern, recursive=recursive)
        video_files.extend(files)
    
    return sorted(set(video_files))


def organize_videos_by_camera(video_files, filename_pattern="CAMERA_DATETIME_DATETIME", base_directory=None):
    """Organize video files by camera and time ranges."""
    cameras = defaultdict(list)
    
    for video_file in video_files:
        info = parse_filename(video_file, filename_pattern)
        if info:
            if base_directory:
                video_path = Path(video_file)
                try:
                    rel_path = video_path.relative_to(Path(base_directory))
                    parent_parts = rel_path.parent.parts
                    if parent_parts:
                        folder_prefix = "_".join(parent_parts).replace(" ", "_")
                        info['camera'] = f"{folder_prefix}_{info['camera']}"
                except ValueError:
                    pass
            
            cameras[info['camera']].append(info)
        else:
            print(f"Warning: Could not parse filename: {Path(video_file).name}")
    
    for camera in cameras:
        cameras[camera].sort(key=lambda x: x['start_time'])
    
    return cameras


def find_video_for_clock_time(camera_videos, target_time):
    """Find the video file that contains the target clock time."""
    for video_info in camera_videos:
        if video_info['start_time'] <= target_time <= video_info['end_time']:
            return video_info
    return None


def extract_frame_at_clock_time(cap, camera, video_info, target_time, output_dir, output_format, time_subfolder=None):
    """Extract a frame from an already-opened video at a specific clock time."""
    start_time = video_info['start_time']
    
    time_offset = target_time - start_time
    offset_seconds = time_offset.total_seconds()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    if offset_seconds > duration or offset_seconds < 0:
        print(f"  ✗ Time {target_time.strftime('%H:%M:%S')} out of range")
        return False
    
    frame_number = int(offset_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"  ✗ Could not extract frame at {target_time.strftime('%H:%M:%S')}")
        return False
    
    actual_output_dir = output_dir
    if time_subfolder:
        actual_output_dir = os.path.join(output_dir, time_subfolder)
        Path(actual_output_dir).mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{camera}.{output_format}"
    output_path = os.path.join(actual_output_dir, output_filename)
    
    if output_format == 'png':
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif output_format in ['jpg', 'jpeg']:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif output_format == 'tiff':
        save_params = []
    elif output_format == 'bmp':
        save_params = []
    else:
        save_params = []
    
    if cv2.imwrite(output_path, frame, save_params):
        print(f"  ✓ {target_time.strftime('%H:%M:%S')} → {time_subfolder}/{output_filename}")
        return True
    else:
        print(f"  ✗ Failed to save frame at {target_time.strftime('%H:%M:%S')}")
        return False

def _process_single_camera(args):
    """Worker function to process a single camera in parallel"""
    camera, camera_videos, target_times, output_dir, output_format = args
    
    video_timestamps = defaultdict(list)
    
    for target_time in target_times:
        video_info = find_video_for_clock_time(camera_videos, target_time)
        if video_info:
            video_timestamps[video_info['filename']].append(target_time)
    
    if not video_timestamps:
        return {'camera': camera, 'success': 0}
    
    success_count = 0
    
    for video_file, timestamps_for_video in video_timestamps.items():
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            continue
        
        video_info = None
        for vi in camera_videos:
            if vi['filename'] == video_file:
                video_info = vi
                break
        
        if not video_info:
            cap.release()
            continue
        
        for target_time in timestamps_for_video:
            time_subfolder = target_time.strftime('%Y%m%d_%H%M%S')
            if extract_frame_at_clock_time(cap, camera, video_info, target_time, 
                                          output_dir, output_format, time_subfolder):
                success_count += 1
        
        cap.release()
    
    return {'camera': camera, 'success': success_count}



def extract_frames_at_clock_times(video_directory, clock_times, output_dir, 
                                output_format, recursive=False, 
                                filename_pattern="CAMERA_DATETIME_DATETIME",
                                n_jobs=None):
    """Extract frames from multiple cameras at specific clock times - OPTIMIZED VERSION."""
    # Parse and sort clock times
    target_times = []
    for clock_time in clock_times:
        if isinstance(clock_time, datetime):
            target_times.append(clock_time)
        else:
            try:
                target_time = parse_clock_time(clock_time)
                target_times.append(target_time)
            except ValueError as e:
                print(f"Error parsing clock time: {e}")
                return False
    
    # Sort timestamps for efficient sequential access
    target_times.sort()
    
    video_files = find_video_files(video_directory, recursive)
    if not video_files:
        print(f"No video files found")
        return False
    
    print(f"Found {len(video_files)} video file(s)")
    
    cameras = organize_videos_by_camera(video_files, filename_pattern, 
                                       base_directory=video_directory if recursive else None)
    
    if not cameras:
        print("No videos could be parsed")
        return False
    
    print(f"Found {len(cameras)} camera(s): {', '.join(cameras.keys())}")
    print(f"Processing {len(target_times)} timestamp(s)")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build a mapping of which timestamps need which videos for each camera
    # This allows us to open each video file only once
    total_success = 0
    
    print(f"\n{'='*60}")
    print("OPTIMIZED PROCESSING: Opening each video once")
    print(f"{'='*60}")
    

    if n_jobs is None:
        n_jobs = cpu_count()
    
    print(f"Using {n_jobs} CPU cores")
    print(f"\n{'='*60}")
    print("PARALLEL PROCESSING")
    print(f"{'='*60}\n")
    
    # Prepare arguments for parallel processing
    camera_args = [
        (camera, camera_videos, target_times, output_dir, output_format)
        for camera, camera_videos in cameras.items()
    ]
    
    # Process cameras in parallel
    with Pool(n_jobs) as pool:
        results = pool.map(_process_single_camera, camera_args)
    
    # Report results
    total_success = sum(r['success'] for r in results)
    for result in results:
        if result['success'] > 0:
            print(f"OK {result['camera']}: {result['success']} frames")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_success} frames extracted")
    print(f"{'='*60}")
    
    return total_success > 0


def extract_frames_from_video(video_file, timestamps, output_dir, output_format, base_directory=None):
    """Extract frames from a single video file."""
    video_path = Path(video_file)
    print(f"\n{'='*20} Processing: {video_path.name} {'='*20}")
    
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    if base_directory:
        try:
            rel_path = video_path.relative_to(Path(base_directory))
            parent_parts = rel_path.parent.parts
            if parent_parts:
                folder_prefix = "_".join(parent_parts).replace(" ", "_")
                video_name_with_path = f"{folder_prefix}_{video_path.stem}"
            else:
                video_name_with_path = video_path.stem
        except ValueError:
            video_name_with_path = video_path.stem
    else:
        video_name_with_path = video_path.stem
    
    if base_directory and video_name_with_path != video_path.stem:
        print(f"Output prefix: {video_name_with_path}")
    
    success_count = 0
    for i, timestamp_str in enumerate(timestamps):
        try:
            timestamp_seconds = parse_timestamp(timestamp_str)
            
            if timestamp_seconds > duration:
                print(f"Warning: Timestamp {timestamp_str} exceeds duration")
                continue
            
            frame_number = int(timestamp_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not extract frame at {timestamp_str}")
                continue
            
            timestamp_folder = f"timestamp_{timestamp_str.replace(':', 'm').replace('.', 's')}"
            timestamp_output_dir = os.path.join(output_dir, timestamp_folder)
            Path(timestamp_output_dir).mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{video_name_with_path}.{output_format}"
            output_path = os.path.join(timestamp_output_dir, output_filename)
            
            if output_format == 'png':
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            elif output_format in ['jpg', 'jpeg']:
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            elif output_format == 'tiff':
                save_params = []
            elif output_format == 'bmp':
                save_params = []
            else:
                save_params = []
            
            if cv2.imwrite(output_path, frame, save_params):
                print(f"✓ Extracted frame at {timestamp_str} → {output_filename}")
                success_count += 1
            else:
                print(f"✗ Failed to save frame at {timestamp_str}")
                
        except ValueError as e:
            print(f"✗ Error: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    
    cap.release()
    print(f"Complete: {success_count}/{len(timestamps)} frames")
    return success_count


def extract_frames(config_path):
    """Extract frames based on configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        return False
    
    has_single_file = 'video_file' in config
    has_directory = 'video_directory' in config
    has_clock_times = 'clock_times' in config
    has_time_range = 'time_range' in config
    has_timestamps = 'timestamps' in config
    mode = config.get('mode', None)
    
    if not has_single_file and not has_directory:
        print("Error: Must have 'video_file' or 'video_directory'")
        return False
    
    if has_single_file and has_directory:
        print("Error: Cannot have both 'video_file' and 'video_directory'")
        return False
    
    using_clock_times = False
    using_time_range = False
    using_timestamps = False
    
    if mode:
        if mode == "clock_times":
            if not has_clock_times:
                print("Error: mode is 'clock_times' but field missing")
                return False
            using_clock_times = True
        elif mode == "time_range":
            if not has_time_range:
                print("Error: mode is 'time_range' but field missing")
                return False
            using_time_range = True
        elif mode == "timestamps":
            if not has_timestamps:
                print("Error: mode is 'timestamps' but field missing")
                return False
            using_timestamps = True
        else:
            print(f"Error: Invalid mode '{mode}'")
            return False
    else:
        time_specs = sum([has_clock_times, has_time_range, has_timestamps])
        
        if time_specs == 0:
            print("Error: Must have time specification")
            return False
        
        if time_specs > 1:
            print("Error: Multiple time specs - specify 'mode' field")
            return False
        
        if has_clock_times:
            using_clock_times = True
        elif has_time_range:
            using_time_range = True
        elif has_timestamps:
            using_timestamps = True
    
    output_dir = config.get('output_directory', 'frames')
    output_format = config.get('output_format', 'png').lower()
    
    valid_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    if output_format not in valid_formats:
        print(f"Error: Unsupported format '{output_format}'")
        return False
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_success = 0
    
    if using_clock_times or using_time_range:
        if not has_directory:
            print("Error: Clock times require 'video_directory'")
            return False
        
        video_directory = config['video_directory']
        recursive = config.get('recursive', False)
        filename_pattern = config.get('filename_pattern', 'CAMERA_DATETIME_DATETIME')
        
        if using_time_range:
            time_range = config['time_range']
            
            if 'start' not in time_range or 'end' not in time_range or 'interval' not in time_range:
                print("Error: time_range needs 'start', 'end', 'interval'")
                return False
            
            try:
                clock_times_dt = generate_clock_times_from_range(
                    time_range['start'],
                    time_range['end'],
                    time_range['interval']
                )
                
                print(f"Time range mode: {len(clock_times_dt)} timestamps")
                print(f"Start: {time_range['start']}")
                print(f"End: {time_range['end']}")
                print(f"Interval: {time_range['interval']}")
                
            except ValueError as e:
                print(f"Error in time_range: {e}")
                return False
        else:
            clock_times_dt = config['clock_times']
            print(f"Clock times mode: {len(clock_times_dt)} timestamp(s)")
        
        print(f"Video directory: {video_directory}")
        print(f"Output directory: {output_dir}")
        print(f"Format: {output_format.upper()}")
        print(f"Recursive: {recursive}")
        
        return extract_frames_at_clock_times(
            video_directory, clock_times_dt, output_dir, output_format, 
            recursive, filename_pattern
        )
    
    else:
        timestamps = config['timestamps']
        
        print(f"Output directory: {output_dir}")
        print(f"Format: {output_format.upper()}")
        print(f"Timestamps: {', '.join(timestamps)}")
        
        if has_single_file:
            video_file = config['video_file']
            if not os.path.exists(video_file):
                print(f"Error: Video file not found")
                return False
            
            total_success = extract_frames_from_video(video_file, timestamps, output_dir, output_format, None)
            
        else:
            video_directory = config['video_directory']
            recursive = config.get('recursive', False)
            
            if not os.path.exists(video_directory):
                print(f"Error: Directory not found")
                return False
            
            if not os.path.isdir(video_directory):
                print(f"Error: Not a directory")
                return False
            
            video_files = find_video_files(video_directory, recursive)
            
            if not video_files:
                print(f"No video files found")
                return False
            
            print(f"Found {len(video_files)} video file(s)")
            
            for video_file in video_files:
                success_count = extract_frames_from_video(
                    video_file, timestamps, output_dir, output_format,
                    base_directory=video_directory if recursive else None
                )
                total_success += success_count
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_success} frames extracted")
    
    return total_success > 0


def create_sample_config(filename="config.json", config_type="single"):
    """Create a sample configuration file."""
    if config_type == "clock":
        sample_config = {
            "video_directory": "path/to/videos",
            "mode": "clock_times",
            "clock_times": ["20250925151530", "20250925152000"],
            "time_range": {
                "start": "2025-09-25 15:15:00",
                "end": "2025-09-25 16:00:00",
                "interval": "2min"
            },
            "output_directory": "frames",
            "output_format": "png",
            "recursive": False,
            "filename_pattern": "CAMERA_DATETIME_DATETIME"
        }
        config_desc = "clock time extraction"
    elif config_type == "batch":
        sample_config = {
            "video_directory": "path/to/videos",
            "timestamps": ["0:10", "1:30", "2:45.5"],
            "output_directory": "frames",
            "output_format": "png",
            "recursive": False
        }
        config_desc = "batch processing"
    else:
        sample_config = {
            "video_file": "example.avi",
            "timestamps": ["0:10", "1:30", "2:45.5"],
            "output_directory": "frames",
            "output_format": "png"
        }
        config_desc = "single file"
    
    with open(filename, 'w') as f:
        json.dump(sample_config, f, indent=4)
    
    print(f"Sample config for {config_desc} created: {filename}")


def process_directory_direct(video_directory, timestamps_str=None, clock_times_str=None,
                           output_dir="frames", output_format="png", recursive=False):
    """Process directory from command line."""
    if timestamps_str and clock_times_str:
        print("Error: Cannot specify both timestamps and clock-times")
        return False
    
    if not timestamps_str and not clock_times_str:
        print("Error: Must specify timestamps or clock-times")
        return False
    
    if not os.path.exists(video_directory):
        print(f"Error: Directory not found")
        return False
    
    if not os.path.isdir(video_directory):
        print(f"Error: Not a directory")
        return False
    
    valid_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    if output_format.lower() not in valid_formats:
        print(f"Error: Unsupported format")
        return False
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if clock_times_str:
        clock_times = [ct.strip() for ct in clock_times_str.split(',')]
        print(f"Clock time processing mode")
        
        return extract_frames_at_clock_times(
            video_directory, clock_times, output_dir, output_format.lower(), recursive
        )
    
    else:
        timestamps = [ts.strip() for ts in timestamps_str.split(',')]
        video_files = find_video_files(video_directory, recursive)
        
        if not video_files:
            print(f"No video files found")
            return False
        
        print(f"Processing {len(video_files)} file(s)")
        
        total_success = 0
        for video_file in video_files:
            success_count = extract_frames_from_video(
                video_file, timestamps, output_dir, output_format.lower(),
                base_directory=video_directory if recursive else None
            )
            total_success += success_count
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: {total_success} frames extracted")
        
        return total_success > 0


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    
    parser.add_argument('config_file', nargs='?', help='JSON config file')
    parser.add_argument('-d', '--directory', help='Video directory')
    parser.add_argument('-t', '--timestamps', help='Comma-separated timestamps')
    parser.add_argument('-c', '--clock-times', help='Comma-separated clock times')
    parser.add_argument('-o', '--output', default='frames', help='Output directory')
    parser.add_argument('-f', '--format', default='png', choices=['png', 'jpg', 'jpeg', 'tiff', 'bmp'])
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('--sample-config', action='store_true')
    parser.add_argument('--sample-config-batch', action='store_true')
    parser.add_argument('--sample-config-clock', action='store_true')
    
    args = parser.parse_args()
    
    if args.sample_config:
        create_sample_config("config.json", "single")
        return
    
    if args.sample_config_batch:
        create_sample_config("config_batch.json", "batch")
        return
        
    if args.sample_config_clock:
        create_sample_config("config_clock.json", "clock")
        return
    
    if args.directory:
        if not args.timestamps and not args.clock_times:
            print("Error: Need --timestamps or --clock-times")
            return
        if args.config_file:
            print("Warning: config_file ignored")
        
        success = process_directory_direct(
            args.directory, args.timestamps, args.clock_times,
            args.output, args.format, args.recursive
        )
        sys.exit(0 if success else 1)
    
    if not args.config_file:
        parser.print_help()
        return
    
    success = extract_frames(args.config_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()