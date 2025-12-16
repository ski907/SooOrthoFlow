"""
Video Mosaic Processor - Main Orchestrator

Coordinates the entire streaming video processing pipeline:
- Frame synchronization across multiple cameras
- Orthorectification using pre-computed caches
- In-memory mosaicking
- Video output writing

Author: SooOrthoFlow Team
Version: 0.1.0
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calibration.calibration_io import load_camera_calibrations, load_ortho_cache
from analysis.video_mosaic.camera_video_reader import CameraVideoReader
from analysis.video_mosaic.in_memory_mosaic import InMemoryMosaicEngine


class VideoMosaicProcessor:
    """
    Main orchestrator for streaming video mosaic processing.

    Implements single-pass processing:
    read frames → orthorectify → mosaic → write to video
    """

    def __init__(self, config: Dict):
        """
        Initialize processor with configuration.

        Parameters:
            config: Configuration dictionary with keys:
                    - input: video_dir, camera_ids, start_time, end_time, interval_seconds
                    - paths: calibration_file, dsm_file
                    - processing: ortho_resolution, mosaic_method, zone_map_shapefile
                    - output: output_dir, video_filename, video_fps, video_codec
                    - camera_time_offsets: dict mapping NVR names to offsets
        """
        self.config = config

        # Parse configuration sections
        self.video_dir = Path(config['input']['video_dir'])
        self.camera_ids = config['input']['camera_ids']
        self.start_time = self._parse_time(config['input']['start_time'])
        self.end_time = self._parse_time(config['input']['end_time'])
        self.interval_seconds = config['input']['interval_seconds']

        self.calibration_file = config['paths']['calibration_file']
        self.dsm_file = config['paths']['dsm_file']

        self.ortho_resolution = config['processing']['ortho_resolution']
        self.mosaic_method = config['processing']['mosaic_method']
        self.zone_map_shapefile = config['processing'].get('zone_map_shapefile')

        self.output_dir = Path(config['output']['output_dir'])
        self.video_filename = config['output']['video_filename']
        self.video_fps = config['output']['video_fps']
        self.video_codec = config['output']['video_codec']

        self.camera_time_offsets = config.get('camera_time_offsets', {})

        # Will be initialized in setup()
        self.calibrations = None
        self.ortho_caches = {}
        self.video_reader = None
        self.mosaic_engine = None
        self.video_writer = None

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime."""
        return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

    def _generate_timestamps(self) -> List[datetime]:
        """
        Generate list of timestamps at specified interval.

        Returns:
            List of datetime objects
        """
        timestamps = []
        current = self.start_time
        while current <= self.end_time:
            timestamps.append(current)
            current += timedelta(seconds=self.interval_seconds)
        return timestamps

    def setup(self):
        """
        Initialize all components (calibrations, caches, readers, engines).

        Returns:
            bool: True if successful
        """
        print("="*70)
        print("VIDEO MOSAIC PROCESSOR - SETUP")
        print("="*70)

        # 1. Load calibrations
        print(f"\n1. Loading calibrations from {self.calibration_file}...")
        try:
            self.calibrations = load_camera_calibrations(self.calibration_file)
            print(f"  Loaded calibrations for {len(self.calibrations)} cameras")
        except Exception as e:
            print(f"  ERROR: Could not load calibrations - {e}")
            return False

        # 2. Load ortho caches for each camera
        print(f"\n2. Loading ortho caches...")
        cache_dir = Path('orthorectification/ortho_cache')

        # Determine resolution name for cache
        if self.ortho_resolution <= 0.003:
            resolution_name = 'hires'
        else:
            resolution_name = 'lowres'

        for camera_id in self.camera_ids:
            if camera_id not in self.calibrations:
                print(f"  ERROR: No calibration found for {camera_id}")
                return False

            cal = self.calibrations[camera_id]

            # Create a temporary geotransform with correct pixel sizes for cache lookup
            # This fixes the hash mismatch issue when loading caches at different resolutions
            geotransform_for_cache = cal['geotransform'].copy()
            geotransform_for_cache['pixel_width'] = self.ortho_resolution
            geotransform_for_cache['pixel_height'] = -self.ortho_resolution

            # Try to load cache with corrected geotransform
            cache = load_ortho_cache(
                camera_id,
                cal['K'],
                cal['D'],
                cal['rvec'],
                cal['tvec'],
                geotransform_for_cache,  # Use corrected geotransform for hash computation
                self.ortho_resolution,
                resolution_name,
                cache_dir
            )

            if cache is None:
                print(f"  ERROR: No ortho cache found for {camera_id}")
                print(f"    Resolution: {self.ortho_resolution} ({resolution_name})")
                print(f"    Run orthorectification first to generate caches")
                return False

            # Store cache with corrected geotransform (keep original x_min/y_max bounds)
            self.ortho_caches[camera_id] = {
                'map_x': cache['map_x'],
                'map_y': cache['map_y'],
                'output_width': cache['output_width'],
                'output_height': cache['output_height'],
                'geotransform': {
                    'x_min': cal['geotransform']['x_min'],     # Keep original bounds
                    'y_max': cal['geotransform']['y_max'],     # Keep original bounds
                    'pixel_width': self.ortho_resolution,      # Use requested resolution
                    'pixel_height': -self.ortho_resolution     # Use requested resolution
                }
            }

            print(f"  OK {camera_id}: {cache['output_width']}x{cache['output_height']} px")

        # 3. Compute mosaic bounds from ortho caches
        print(f"\n3. Computing mosaic bounds...")
        mosaic_bounds = self._compute_mosaic_bounds()
        x_min, x_max, y_min, y_max = mosaic_bounds
        print(f"  Bounds: ({x_min:.2f}, {x_max:.2f}, {y_min:.2f}, {y_max:.2f})")

        # 4. Initialize video reader
        print(f"\n4. Initializing video reader...")
        try:
            self.video_reader = CameraVideoReader(
                self.video_dir,
                self.camera_ids,
                self.camera_time_offsets
            )
            print(f"  OK Video reader initialized")
        except Exception as e:
            print(f"  ERROR: Could not initialize video reader - {e}")
            return False

        # 5. Initialize mosaic engine
        print(f"\n5. Initializing mosaic engine ({self.mosaic_method} method)...")
        try:
            self.mosaic_engine = InMemoryMosaicEngine(
                self.camera_ids,
                mosaic_bounds,
                self.ortho_resolution,
                self.mosaic_method,
                self.zone_map_shapefile
            )
            print(f"  OK Mosaic engine initialized")
        except Exception as e:
            print(f"  ERROR: Could not initialize mosaic engine - {e}")
            return False

        # 6. Create output directory
        print(f"\n6. Creating output directory...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  OK Output: {self.output_dir}")

        # 7. Create video writer
        print(f"\n7. Initializing video writer...")
        video_path = self.output_dir / self.video_filename
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)

        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.video_fps,
            (self.mosaic_engine.mosaic_width, self.mosaic_engine.mosaic_height)
        )

        if not self.video_writer.isOpened():
            print(f"  ERROR: Could not create video writer")
            return False

        print(f"  OK Video: {video_path}")
        print(f"  Resolution: {self.mosaic_engine.mosaic_width}x{self.mosaic_engine.mosaic_height}")
        print(f"  FPS: {self.video_fps}, Codec: {self.video_codec}")

        print(f"\nSetup complete!\n")
        return True

    def _compute_mosaic_bounds(self):
        """
        Compute mosaic bounds from ortho cache geotransforms.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        x_mins, x_maxs, y_mins, y_maxs = [], [], [], []

        for camera_id, cache in self.ortho_caches.items():
            geotransform = cache['geotransform']
            width = cache['output_width']
            height = cache['output_height']

            x_min = geotransform['x_min']
            y_max = geotransform['y_max']
            x_max = x_min + width * geotransform['pixel_width']
            y_min = y_max + height * geotransform['pixel_height']

            x_mins.append(x_min)
            x_maxs.append(x_max)
            y_mins.append(y_min)
            y_maxs.append(y_max)

        # Return overall bounds
        return (min(x_mins), max(x_maxs), min(y_mins), max(y_maxs))

    def process(self):
        """
        Main processing loop.

        Returns:
            bool: True if successful
        """
        print("="*70)
        print("VIDEO MOSAIC PROCESSING")
        print("="*70)

        # Generate timestamps
        timestamps = self._generate_timestamps()
        print(f"\nProcessing {len(timestamps)} frames")
        print(f"  Start: {self.start_time}")
        print(f"  End:   {self.end_time}")
        print(f"  Interval: {self.interval_seconds} seconds\n")

        frames_processed = 0
        frames_skipped = 0

        for i, timestamp in enumerate(timestamps):
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(timestamps)}] {ts_str}")

            try:
                # 1. Get synchronized raw frames
                raw_frames = self.video_reader.get_frames_at_timestamp(timestamp)
                if raw_frames is None:
                    frames_skipped += 1
                    continue

                # 2. Orthorectify each frame
                ortho_images = {}
                for camera_id, raw_frame in raw_frames.items():
                    cache = self.ortho_caches[camera_id]

                    # Apply orthorectification using cached lookup tables
                    ortho = cv2.remap(
                        raw_frame,
                        cache['map_x'],
                        cache['map_y'],
                        cv2.INTER_LINEAR
                    )

                    ortho_images[camera_id] = (ortho, cache['geotransform'])

                # 3. Mosaic in memory
                mosaicked_frame = self.mosaic_engine.mosaic_frame(ortho_images)

                # 4. Write to output video
                self.video_writer.write(mosaicked_frame)

                frames_processed += 1

            except Exception as e:
                print(f"  ERROR at {ts_str}: {e}")
                frames_skipped += 1
                continue

        # Summary
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Frames processed: {frames_processed}")
        print(f"Frames skipped:   {frames_skipped}")

        return True

    def cleanup(self):
        """Release all resources."""
        print(f"\nCleaning up...")

        if self.video_writer:
            self.video_writer.release()
            print("  OK Video writer released")

        if self.video_reader:
            self.video_reader.close()
            print("  OK Video reader closed")

    def save_metadata(self):
        """Save geotransform and processing metadata to JSON."""
        print(f"\nSaving metadata...")

        metadata = {
            'processing_info': {
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': self.end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'interval_seconds': self.interval_seconds,
                'camera_ids': self.camera_ids,
                'ortho_resolution': self.ortho_resolution,
                'mosaic_method': self.mosaic_method,
                'video_fps': self.video_fps
            },
            'geotransform': self.mosaic_engine.get_geotransform(),
            'mosaic_bounds': {
                'x_min': self.mosaic_engine.mosaic_bounds[0],
                'x_max': self.mosaic_engine.mosaic_bounds[1],
                'y_min': self.mosaic_engine.mosaic_bounds[2],
                'y_max': self.mosaic_engine.mosaic_bounds[3]
            },
            'crs': 'EPSG:26919'
        }

        metadata_path = self.output_dir / 'mosaic_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  OK Metadata saved to {metadata_path}")

    def run(self):
        """
        Run the complete pipeline.

        Returns:
            bool: True if successful
        """
        try:
            # Setup
            if not self.setup():
                return False

            # Process
            if not self.process():
                return False

            # Cleanup
            self.cleanup()

            # Save metadata
            self.save_metadata()

            print(f"\n{'='*70}")
            print("SUCCESS")
            print(f"{'='*70}")
            print(f"Output video: {self.output_dir / self.video_filename}")
            print(f"Metadata:     {self.output_dir / 'mosaic_metadata.json'}\n")

            return True

        except Exception as e:
            print(f"\n{'='*70}")
            print("ERROR")
            print(f"{'='*70}")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Always cleanup
            if self.video_writer:
                self.video_writer.release()
            if self.video_reader:
                self.video_reader.close()
