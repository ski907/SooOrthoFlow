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
from analysis.video_mosaic.camera_selector import CameraSelector


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
                    - input: video_dir, camera_selection (or camera_ids for backward compat),
                            start_time, end_time, interval_seconds, error_handling
                    - paths: calibration_file, dsm_file
                    - processing: ortho_resolution, mosaic_method, zone_map_shapefile
                    - output: output_dir, video_filename, video_fps, video_codec
                    - camera_time_offsets: dict mapping NVR names to offsets
        """
        self.config = config

        # Parse configuration sections
        self.video_dir = Path(config['input']['video_dir'])

        # Handle backward compatibility: camera_ids → camera_selection
        if 'camera_ids' in config['input'] and 'camera_selection' not in config['input']:
            # Old format: migrate to new format
            config['input']['camera_selection'] = {
                'mode': 'list',
                'cameras': config['input']['camera_ids']
            }

        # Parse camera selection config (will be resolved in setup())
        self.camera_selection_config = config['input'].get('camera_selection', {'mode': 'list', 'cameras': []})
        self.error_handling_config = config['input'].get('error_handling', {
            'skip_missing_calibration': False,
            'skip_missing_cache': False,
            'skip_missing_videos': True,
            'min_cameras_required': 1
        })

        # camera_ids will be set during setup() after camera selection
        self.camera_ids = []

        self.start_time = self._parse_time(config['input']['start_time'])
        self.end_time = self._parse_time(config['input']['end_time'])
        self.interval_seconds = config['input']['interval_seconds']

        self.calibration_file = config['paths']['calibration_file']
        self.dsm_file = config['paths']['dsm_file']

        self.ortho_resolution = config['processing']['ortho_resolution']
        self.mosaic_method = config['processing']['mosaic_method']
        self.zone_map_shapefile = config['processing'].get('zone_map_shapefile')
        self.rotation_angle_deg = config['processing'].get('rotation_angle_deg', 0.0)
        self.clip_shapefile = config['processing'].get('clip_shapefile')

        self.output_dir = Path(config['output']['output_dir'])
        self.video_filename = config['output']['video_filename']
        self.video_fps = config['output']['video_fps']
        self.video_codec = config['output']['video_codec']
        self.verbose_frames = config['output'].get('verbose_frames', False)

        self.camera_time_offsets = config.get('camera_time_offsets', {})

        # Ice flux analysis configuration
        self.ice_flux_config = config.get('ice_flux', {'enabled': False})
        self.ice_flux_analyzer = None

        # Will be initialized in setup()
        self.calibrations = None
        self.ortho_caches = {}
        self.video_reader = None
        self.mosaic_engine = None
        self.video_writer = None
        self.clip_mask = None  # Pre-computed clip mask if using shapefile (full size)
        self.cropped_clip_mask = None  # Clip mask cropped to content bounds (set after first frame)
        self.final_crop_bounds = None  # Crop bounds after clipping (set after first frame)

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
        Initialize all components with flexible camera selection.

        Returns:
            bool: True if successful
        """
        print("="*70)
        print("VIDEO MOSAIC PROCESSOR - SETUP")
        print("="*70)

        # 1. Resolve camera selection and validate dependencies
        print(f"\n1. Resolving camera selection...")
        mode = self.camera_selection_config.get('mode', 'list')
        print(f"  Mode: {mode}")

        selector = CameraSelector(
            self.camera_selection_config,
            self.calibration_file,
            self.ortho_resolution,
            self.error_handling_config,
            dsm_file=self.dsm_file  # Pass DSM for automatic cache generation
        )

        selection_result = selector.select_cameras()

        # 2. Display selection results
        print(f"\n2. Camera selection results:")
        print(f"  Selected: {len(selection_result.selected_cameras)} cameras")

        if selection_result.selected_cameras:
            for camera_id in selection_result.selected_cameras:
                status = selection_result.validation_status.get(camera_id, {})
                cache_status = "OK" if status.get('has_cache') else "X"
                video_status = "OK" if status.get('has_videos') else "?"
                print(f"    {camera_id} (cache: {cache_status}, videos: {video_status})")

        # 3. Show warnings
        if selection_result.warnings:
            print(f"\n  Warnings:")
            for warning in selection_result.warnings:
                print(f"    WARNING: {warning}")

        # 4. Check for errors
        if selection_result.errors:
            print(f"\n  Errors:")
            for error in selection_result.errors:
                print(f"    ERROR: {error}")
            return False

        if not selection_result.selected_cameras:
            print(f"  ERROR: No cameras selected")
            return False

        # 5. Update processor state with selected cameras
        self.camera_ids = selection_result.selected_cameras
        self.calibrations = selector.calibrations  # Already loaded by selector
        self.ortho_caches = selector.ortho_caches  # Already loaded by selector

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
                self.camera_time_offsets,
                verbose_frames=self.verbose_frames
            )
            print(f"  OK Video reader initialized")
            if self.verbose_frames:
                print(f"  Verbose frame logging enabled")
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

        # 5.5. Initialize ice flux analyzer if enabled
        if self.ice_flux_config.get('enabled', False):
            print(f"\n5.5. Initializing ice flux analyzer...")
            try:
                from analysis.video_mosaic.ice_flux import IceFluxAnalyzer

                # Add time delta and output directory to config
                ice_flux_config = self.ice_flux_config.copy()
                ice_flux_config['time_delta_seconds'] = self.interval_seconds
                ice_flux_config['output_dir'] = self.output_dir / 'ice_flux'
                ice_flux_config['video_fps'] = self.video_fps
                ice_flux_config['video_codec'] = self.video_codec

                self.ice_flux_analyzer = IceFluxAnalyzer(
                    ice_flux_config,
                    mosaic_geotransform_getter=self.mosaic_engine.get_geotransform
                )
                self.ice_flux_analyzer.setup()
                print(f"  OK Ice flux analyzer initialized")
            except Exception as e:
                print(f"  ERROR: Could not initialize ice flux analyzer - {e}")
                if not self.ice_flux_config.get('optional', True):
                    return False
                else:
                    print(f"  Continuing without ice flux analysis (optional=True)")
                    self.ice_flux_analyzer = None

        # 6. Create output directory
        print(f"\n6. Creating output directory...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  OK Output: {self.output_dir}")

        # 7. Determine output dimensions (may be cropped for zone_map method)
        print(f"\n7. Determining output dimensions...")
        self.output_width = self.mosaic_engine.mosaic_width
        self.output_height = self.mosaic_engine.mosaic_height

        # Note: For zone_map with auto-crop, dimensions will be updated after first frame
        self.video_writer = None
        self.video_path = self.output_dir / self.video_filename

        # 8. Create clip mask if shapefile provided
        if self.clip_shapefile:
            print(f"\n8. Loading clip shapefile...")
            try:
                self.clip_mask = self._create_clip_mask(
                    self.clip_shapefile,
                    self.mosaic_engine.mosaic_bounds,
                    self.mosaic_engine.mosaic_width,
                    self.mosaic_engine.mosaic_height,
                    self.ortho_resolution
                )
                clip_pixels = np.sum(self.clip_mask)
                total_pixels = self.clip_mask.size
                print(f"  OK Clip mask created: {clip_pixels}/{total_pixels} pixels ({100*clip_pixels/total_pixels:.1f}%) in analysis area")
            except Exception as e:
                print(f"  ERROR: Could not create clip mask - {e}")
                return False

        print(f"\n{'8. ' if not self.clip_shapefile else '9. '}Determining output dimensions...")
        print(f"  Initial dimensions: {self.output_width}x{self.output_height}")
        if self.mosaic_method == 'zone_map':
            print(f"  Note: Will auto-crop to content after first frame")
        if self.clip_shapefile:
            print(f"  Note: Will clip to shapefile analysis area")
        if self.rotation_angle_deg != 0.0:
            print(f"  Note: Will rotate frames by {self.rotation_angle_deg}° counterclockwise")

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

    def _compute_content_bounds(self, frame: np.ndarray):
        """
        Compute the bounding box of non-zero content in a frame.

        Parameters:
            frame: (H, W, 3) numpy array

        Returns:
            Tuple of (row_min, row_max, col_min, col_max)
        """
        # Find pixels where any channel has non-zero values
        content_mask = np.any(frame > 0, axis=2)

        # Find the bounding box
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # No content, return full frame
            return (0, frame.shape[0], 0, frame.shape[1])

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Add 1 to max indices for slicing (Python slice is exclusive)
        row_max += 1
        col_max += 1

        return (row_min, row_max, col_min, col_max)

    def _rotate_frame(self, frame: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate frame counterclockwise by specified angle.

        Parameters:
            frame: Input frame (H, W, 3)
            angle_deg: Rotation angle in degrees (counterclockwise)

        Returns:
            Rotated frame
        """
        height, width = frame.shape[:2]
        center = (width / 2, height / 2)

        # Get rotation matrix (counterclockwise rotation)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # Calculate new bounding box size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation
        rotated = cv2.warpAffine(frame, rotation_matrix, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

        return rotated

    def _create_clip_mask(self, shapefile_path: str, mosaic_bounds: tuple,
                         mosaic_width: int, mosaic_height: int, resolution: float) -> np.ndarray:
        """
        Create a binary mask from shapefile for clipping.

        Parameters:
            shapefile_path: Path to shapefile
            mosaic_bounds: (x_min, x_max, y_min, y_max) in model coordinates
            mosaic_width, mosaic_height: Dimensions of mosaic
            resolution: Pixel resolution

        Returns:
            Boolean mask array (H, W) where True = keep, False = clip
        """
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import Affine

        # Load shapefile
        gdf = gpd.read_file(shapefile_path)

        # Create affine transform for rasterization
        x_min, x_max, y_min, y_max = mosaic_bounds
        transform = Affine(resolution, 0, x_min,
                          0, -resolution, y_max)

        # Rasterize the shapefile polygons
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(mosaic_height, mosaic_width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        return mask.astype(bool)

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

        # Initialize per-camera frame counters
        camera_frame_counts = {cam_id: 0 for cam_id in self.camera_ids}

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

                # Track which cameras provided frames
                for camera_id in raw_frames:
                    camera_frame_counts[camera_id] += 1

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

                # 3.5. Apply shapefile clipping if specified (before rotation)
                if self.clip_mask is not None:
                    # After first frame, crop the clip mask to match content bounds
                    if self.cropped_clip_mask is None and self.mosaic_engine.content_bounds is not None:
                        row_min, row_max, col_min, col_max = self.mosaic_engine.content_bounds
                        self.cropped_clip_mask = self.clip_mask[row_min:row_max, col_min:col_max]
                        print(f"Cropped clip mask to content bounds: {self.cropped_clip_mask.shape}")

                    # Apply the appropriate mask (cropped if available, otherwise full)
                    mask_to_use = self.cropped_clip_mask if self.cropped_clip_mask is not None else self.clip_mask
                    clipped_frame = mosaicked_frame.copy()
                    clipped_frame[~mask_to_use] = 0  # Black out pixels outside analysis area
                    mosaicked_frame = clipped_frame

                    # After first frame with clipping, compute final crop bounds
                    if self.final_crop_bounds is None:
                        self.final_crop_bounds = self._compute_content_bounds(mosaicked_frame)
                        row_min, row_max, col_min, col_max = self.final_crop_bounds
                        print(f"Final crop bounds after clipping: [{row_min}:{row_max}, {col_min}:{col_max}]")
                        print(f"Final cropped dimensions: {col_max-col_min} x {row_max-row_min} pixels")

                    # Crop to final content area (removes black padding from clipped regions)
                    row_min, row_max, col_min, col_max = self.final_crop_bounds
                    mosaicked_frame = mosaicked_frame[row_min:row_max, col_min:col_max]

                # 3.55. Ice flux analysis (in UTM coordinates, before rotation)
                if self.ice_flux_analyzer:
                    try:
                        overlay_frame = self.ice_flux_analyzer.process_frame(mosaicked_frame, timestamp)
                        # overlay_frame is used later if creating overlay video
                    except Exception as e:
                        print(f"  WARNING: Ice flux analysis failed at {timestamp.strftime('%H:%M:%S')} - {e}")

                # 3.6. Apply rotation if specified
                if self.rotation_angle_deg != 0.0:
                    mosaicked_frame = self._rotate_frame(mosaicked_frame, self.rotation_angle_deg)

                # 4. Create video writer after first frame (now we know the output dimensions)
                if self.video_writer is None:
                    actual_height, actual_width = mosaicked_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                    self.video_writer = cv2.VideoWriter(
                        str(self.video_path),
                        fourcc,
                        self.video_fps,
                        (actual_width, actual_height)
                    )
                    if not self.video_writer.isOpened():
                        raise RuntimeError("Could not create video writer")
                    print(f"Video writer created: {actual_width}x{actual_height} @ {self.video_fps} fps\n")

                # 5. Write to output video
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

        # Per-camera statistics
        print(f"\nCamera frame statistics:")
        for camera_id in self.camera_ids:
            count = camera_frame_counts.get(camera_id, 0)
            total = frames_processed
            pct = 100.0 * count / total if total > 0 else 0
            status = "OK" if pct > 95 else "!"
            print(f"  {camera_id}: {count}/{total} ({pct:.1f}%) {status}")

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

        if self.ice_flux_analyzer:
            self.ice_flux_analyzer.cleanup()
            print("  OK Ice flux analyzer cleaned up")

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
