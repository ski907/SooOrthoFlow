"""
Camera Selection and Validation Module

Provides flexible camera selection from config with comprehensive validation.
Supports multiple selection modes: all, list, exclude, spatial (future).

Author: SooOrthoFlow Team
Version: 0.1.0
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calibration.calibration_io import (
    load_camera_calibrations,
    load_ortho_cache,
    save_ortho_cache
)
import numpy as np


@dataclass
class CameraSelectionResult:
    """Results of camera selection and validation."""
    selected_cameras: List[str] = field(default_factory=list)
    validation_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class CameraSelector:
    """
    Resolve camera selection from config and validate dependencies.

    Supports multiple selection modes:
    - 'all': Select all cameras from calibration file
    - 'list': Explicit list of camera IDs
    - 'exclude': All cameras except those in exclude list
    - 'spatial': (Future) Select by geographic bounds
    """

    def __init__(self, selection_config: Dict, calibration_file: str,
                 ortho_resolution: float, error_handling: Optional[Dict] = None,
                 dsm_file: Optional[str] = None):
        """
        Initialize camera selector.

        Parameters:
            selection_config: Camera selection configuration with:
                - mode: 'all', 'list', 'exclude', 'spatial'
                - cameras: List of camera IDs (for mode='list')
                - exclude: List of camera IDs to exclude (for mode='exclude')
            calibration_file: Path to camera_calibrations_YYYYMMDD.csv
            ortho_resolution: Orthorectification resolution in meters/pixel
            error_handling: Optional error handling config with:
                - skip_missing_calibration: bool (default False)
                - skip_missing_cache: bool (default False)
                - skip_missing_videos: bool (default True)
                - min_cameras_required: int (default 1)
            dsm_file: Optional path to DSM file for generating missing caches
        """
        self.selection_config = selection_config
        self.calibration_file = calibration_file
        self.ortho_resolution = ortho_resolution
        self.dsm_file = dsm_file

        # Parse error handling config
        error_handling = error_handling or {}
        self.skip_missing_calibration = error_handling.get('skip_missing_calibration', False)
        self.skip_missing_cache = error_handling.get('skip_missing_cache', False)
        self.skip_missing_videos = error_handling.get('skip_missing_videos', True)
        self.min_cameras_required = error_handling.get('min_cameras_required', 1)

        # Will be loaded during selection
        self.calibrations = None
        self.ortho_caches = {}

    def select_cameras(self) -> CameraSelectionResult:
        """
        Main entry point - resolve and validate camera selection.

        Returns:
            CameraSelectionResult with selected cameras and validation details
        """
        result = CameraSelectionResult()

        # 1. Load calibrations (all cameras available)
        try:
            self.calibrations = load_camera_calibrations(self.calibration_file)
        except Exception as e:
            result.errors.append(f"Failed to load calibration file: {e}")
            return result

        # 2. Resolve selection mode to candidate list
        try:
            candidates = self._resolve_selection_mode()
        except Exception as e:
            result.errors.append(f"Failed to resolve selection mode: {e}")
            return result

        if not candidates:
            result.errors.append("No cameras in candidate list")
            return result

        # 3. Validate each candidate camera
        result = self._validate_cameras(candidates)

        # 4. Check minimum cameras requirement
        self._check_minimum_cameras(result)

        return result

    def _resolve_selection_mode(self) -> List[str]:
        """
        Convert selection config to camera ID list.

        Returns:
            List of candidate camera IDs
        """
        mode = self.selection_config.get('mode', 'list')

        if mode == 'all':
            # Select all cameras from calibration file
            return list(self.calibrations.keys())

        elif mode == 'list':
            # Explicit list of cameras
            cameras = self.selection_config.get('cameras', [])
            if not cameras:
                raise ValueError("mode='list' requires 'cameras' array")
            return cameras

        elif mode == 'exclude':
            # All cameras except excluded ones
            all_cameras = list(self.calibrations.keys())
            exclude = set(self.selection_config.get('exclude', []))
            return [cam for cam in all_cameras if cam not in exclude]

        elif mode == 'spatial':
            # Future: Select by geographic bounds
            return self._select_by_spatial_bounds()

        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    def _validate_cameras(self, candidates: List[str]) -> CameraSelectionResult:
        """
        Three-phase validation: calibration, cache, videos.

        Parameters:
            candidates: List of candidate camera IDs

        Returns:
            CameraSelectionResult with validation details
        """
        result = CameraSelectionResult()

        # Determine resolution name for cache lookup
        if self.ortho_resolution <= 0.003:
            resolution_name = 'hires'
        else:
            resolution_name = 'lowres'

        cache_dir = Path('orthorectification/ortho_cache')

        for camera_id in candidates:
            status = {
                'has_calibration': False,
                'has_cache': False,
                'has_videos': False,  # Will be checked at runtime
                'reasons': []
            }

            # Phase 1: Check calibration exists (hard requirement by default)
            if camera_id not in self.calibrations:
                status['reasons'].append('No calibration found')
                if not self.skip_missing_calibration:
                    result.errors.append(f"{camera_id}: Missing calibration")
                    result.validation_status[camera_id] = status
                    continue
                else:
                    result.warnings.append(f"{camera_id}: Missing calibration (skipped)")
                    result.validation_status[camera_id] = status
                    continue

            status['has_calibration'] = True
            cal = self.calibrations[camera_id]

            # Phase 2: Check ortho cache exists (hard requirement by default)
            # Create temporary geotransform with correct pixel sizes for cache lookup
            geotransform_for_cache = cal['geotransform'].copy()
            geotransform_for_cache['pixel_width'] = self.ortho_resolution
            geotransform_for_cache['pixel_height'] = -self.ortho_resolution

            try:
                cache = load_ortho_cache(
                    camera_id,
                    cal['K'],
                    cal['D'],
                    cal['rvec'],
                    cal['tvec'],
                    geotransform_for_cache,
                    self.ortho_resolution,
                    resolution_name,
                    cache_dir
                )

                # If cache missing, try to generate it
                if cache is None:
                    if self.dsm_file:
                        print(f"    Generating missing cache for {camera_id}...")
                        try:
                            self._generate_cache_for_camera(
                                camera_id,
                                cal,
                                geotransform_for_cache,
                                resolution_name,
                                cache_dir
                            )
                            # Try loading again after generation
                            cache = load_ortho_cache(
                                camera_id,
                                cal['K'],
                                cal['D'],
                                cal['rvec'],
                                cal['tvec'],
                                geotransform_for_cache,
                                self.ortho_resolution,
                                resolution_name,
                                cache_dir
                            )
                            if cache is None:
                                raise RuntimeError("Cache generation failed")
                            print(f"    OK Cache generated successfully")
                        except Exception as gen_error:
                            raise FileNotFoundError(f"Cache generation failed: {gen_error}")
                    else:
                        raise FileNotFoundError("Cache file not found and DSM not provided for generation")

                status['has_cache'] = True

                # Store cache with corrected geotransform
                self.ortho_caches[camera_id] = {
                    'map_x': cache['map_x'],
                    'map_y': cache['map_y'],
                    'output_width': cache['output_width'],
                    'output_height': cache['output_height'],
                    'geotransform': {
                        'x_min': cal['geotransform']['x_min'],
                        'y_max': cal['geotransform']['y_max'],
                        'pixel_width': self.ortho_resolution,
                        'pixel_height': -self.ortho_resolution
                    }
                }

            except Exception as e:
                status['reasons'].append(f'No ortho cache at {self.ortho_resolution}m ({resolution_name})')
                if not self.skip_missing_cache:
                    result.errors.append(f"{camera_id}: Missing ortho cache - {e}")
                    result.validation_status[camera_id] = status
                    continue
                else:
                    result.warnings.append(f"{camera_id}: Missing ortho cache (skipped) - {e}")
                    result.validation_status[camera_id] = status
                    continue

            # Phase 3: Videos will be checked at runtime
            # We can't validate video availability here without knowing time range
            # Mark as True for now, will be checked during frame extraction
            status['has_videos'] = True  # Assumed, verified at runtime

            # Camera passed all required checks
            result.selected_cameras.append(camera_id)
            result.validation_status[camera_id] = status

        return result

    def _check_minimum_cameras(self, result: CameraSelectionResult):
        """
        Check if minimum cameras requirement is met.

        Parameters:
            result: CameraSelectionResult to check and update
        """
        if len(result.selected_cameras) < self.min_cameras_required:
            result.errors.append(
                f"Only {len(result.selected_cameras)} cameras selected, "
                f"but {self.min_cameras_required} required"
            )

    def _generate_cache_for_camera(self, camera_id: str, calib: Dict,
                                   geotransform: Dict, resolution_name: str,
                                   cache_dir: Path):
        """
        Generate ortho cache for a camera (similar to main pipeline).

        Parameters:
            camera_id: Camera identifier
            calib: Calibration dictionary with K, D, rvec, tvec, etc.
            geotransform: Geotransform dict with x_min, y_max, pixel_width, pixel_height
            resolution_name: 'hires' or 'lowres'
            cache_dir: Directory to save cache file
        """
        from orthorectification.undistort_and_orthorectify import (
            create_ortho_lookup_tables_with_dem,
            load_dem_from_tiff
        )

        # Compute output dimensions based on resolution and geotransform
        x_min = geotransform['x_min']
        y_max = geotransform['y_max']
        pixel_width = geotransform['pixel_width']
        pixel_height = geotransform['pixel_height']

        # Get original dimensions from calibration
        orig_width = calib['output_width']
        orig_height = calib['output_height']
        orig_geotransform = calib['geotransform']

        # Calculate new dimensions at target resolution
        x_extent = orig_width * abs(orig_geotransform['pixel_width'])
        y_extent = orig_height * abs(orig_geotransform['pixel_height'])

        width = int(x_extent / abs(pixel_width))
        height = int(y_extent / abs(pixel_height))

        # Load DEM data (resampled to output grid)
        dem_array = load_dem_from_tiff(
            self.dsm_file,
            width,
            height,
            geotransform
        )

        # Get local origin from calibration
        local_origin = np.array([
            calib.get('local_origin_x', 0.0),
            calib.get('local_origin_y', 0.0),
            calib.get('local_origin_z', 0.0)
        ])

        # Generate ortho lookup tables
        map_x, map_y = create_ortho_lookup_tables_with_dem(
            calib['K'],
            calib['D'],
            calib['rvec'],
            calib['tvec'],
            width,
            height,
            geotransform,
            dem_array,
            local_origin=local_origin
        )

        # Prepare cache data
        cache_data = {
            'map_x': map_x,
            'map_y': map_y,
            'output_width': width,
            'output_height': height
        }

        # Save cache
        save_ortho_cache(
            camera_id,
            calib['K'],
            calib['D'],
            calib['rvec'],
            calib['tvec'],
            geotransform,
            self.ortho_resolution,
            resolution_name,
            cache_data,
            cache_dir
        )

    def _select_by_spatial_bounds(self) -> List[str]:
        """
        Select cameras whose zones intersect spatial bounds (future feature).

        Returns:
            List of camera IDs in spatial bounds
        """
        # Future implementation:
        # 1. Load zone_map_shapefile
        # 2. Get spatial_bounds from config
        # 3. Filter cameras by geometric intersection
        # 4. Return matching camera IDs

        raise NotImplementedError("Spatial selection not yet implemented")
