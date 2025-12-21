"""
In-Memory Mosaic Engine

Efficiently mosaic orthorectified frames without GeoTIFF I/O.
Uses pre-computed zone maps for camera assignment.

Author: SooOrthoFlow Team
Version: 0.1.0
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orthorectification.ortho_mosaic import load_zone_map_raster, get_image_bounds


class InMemoryMosaicEngine:
    """
    Mosaic orthorectified frames in memory without GeoTIFF I/O.

    Uses zone map to determine which camera's pixels to use in each region.
    """

    def __init__(self, camera_ids: list, mosaic_bounds: Tuple[float, float, float, float],
                 resolution: float, mosaic_method: str = 'zone_map',
                 zone_map_shapefile: Optional[str] = None):
        """
        Initialize mosaic engine.

        Parameters:
            camera_ids: List of camera IDs
            mosaic_bounds: Tuple of (x_min, x_max, y_min, y_max) in model coordinates
            resolution: Pixel resolution in meters/pixel
            mosaic_method: 'zone_map' or 'center' (default: 'zone_map')
            zone_map_shapefile: Path to zone map shapefile (required if mosaic_method='zone_map')
        """
        self.camera_ids = camera_ids
        self.mosaic_bounds = mosaic_bounds
        self.resolution = resolution
        self.mosaic_method = mosaic_method

        x_min, x_max, y_min, y_max = mosaic_bounds

        # Calculate mosaic dimensions
        self.mosaic_width = int((x_max - x_min) / resolution)
        self.mosaic_height = int((y_max - y_min) / abs(resolution))

        print(f"Mosaic dimensions: {self.mosaic_width} x {self.mosaic_height} pixels")
        print(f"Mosaic bounds: ({x_min:.2f}, {x_max:.2f}, {y_min:.2f}, {y_max:.2f})")

        # Load zone map if using zone-based mosaicking
        if mosaic_method == 'zone_map':
            if not zone_map_shapefile:
                raise ValueError("zone_map_shapefile is required for zone_map method")

            print(f"Loading zone map from {zone_map_shapefile}...")
            self.zone_array, self.camera_id_to_name, self.zone_transform = load_zone_map_raster(
                zone_map_shapefile,
                mosaic_bounds,
                resolution
            )

            # CRITICAL: Use zone map bounds instead of provided mosaic_bounds
            # The zone map was created for ALL cameras, so we need to use its full extent
            # to ensure correct alignment even when processing a subset of cameras
            import rasterio

            res_mm_value = resolution * 1000
            if res_mm_value % 1 == 0:
                # Integer value (e.g., 10.0 -> "10mm")
                res_str = f"{int(res_mm_value)}mm"
            else:
                # Decimal value (e.g., 2.5 -> "2_5mm")
                res_str = f"{res_mm_value:.10g}".replace('.', '_') + "mm"

            zone_raster_path = Path(zone_map_shapefile).parent / f"{Path(zone_map_shapefile).stem}_{res_str}.tif"
            with rasterio.open(zone_raster_path) as src:
                zone_bounds = src.bounds
                self.mosaic_bounds = (zone_bounds.left, zone_bounds.right, zone_bounds.bottom, zone_bounds.top)
                self.mosaic_width = src.width
                self.mosaic_height = src.height
                print(f"Using zone map bounds: ({self.mosaic_bounds[0]:.2f}, {self.mosaic_bounds[1]:.2f}, {self.mosaic_bounds[2]:.2f}, {self.mosaic_bounds[3]:.2f})")
                print(f"Zone map dimensions: {self.mosaic_width} x {self.mosaic_height} pixels")

            # Create reverse lookup: camera_name -> zone_id
            self.camera_name_to_id = {name: zone_id for zone_id, name in self.camera_id_to_name.items()}

            print(f"  Loaded {len(self.camera_id_to_name)} camera zones")

        else:
            # For 'center' method, no zone map needed
            self.zone_array = None
            self.camera_id_to_name = None
            self.camera_name_to_id = None

        # Create reusable mosaic template
        self.mosaic_template = np.zeros((self.mosaic_height, self.mosaic_width, 3), dtype=np.uint8)

        # Track content bounds for auto-cropping (when using subset of cameras)
        self.content_bounds = None  # Will be set to (row_min, row_max, col_min, col_max) after first frame

    def mosaic_frame(self, ortho_images: Dict[str, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """
        Mosaic multiple orthorectified images into single frame.

        Parameters:
            ortho_images: Dict mapping camera_id to (ortho_array, geotransform)
                         where ortho_array is (H, W, 3) numpy array
                         and geotransform is dict with x_min, y_max, pixel_width, pixel_height

        Returns:
            mosaic_array: (H, W, 3) numpy array (auto-cropped to content if using zone_map)
        """
        # Clear previous frame
        self.mosaic_template.fill(0)

        if self.mosaic_method == 'zone_map':
            full_mosaic = self._mosaic_with_zone_map(ortho_images)

            # Auto-crop to content region (compute bounds from first frame)
            if self.content_bounds is None:
                self.content_bounds = self._compute_content_bounds(full_mosaic)
                print(f"Auto-crop enabled: content region is [{self.content_bounds[0]}:{self.content_bounds[1]}, {self.content_bounds[2]}:{self.content_bounds[3]}]")
                print(f"Cropped dimensions: {self.content_bounds[3]-self.content_bounds[2]} x {self.content_bounds[1]-self.content_bounds[0]} pixels")

            # Crop to content
            return full_mosaic[self.content_bounds[0]:self.content_bounds[1],
                              self.content_bounds[2]:self.content_bounds[3]]

        elif self.mosaic_method == 'center':
            return self._mosaic_with_center_weighting(ortho_images)
        else:
            raise ValueError(f"Unknown mosaic method: {self.mosaic_method}")

    def _mosaic_with_zone_map(self, ortho_images: Dict[str, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """
        Mosaic using zone map (assign each pixel based on zone).

        Parameters:
            ortho_images: Dict mapping camera_id to (ortho_array, geotransform)

        Returns:
            mosaicked frame
        """
        x_min, x_max, y_min, y_max = self.mosaic_bounds

        for camera_id, (ortho_img, geotransform) in ortho_images.items():
            # Get zone ID for this camera
            if camera_id not in self.camera_name_to_id:
                print(f"  WARNING: {camera_id} not in zone map, skipping")
                continue

            zone_id = self.camera_name_to_id[camera_id]

            # Get image bounds
            img_bounds = get_image_bounds(ortho_img.shape, geotransform)
            img_x_min, img_x_max, img_y_min, img_y_max = img_bounds

            # Convert to mosaic pixel coordinates
            mosaic_col_start = max(0, int((img_x_min - x_min) / self.resolution))
            mosaic_row_start = max(0, int((y_max - img_y_max) / abs(self.resolution)))
            mosaic_col_end = min(self.mosaic_width, mosaic_col_start + ortho_img.shape[1])
            mosaic_row_end = min(self.mosaic_height, mosaic_row_start + ortho_img.shape[0])

            # Calculate corresponding image region
            img_col_start = max(0, -int((img_x_min - x_min) / self.resolution))
            img_row_start = max(0, -int((y_max - img_y_max) / abs(self.resolution)))
            img_col_end = img_col_start + (mosaic_col_end - mosaic_col_start)
            img_row_end = img_row_start + (mosaic_row_end - mosaic_row_start)

            # Extract the region
            img_region = ortho_img[img_row_start:img_row_end, img_col_start:img_col_end]

            # Find valid (non-black) pixels
            valid_mask = np.any(img_region > 0, axis=2)

            # Get zone map region
            zone_region = self.zone_array[mosaic_row_start:mosaic_row_end,
                                         mosaic_col_start:mosaic_col_end]

            # Only update where zone matches this camera AND pixel is valid
            zone_match = (zone_region == zone_id)
            update_mask = valid_mask & zone_match

            # Update mosaic
            if np.any(update_mask):
                mosaic_region = self.mosaic_template[mosaic_row_start:mosaic_row_end,
                                                     mosaic_col_start:mosaic_col_end]

                for c in range(3):
                    mosaic_region[:, :, c][update_mask] = img_region[:, :, c][update_mask]

                self.mosaic_template[mosaic_row_start:mosaic_row_end,
                                    mosaic_col_start:mosaic_col_end] = mosaic_region

        return self.mosaic_template.copy()

    def _compute_content_bounds(self, mosaic_array: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Compute the bounding box of non-zero content in the mosaic.

        Parameters:
            mosaic_array: (H, W, 3) mosaic array

        Returns:
            Tuple of (row_min, row_max, col_min, col_max)
        """
        # Find pixels where any channel has non-zero values
        content_mask = np.any(mosaic_array > 0, axis=2)

        # Find the bounding box
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # No content, return full frame
            return (0, mosaic_array.shape[0], 0, mosaic_array.shape[1])

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Add 1 to max indices for slicing (Python slice is exclusive)
        row_max += 1
        col_max += 1

        return (row_min, row_max, col_min, col_max)

    def _mosaic_with_center_weighting(self, ortho_images: Dict[str, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """
        Mosaic using center weighting (prefer pixels near image centers).

        For 2-camera mosaic, this creates a simple blend favoring each camera's center.

        Parameters:
            ortho_images: Dict mapping camera_id to (ortho_array, geotransform)

        Returns:
            mosaicked frame
        """
        x_min, x_max, y_min, y_max = self.mosaic_bounds

        # Track accumulated weights
        weight_map = np.zeros((self.mosaic_height, self.mosaic_width), dtype=np.float32)
        accumulator = np.zeros((self.mosaic_height, self.mosaic_width, 3), dtype=np.float32)

        for camera_id, (ortho_img, geotransform) in ortho_images.items():
            # Get image bounds
            img_bounds = get_image_bounds(ortho_img.shape, geotransform)
            img_x_min, img_x_max, img_y_min, img_y_max = img_bounds

            # Convert to mosaic pixel coordinates
            mosaic_col_start = max(0, int((img_x_min - x_min) / self.resolution))
            mosaic_row_start = max(0, int((y_max - img_y_max) / abs(self.resolution)))
            mosaic_col_end = min(self.mosaic_width, mosaic_col_start + ortho_img.shape[1])
            mosaic_row_end = min(self.mosaic_height, mosaic_row_start + ortho_img.shape[0])

            # Calculate corresponding image region
            img_col_start = max(0, -int((img_x_min - x_min) / self.resolution))
            img_row_start = max(0, -int((y_max - img_y_max) / abs(self.resolution)))
            img_col_end = img_col_start + (mosaic_col_end - mosaic_col_start)
            img_row_end = img_row_start + (mosaic_row_end - mosaic_row_start)

            # Extract the region
            img_region = ortho_img[img_row_start:img_row_end, img_col_start:img_col_end]

            # Find valid (non-black) pixels
            valid_mask = np.any(img_region > 0, axis=2)

            # Create center weight (pixels closer to center get higher weight)
            h, w = img_region.shape[:2]
            cy, cx = h / 2, w / 2
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            center_weight = 1.0 - (dist_from_center / (max_dist + 1e-6))

            # Apply valid mask to weight
            center_weight[~valid_mask] = 0

            # Accumulate weighted pixels
            mosaic_region = accumulator[mosaic_row_start:mosaic_row_end,
                                       mosaic_col_start:mosaic_col_end]
            weight_region = weight_map[mosaic_row_start:mosaic_row_end,
                                      mosaic_col_start:mosaic_col_end]

            for c in range(3):
                mosaic_region[:, :, c] += img_region[:, :, c] * center_weight

            weight_region += center_weight

            accumulator[mosaic_row_start:mosaic_row_end,
                       mosaic_col_start:mosaic_col_end] = mosaic_region
            weight_map[mosaic_row_start:mosaic_row_end,
                      mosaic_col_start:mosaic_col_end] = weight_region

        # Normalize by weights
        valid = weight_map > 0
        for c in range(3):
            self.mosaic_template[:, :, c][valid] = (accumulator[:, :, c][valid] / weight_map[valid]).astype(np.uint8)

        return self.mosaic_template.copy()

    def get_geotransform(self) -> Dict[str, float]:
        """
        Get geotransform for the mosaic (adjusted for cropping if applicable).

        Returns:
            Dict with x_min, y_max, pixel_width, pixel_height
        """
        x_min, x_max, y_min, y_max = self.mosaic_bounds

        # Adjust geotransform if cropping was applied
        if self.content_bounds is not None:
            row_min, row_max, col_min, col_max = self.content_bounds
            # Adjust x_min and y_max based on crop offset
            x_min += col_min * self.resolution
            y_max -= row_min * self.resolution  # Subtract because y increases downward in image

        return {
            'x_min': x_min,
            'y_max': y_max,
            'pixel_width': self.resolution,
            'pixel_height': -self.resolution  # Negative for north-up
        }
