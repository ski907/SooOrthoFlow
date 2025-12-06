"""
Camera Zone Map Module

Manages persistent camera zone maps for consistent mosaicing.
Supports both raster (GeoTIFF) and vector (Shapefile) formats.
"""

import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes, rasterize
from rasterio.crs import CRS
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping
import logging

logger = logging.getLogger(__name__)


class CameraZoneMap:
    """
    Manages camera zone maps for consistent mosaicing.

    The zone map defines which camera should be used for each pixel location.
    Supports generation from center method, conversion between raster/vector formats,
    and manual editing in QGIS.
    """

    def __init__(self, zone_map_dir, num_cameras, resolution=0.05, camera_names=None):
        """
        Initialize CameraZoneMap handler.

        Args:
            zone_map_dir: Directory to store zone map files (global location for project)
            num_cameras: Number of cameras in the system
            resolution: Zone map resolution in meters per pixel (default 0.05 = 5cm)
            camera_names: List of camera names (e.g., ['NVR1_ch1', 'NVR1_ch2', ...])
        """
        self.zone_map_dir = Path(zone_map_dir)
        self.num_cameras = num_cameras
        self.resolution = resolution
        self.camera_names = camera_names if camera_names else [f"CAM_{i}" for i in range(num_cameras)]

        # Create zone_maps directory if it doesn't exist
        self.zone_map_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.raster_path = self.zone_map_dir / "camera_zone_map.tif"
        self.vector_path = self.zone_map_dir / "camera_zone_map.shp"

        # Cached zone map
        self._zone_map = None
        self._zone_map_transform = None
        self._zone_map_crs = None
        self._zone_map_bounds = None

    def exists(self):
        """Check if a zone map exists (either raster or vector)."""
        return self.raster_path.exists() or self.vector_path.exists()

    def generate_from_center_method(self, ortho_images, output_bounds, output_crs, camera_centers):
        """
        Generate zone map using center-distance method.

        Each pixel is assigned to the nearest camera center.

        Args:
            ortho_images: List of orthorectified image arrays (not used directly, for reference)
            output_bounds: Tuple (minx, miny, maxx, maxy) in output CRS
            output_crs: Output coordinate reference system
            camera_centers: List of (x, y) camera center coordinates in output CRS

        Returns:
            numpy array of camera indices
        """
        logger.info(f"Generating camera zone map at {self.resolution}m resolution...")

        minx, miny, maxx, maxy = output_bounds

        # Calculate dimensions at coarse resolution
        width = int(np.ceil((maxx - minx) / self.resolution))
        height = int(np.ceil((maxy - miny) / self.resolution))

        logger.info(f"Zone map dimensions: {width}x{height} pixels")
        logger.info(f"Output bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")
        logger.info(f"Number of cameras: {self.num_cameras}")

        # Create transform
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Create coordinate grids
        # Generate world coordinates for each pixel center using vectorized operations
        cols = np.arange(width)
        rows = np.arange(height)

        # Create 2D arrays of row and column indices
        cols_grid, rows_grid = np.meshgrid(cols, rows)

        # Apply transform manually: x = a*col + b*row + c, y = d*col + e*row + f
        xs = transform.a * cols_grid + transform.b * rows_grid + transform.c
        ys = transform.d * cols_grid + transform.e * rows_grid + transform.f

        # Initialize zone map with invalid value
        zone_map = np.full((height, width), 255, dtype=np.uint8)

        # For each camera, compute distance to all pixels
        min_distances = np.full((height, width), np.inf)

        for cam_idx, (cx, cy) in enumerate(camera_centers):
            # Euclidean distance from camera center
            distances = np.sqrt((xs - cx)**2 + (ys - cy)**2)

            # Update zone map where this camera is closer
            closer_mask = distances < min_distances
            zone_map[closer_mask] = cam_idx
            min_distances[closer_mask] = distances[closer_mask]

        # Count assignments
        for cam_idx in range(self.num_cameras):
            pixel_count = np.sum(zone_map == cam_idx)
            logger.info(f"  Camera {cam_idx}: {pixel_count} pixels ({pixel_count/(width*height)*100:.1f}%)")

        # Save raster
        self._save_raster(zone_map, transform, output_crs)

        # Cache for immediate use
        self._zone_map = zone_map
        self._zone_map_transform = transform
        self._zone_map_crs = output_crs
        self._zone_map_bounds = output_bounds

        logger.info(f"Zone map saved to {self.raster_path}")

        return zone_map

    def _save_raster(self, zone_map, transform, crs):
        """Save zone map as GeoTIFF."""
        with rasterio.open(
            self.raster_path,
            'w',
            driver='GTiff',
            height=zone_map.shape[0],
            width=zone_map.shape[1],
            count=1,
            dtype=zone_map.dtype,
            crs=crs,
            transform=transform,
            compress='lzw',
            nodata=255
        ) as dst:
            dst.write(zone_map, 1)
            dst.set_band_description(1, "Camera Index")

        # Give file system time to flush (prevents race condition when immediately reading)
        # Need extra time for LZW compression to complete
        import time
        time.sleep(0.5)

    def raster_to_vector(self, simplify_tolerance=None):
        """
        Convert raster zone map to vector shapefile.

        Creates simplified polygon features for each camera zone.

        Args:
            simplify_tolerance: Tolerance for polygon simplification in map units.
                              If None, uses adaptive tolerance based on resolution.
        """
        if not self.raster_path.exists():
            logger.error("Raster zone map does not exist. Cannot convert to vector.")
            return False

        logger.info("Converting raster zone map to vector polygons...")

        # Read raster
        with rasterio.open(self.raster_path) as src:
            zone_data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Adaptive simplification tolerance based on resolution
        # For fine resolutions (< 0.01m), use larger multiplier to get straight edges
        # For coarse resolutions (>= 0.01m), use smaller multiplier
        if simplify_tolerance is None:
            if self.resolution < 0.01:
                # Fine resolution (e.g., 0.0025m = 2.5mm): use 10x for ~2.5cm tolerance
                simplify_tolerance = self.resolution * 10
            else:
                # Coarse resolution (e.g., 0.05m = 5cm): use 2x
                simplify_tolerance = self.resolution * 2

        # Generate polygon shapes and simplify them
        polygons = []
        for geom, value in shapes(zone_data, mask=(zone_data != 255), transform=transform):
            cam_idx = int(value)
            camera_name = self.camera_names[cam_idx] if cam_idx < len(self.camera_names) else f"CAM_{cam_idx}"

            # Convert to shapely geometry for simplification
            poly = shape(geom)

            # Simplify to remove stair-stepping (Douglas-Peucker algorithm)
            simplified_poly = poly.simplify(simplify_tolerance, preserve_topology=True)

            polygons.append({
                'geometry': mapping(simplified_poly),
                'properties': {
                    'camera_id': cam_idx,
                    'camera_name': camera_name,
                    'priority': cam_idx  # Default priority = camera index
                }
            })

        logger.info(f"Generated {len(polygons)} simplified polygon features")
        logger.info(f"  Simplification tolerance: {simplify_tolerance:.4f}m")

        # Define schema
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'camera_id': 'int',
                'camera_name': 'str:20',  # String with max length 20
                'priority': 'int'
            }
        }

        # Write shapefile
        with fiona.open(
            self.vector_path,
            'w',
            driver='ESRI Shapefile',
            crs=crs.to_dict() if crs else from_epsg(32616),  # Default to UTM 16N
            schema=schema
        ) as dst:
            dst.writerecords(polygons)

        logger.info(f"Vector zone map saved to {self.vector_path}")
        return True

    def vector_to_raster(self):
        """
        Rasterize vector shapefile back to GeoTIFF.

        Used after manual editing in QGIS.
        """
        if not self.vector_path.exists():
            logger.error("Vector zone map does not exist. Cannot rasterize.")
            return False

        logger.info("Rasterizing edited vector zone map...")

        # Read reference raster for dimensions and transform
        if self.raster_path.exists():
            with rasterio.open(self.raster_path) as ref:
                transform = ref.transform
                width = ref.width
                height = ref.height
                crs = ref.crs
        else:
            logger.error("Reference raster not found. Cannot determine output dimensions.")
            return False

        # Read vector features
        with fiona.open(self.vector_path) as src:
            geometries = []
            for feature in src:
                geom = shape(feature['geometry'])
                camera_id = feature['properties']['camera_id']
                geometries.append((geom, camera_id))

        # Rasterize
        zone_map = rasterize(
            geometries,
            out_shape=(height, width),
            transform=transform,
            fill=255,  # Nodata value
            dtype=np.uint8
        )

        # Save updated raster
        self._save_raster(zone_map, transform, crs)

        # Clear cache to force reload
        self._zone_map = None

        logger.info(f"Updated raster zone map saved to {self.raster_path}")
        return True

    def load_zone_map(self, force_reload=False):
        """
        Load zone map for use in mosaicing.

        Automatically detects if vector file is newer and rasterizes if needed.

        Args:
            force_reload: Force reload from disk even if cached

        Returns:
            Tuple of (zone_map array, transform, crs, bounds)
        """
        # Check if vector is newer than raster
        if self.vector_path.exists() and self.raster_path.exists():
            vector_mtime = self.vector_path.stat().st_mtime
            raster_mtime = self.raster_path.stat().st_mtime

            if vector_mtime > raster_mtime:
                logger.info("Vector zone map is newer than raster. Rasterizing...")
                self.vector_to_raster()

        # Return cached if available
        if not force_reload and self._zone_map is not None:
            return self._zone_map, self._zone_map_transform, self._zone_map_crs, self._zone_map_bounds

        # Load from raster
        if not self.raster_path.exists():
            logger.error("Zone map raster file not found.")
            return None, None, None, None

        with rasterio.open(self.raster_path) as src:
            self._zone_map = src.read(1)
            self._zone_map_transform = src.transform
            self._zone_map_crs = src.crs
            self._zone_map_bounds = src.bounds

        logger.info(f"Loaded zone map: {self._zone_map.shape}")

        return self._zone_map, self._zone_map_transform, self._zone_map_crs, self._zone_map_bounds

    def get_camera_for_location(self, x, y, zone_map, transform):
        """
        Query which camera should be used at world coordinate (x, y).

        Args:
            x, y: World coordinates in same CRS as zone map
            zone_map: Zone map array
            transform: Rasterio transform

        Returns:
            Camera index (0 to num_cameras-1) or None if out of bounds
        """
        # Convert world coords to pixel coords
        row, col = rasterio.transform.rowcol(transform, x, y)

        # Check bounds
        if 0 <= row < zone_map.shape[0] and 0 <= col < zone_map.shape[1]:
            cam_idx = zone_map[row, col]
            if cam_idx != 255:  # Not nodata
                return int(cam_idx)

        return None

    def upsample_to_resolution(self, zone_map, zone_transform, target_resolution, target_bounds, crs=None):
        """
        Upsample zone map to match higher resolution output.

        Uses nearest-neighbor interpolation to maintain discrete camera indices.

        Args:
            zone_map: Coarse zone map array
            zone_transform: Transform of coarse zone map
            target_resolution: Desired resolution in meters
            target_bounds: Target bounds (minx, miny, maxx, maxy)
            crs: Coordinate reference system (optional)

        Returns:
            Tuple of (upsampled_zone_map, new_transform)
        """
        from rasterio.warp import reproject, Resampling

        minx, miny, maxx, maxy = target_bounds

        # Calculate target dimensions
        width = int(np.ceil((maxx - minx) / target_resolution))
        height = int(np.ceil((maxy - miny) / target_resolution))

        target_transform = from_bounds(minx, miny, maxx, maxy, width, height)

        # Create output array
        upsampled = np.zeros((height, width), dtype=np.uint8)

        # Use cached CRS if not provided
        if crs is None:
            crs = self._zone_map_crs

        # Reproject using nearest neighbor
        reproject(
            zone_map,
            upsampled,
            src_transform=zone_transform,
            src_crs=crs,
            dst_transform=target_transform,
            dst_crs=crs,
            src_nodata=255,
            dst_nodata=255,
            resampling=Resampling.nearest
        )

        logger.info(f"Upsampled zone map from {zone_map.shape} to {upsampled.shape}")

        return upsampled, target_transform
