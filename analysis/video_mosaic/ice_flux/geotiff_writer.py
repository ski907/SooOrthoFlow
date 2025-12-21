"""
GeoTIFF writer for velocity fields.

Writes 2-band float32 GeoTIFFs with geotransform and CRS preservation.

Author: SooOrthoFlow Team
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
import rasterio
from rasterio.transform import Affine


class GeoTIFFWriter:
    """Writes velocity fields as georeferenced GeoTIFF files."""

    def __init__(self, output_dir: Path, compress: bool = True):
        """
        Initialize GeoTIFF writer.

        Parameters:
            output_dir: Directory for output GeoTIFF files
            compress: Enable LZW compression (default: True)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress

    def write_velocity_field(self, u_velocity: np.ndarray, v_velocity: np.ndarray,
                             geotransform: Dict, timestamp: datetime, crs: str = 'EPSG:26919'):
        """
        Write velocity field as 2-band float32 GeoTIFF.

        Parameters:
            u_velocity: (H, W) eastward velocity component in m/s
            v_velocity: (H, W) northward velocity component in m/s
            geotransform: Dict with x_min, y_max, pixel_width, pixel_height
            timestamp: Frame timestamp
            crs: Coordinate reference system (default: EPSG:26919)

        Returns:
            Path to written GeoTIFF file
        """
        # Create filename
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"velocity_{timestamp_str}.tif"
        output_path = self.output_dir / filename

        # Stack bands: Band 1 = u (east), Band 2 = v (north)
        velocity_data = np.stack([u_velocity, v_velocity], axis=0).astype(np.float32)

        # Create rasterio transform
        transform = Affine(
            geotransform['pixel_width'], 0, geotransform['x_min'],
            0, geotransform['pixel_height'], geotransform['y_max']
        )

        # Build profile
        profile = {
            'driver': 'GTiff',
            'height': u_velocity.shape[0],
            'width': u_velocity.shape[1],
            'count': 2,
            'dtype': 'float32',
            'crs': crs,
            'transform': transform,
            'compress': 'lzw' if self.compress else None,
            'tiled': False,
            'interleave': 'band'
        }

        # Write GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(velocity_data)
            dst.set_band_description(1, 'u_velocity_east_m_s')
            dst.set_band_description(2, 'v_velocity_north_m_s')

            # Add metadata
            dst.update_tags(
                1,
                units='m/s',
                description='Eastward (u) velocity component',
                timestamp=timestamp.isoformat()
            )
            dst.update_tags(
                2,
                units='m/s',
                description='Northward (v) velocity component',
                timestamp=timestamp.isoformat()
            )

        return output_path
