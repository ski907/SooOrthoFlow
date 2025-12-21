"""
Velocity tracker and ice flux analyzer.

Main coordinator for optical flow-based ice velocity tracking.

Author: SooOrthoFlow Team
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable, Optional, Tuple

from analysis.video_mosaic.ice_flux.geotiff_writer import GeoTIFFWriter
from analysis.video_mosaic.ice_flux.visualization import Visualizer
from analysis.video_mosaic.ice_flux.config_schema import validate_ice_flux_config


class VelocityTracker:
    """Computes dense optical flow and converts to UTM velocities."""

    def __init__(self, farneback_params: Dict, time_delta_seconds: float):
        """
        Initialize velocity tracker.

        Parameters:
            farneback_params: Parameters for cv2.calcOpticalFlowFarneback
            time_delta_seconds: Time between consecutive frames
        """
        self.farneback_params = farneback_params
        self.time_delta_seconds = time_delta_seconds

    def compute_flow(self, frame1_gray: np.ndarray, frame2_gray: np.ndarray,
                     geotransform: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow and convert to UTM velocities.

        Parameters:
            frame1_gray: Previous frame (H, W) uint8 grayscale
            frame2_gray: Current frame (H, W) uint8 grayscale
            geotransform: Dict with pixel_width, pixel_height (m/pixel)

        Returns:
            Tuple of (u_velocity, v_velocity) in m/s
            - u_velocity: (H, W) float32 - eastward velocity component
            - v_velocity: (H, W) float32 - northward velocity component
        """
        # 1. Compute optical flow (pixel displacements)
        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray, None,
            pyr_scale=self.farneback_params['pyr_scale'],
            levels=self.farneback_params['levels'],
            winsize=self.farneback_params['winsize'],
            iterations=self.farneback_params['iterations'],
            poly_n=self.farneback_params['poly_n'],
            poly_sigma=self.farneback_params['poly_sigma'],
            flags=self.farneback_params['flags']
        )

        # 2. Convert pixel displacement to UTM displacement (meters)
        pixel_width = geotransform['pixel_width']  # m/pixel (positive eastward)
        pixel_height = geotransform['pixel_height']  # m/pixel (negative for north-up)

        u_displacement = flow[..., 0] * pixel_width  # meters east
        v_displacement = flow[..., 1] * pixel_height  # meters north (negative * negative = positive north)

        # 3. Convert displacement to velocity (m/s)
        u_velocity = u_displacement / self.time_delta_seconds
        v_velocity = v_displacement / self.time_delta_seconds

        return u_velocity.astype(np.float32), v_velocity.astype(np.float32)


class IceFluxAnalyzer:
    """
    Main coordinator for ice flux analysis.

    Manages frame buffering, velocity computation, and output writing.
    """

    def __init__(self, config: Dict, mosaic_geotransform_getter: Callable):
        """
        Initialize ice flux analyzer.

        Parameters:
            config: Ice flux configuration dict (includes time_delta_seconds, output_dir)
            mosaic_geotransform_getter: Callable that returns current geotransform dict
        """
        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return

        # Validate configuration
        is_valid, errors = validate_ice_flux_config(config)
        if not is_valid:
            raise ValueError(f"Invalid ice flux configuration: {', '.join(errors)}")

        self.config = config
        self.get_geotransform = mosaic_geotransform_getter

        # Extract configuration
        self.time_delta_seconds = config['time_delta_seconds']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_velocity_geotiffs = config.get('save_velocity_geotiffs', True)
        self.create_validation_plots = config.get('create_validation_plots', True)
        self.validation_plot_interval = config.get('validation_plot_interval', 10)
        self.create_overlay_video = config.get('create_overlay_video', False)
        self.rotation_angle_deg = config.get('rotation_angle_deg', 0.0)
        self.velocity_clip_shapefile = config.get('velocity_clip_shapefile')

        # Velocity clipping mask (created on first frame)
        self.velocity_clip_mask = None

        # Initialize components
        self.velocity_tracker = VelocityTracker(
            config['farneback_params'],
            self.time_delta_seconds
        )

        if self.save_velocity_geotiffs:
            self.geotiff_writer = GeoTIFFWriter(
                self.output_dir / 'velocity_fields',
                compress=config.get('compress_geotiffs', True)
            )
        else:
            self.geotiff_writer = None

        self.visualizer = Visualizer(
            self.output_dir,
            create_plots=self.create_validation_plots,
            plot_interval=self.validation_plot_interval,
            create_overlay_video=self.create_overlay_video,
            overlay_subsample=config.get('overlay_video_subsample', 20),
            video_fps=config.get('video_fps', 10),
            video_codec=config.get('video_codec', 'mp4v'),
            rotation_angle_deg=self.rotation_angle_deg
        )

        # Frame buffering
        self.previous_frame_gray = None
        self.previous_timestamp = None

        # Statistics tracking
        self.frames_processed = 0

    def _create_velocity_clip_mask(self, frame_shape: tuple, geotransform: Dict) -> np.ndarray:
        """
        Create velocity clipping mask from shapefile.

        Parameters:
            frame_shape: (height, width) of velocity fields
            geotransform: Geotransform dict with pixel_width, pixel_height, x_min, y_max

        Returns:
            Boolean mask (H, W) where True = analyze, False = clip (set to zero)
        """
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import Affine

        # Load shapefile
        gdf = gpd.read_file(self.velocity_clip_shapefile)

        # Create affine transform for rasterization
        height, width = frame_shape
        pixel_width = geotransform['pixel_width']
        pixel_height = geotransform['pixel_height']  # Negative for north-up
        x_min = geotransform.get('x_min', 0)
        y_max = geotransform.get('y_max', 0)

        transform = Affine(pixel_width, 0, x_min,
                          0, pixel_height, y_max)

        # Rasterize the shapefile polygons
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        return mask.astype(bool)

    def setup(self):
        """Initialize resources."""
        if not self.enabled:
            return

        print("\n" + "="*70)
        print("ICE FLUX ANALYZER - SETUP")
        print("="*70)
        print(f"  Optical flow method: Farneback")
        print(f"  Time delta: {self.time_delta_seconds:.2f} seconds")
        print(f"  Window size: {self.config['farneback_params']['winsize']}")
        print(f"  Pyramid levels: {self.config['farneback_params']['levels']}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Save velocity GeoTIFFs: {self.save_velocity_geotiffs}")
        print(f"  Create validation plots: {self.create_validation_plots}")
        if self.create_validation_plots:
            print(f"    Plot interval: every {self.validation_plot_interval} frames")
        print(f"  Create overlay video: {self.create_overlay_video}")
        if self.velocity_clip_shapefile:
            print(f"  Velocity clipping: {self.velocity_clip_shapefile}")

        # Open statistics CSV
        self.visualizer.open_stats_csv()
        print(f"  Statistics CSV: {self.visualizer.stats_csv_path}")

    def process_frame(self, mosaicked_frame: np.ndarray, timestamp: datetime) -> Optional[np.ndarray]:
        """
        Process a mosaicked frame for ice flux analysis.

        Parameters:
            mosaicked_frame: (H, W, 3) BGR mosaic frame
            timestamp: Frame timestamp

        Returns:
            Overlay frame if overlay video enabled, otherwise None
        """
        if not self.enabled:
            return None

        # Convert to grayscale
        current_frame_gray = cv2.cvtColor(mosaicked_frame, cv2.COLOR_BGR2GRAY)

        # Skip first frame (need pair for optical flow)
        # Note: overlay video writer will be lazily initialized on first write
        if self.previous_frame_gray is None:
            self.previous_frame_gray = current_frame_gray.copy()
            self.previous_timestamp = timestamp
            return None

        # Compute optical flow
        geotransform = self.get_geotransform()

        try:
            u_velocity, v_velocity = self.velocity_tracker.compute_flow(
                self.previous_frame_gray,
                current_frame_gray,
                geotransform
            )
        except Exception as e:
            print(f"  WARNING: Optical flow computation failed - {e}")
            self.previous_frame_gray = current_frame_gray.copy()
            self.previous_timestamp = timestamp
            return None

        # Create velocity clip mask on first frame if shapefile specified
        if self.velocity_clip_shapefile and self.velocity_clip_mask is None:
            try:
                self.velocity_clip_mask = self._create_velocity_clip_mask(
                    u_velocity.shape,
                    geotransform
                )
                clip_pixels = np.sum(self.velocity_clip_mask)
                total_pixels = self.velocity_clip_mask.size
                print(f"  Velocity clip mask created: {clip_pixels}/{total_pixels} pixels ({100*clip_pixels/total_pixels:.1f}%) in analysis area")
            except Exception as e:
                print(f"  WARNING: Could not create velocity clip mask - {e}")
                self.velocity_clip_mask = None

        # Apply velocity clip mask if present
        if self.velocity_clip_mask is not None:
            u_velocity = np.where(self.velocity_clip_mask, u_velocity, 0.0)
            v_velocity = np.where(self.velocity_clip_mask, v_velocity, 0.0)

        # Compute statistics and print
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        p95_mag = np.percentile(magnitude, 95)

        if self.frames_processed % 10 == 0:
            print(f"  Frame {self.frames_processed+1}: "
                  f"vel_mean={mean_mag:.4f} m/s, vel_max={max_mag:.4f} m/s, "
                  f"vel_p95={p95_mag:.4f} m/s")

        # Write statistics CSV
        self.visualizer.write_statistics(timestamp, u_velocity, v_velocity)

        # Write velocity GeoTIFF
        if self.geotiff_writer:
            self.geotiff_writer.write_velocity_field(
                u_velocity, v_velocity, geotransform, timestamp
            )

        # Create validation plots (at specified interval)
        if self.create_validation_plots and (self.frames_processed % self.validation_plot_interval == 0):
            self.visualizer.create_validation_plots(
                mosaicked_frame, u_velocity, v_velocity, timestamp
            )

        # Create overlay frame
        overlay_frame = None
        if self.create_overlay_video:
            overlay_frame = self.visualizer.create_overlay_frame(
                mosaicked_frame, u_velocity, v_velocity, timestamp
            )
            self.visualizer.write_overlay_frame(overlay_frame)

        # Update buffer
        self.previous_frame_gray = current_frame_gray.copy()
        self.previous_timestamp = timestamp
        self.frames_processed += 1

        return overlay_frame

    def cleanup(self):
        """Release resources."""
        if not self.enabled:
            return

        # Close statistics CSV
        self.visualizer.close_stats_csv()

        # Close overlay video
        if self.create_overlay_video:
            self.visualizer.close_overlay_video()

        print("\n" + "="*70)
        print("ICE FLUX ANALYZER - SUMMARY")
        print("="*70)
        print(f"  Velocity fields computed: {self.frames_processed}")
        print(f"  Output directory: {self.output_dir}")
        if self.save_velocity_geotiffs:
            print(f"  Velocity GeoTIFFs: {self.output_dir / 'velocity_fields'}")
        if self.create_validation_plots:
            print(f"  Validation plots: {self.output_dir / 'validation_plots'}")
        if self.create_overlay_video:
            print(f"  Overlay video: {self.visualizer.overlay_video_path}")
        print(f"  Statistics CSV: {self.visualizer.stats_csv_path}")
