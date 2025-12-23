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

        # Velocity clipping mask (unrotated cache)
        self.unrotated_clip_mask = None
        
        # Rotated mask (cached per frame size/rotation)
        self.rotated_clip_mask = None
        self.mask_bbox = None  # (y_min, y_max, x_min, x_max) for optimization

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

    def _create_unrotated_mask(self, shape: tuple, geotransform: Dict) -> np.ndarray:
        """
        Create velocity clipping mask in standard (unrotated) coordinates.

        Parameters:
            shape: (height, width) of the unrotated frame
            geotransform: Geotransform dict for the unrotated frame

        Returns:
            Boolean mask (H, W) where True = analyze, False = clip
        """
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import Affine

        # Load shapefile (World Coordinates)
        gdf = gpd.read_file(self.velocity_clip_shapefile)

        # Create affine transform for rasterization
        height, width = shape
        pixel_width = geotransform['pixel_width']
        pixel_height = geotransform['pixel_height']
        x_min = geotransform.get('x_min', 0)
        y_max = geotransform.get('y_max', 0)

        transform = Affine(pixel_width, 0, x_min,
                          0, pixel_height, y_max)

        # Rasterize the shapefile polygons directly onto unrotated grid
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        return mask.astype(bool)

    def _generate_methods_summary(self) -> str:
        """
        Generate scholarly summary of velocity field computation methods.

        Returns:
            Formatted methods description
        """
        rotation_text = ""
        if self.rotation_angle_deg != 0.0:
            rotation_text = (
                f"Prior to optical flow computation, mosaic frames were rotated by "
                f"{self.rotation_angle_deg:.1f} degrees counterclockwise to align the coordinate "
                f"system with the principal flow direction, ensuring velocity grids are "
                f"oriented along the channel axis. "
            )

        clip_text = ""
        if self.velocity_clip_shapefile:
            clip_text = (
                f"Velocity fields were spatially clipped to the analysis domain defined by "
                f"the shapefile boundary, with velocities outside the domain set to zero. "
                f"Computational efficiency was optimized by restricting optical flow "
                f"calculations to the bounding box of the analysis domain. "
            )

        methods = f"""
METHODS SUMMARY - ICE VELOCITY FIELD COMPUTATION

Ice velocity fields were computed from orthorectified multi-camera mosaics using
dense optical flow analysis. Consecutive mosaic frames, separated by {self.time_delta_seconds:.1f}
seconds, were processed to extract two-dimensional velocity vectors across the
analysis domain.

{rotation_text}Optical Flow Algorithm:
The Farneback dense optical flow algorithm (Farneback, 2003) was applied to compute
pixel displacement fields between consecutive grayscale frames. The method employs
polynomial expansion to represent the neighborhood of each pixel, enabling sub-pixel
motion estimation. Key parameters included: pyramid levels = {self.config['farneback_params']['levels']},
pyramid scale = {self.config['farneback_params']['pyr_scale']}, window size = {self.config['farneback_params']['winsize']} pixels,
iterations = {self.config['farneback_params']['iterations']}, polynomial neighborhood = {self.config['farneback_params']['poly_n']},
and polynomial sigma = {self.config['farneback_params']['poly_sigma']}.

Coordinate Transformation:
Pixel displacements were converted to metric displacements using the georeferenced
mosaic pixel resolution (meters/pixel) derived from the UTM coordinate system
(EPSG:26919, NAD83 UTM Zone 19N). The u-component represents eastward velocity
(positive east), and the v-component represents northward velocity (positive north).

Velocity Computation:
Velocities were obtained by dividing metric displacements by the inter-frame time
interval ({self.time_delta_seconds:.1f} s), yielding velocity components in m/s. The velocity
magnitude |V| = sqrt(u^2 + v^2) and direction theta = arctan2(v, u) were computed from the
orthogonal components.

{clip_text}Output Products:
Velocity fields were saved as dual-band GeoTIFF files (u, v components), preserving
spatial reference information for GIS integration. Validation visualizations include
quiver plots overlaid on mosaic imagery, velocity magnitude heatmaps, and directional
HSV representations. Statistical metrics (mean, maximum, 95th percentile velocities)
were computed for each frame pair and exported as time-series data.

Software Implementation:
Optical flow computation was performed using OpenCV (cv2.calcOpticalFlowFarneback)
version {cv2.__version__}. Numerical operations utilized NumPy version {np.__version__}.
Geospatial data handling employed Rasterio and GeoPandas for coordinate transformations
and shapefile operations. Visualization products were generated using Matplotlib.

References:
Farneback, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion.
In Scandinavian Conference on Image Analysis (SCIA), pp. 363-370.

Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

Harris, C.R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return methods

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Rotate frame by rotation_angle_deg.

        Parameters:
            frame: (H, W, 3) BGR or (H, W) grayscale/mask frame

        Returns:
            Rotated frame with expanded dimensions
        """
        h, w = frame.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle_deg, 1.0)

        # Calculate new dimensions after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((h * sin) + (w * cos))
        new_height = int((h * cos) + (w * sin))

        # Adjust rotation matrix for translation to center rotated image
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Handle channel count for border value
        border_val = (0, 0, 0) if len(frame.shape) == 3 else 0

        # Rotate
        rotated = cv2.warpAffine(
            frame, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val
        )

        return rotated

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

        # Generate and save methods summary
        methods_summary = self._generate_methods_summary()
        methods_path = self.output_dir / 'methods_summary.txt'
        with open(methods_path, 'w') as f:
            f.write(methods_summary)
        print(f"  Methods summary: {methods_path}")

        # Print methods summary to console
        print(methods_summary)

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

        # 1. Create/Get Unrotated Mask (if needed)
        if self.velocity_clip_shapefile and self.unrotated_clip_mask is None:
            geotransform = self.get_geotransform()
            try:
                self.unrotated_clip_mask = self._create_unrotated_mask(
                    mosaicked_frame.shape[:2],
                    geotransform
                )
                print(f"  Velocity clip mask created (unrotated)")
            except Exception as e:
                print(f"  WARNING: Could not create velocity clip mask - {e}")
                self.unrotated_clip_mask = None

        # 2. Rotate Frame
        if self.rotation_angle_deg != 0.0:
            rotated_frame = self._rotate_frame(mosaicked_frame)
        else:
            rotated_frame = mosaicked_frame

        # 3. Rotate Mask (to match frame alignment perfectly)
        if self.unrotated_clip_mask is not None:
            if self.rotated_clip_mask is None:
                # Rotate the mask using the same transform as the image
                # Convert bool -> uint8 for rotation -> bool
                mask_uint8 = self.unrotated_clip_mask.astype(np.uint8) * 255
                if self.rotation_angle_deg != 0.0:
                    rotated_mask_uint8 = self._rotate_frame(mask_uint8)
                else:
                    rotated_mask_uint8 = mask_uint8
                
                # Threshold back to boolean
                self.rotated_clip_mask = rotated_mask_uint8 > 127
                
                # Compute optimization bbox on the ROTATED mask
                rows = np.any(self.rotated_clip_mask, axis=1)
                cols = np.any(self.rotated_clip_mask, axis=0)
                
                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    # Add margin
                    margin = 20
                    y_min = max(0, y_min - margin)
                    y_max = min(self.rotated_clip_mask.shape[0], y_max + margin)
                    x_min = max(0, x_min - margin)
                    x_max = min(self.rotated_clip_mask.shape[1], x_max + margin)
                    
                    self.mask_bbox = (y_min, y_max, x_min, x_max)
                    
                    # Print stats
                    clip_pixels = np.sum(self.rotated_clip_mask)
                    total_pixels = self.rotated_clip_mask.size
                    bbox_area = (y_max - y_min) * (x_max - x_min)
                    print(f"  Velocity clip mask rotated & ready: {clip_pixels}/{total_pixels} pixels ({100*clip_pixels/total_pixels:.1f}%)")
                    print(f"  Optimization enabled: Processing {bbox_area} pixels (bbox) vs {total_pixels} (full) - {100*(1 - bbox_area/total_pixels):.1f}% reduction")
                else:
                    print("  WARNING: Rotated mask is empty!")
                    self.mask_bbox = None
        else:
            self.rotated_clip_mask = None

        # Convert to grayscale
        current_frame_gray = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)

        # Skip first frame (need pair for optical flow)
        if self.previous_frame_gray is None:
            self.previous_frame_gray = current_frame_gray.copy()
            self.previous_timestamp = timestamp
            return None

        geotransform = self.get_geotransform()

        try:
            # OPTIMIZATION: Crop to mask bounding box if available
            if self.mask_bbox:
                y_min, y_max, x_min, x_max = self.mask_bbox
                
                # Crop frames
                prev_crop = self.previous_frame_gray[y_min:y_max, x_min:x_max]
                curr_crop = current_frame_gray[y_min:y_max, x_min:x_max]
                
                # Compute flow on crop
                u_vel_crop, v_vel_crop = self.velocity_tracker.compute_flow(
                    prev_crop,
                    curr_crop,
                    geotransform
                )
                
                # Create full size arrays
                u_velocity = np.zeros(current_frame_gray.shape, dtype=np.float32)
                v_velocity = np.zeros(current_frame_gray.shape, dtype=np.float32)
                
                # Place cropped result back
                u_velocity[y_min:y_max, x_min:x_max] = u_vel_crop
                v_velocity[y_min:y_max, x_min:x_max] = v_vel_crop
                
            else:
                # Full frame computation
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

        # Apply precise velocity clip mask if present
        if self.rotated_clip_mask is not None:
            u_velocity = np.where(self.rotated_clip_mask, u_velocity, 0.0)
            v_velocity = np.where(self.rotated_clip_mask, v_velocity, 0.0)

        # Compute statistics and print
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
        
        # Only compute stats for valid area if masked
        if self.rotated_clip_mask is not None:
            valid_mags = magnitude[self.rotated_clip_mask]
            if valid_mags.size > 0:
                mean_mag = np.mean(valid_mags)
                max_mag = np.max(valid_mags)
                p95_mag = np.percentile(valid_mags, 95)
            else:
                mean_mag = 0
                max_mag = 0
                p95_mag = 0
        else:
            mean_mag = np.mean(magnitude)
            max_mag = np.max(magnitude)
            p95_mag = np.percentile(magnitude, 95)

        if self.frames_processed % 10 == 0:
            print(f"  Frame {self.frames_processed+1}: "
                  f"vel_mean={mean_mag:.4f} m/s, vel_max={max_mag:.4f} m/s, "
                  f"vel_p95={p95_mag:.4f} m/s")

        # Write statistics CSV
        self.visualizer.write_statistics(timestamp, u_velocity, v_velocity, mask=self.rotated_clip_mask)

        # Write velocity GeoTIFF
        if self.geotiff_writer:
            self.geotiff_writer.write_velocity_field(
                u_velocity, v_velocity, geotransform, timestamp
            )

        # Create validation plots (at specified interval)
        if self.create_validation_plots and (self.frames_processed % self.validation_plot_interval == 0):
            self.visualizer.create_validation_plots(
                rotated_frame, u_velocity, v_velocity, timestamp, mask=self.rotated_clip_mask
            )

        # Create overlay frame
        overlay_frame = None
        if self.create_overlay_video:
            overlay_frame = self.visualizer.create_overlay_frame(
                rotated_frame, u_velocity, v_velocity, timestamp, mask=self.rotated_clip_mask
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
