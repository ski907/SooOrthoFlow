"""
Visualization tools for velocity field validation.

Creates validation plots, overlay videos, and statistics.

Author: SooOrthoFlow Team
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import csv


class Visualizer:
    """Creates validation visualizations for velocity fields."""

    def __init__(self, output_dir: Path, create_plots: bool = True,
                 plot_interval: int = 10, create_overlay_video: bool = False,
                 overlay_subsample: int = 20, video_fps: int = 10,
                 video_codec: str = 'mp4v', rotation_angle_deg: float = 0.0):
        """
        Initialize visualizer.

        Parameters:
            output_dir: Output directory for validation plots (ice_flux subfolder)
            create_plots: Enable plot creation
            plot_interval: Create plot every N frames
            create_overlay_video: Enable overlay video creation
            overlay_subsample: Subsample factor for overlay vectors
            video_fps: Frame rate for overlay video
            video_codec: Codec for overlay video
            rotation_angle_deg: Rotation angle for visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.create_plots = create_plots
        self.plot_interval = plot_interval
        self.create_overlay_video = create_overlay_video
        self.overlay_subsample = overlay_subsample
        self.video_fps = video_fps
        self.video_codec = video_codec
        self.rotation_angle_deg = rotation_angle_deg

        # Validation plots go to main output directory (parent of ice_flux)
        if self.create_plots:
            self.plots_dir = self.output_dir.parent / 'validation_plots'
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats_csv_path = self.output_dir / 'velocity_statistics.csv'
        self.stats_file = None
        self.stats_writer = None
        self.stats_header_written = False

        # Overlay video writer
        self.overlay_video_writer = None
        self.overlay_video_path = None
        self.frame_count = 0

        # Color scale tracking for consistent visualization
        self.max_magnitude_seen = 0.0

    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate image by rotation_angle_deg with expanded canvas to fit entire rotated image.

        Parameters:
            image: Input image (grayscale or BGR)

        Returns:
            Rotated image with expanded dimensions
        """
        if self.rotation_angle_deg == 0.0:
            return image

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle_deg, 1.0)

        # Calculate new dimensions to fit entire rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((h * sin) + (w * cos))
        new_height = int((h * cos) + (w * sin))

        # Adjust rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform rotation with expanded canvas
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return rotated

    def _rotate_velocity_components(self, u: np.ndarray, v: np.ndarray) -> tuple:
        """
        Rotate velocity vector components by rotation_angle_deg.

        When rotating velocity fields, we need to:
        1. Rotate the spatial positions (done by _rotate_image)
        2. Rotate the vector components themselves (done here)

        Parameters:
            u: Eastward velocity component
            v: Northward velocity component

        Returns:
            Tuple of (u_rotated, v_rotated) in the rotated coordinate frame
        """
        if self.rotation_angle_deg == 0.0:
            return u, v

        # Convert angle to radians
        theta_rad = np.deg2rad(self.rotation_angle_deg)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        # Rotate vector components
        # Standard 2D rotation matrix applied to each vector
        u_rotated = u * cos_theta - v * sin_theta
        v_rotated = u * sin_theta + v * cos_theta

        return u_rotated, v_rotated

    def _add_timestamp(self, image: np.ndarray, timestamp: datetime) -> np.ndarray:
        """
        Add timestamp overlay to image.

        Parameters:
            image: Input image (BGR)
            timestamp: Timestamp to display

        Returns:
            Image with timestamp overlay
        """
        # Format timestamp
        timestamp_str = timestamp.strftime('%Y_%m_%d %H:%M:%S')

        # Create a copy to avoid modifying original
        img_with_text = image.copy()

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            timestamp_str, font, font_scale, font_thickness
        )

        # Position: top-right with padding
        padding = 10
        frame_width = image.shape[1]
        x = frame_width - text_width - padding
        y = padding + text_height

        # Draw background rectangle
        cv2.rectangle(
            img_with_text,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            img_with_text,
            timestamp_str,
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

        return img_with_text

    def open_stats_csv(self):
        """Open statistics CSV file for writing."""
        self.stats_file = open(self.stats_csv_path, 'w', newline='')
        self.stats_writer = csv.writer(self.stats_file)

    def write_statistics(self, timestamp: datetime, u_velocity: np.ndarray,
                        v_velocity: np.ndarray):
        """
        Write velocity statistics to CSV.

        Parameters:
            timestamp: Frame timestamp
            u_velocity: Eastward velocity component
            v_velocity: Northward velocity component
        """
        # Compute statistics
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)
        mean_u = np.mean(u_velocity)
        mean_v = np.mean(v_velocity)
        mean_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        std_mag = np.std(magnitude)
        p95_mag = np.percentile(magnitude, 95)

        # Write header if first time
        if not self.stats_header_written:
            self.stats_writer.writerow([
                'timestamp', 'mean_u', 'mean_v', 'mean_magnitude',
                'max_magnitude', 'std_magnitude', 'p95_magnitude'
            ])
            self.stats_header_written = True

        # Write data
        self.stats_writer.writerow([
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            f'{mean_u:.6f}',
            f'{mean_v:.6f}',
            f'{mean_mag:.6f}',
            f'{max_mag:.6f}',
            f'{std_mag:.6f}',
            f'{p95_mag:.6f}'
        ])
        self.stats_file.flush()

    def create_validation_plots(self, mosaic_frame: np.ndarray,
                                u_velocity: np.ndarray, v_velocity: np.ndarray,
                                timestamp: datetime):
        """
        Create validation plots (quiver, magnitude, direction).

        Parameters:
            mosaic_frame: Mosaic frame for background
            u_velocity: Eastward velocity component
            v_velocity: Northward velocity component
            timestamp: Frame timestamp
        """
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)

        # Update max magnitude seen for consistent color scaling
        current_max = np.max(magnitude)
        if current_max > self.max_magnitude_seen:
            self.max_magnitude_seen = current_max

        # No rotation needed - velocities are already computed on rotated frames
        # mosaic_frame is already rotated at this point (rotation happens before optical flow)

        # 1. Quiver plot
        self._create_quiver_plot(
            mosaic_frame, u_velocity, v_velocity, magnitude, timestamp,
            self.plots_dir / f'validation_{timestamp_str}_quiver.png',
            vmax=self.max_magnitude_seen
        )

        # 2. Magnitude heatmap
        self._create_magnitude_plot(
            magnitude, timestamp,
            self.plots_dir / f'validation_{timestamp_str}_magnitude.png',
            vmax=self.max_magnitude_seen
        )

        # 3. Direction HSV
        self._create_direction_plot(
            u_velocity, v_velocity, magnitude, timestamp,
            self.plots_dir / f'validation_{timestamp_str}_direction.png'
        )

    def _create_quiver_plot(self, background: np.ndarray, u: np.ndarray,
                           v: np.ndarray, magnitude: np.ndarray,
                           timestamp: datetime, output_path: Path,
                           vmax: float = None):
        """Create quiver plot with velocity vectors over background image."""
        fig, ax = plt.subplots(figsize=(16, 12), dpi=100)

        # Show background (convert BGR to RGB)
        if len(background.shape) == 3:
            bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb, alpha=0.7)
        else:
            ax.imshow(background, cmap='gray', alpha=0.7)

        # Create subsampled grid
        step = 20
        h, w = u.shape
        y, x = np.mgrid[0:h:step, 0:w:step]
        u_sub = u[::step, ::step]
        v_sub = v[::step, ::step]
        mag_sub = magnitude[::step, ::step]

        # Calculate reasonable arrow scale
        # In matplotlib quiver: scale = data units per arrow length unit
        # Larger scale = shorter arrows
        # For velocities in m/s range (0.01-0.1), we want reasonably sized arrows
        mean_mag = np.mean(magnitude)
        if mean_mag > 0:
            # FIX: Increased from 15.0 to 150.0 (10x larger = 10x shorter arrows)
            # This prevents arrows from obscuring the entire plot
            scale = mean_mag * 150.0
        else:
            scale = 1.0

        # Quiver plot with consistent color scale
        # matplotlib quiver on imshow: U,V are in data coordinates (not image pixel coordinates)
        # Our u,v are already in correct orientation: u=east (right), v=north (up in data space)
        # matplotlib handles the image coordinate inversion automatically
        q = ax.quiver(x, y, u_sub, v_sub, mag_sub,
                     cmap='jet', scale=scale, width=0.002,
                     headwidth=3, headlength=4, alpha=0.9,
                     clim=(0, vmax) if vmax is not None else None)

        # Colorbar
        cbar = plt.colorbar(q, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Velocity (m/s)', fontsize=12, weight='bold')

        # Add title with timestamp
        timestamp_str = timestamp.strftime('%Y_%m_%d %H:%M:%S')
        ax.set_title(f'Velocity Vectors - {timestamp_str}', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_magnitude_plot(self, magnitude: np.ndarray, timestamp: datetime,
                               output_path: Path, vmax: float = None):
        """Create magnitude heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # Use consistent vmax if provided, otherwise use 95th percentile
        if vmax is None:
            vmax = np.percentile(magnitude, 95)

        im = ax.imshow(magnitude, cmap='jet', vmin=0, vmax=vmax, origin='upper')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity magnitude (m/s)', fontsize=12, weight='bold')

        # Add title with timestamp
        timestamp_str = timestamp.strftime('%Y_%m_%d %H:%M:%S')
        ax.set_title(f'Velocity Magnitude - {timestamp_str}', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_direction_plot(self, u: np.ndarray, v: np.ndarray,
                              magnitude: np.ndarray, timestamp: datetime,
                              output_path: Path):
        """Create HSV direction visualization."""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # HSV: Hue = direction, Value = magnitude
        angle = np.arctan2(v, u)
        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ((angle * 180 / np.pi) % 360) / 2  # Hue (0-180)
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        ax.imshow(rgb)

        # Add title with timestamp
        timestamp_str = timestamp.strftime('%Y_%m_%d %H:%M:%S')
        ax.set_title(f'Flow Direction & Speed - {timestamp_str}', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add legend
        ax.text(0.02, 0.98, 'Hue = Direction\nBrightness = Speed',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_overlay_frame(self, mosaic_frame: np.ndarray,
                            u_velocity: np.ndarray, v_velocity: np.ndarray,
                            timestamp: datetime) -> np.ndarray:
        """
        Create frame with velocity vectors overlaid.

        Parameters:
            mosaic_frame: Background mosaic frame (already rotated)
            u_velocity: Eastward velocity component (computed on rotated frame)
            v_velocity: Northward velocity component (computed on rotated frame)
            timestamp: Frame timestamp

        Returns:
            Frame with overlay vectors and timestamp
        """
        # mosaic_frame is already rotated
        # u_velocity, v_velocity are already computed on rotated frame
        overlay = mosaic_frame.copy()
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)

        # Update max magnitude for consistent color scaling
        current_max = np.max(magnitude)
        if current_max > self.max_magnitude_seen:
            self.max_magnitude_seen = current_max

        # Subsample
        step = self.overlay_subsample
        h, w = u_velocity.shape
        y_coords = np.arange(step//2, h, step)
        x_coords = np.arange(step//2, w, step)

        # Normalize magnitude for color mapping using consistent scale
        if self.max_magnitude_seen > 0:
            mag_norm = np.clip(magnitude / self.max_magnitude_seen * 255, 0, 255).astype(np.uint8)
        else:
            mag_norm = np.zeros_like(magnitude, dtype=np.uint8)

        # Apply colormap
        mag_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        # Compute adaptive scale: target arrow length of 20-30 pixels for mean velocity
        mean_mag = np.mean(magnitude)
        if mean_mag > 0:
            # Scale to make mean velocity show as ~25 pixel arrows
            scale = 25.0 / mean_mag
        else:
            scale = 500.0  # Default if no motion detected

        # Clamp scale to reasonable range
        scale = np.clip(scale, 100.0, 1000.0)

        # Draw vectors
        for yi in y_coords:
            for xi in x_coords:
                if yi >= h or xi >= w:
                    continue

                u_val = u_velocity[yi, xi]
                v_val = v_velocity[yi, xi]

                # Arrow endpoints
                # u: positive = east = right in image (positive x)
                # v: positive = north = up in image (negative y, since y increases downward)
                end_x = int(xi + u_val * scale)
                end_y = int(yi - v_val * scale)  # Subtract because y increases downward

                # Get color from magnitude
                color = tuple(int(c) for c in mag_color[yi, xi])

                # Draw arrow with thicker line for visibility
                cv2.arrowedLine(overlay, (xi, yi), (end_x, end_y),
                              color, thickness=2, tipLength=0.3)

        # NO ROTATION HERE - frame is already rotated before optical flow computation
        # Add timestamp overlay (already horizontal since frame is rotated)
        overlay = self._add_timestamp(overlay, timestamp)

        return overlay

    def init_overlay_video(self, frame_shape: Tuple[int, int]):
        """
        Initialize overlay video writer (DEPRECATED - now uses lazy initialization).

        Video writer is now automatically initialized on first frame write to ensure
        correct dimensions after rotation is applied.

        Parameters:
            frame_shape: (height, width) of frames (ignored)
        """
        # No-op - lazy initialization in write_overlay_frame handles this
        pass

    def write_overlay_frame(self, overlay_frame: np.ndarray):
        """Write frame to overlay video with lazy initialization."""
        if not self.create_overlay_video:
            return

        # Lazy initialization: create video writer on first frame (after rotation)
        if self.overlay_video_writer is None:
            self.overlay_video_path = self.output_dir / 'mosaic_with_velocity_overlay.mp4'
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            height, width = overlay_frame.shape[:2]

            self.overlay_video_writer = cv2.VideoWriter(
                str(self.overlay_video_path),
                fourcc,
                self.video_fps,
                (width, height)
            )

            if not self.overlay_video_writer.isOpened():
                raise RuntimeError("Could not create overlay video writer")

            print(f"  Overlay video writer initialized: {width}x{height} @ {self.video_fps} fps")

        self.overlay_video_writer.write(overlay_frame)
        self.frame_count += 1

    def close_stats_csv(self):
        """Close statistics CSV file."""
        if self.stats_file:
            self.stats_file.close()

    def close_overlay_video(self):
        """Close overlay video writer."""
        if self.overlay_video_writer:
            self.overlay_video_writer.release()
            print(f"  Overlay video saved: {self.overlay_video_path} ({self.frame_count} frames)")
