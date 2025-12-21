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
                 video_codec: str = 'mp4v'):
        """
        Initialize visualizer.

        Parameters:
            output_dir: Output directory for validation plots
            create_plots: Enable plot creation
            plot_interval: Create plot every N frames
            create_overlay_video: Enable overlay video creation
            overlay_subsample: Subsample factor for overlay vectors
            video_fps: Frame rate for overlay video
            video_codec: Codec for overlay video
        """
        self.output_dir = Path(output_dir)
        self.create_plots = create_plots
        self.plot_interval = plot_interval
        self.create_overlay_video = create_overlay_video
        self.overlay_subsample = overlay_subsample
        self.video_fps = video_fps
        self.video_codec = video_codec

        # Create subdirectories
        if self.create_plots:
            self.plots_dir = self.output_dir / 'validation_plots'
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

        # 1. Quiver plot
        self._create_quiver_plot(
            mosaic_frame, u_velocity, v_velocity, magnitude,
            self.plots_dir / f'validation_{timestamp_str}_quiver.png'
        )

        # 2. Magnitude heatmap
        self._create_magnitude_plot(
            magnitude,
            self.plots_dir / f'validation_{timestamp_str}_magnitude.png'
        )

        # 3. Direction HSV
        self._create_direction_plot(
            u_velocity, v_velocity, magnitude,
            self.plots_dir / f'validation_{timestamp_str}_direction.png'
        )

    def _create_quiver_plot(self, background: np.ndarray, u: np.ndarray,
                           v: np.ndarray, magnitude: np.ndarray,
                           output_path: Path):
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

        # Quiver plot
        # matplotlib quiver on imshow: U,V are in data coordinates (not image pixel coordinates)
        # Our u,v are already in correct orientation: u=east (right), v=north (up in data space)
        # matplotlib handles the image coordinate inversion automatically
        q = ax.quiver(x, y, u_sub, v_sub, mag_sub,
                     cmap='jet', scale=None, width=0.002,
                     headwidth=3, headlength=4, alpha=0.9)

        # Colorbar
        cbar = plt.colorbar(q, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Velocity (m/s)', fontsize=12, weight='bold')

        ax.set_title(f'Velocity Vectors ({u_sub.size} arrows)', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_magnitude_plot(self, magnitude: np.ndarray, output_path: Path):
        """Create magnitude heatmap."""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        vmax = np.percentile(magnitude, 95)
        im = ax.imshow(magnitude, cmap='jet', vmin=0, vmax=vmax, origin='upper')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Velocity magnitude (m/s)', fontsize=12, weight='bold')

        ax.set_title('Velocity Magnitude', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_direction_plot(self, u: np.ndarray, v: np.ndarray,
                              magnitude: np.ndarray, output_path: Path):
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

        ax.set_title('Flow Direction (Color) & Speed (Brightness)', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add legend
        ax.text(0.02, 0.98, 'Hue = Direction\nBrightness = Speed',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_overlay_frame(self, mosaic_frame: np.ndarray,
                            u_velocity: np.ndarray, v_velocity: np.ndarray) -> np.ndarray:
        """
        Create frame with velocity vectors overlaid.

        Parameters:
            mosaic_frame: Background mosaic frame
            u_velocity: Eastward velocity component
            v_velocity: Northward velocity component

        Returns:
            Frame with overlay vectors
        """
        overlay = mosaic_frame.copy()
        magnitude = np.sqrt(u_velocity**2 + v_velocity**2)

        # Subsample
        step = self.overlay_subsample
        h, w = u_velocity.shape
        y_coords = np.arange(step//2, h, step)
        x_coords = np.arange(step//2, w, step)

        # Normalize magnitude for color mapping
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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

        return overlay

    def init_overlay_video(self, frame_shape: Tuple[int, int]):
        """
        Initialize overlay video writer.

        Parameters:
            frame_shape: (height, width) of frames
        """
        if not self.create_overlay_video:
            return

        self.overlay_video_path = self.output_dir / 'mosaic_with_velocity_overlay.mp4'
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        height, width = frame_shape

        self.overlay_video_writer = cv2.VideoWriter(
            str(self.overlay_video_path),
            fourcc,
            self.video_fps,
            (width, height)
        )

        if not self.overlay_video_writer.isOpened():
            raise RuntimeError("Could not create overlay video writer")

        print(f"  Overlay video writer initialized: {width}x{height} @ {self.video_fps} fps")

    def write_overlay_frame(self, overlay_frame: np.ndarray):
        """Write frame to overlay video."""
        if self.overlay_video_writer:
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
