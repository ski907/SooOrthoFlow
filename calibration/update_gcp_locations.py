"""
GCP Update Tool for Relocated Cameras

Interactive tool to update Ground Control Point (GCP) pixel coordinates when cameras
are physically relocated. Uses a side-by-side interface showing:
- Left: Reference georeferenced image (mosaic/ortho) for selecting world coordinates
- Right: Camera frame (for selecting corresponding pixel coordinates)

Author: SooOrthoFlow Team
Version: 0.2.0

Usage:
python calibration/update_gcp_locations.py `
        --camera NVR3_N910A6_ch2_main `
        --video "test_videos/12-18-25 Camera Reset/N910A6_ch2_main_20251218105300_20251218105400.avi" `
        --timestamp "2025-12-18 10:53:30" `
        --reference-image "output_data/Initial Ice Pack/mosaics/mosaic_20251201_102630.tif" `
        --dem inputs/TLS_DTM_cropped_filled_utmNAD8319N.tif `
        --gcp-file inputs/GCP_24cameras_utm.csv
"""

import sys
import cv2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import rasterio
import rasterio.transform


class DualImageGCPPicker:
    """
    Interactive dual-image GCP picker for relocated cameras.

    Workflow:
    1. User clicks point in reference image (left panel) → extracts world coordinates (X, Y)
    2. Samples elevation (Z) from DEM at those coordinates
    3. User clicks corresponding point in camera frame (right panel) → captures pixel coords
    4. Repeat for all desired GCPs
    5. Return list of complete GCP mappings
    """

    def __init__(self, reference_tif_path: str, camera_frame: np.ndarray, dem_tif_path: str,
                 existing_gcps: Optional[List[Dict]] = None, camera_name: Optional[str] = None):
        """
        Initialize dual-image GCP picker.

        Parameters:
            reference_tif_path: Path to georeferenced reference image (mosaic, ortho, etc.)
            camera_frame: Camera frame image (BGR format from OpenCV)
            dem_tif_path: Path to DEM GeoTIFF for elevation sampling
            existing_gcps: Optional list of existing GCPs for this camera
            camera_name: Camera name for display
        """
        self.reference_tif_path = Path(reference_tif_path)
        self.camera_frame = camera_frame
        self.dem_path = Path(dem_tif_path)
        self.existing_gcps = existing_gcps or []
        self.camera_name = camera_name or "Unknown"

        # Images
        self.reference_img = None      # Reference image (RGB or grayscale)

        # GeoTIFF metadata
        self.reference_transform = None    # Rasterio affine transform
        self.reference_crs = None          # Coordinate reference system

        # DEM for elevation sampling
        self.dem_dataset = None        # Rasterio dataset (kept open)

        # State machine
        self.current_state = 'reference'   # 'reference' → 'camera' → 'reference'
        self.temp_reference_point = None   # Temporary storage for current GCP being picked
        self.picked_points = []            # Completed GCPs

        # Matplotlib UI
        self.fig = None
        self.ax_reference = None      # Left subplot
        self.ax_camera = None         # Right subplot
        self.reference_markers = None # Scatter plot for reference markers
        self.camera_markers = None    # Scatter plot for camera markers
        self.reference_labels = []    # Text labels for reference points
        self.camera_labels = []       # Text labels for camera points

        # Load data
        self._load_reference_geotiff()
        self._load_dem()

    def _load_reference_geotiff(self):
        """Load reference GeoTIFF with rasterio (handles RGB or grayscale)."""
        try:
            with rasterio.open(self.reference_tif_path) as src:
                self.reference_transform = src.transform  # Affine transform
                self.reference_crs = src.crs

                # Read image data - handle multi-band (RGB) or single-band (grayscale)
                if src.count == 1:
                    # Grayscale
                    self.reference_img = src.read(1)
                elif src.count >= 3:
                    # RGB - read first 3 bands and transpose to (height, width, channels)
                    self.reference_img = np.dstack([src.read(i) for i in range(1, 4)])
                else:
                    # Fallback - just read first band
                    self.reference_img = src.read(1)

            print(f"  ✓ Loaded reference image: {self.reference_img.shape}, CRS: {self.reference_crs}")
        except Exception as e:
            print(f"✗ Error loading reference GeoTIFF: {e}")
            raise

    def _load_dem(self):
        """Open DEM with rasterio (keep open for sampling)."""
        try:
            self.dem_dataset = rasterio.open(self.dem_path)
            print(f"  ✓ Loaded DEM: {self.dem_dataset.shape}, CRS: {self.dem_dataset.crs}")
        except Exception as e:
            print(f"✗ Error loading DEM: {e}")
            raise

    def _pixel_to_world(self, col: float, row: float) -> Tuple[float, float]:
        """
        Convert reference image pixel coordinates to world coordinates using affine transform.

        Parameters:
            col: Column (x) coordinate in pixel space
            row: Row (y) coordinate in pixel space

        Returns:
            (world_x, world_y): World coordinates in the reference image's CRS
        """
        world_x, world_y = self.reference_transform * (col, row)
        return world_x, world_y

    def _sample_elevation(self, world_x: float, world_y: float) -> float:
        """
        Sample elevation (Z) from DEM at world coordinates.

        Parameters:
            world_x: World X coordinate
            world_y: World Y coordinate

        Returns:
            Elevation value from DEM, or 0.0 if out of bounds or nodata
        """
        try:
            # World coords → DEM pixel coords
            row, col = rasterio.transform.rowcol(
                self.dem_dataset.transform, world_x, world_y
            )

            # Bounds check
            if (0 <= row < self.dem_dataset.height and
                0 <= col < self.dem_dataset.width):

                # Read elevation value
                z_value = self.dem_dataset.read(1, window=((row, row+1), (col, col+1)))[0, 0]

                # Check nodata
                if (self.dem_dataset.nodata is not None and
                    z_value == self.dem_dataset.nodata):
                    return 0.0

                return float(z_value)

            return 0.0  # Out of bounds

        except Exception as e:
            print(f"  Warning: Could not sample elevation at ({world_x:.2f}, {world_y:.2f}): {e}")
            return 0.0

    def _on_click(self, event):
        """
        Handle mouse click events - state machine for alternating picks.

        State machine:
        - 'reference' state: Click in reference panel → extract world coords → go to 'camera' state
        - 'camera' state: Click in camera panel → capture pixel coords → complete GCP → go to 'reference' state
        """
        # Ignore clicks when matplotlib toolbar is active (zoom/pan)
        toolbar = self.fig.canvas.toolbar
        if toolbar.mode != '':
            return

        if event.inaxes == self.ax_reference and self.current_state == 'reference':
            # REFERENCE PICK: Extract world coordinates
            col, row = event.xdata, event.ydata

            world_x, world_y = self._pixel_to_world(col, row)
            world_z = self._sample_elevation(world_x, world_y)

            # Store temporarily
            self.temp_reference_point = {
                'X': world_x,
                'Y': world_y,
                'Z': world_z,
                'reference_col': col,
                'reference_row': row
            }

            print(f"✓ Reference: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")

            # Update UI
            self._highlight_camera_panel()
            self.current_state = 'camera'

        elif event.inaxes == self.ax_camera and self.current_state == 'camera':
            # CAMERA PICK: Extract pixel coordinates
            col, row = event.xdata, event.ydata

            # Complete the GCP
            gcp = self.temp_reference_point.copy()
            gcp['col_sample'] = col
            gcp['row_sample'] = row

            self.picked_points.append(gcp)

            print(f"✓ Camera: ({col:.1f}, {row:.1f})")
            print(f"  → GCP #{len(self.picked_points)} complete\n")

            # Update displays
            self._update_displays()

            # Reset state
            self.temp_reference_point = None
            self.current_state = 'reference'
            self._highlight_reference_panel()

    def _on_key(self, event):
        """
        Handle keyboard events.

        Shortcuts:
        - 'u': Undo last GCP
        - 'z': Toggle zoom mode
        - 'q': Quit and save
        - 'escape': Cancel without saving
        """
        if event.key == 'u':
            # Undo last GCP
            if self.picked_points:
                removed = self.picked_points.pop()
                print(f"↶ Undid GCP #{len(self.picked_points) + 1}")
                self._update_displays()
            else:
                print("  No GCPs to undo")

        elif event.key == 'z':
            # Toggle zoom mode
            toolbar = self.fig.canvas.toolbar
            if toolbar.mode == 'zoom rect':
                toolbar.zoom()
                print("  Zoom mode OFF")
            else:
                toolbar.zoom()
                print("  Zoom mode ON")

        elif event.key == 'q':
            # Quit and save
            print(f"\n✓ Saving {len(self.picked_points)} GCPs...")
            plt.close(self.fig)

        elif event.key == 'escape':
            # Cancel without saving
            print("\n✗ Cancelled. Discarding all GCPs.")
            self.picked_points = []
            plt.close(self.fig)

    def _highlight_reference_panel(self):
        """Highlight reference panel as active."""
        self.ax_reference.set_title('Reference Image (Mosaic/Ortho)\n(Click to select point)',
                                fontsize=12, fontweight='bold', color='blue')
        self.ax_camera.set_title('Camera Frame\n(Waiting...)',
                                 fontsize=12, fontweight='normal', color='gray')
        self.fig.canvas.draw_idle()

    def _highlight_camera_panel(self):
        """Highlight camera panel as active."""
        self.ax_reference.set_title('Reference Image (Mosaic/Ortho)\n(Waiting...)',
                                fontsize=12, fontweight='normal', color='gray')
        self.ax_camera.set_title('Camera Frame\n(Click corresponding point)',
                                 fontsize=12, fontweight='bold', color='blue')
        self.fig.canvas.draw_idle()

    def _update_displays(self):
        """Update marker displays with current GCPs."""
        # Extract coordinates for markers
        reference_cols = [gcp['reference_col'] for gcp in self.picked_points]
        reference_rows = [gcp['reference_row'] for gcp in self.picked_points]
        camera_cols = [gcp['col_sample'] for gcp in self.picked_points]
        camera_rows = [gcp['row_sample'] for gcp in self.picked_points]

        # Update scatter plots
        self.reference_markers.set_offsets(np.c_[reference_cols, reference_rows])
        self.camera_markers.set_offsets(np.c_[camera_cols, camera_rows])

        # Remove old labels
        for label in self.reference_labels:
            label.remove()
        for label in self.camera_labels:
            label.remove()
        self.reference_labels = []
        self.camera_labels = []

        # Add new labels
        for i, gcp in enumerate(self.picked_points):
            # Reference label
            label = self.ax_reference.text(
                gcp['reference_col'] + 10, gcp['reference_row'] - 10,
                str(i + 1), fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
            self.reference_labels.append(label)

            # Camera label
            label = self.ax_camera.text(
                gcp['col_sample'] + 10, gcp['row_sample'] - 10,
                str(i + 1), fontsize=10, color='lime', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
            self.camera_labels.append(label)

        # Update figure title
        self.fig.suptitle(f'GCP Picker: {self.camera_name} | Picked {len(self.picked_points)} GCPs',
                         fontsize=14, fontweight='bold')

        self.fig.canvas.draw_idle()

    def start_picking(self) -> List[Dict]:
        """
        Launch interactive picking session.

        Returns:
            List of GCP dicts with keys: X, Y, Z, col_sample, row_sample
        """
        # Create 1 row, 2 columns (20x10 inch figure)
        self.fig, (self.ax_reference, self.ax_camera) = plt.subplots(
            1, 2, figsize=(20, 10)
        )

        # LEFT: Reference image (mosaic/ortho)
        # Display image - handle RGB or grayscale
        if len(self.reference_img.shape) == 3:
            # RGB image
            self.ax_reference.imshow(self.reference_img)
        else:
            # Grayscale image
            self.ax_reference.imshow(self.reference_img, cmap='gray')

        self.ax_reference.set_title('Reference Image (Mosaic/Ortho)\n(Click to select point)',
                                fontsize=12, fontweight='bold', color='blue')
        self.ax_reference.set_xlabel('Column (pixels)')
        self.ax_reference.set_ylabel('Row (pixels)')

        # RIGHT: Camera frame
        camera_rgb = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
        self.ax_camera.imshow(camera_rgb)
        self.ax_camera.set_title('Camera Frame\n(Waiting...)',
                                 fontsize=12, color='gray')
        self.ax_camera.set_xlabel('Column (pixels)')
        self.ax_camera.set_ylabel('Row (pixels)')

        # Markers
        self.reference_markers = self.ax_reference.scatter(
            [], [], c='red', s=200, marker='+', linewidths=3
        )
        self.camera_markers = self.ax_camera.scatter(
            [], [], c='lime', s=200, marker='+', linewidths=3
        )

        # Instructions
        instruction_text = (
            "WORKFLOW:\n"
            "1. Click point in reference image (LEFT)\n"
            "2. Click corresponding point in camera (RIGHT)\n"
            "3. Repeat for all GCPs\n\n"
            "SHORTCUTS: u=undo, z=zoom toggle, q=save & quit, ESC=cancel"
        )
        self.fig.text(0.02, 0.98, instruction_text,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Initial title
        self.fig.suptitle(f'GCP Picker: {self.camera_name} | Picked 0 GCPs',
                         fontsize=14, fontweight='bold')

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.show()

        # Close DEM dataset when done
        if self.dem_dataset:
            self.dem_dataset.close()

        return self.picked_points


def extract_frame_from_video(video_path: str, camera_name: str, timestamp: datetime,
                             time_offset: float = 0.0) -> Optional[np.ndarray]:
    """
    Extract frame from video using CameraVideoReader.

    Parameters:
        video_path: Path to video file or directory
        camera_name: Camera ID (e.g., "NVR1_N910A6_ch4_main")
        timestamp: Target timestamp for frame extraction
        time_offset: Camera time offset in seconds

    Returns:
        Frame as numpy array (BGR format), or None if extraction fails
    """
    # Import here to avoid circular dependency
    sys.path.append(str(Path(__file__).parent.parent / 'analysis' / 'video_mosaic'))
    from camera_video_reader import CameraVideoReader

    video_path = Path(video_path)
    video_dir = video_path if video_path.is_dir() else video_path.parent

    # Extract NVR name for time offset
    nvr_name = camera_name.split('_')[0]  # 'NVR1', 'NVR2', etc.
    camera_time_offsets = {nvr_name: time_offset}

    reader = CameraVideoReader(
        video_dir=video_dir,
        camera_ids=[camera_name],
        camera_time_offsets=camera_time_offsets,
        verbose_frames=True
    )

    try:
        frames = reader.get_frames_at_timestamp(timestamp)
        return frames.get(camera_name) if frames else None
    finally:
        reader.close()


def load_gcp_csv(gcp_file_path: str) -> pd.DataFrame:
    """
    Load GCP CSV file.

    Parameters:
        gcp_file_path: Path to GCP CSV file

    Returns:
        DataFrame with GCP data
    """
    try:
        gcp_df = pd.read_csv(gcp_file_path)
        print(f"  ✓ Loaded {len(gcp_df)} total GCPs from {gcp_file_path}")
        return gcp_df
    except Exception as e:
        print(f"✗ Error loading GCP CSV: {e}")
        raise


def update_gcp_csv(gcp_df: pd.DataFrame, camera_name: str, new_gcps: List[Dict],
                  output_path: str) -> pd.DataFrame:
    """
    Update GCP CSV with new coordinates for specified camera.

    Steps:
    1. Remove old rows for this camera
    2. Create new rows with updated pixel coordinates
    3. Backup original file
    4. Save updated CSV

    Parameters:
        gcp_df: Original GCP DataFrame
        camera_name: Camera ID to update
        new_gcps: List of new GCP dicts
        output_path: Path for output CSV

    Returns:
        Updated DataFrame
    """
    # Remove old rows
    gcp_df_filtered = gcp_df[gcp_df['camera_name'] != camera_name].copy()
    removed_count = len(gcp_df) - len(gcp_df_filtered)
    print(f"  Removed {removed_count} old rows for {camera_name}")

    # Create new rows
    new_rows = []
    for gcp in new_gcps:
        # Extract channel number from camera name
        channel = camera_name.split('_ch')[1].split('_')[0] if '_ch' in camera_name else '1'

        new_row = {
            'image_name': f"{camera_name}.tiff",
            'channel': int(channel),
            'camera_name': camera_name,
            'X': gcp['X'],
            'Y': gcp['Y'],
            'Z': gcp['Z'],
            'col_sample': gcp['col_sample'],
            'row_sample': gcp['row_sample']
        }
        new_rows.append(new_row)

    # Append new rows
    new_df = pd.DataFrame(new_rows)
    updated_gcp_df = pd.concat([gcp_df_filtered, new_df], ignore_index=True)
    updated_gcp_df = updated_gcp_df.sort_values(['camera_name', 'X', 'Y'])

    # Create backup of original
    output_path = Path(output_path)
    if output_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = output_path.with_name(
            f"{output_path.stem}_backup_{timestamp}{output_path.suffix}"
        )
        output_path.rename(backup_path)
        print(f"  ✓ Created backup: {backup_path}")

    # Save updated CSV
    updated_gcp_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved updated CSV: {output_path}")
    print(f"  New GCP count: {len(new_gcps)}")

    return updated_gcp_df


def main():
    """Command-line interface for GCP update tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Update GCP pixel coordinates for relocated cameras',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python update_gcp_locations.py \\
      --camera NVR2_N910A6_ch1_main \\
      --video /path/to/videos/ \\
      --timestamp "2025-12-05 14:30:00" \\
      --reference-image output_data/video_mosaic/mosaic_frame_001.tif \\
      --dem inputs/TLS_DTM_cropped_filled_utmNAD8319N.tif \\
      --gcp-file inputs/GCP_24cameras_utm.csv
        """
    )

    # Required arguments
    parser.add_argument('--camera', required=True,
                       help='Camera name (e.g., NVR2_N910A6_ch1_main)')
    parser.add_argument('--video', required=True,
                       help='Path to video file or directory')
    parser.add_argument('--timestamp', required=True,
                       help='Frame timestamp: "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--reference-image', required=True,
                       help='Georeferenced reference image (mosaic, ortho, etc.)')

    # Optional with defaults
    parser.add_argument('--dem',
                       default='inputs/TLS_DTM_cropped_filled_utmNAD8319N.tif',
                       help='DEM GeoTIFF for elevation sampling')
    parser.add_argument('--gcp-file',
                       default='inputs/GCP_24cameras_utm.csv',
                       help='Input GCP CSV file')
    parser.add_argument('--output', default=None,
                       help='Output CSV path (default: auto-generated)')
    parser.add_argument('--time-offset', type=float, default=0.0,
                       help='Camera time offset in seconds')

    args = parser.parse_args()

    # Parse timestamp
    timestamp = datetime.strptime(args.timestamp, '%Y-%m-%d %H:%M:%S')

    # Auto-generate output path if not specified
    if args.output is None:
        timestamp_str = datetime.now().strftime('%Y%m%d')
        gcp_path = Path(args.gcp_file)
        args.output = gcp_path.parent / f"{gcp_path.stem}_updated_{args.camera}_{timestamp_str}.csv"

    print(f"\n{'='*60}")
    print(f"GCP Update Tool - {args.camera}")
    print(f"{'='*60}\n")

    # Extract frame from video
    print(f"[1/4] Extracting frame at {args.timestamp}...")
    camera_frame = extract_frame_from_video(
        args.video, args.camera, timestamp, args.time_offset
    )

    if camera_frame is None:
        print("✗ Error: Could not extract frame from video")
        return 1

    print(f"  ✓ Frame extracted: {camera_frame.shape}")

    # Load GCP CSV
    print(f"\n[2/4] Loading GCP CSV...")
    gcp_df = load_gcp_csv(args.gcp_file)
    existing_gcps = gcp_df[gcp_df['camera_name'] == args.camera]
    print(f"  Existing GCPs for {args.camera}: {len(existing_gcps)}")

    # Launch interactive picker
    print(f"\n[3/4] Launching interactive picker...")
    print("  Use matplotlib toolbar for zoom/pan")
    print("  Click reference image (left) then camera (right) for each GCP")
    print("  Press 'q' when done, ESC to cancel\n")

    picker = DualImageGCPPicker(
        reference_tif_path=args.reference_image,
        camera_frame=camera_frame,
        dem_tif_path=args.dem,
        existing_gcps=existing_gcps.to_dict('records'),
        camera_name=args.camera
    )

    picked_gcps = picker.start_picking()

    if not picked_gcps:
        print("\n✗ No GCPs picked. Exiting without changes.")
        return 0

    # Update CSV
    print(f"\n[4/4] Updating GCP CSV...")
    updated_gcp_df = update_gcp_csv(
        gcp_df, args.camera, picked_gcps, args.output
    )

    print(f"\n{'='*60}")
    print(f"✓ SUCCESS! Updated {len(picked_gcps)} GCPs for {args.camera}")
    print(f"  Output file: {args.output}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
