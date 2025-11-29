import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import argparse
from datetime import datetime

# Import new calibration I/O utilities
from calibration_io import (
    load_camera_calibrations,
    save_camera_calibrations,
    update_camera_calibration,
    find_most_recent_calibration_csv,
    save_ortho_cache,
    delete_old_ortho_cache
)

def find_most_recent_calibration(calibration_file, date=None):
    """
    Find the most recent calibration CSV file to load from.
    Priority:
    1. File with specified date (if it exists)
    2. Most recent dated file
    3. Fall back to calibration directory search

    Args:
        calibration_file: Path to calibration CSV file
        date: Date string (YYYYMMDD). If None, uses today.

    Returns:
        Path to the calibration file to load from
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    cal_path = Path(calibration_file)

    # Check if specified date's file exists
    dated_file = cal_path.parent / f"camera_calibrations_{date}.csv"
    if dated_file.exists():
        print(f"Found existing calibration for {date}, will update it")
        return dated_file

    # Find all dated calibration CSV files
    dated_files = sorted(
        p for p in cal_path.parent.iterdir()
        if p.is_file()
        and p.name.startswith("camera_calibrations_")
        and p.name.endswith(".csv")
        and len(p.stem.split("_")[-1]) == 8
        and p.stem.split("_")[-1].isdigit()
    )

    if dated_files:
        # Return the most recent dated file
        most_recent = dated_files[-1]
        recent_date = most_recent.stem.split('_')[-1]
        print(f"Loading from most recent calibration: {recent_date}")
        return most_recent

    # Try to find any calibration file
    any_cal = find_most_recent_calibration_csv(str(cal_path.parent))
    if any_cal:
        print(f"Loading from found calibration file: {any_cal}")
        return any_cal

    raise FileNotFoundError(f"No calibration CSV files found in {cal_path.parent}")


def load_gcp_targets(gcp_file, camera_id, calibration_file=None):
    """
    Load GCP world coordinates and pixel coordinates for a specific camera.

    Priority order:
    1. If calibration_file provided: Check calibration CSV for gcp_pixel_coords
    2. Otherwise: Check for camera-specific updated GCP file (from old recalibrations)
    3. Fall back to main GCP file

    Note: World coordinates (X, Y, Z) always come from the main GCP file (they don't change).
    Only pixel coordinates may come from the calibration CSV.

    Returns list of GCP names and their X,Y,Z coordinates and pixel coordinates
    """
    # Always load world coordinates from main GCP file (they never change)
    gcp_data = pd.read_csv(gcp_file)
    camera_gcps = gcp_data[gcp_data['camera_name'].str.contains(camera_id)]

    # Try to get pixel coordinates from calibration CSV if provided
    pixel_coords_from_calib = None
    if calibration_file:
        try:
            from calibration_io import load_camera_calibrations
            calibrations = load_camera_calibrations(calibration_file)
            if camera_id in calibrations:
                gcp_pixel_coords = calibrations[camera_id].get('gcp_pixel_coords', [])
                if gcp_pixel_coords:
                    print(f"  Using GCP pixel coordinates from calibration CSV")
                    # Convert to dict for easy lookup by world coords
                    pixel_coords_from_calib = {
                        (gcp['X'], gcp['Y'], gcp['Z']): (gcp['col'], gcp['row'])
                        for gcp in gcp_pixel_coords
                    }
        except Exception as e:
            print(f"  Warning: Could not load GCP pixel coords from calibration: {e}")

    # If no pixel coords from calibration, check for old-style updated GCP file
    if pixel_coords_from_calib is None:
        gcp_path = Path(gcp_file)
        updated_gcp_file = gcp_path.parent / f'GCP_{camera_id}_updated.csv'

        if updated_gcp_file.exists():
            print(f"  Using updated GCP locations from {updated_gcp_file.name}")
            updated_data = pd.read_csv(updated_gcp_file)
            camera_gcps = updated_data

    # Build GCP list with world coords and pixel coords
    gcps = []
    for idx, row in camera_gcps.iterrows():
        # Use pixel coords from calibration if available, otherwise from GCP file
        if pixel_coords_from_calib:
            key = (row['X'], row['Y'], row['Z'])
            if key in pixel_coords_from_calib:
                col, row_coord = pixel_coords_from_calib[key]
            else:
                # Fall back to original if not in calibration
                col, row_coord = row['col_sample'], row['row_sample']
        else:
            col, row_coord = row['col_sample'], row['row_sample']

        gcps.append({
            'name': f"GCP_{idx}",
            'X': row['X'],
            'Y': row['Y'],
            'Z': row['Z'],
            'original_col': col,
            'original_row': row_coord
        })

    return gcps


class InteractiveGCPPicker:
    """
    Interactive tool to pick GCP locations in an image
    """
    def __init__(self, image_path, gcps, zoom_radius=300):
        self.image_path = image_path
        self.img = cv2.imread(str(image_path))
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.gcps = gcps
        self.current_gcp_idx = 0
        self.picked_points = []
        self.skipped_gcps = []
        self.fig = None
        self.ax = None
        self.point_plot = None
        self.text_display = None
        self.hint_plot = None
        self.point_labels = []  # Track text labels we create
        self.zoom_radius = zoom_radius  # Pixels to show around hint
        self.zoomed = False  # Track if currently zoomed
        
    def start_picking(self):
        """Start the interactive picking session"""
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title(f'GCP Picker - {self.image_path.name}')
        
        # Display image
        self.ax.imshow(self.img_rgb)
        self.ax.set_title(self._get_title(), fontsize=14, fontweight='bold')
        
        # Create empty scatter plot for picked points
        self.point_plot = self.ax.scatter([], [], c='red', s=200, marker='+', linewidths=3)
        
        # Add text display for instructions
        instruction_text = (
            "Click on the target shown below\n"
            "Press 'n' to SKIP if target not visible\n"
            "Press 'u' to undo last action\n"
            "Press 'o' to zoom OUT (full view)\n"
            "Press 'q' to quit (saves progress)\n"
            "Close window to cancel"
        )
        self.text_display = self.fig.text(0.02, 0.98, instruction_text,
                                          transform=self.fig.transFigure,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Show original location hint and zoom to it
        self._show_hint()
        self._zoom_to_hint()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.tight_layout()
        plt.show()
        
        return self.picked_points, self.skipped_gcps
    
    def _show_hint(self):
        """Show hint for current GCP location"""
        if self.current_gcp_idx < len(self.gcps):
            if 'original_col' in self.gcps[self.current_gcp_idx] and \
               self.gcps[self.current_gcp_idx]['original_col'] is not None:
                orig_col = self.gcps[self.current_gcp_idx]['original_col']
                orig_row = self.gcps[self.current_gcp_idx]['original_row']

                # Clear previous hint
                if self.hint_plot:
                    self.hint_plot.remove()

                self.hint_plot = self.ax.plot(orig_col, orig_row, 'ro', markersize=15,
                            fillstyle='none', markeredgewidth=2,
                            label='Original location (hint)')[0]
                self.ax.legend()

    def _zoom_to_hint(self):
        """Zoom into the area around the hint location"""
        if self.current_gcp_idx < len(self.gcps):
            gcp = self.gcps[self.current_gcp_idx]
            if 'original_col' in gcp and gcp['original_col'] is not None:
                orig_col = gcp['original_col']
                orig_row = gcp['original_row']

                # Calculate zoom bounds
                img_height, img_width = self.img_rgb.shape[:2]

                x_min = max(0, orig_col - self.zoom_radius)
                x_max = min(img_width, orig_col + self.zoom_radius)
                y_min = max(0, orig_row - self.zoom_radius)
                y_max = min(img_height, orig_row + self.zoom_radius)

                # Set axis limits to zoom in
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_max, y_min)  # Y is inverted in images
                self.zoomed = True
                self.fig.canvas.draw()

    def _zoom_out(self):
        """Zoom out to show full image"""
        img_height, img_width = self.img_rgb.shape[:2]
        self.ax.set_xlim(0, img_width)
        self.ax.set_ylim(img_height, 0)  # Y is inverted in images
        self.zoomed = False
        self.fig.canvas.draw()
    
    def _get_title(self):
        """Generate title showing current GCP"""
        if self.current_gcp_idx < len(self.gcps):
            gcp = self.gcps[self.current_gcp_idx]
            picked_count = len(self.picked_points)
            skipped_count = len(self.skipped_gcps)
            
            return (f"GCP {self.current_gcp_idx + 1}/{len(self.gcps)} "
                   f"(Picked: {picked_count}, Skipped: {skipped_count})\n"
                   f"X={gcp['X']:.2f}, Y={gcp['Y']:.2f}, Z={gcp['Z']:.2f}\n"
                   f"Click target location or press 'n' to skip if not visible")
        else:
            picked_count = len(self.picked_points)
            skipped_count = len(self.skipped_gcps)
            return (f"✓ All GCPs processed! "
                   f"Picked: {picked_count}, Skipped: {skipped_count}\n"
                   f"Close window or press 'q' to finish")
    
    def _on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax:
            return
        
        # CRITICAL: Ignore clicks when toolbar is active
        toolbar = self.fig.canvas.toolbar
        if toolbar.mode != '':
            return
        
        if self.current_gcp_idx >= len(self.gcps):
            return
        
        # Record the point
        col = event.xdata
        row = event.ydata
        
        self.picked_points.append({
            'gcp': self.gcps[self.current_gcp_idx],
            'col': col,
            'row': row
        })
        
        print(f"✓ Picked GCP {self.current_gcp_idx + 1}: "
              f"col={col:.1f}, row={row:.1f}")
        
        # Update display
        self._update_display()
        
        # Move to next GCP
        self._advance_to_next_gcp()
    
    def _on_key(self, event):
        """Handle key press"""
        if event.key == 'n':
            # Skip this GCP
            if self.current_gcp_idx < len(self.gcps):
                skipped_gcp = self.gcps[self.current_gcp_idx]
                self.skipped_gcps.append(skipped_gcp)
                print(f"⊘ Skipped GCP {self.current_gcp_idx + 1} "
                      f"(X={skipped_gcp['X']:.2f}, Y={skipped_gcp['Y']:.2f})")
                self._advance_to_next_gcp()

        elif event.key == 'o':
            # Toggle zoom out to full view
            if self.zoomed:
                self._zoom_out()
                print("Zoomed out to full view")
            else:
                self._zoom_to_hint()
                print("Zoomed to hint location")

        elif event.key == 'u':
            # Undo last action (either pick or skip)
            if self.picked_points or self.skipped_gcps:
                if self.current_gcp_idx > 0:
                    self.current_gcp_idx -= 1
                
                # Determine what to undo
                if self.picked_points and (not self.skipped_gcps or 
                   self.picked_points[-1]['gcp'] == self.gcps[self.current_gcp_idx]):
                    self.picked_points.pop()
                    print(f"↶ Undid pick. Now at GCP {self.current_gcp_idx + 1}")
                elif self.skipped_gcps:
                    self.skipped_gcps.pop()
                    print(f"↶ Undid skip. Now at GCP {self.current_gcp_idx + 1}")
                
                self._update_display()
                self.ax.set_title(self._get_title(), fontsize=14, fontweight='bold')
                self._show_hint()
                self._zoom_to_hint()  # Zoom to hint after undo
                self.fig.canvas.draw()
        
        elif event.key == 'q':
            # Quit
            plt.close(self.fig)
    
    def _advance_to_next_gcp(self):
        """Move to the next GCP and update display"""
        self.current_gcp_idx += 1

        if self.current_gcp_idx < len(self.gcps):
            self.ax.set_title(self._get_title(), fontsize=14, fontweight='bold')
            self._show_hint()
            self._zoom_to_hint()  # Auto-zoom to next hint
        else:
            picked_count = len(self.picked_points)
            skipped_count = len(self.skipped_gcps)
            self.ax.set_title(f"✓ All GCPs processed! "
                            f"Picked: {picked_count}, Skipped: {skipped_count}\n"
                            f"Close window to finish",
                            fontsize=14, fontweight='bold', color='green')
            if self.hint_plot:
                self.hint_plot.remove()
                self.hint_plot = None
            self._zoom_out()  # Zoom out when done

        self.fig.canvas.draw()
    
    def _update_display(self):
        """Update the visualization of picked points"""
        # Clear old point labels
        for label in self.point_labels:
            label.remove()
        self.point_labels = []
        
        if self.picked_points:
            cols = [p['col'] for p in self.picked_points]
            rows = [p['row'] for p in self.picked_points]
            self.point_plot.set_offsets(np.c_[cols, rows])
            
            # Add new labels
            for i, (col, row) in enumerate(zip(cols, rows)):
                label = self.ax.text(col + 10, row, f"{i+1}", color='red', 
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                self.point_labels.append(label)


def validate_gcp_distribution(image_points, min_col_span=1000, min_row_span=800, mode='full'):
    """
    Validate GCP spatial distribution

    Args:
        image_points: Nx2 array of (col, row) coordinates
        min_col_span: Minimum column span in pixels
        min_row_span: Minimum row span in pixels
        mode: 'pose-only' or 'full' (affects thresholds)

    Returns:
        (bool, str): (is_valid, warning_message)
    """
    cols = image_points[:, 0]
    rows = image_points[:, 1]
    col_span = cols.max() - cols.min()
    row_span = rows.max() - rows.min()

    # More lenient thresholds for pose-only mode
    if mode == 'pose-only':
        min_col_span = max(800, min_col_span * 0.8)
        min_row_span = max(600, min_row_span * 0.75)

    print(f"\n  Point distribution:")
    print(f"    Column span: {col_span:.1f} pixels")
    print(f"    Row span: {row_span:.1f} pixels")
    print(f"    Coverage: {'GOOD' if (col_span > 1500 and row_span > 1000) else 'MARGINAL'}")

    if col_span < min_col_span or row_span < min_row_span:
        warning = (f"Points are very clustered! "
                  f"Col span: {col_span:.0f} < {min_col_span}, "
                  f"Row span: {row_span:.0f} < {min_row_span}")
        return False, warning

    return True, None


def solve_pose_only(gcp_world_coords, image_points, K, D, image_path,
                    min_inlier_ratio=0.7, max_rms=10.0):
    """
    Solve for camera pose (R, t) only, keeping intrinsics (K, D) fixed

    Correct workflow for fisheye camera pose estimation:
    1. Undistort distorted points to normalized rays
    2. Convert normalized rays to pinhole pixel coordinates
    3. Use solvePnPRansac with pinhole model for initial robust estimate
    4. Refine pose using fisheye.calibrate with fixed intrinsics
    5. Validate with fisheye projection

    This is appropriate when cameras have shifted slightly but lens parameters haven't changed.

    Args:
        gcp_world_coords: Nx3 array of world coordinates (X, Y, Z)
        image_points: Nx2 array of image coordinates (col, row)
        K: 3x3 camera intrinsic matrix (FIXED)
        D: Distortion coefficients (FIXED)
        image_path: Path to image (for getting image size)
        min_inlier_ratio: Minimum fraction of inliers required (default 0.7 = 70%)
        max_rms: Maximum acceptable RMS reprojection error in pixels (default 10.0)

    Returns:
        (rvec, tvec, rms, image_size): Rotation vector, translation vector, RMS error, image size

    Raises:
        ValueError: If pose solving fails or solution is degenerate
    """
    print(f"\n{'='*60}")
    print("Pose-Only Refinement (K and D fixed)")
    print(f"{'='*60}")

    # Load image to get size
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_size = (img.shape[1], img.shape[0])

    # Step 1: Undistort to NORMALIZED rays (not pixel coordinates)
    print("Step 1: Undistorting to normalized rays...")
    image_points_normalized = cv2.fisheye.undistortPoints(
        image_points.reshape(-1, 1, 2).astype(np.float32),
        K, D, None, None  # P=None gives normalized coordinates
    ).reshape(-1, 2)

    # Step 2: Create pinhole K matrix and convert to pinhole pixels
    print("Step 2: Converting to pinhole pixel coordinates...")
    fx_pinhole = (K[0, 0] + K[1, 1]) / 2.0  # Mean focal length
    Kp = np.array([
        [fx_pinhole, 0, 0],
        [0, fx_pinhole, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # Convert normalized rays to pinhole pixel coordinates
    image_points_pinhole = image_points_normalized * fx_pinhole

    # Step 3: Solve PnP with RANSAC for initial robust estimate
    print(f"Step 3: Solving PnP with RANSAC ({len(gcp_world_coords)} GCPs)...")
    success, rvec_init, tvec_init, inliers = cv2.solvePnPRansac(
        gcp_world_coords.astype(np.float32),
        image_points_pinhole.astype(np.float32),
        Kp,
        None,  # No distortion for pinhole model
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=8.0,  # Pixels in pinhole space
        confidence=0.99
    )

    if not success or rvec_init is None or tvec_init is None:
        raise ValueError("Pose solving failed - could not find valid solution")

    if inliers is None or len(inliers) < 4:
        raise ValueError(f"Pose solving failed - only {len(inliers) if inliers is not None else 0} inliers found (need >= 4)")

    inlier_ratio = len(inliers) / len(gcp_world_coords)
    print(f"  Initial RANSAC inliers: {len(inliers)}/{len(gcp_world_coords)} ({inlier_ratio*100:.1f}%)")

    if inlier_ratio < min_inlier_ratio:
        raise ValueError(f"Too many outliers ({inlier_ratio*100:.1f}% inliers < {min_inlier_ratio*100:.0f}%)")

    # Step 4: Nonlinear refinement using fisheye model with ALL points
    print("Step 4: Refining pose with fisheye model...")
    object_points = [gcp_world_coords.astype(np.float32)]
    image_points_list = [image_points.astype(np.float32)]

    # Calibration flags: fix intrinsics, only optimize extrinsics
    flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS |
             cv2.fisheye.CALIB_FIX_INTRINSIC |
             cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    try:
        rms_refined, K_out, D_out, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points_list,
            image_size,
            K.copy(),  # Initial guess (will be fixed)
            D.copy(),  # Initial guess (will be fixed)
            rvecs=[rvec_init],  # Initial pose from PnP
            tvecs=[tvec_init],
            flags=flags,
            criteria=criteria
        )

        rvec_refined = rvecs[0]
        tvec_refined = tvecs[0]
        print(f"  Refinement RMS: {rms_refined:.4f} pixels")

    except cv2.error as e:
        print(f"  ⚠ WARNING: Fisheye refinement failed ({e})")
        print(f"  Falling back to initial RANSAC solution")
        rvec_refined = rvec_init
        tvec_refined = tvec_init
        rms_refined = None

    # Step 5: Validate with fisheye projection
    print("Step 5: Validating with fisheye projection...")
    projected_pts, _ = cv2.fisheye.projectPoints(
        gcp_world_coords.reshape(-1, 1, 3).astype(np.float32),
        rvec_refined, tvec_refined, K, D
    )
    projected_pts = projected_pts.reshape(-1, 2)

    errors = np.linalg.norm(projected_pts - image_points, axis=1)
    rms = np.sqrt(np.mean(errors**2))

    print(f"  Final RMS reprojection error: {rms:.4f} pixels")
    print(f"  Max error: {errors.max():.4f} pixels")
    print(f"  Mean error: {errors.mean():.4f} pixels")

    # Validate solution quality
    if rms > max_rms:
        raise ValueError(f"RMS error too high ({rms:.4f} > {max_rms} pixels)")

    if errors.max() > 20.0:
        print(f"\n  ⚠ WARNING: Large max error ({errors.max():.4f} > 20.0 pixels)")
        print("     Check for incorrect GCP picks")

    print(f"{'='*60}\n")

    return rvec_refined, tvec_refined, rms, image_size


def recalibrate_single_camera(image_path, gcp_file, camera_id, dem_path,
                              calibration_file, output_dir=None,
                              resolution=0.005, padding_meters=0.5,
                              min_gcps=6, date=None, mode='pose-only',
                              min_inlier_ratio=0.7, max_rms=10.0):
    """
    Interactively recalibrate a single camera

    Args:
        date: Date string for the new calibration file (YYYYMMDD).
              If None, extracts date from timestamp folder in image path (e.g., frames/20251016_103100/).
              Falls back to image filename, then today's date if not found.
        mode: 'pose-only' or 'full'.
              'pose-only': Only refines camera position/orientation (R, t), keeps lens parameters (K, D) fixed.
                           Faster, requires fewer GCPs (min 4-6), appropriate for small camera shifts.
              'full': Complete recalibration of all parameters (K, D, R, t).
                      Slower, requires more GCPs (min 6-16), use if lens changed or large movements.
        min_inlier_ratio: For pose-only mode, minimum fraction of GCPs that must be inliers (default 0.7)
        max_rms: For pose-only mode, maximum acceptable RMS reprojection error in pixels (default 10.0)

    Returns:
        bool: True if successful, False if failed
    """
    from undistort_and_orthorectify import (calibrate_fisheye_camera, create_orthorectification_params,
                          load_dem_from_tiff, create_ortho_lookup_tables_with_dem,
                          orthorectify_with_lookup, save_geotiff, undistort_fisheye)
    
    print("="*60)
    print(f"Interactive Recalibration for {camera_id}")
    print(f"Mode: {mode.upper()}")
    print("="*60)

    # Adjust minimum GCP requirements based on mode
    if mode == 'pose-only':
        if min_gcps < 4:
            min_gcps = 4  # Absolute minimum for pose
        print(f"\nPose-Only Mode:")
        print(f"  - Only refining camera position/orientation")
        print(f"  - Lens parameters (K, D) remain fixed")
        print(f"  - Faster, requires fewer GCPs (min {min_gcps})")
    else:  # full mode
        if min_gcps < 6:
            min_gcps = 6  # Minimum for fisheye full calibration
        print(f"\nFull Calibration Mode:")
        print(f"  - Complete recalibration of all parameters")
        print(f"  - Includes lens distortion (K, D) and pose (R, t)")
        print(f"  - More robust but requires more GCPs (min {min_gcps})")

    # Load GCPs
    print(f"\nLoading GCP targets from {gcp_file}...")
    gcps = load_gcp_targets(gcp_file, camera_id, calibration_file)
    print(f"Found {len(gcps)} GCP targets")
    print(f"\nFor good calibration with limited targets:")
    print(f"  - Pick at least {min_gcps} points (ideally {min_gcps + 2}-{min_gcps + 4})")
    print(f"  - Spread points across the visible area")
    print(f"  - Skip clustered margin targets if you have better coverage")
    print(f"  - Avoid clicking the same point multiple times!")
    
    # Interactive picking
    print(f"\nOpening image for interactive picking...")
    print("Instructions:")
    print("  - Click on each target in the image")
    print("  - Press 's' to SKIP if target is not visible or too clustered")
    print("  - Press 'u' to undo last action")
    print("  - Press 'q' or close window when done")
    print()
    
    picker = InteractiveGCPPicker(Path(image_path), gcps)
    picked_points, skipped_gcps = picker.start_picking()
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total GCPs: {len(gcps)}")
    print(f"  Picked: {len(picked_points)}")
    print(f"  Skipped: {len(skipped_gcps)}")
    print(f"{'='*60}")
    
    if not picked_points:
        print("\n✗ No points picked. Aborting.")
        return False

    if len(picked_points) < min_gcps:
        print(f"\n✗ Error: Need at least {min_gcps} GCPs for fisheye calibration")
        print(f"  Only have {len(picked_points)} points")
        print("\nAborting. Please run again and pick more points.")
        return False
    
    # Check for duplicate or very close points
    print("\nChecking point quality...")
    image_points = np.array([[p['col'], p['row']] for p in picked_points])
    
    # Check for points that are too close together (within 30 pixels - reduced threshold)
    duplicate_threshold = 30
    close_pairs = []
    for i in range(len(image_points)):
        for j in range(i + 1, len(image_points)):
            dist = np.linalg.norm(image_points[i] - image_points[j])
            if dist < duplicate_threshold:
                close_pairs.append((i+1, j+1, dist))
    
    if close_pairs:
        print(f"  ⚠ Found {len(close_pairs)} pairs of very close points:")
        for i, j, dist in close_pairs[:5]:  # Show first 5
            print(f"    Points {i} and {j}: {dist:.1f} pixels apart")
        print(f"  This may cause calibration to fail.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Try again and avoid clicking near the same location.")
            return False
    
    # Check spatial distribution
    cols = image_points[:, 0]
    rows = image_points[:, 1]
    col_span = cols.max() - cols.min()
    row_span = rows.max() - rows.min()
    
    print(f"\n  Point distribution:")
    print(f"    Column span: {col_span:.1f} pixels")
    print(f"    Row span: {row_span:.1f} pixels")
    print(f"    Coverage: {'GOOD' if (col_span > 1500 and row_span > 1000) else 'MARGINAL'}")
    
    # More lenient check
    if col_span < 1000 or row_span < 800:
        print(f"\n  ⚠ WARNING: Points are very clustered!")
        print(f"    Calibration quality will be poor.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Try to pick points with better spatial coverage.")
            return False
    
    if len(skipped_gcps) > 0:
        print(f"\nSkipped {len(skipped_gcps)} GCPs (clustered margin targets, etc)")
    
    # Create new GCP dataframe (for use in calibration)
    print(f"\nCreating GCP data with {len(picked_points)} points...")
    new_gcp_data = pd.DataFrame([{
        'camera_name': camera_id,
        'X': p['gcp']['X'],
        'Y': p['gcp']['Y'],
        'Z': p['gcp']['Z'],
        'col_sample': p['col'],
        'row_sample': p['row']
    } for p in picked_points])

    # Note: GCP pixel coordinates will be saved to calibration CSV
    # (not to a separate file in inputs/)

    # Save to output directory if specified (for reference)
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(exist_ok=True)
        backup_gcp_file = output_path / f'GCP_{camera_id}_recalibrated_{date}.csv'
        new_gcp_data.to_csv(backup_gcp_file, index=False)
        print(f"Saved backup GCP file: {backup_gcp_file}")
    
    # Perform calibration based on mode
    print(f"\nRecalibrating {camera_id}...")

    # Add camera_id column
    new_gcp_data['camera_id'] = camera_id

    if mode == 'pose-only':
        # Pose-only refinement: Load existing K, D and solve for R, t only
        print(f"\nLoading existing calibration to get K and D...")
        source_file = find_most_recent_calibration(calibration_file, date)

        calibrations = load_camera_calibrations(source_file)

        if camera_id not in calibrations:
            print(f"\n✗ Error: Camera {camera_id} not found in calibration file")
            print(f"  Available cameras: {list(calibrations.keys())}")
            print(f"\n  Cannot use pose-only mode without existing calibration.")
            print(f"  Please run full calibration first, or use --mode full")
            return False

        # Get existing K, D from previous calibration
        K = calibrations[camera_id]['K']
        D = calibrations[camera_id]['D']
        image_size = calibrations[camera_id]['image_size']

        print(f"✓ Loaded K and D from {source_file}")
        print(f"  Using existing lens parameters")

        # Extract GCP world coordinates and image points
        gcp_world = new_gcp_data[['X', 'Y', 'Z']].values
        gcp_image = new_gcp_data[['col_sample', 'row_sample']].values

        # Try pose-only calibration with automatic threshold override option
        try:
            rvec, tvec, rms, image_size = solve_pose_only(
                gcp_world, gcp_image, K, D, image_path,
                min_inlier_ratio, max_rms
            )
            camera_gcps = new_gcp_data  # For orthorectification params
        except ValueError as e:
            print(f"\n⚠ WARNING: Pose refinement validation failed")
            print(f"Error: {str(e)}")
            print("\nThis may indicate:")
            print("  - Some GCP picks were inaccurate")
            print("  - Camera moved more than expected")
            print("  - Outlier points in the selection")

            # Offer to proceed anyway by ignoring thresholds
            proceed = input("\nProceed anyway (ignore thresholds)? (y/n): ").strip().lower()

            if proceed == 'y':
                print("\nRetrying with thresholds disabled...")
                # Retry with very lenient thresholds
                try:
                    rvec, tvec, rms, image_size = solve_pose_only(
                        gcp_world, gcp_image, K, D, image_path,
                        min_inlier_ratio=0.0,  # Accept all points
                        max_rms=1000.0  # Effectively no RMS limit
                    )
                    camera_gcps = new_gcp_data
                    print("✓ Calibration completed (thresholds ignored)")
                except ValueError as e2:
                    print(f"\n✗ Calibration still failed: {str(e2)}")
                    return False
            else:
                print("\nSuggestions:")
                print("  - Re-run and pick points more carefully")
                print("  - Try picking more well-distributed points")
                print("  - If camera moved significantly, use --mode full instead")
                return False

    else:  # full mode
        try:
            K, D, rvec, tvec, rms, image_size, camera_gcps = calibrate_fisheye_camera(
                new_gcp_data, image_path, camera_id
            )
        except cv2.error as e:
            print(f"\n✗ Calibration failed!")
            print(f"Error: {str(e)}")
            print("\nMost common causes:")
            print("  1. Duplicate points - clicked same location multiple times")
            print("  2. Points too clustered - not enough spatial distribution")
            print("  3. Not enough points - try to pick at least 8-9")
            print("\nSuggestions:")
            print("  - Run again and skip more of the clustered margin targets")
            print("  - Focus on well-distributed targets across the image")
            print(f"  - Aim for {min_gcps + 3} or more points if possible")
            return False

    print(f"✓ Calibration complete - RMS: {rms:.4f} pixels")

    # Adjust RMS threshold based on mode
    rms_threshold = 10.0 if mode == 'pose-only' else 5.0

    if rms > rms_threshold:
        print(f"⚠ WARNING: RMS error is high ({rms:.4f} > {rms_threshold} pixels)")
        print("   This may indicate:")
        print("   - Inaccurate point picking (off by a few pixels)")
        if mode == 'full':
            print("   - Camera parameters changed significantly")
        else:
            print("   - Camera moved more than expected (consider full calibration)")
        print("   - Poor point distribution")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Try picking points more carefully.")
            return False
    
    # Create orthorectification parameters
    width, height, geotransform = create_orthorectification_params(
        camera_gcps, resolution, padding_meters
    )
    
    # Load DEM
    print("\nLoading DEM...")
    dem_array = load_dem_from_tiff(dem_path, width, height, geotransform)
    
    # Create lookup tables
    print("\nCreating lookup tables...")
    map_x, map_y = create_ortho_lookup_tables_with_dem(
        K, D, rvec, tvec, width, height, geotransform, dem_array
    )
    
    # Orthorectify
    print("\nOrthorectifying test image...")
    img = cv2.imread(str(image_path))
    ortho_img = orthorectify_with_lookup(img, map_x, map_y)
    
    # Save orthorectified test image for QC (optional, can be disabled if not needed)
    if output_path:
        ortho_dir = output_path / 'orthorectified'
        ortho_dir.mkdir(parents=True, exist_ok=True)
        ortho_path = ortho_dir / f"{camera_id}_recalibrated_ortho.tif"
        save_geotiff(ortho_img, geotransform, ortho_path)
        print(f"Saved test orthorectification: {ortho_path}")
    
    # Create date-stamped calibration file
    if date is None:
        # Try to extract date from timestamp folder path (look for YYYYMMDD pattern)
        import re
        image_path_obj = Path(image_path)
        date = None

        # Check parent directory names for date (e.g., frames/20251016_103100/)
        for parent in image_path_obj.parents:
            date_match = re.search(r'(\d{8})', parent.name)
            if date_match:
                date = date_match.group(1)
                print(f"\nExtracted date {date} from path: {parent.name}")
                break

        # Fall back to image filename if not found in path
        if date is None:
            date_match = re.search(r'(\d{8})', image_path_obj.stem)
            if date_match:
                date = date_match.group(1)
                print(f"\nExtracted date {date} from image filename")

        # Final fallback to today's date
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
            print(f"\nNo date found in path or filename, using today's date: {date}")

    print(f"Creating/updating calibration file for {date}...")

    # Find the most recent calibration file to load from
    source_file = find_most_recent_calibration(calibration_file, date)

    # Load ALL calibrations from the most recent source
    calibrations = load_camera_calibrations(source_file)

    # Delete old ortho cache for this camera
    cache_dir = Path(calibration_file).parent.parent / 'orthorectification' / 'ortho_cache'
    delete_old_ortho_cache(camera_id, cache_dir=str(cache_dir))

    # Update with new calibration for this camera (parameters only, no cache)
    calibrations[camera_id] = {
        'K': K,
        'D': D,
        'rvec': rvec,
        'tvec': tvec,
        'rms': rms,
        'image_size': image_size,
        'n_gcps': len(picked_points),
        'geotransform': geotransform,
        'calibration_date': date,
        'recalibrated': True,
        'recalibration_mode': mode,
        'gcps_skipped': len(skipped_gcps),
        'output_width': width,
        'output_height': height,
        'gcp_pixel_coords': [
            {
                'X': p['gcp']['X'],
                'Y': p['gcp']['Y'],
                'Z': p['gcp']['Z'],
                'col': p['col'],
                'row': p['row']
            }
            for p in picked_points
        ]
    }

    # Save ortho cache separately
    cache_data = {
        'map_x': map_x,
        'map_y': map_y,
        'output_width': width,
        'output_height': height
    }
    save_ortho_cache(camera_id, date, cache_data, cache_dir=str(cache_dir))
    print(f"  ✓ Saved ortho cache: {camera_id}_ortho_cache_{date}.pkl")

    # Save ALL calibrations to dated CSV file
    cal_path = Path(calibration_file)
    if cal_path.suffix == '.pkl':
        # Convert legacy path to CSV
        cal_path = cal_path.with_suffix('.csv')

    new_cal_file = cal_path.parent / f"camera_calibrations_{date}.csv"
    save_camera_calibrations(calibrations, new_cal_file)

    print(f"✓ Saved calibration CSV: {new_cal_file}")
    print(f"  Contains {len(calibrations)} camera(s), updated {camera_id} with date={date}")
    
    print("\n" + "="*60)
    print("Recalibration Complete!")
    print("="*60)
    print(f"Camera: {camera_id}")
    print(f"Mode: {mode.upper()}")
    print(f"Date: {date}")
    print(f"RMS error: {rms:.4f} pixels")
    print(f"GCPs used: {len(picked_points)}")
    print(f"GCPs skipped: {len(skipped_gcps)}")
    print(f"\nOutputs:")
    print(f"  - Updated calibration: {new_cal_file}")
    if camera_id in locals() and 'backup_file' in locals():
        print(f"  - Backup: {backup_file}")
    if output_path:
        print(f"  - Test ortho: {ortho_path}")
        print(f"  - GCP file: {new_gcp_file}")
    print(f"\nNext steps:")
    if output_path:
        print(f"  1. Load {ortho_path} in QGIS and verify alignment with GCPs")
        print(f"  2. If good, use the updated calibration file in your pipeline")
    else:
        print(f"  1. Use the updated calibration file in your pipeline:")
    print(f"     python undistort_and_orthorectify.py process -i images/ -o ortho/ -cal {new_cal_file}")
    if output_path:
        print(f"  3. If alignment is bad, delete {new_cal_file} and try again")
    else:
        print(f"  2. If results are bad, delete {new_cal_file} and try again")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Interactively recalibrate a single camera that has shifted',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python recalibrate_camera.py -i new_images/cam01_shifted.tif -g GCP_merged.csv \\
      -c ch01 -d dem.tif -cal output/camera_calibrations.pkl

Instructions during picking:
  - Click on each target shown
  - Press 'n' to skip targets not visible in the new image
  - Press 'u' to undo last action
  - Press 'q' when done
        """
    )
    
    parser.add_argument('-i', '--image', required=True,
                       help='New image from shifted camera')
    parser.add_argument('-g', '--gcp-file', required=True,
                       help='Original GCP CSV file')
    parser.add_argument('-c', '--camera-id', required=True,
                       help='Camera identifier (e.g., ch01)')
    parser.add_argument('-d', '--dem', required=True,
                       help='DEM TIFF file')
    parser.add_argument('-cal', '--calibration', required=True,
                       help='Existing calibration file to update')
    parser.add_argument('-o', '--output', default=None,
                       help='Output directory for test orthorectification (optional, default: None - skips saving test image)')
    parser.add_argument('-r', '--resolution', type=float, default=0.005,
                       help='Resolution in m/pixel (default: 0.005)')
    parser.add_argument('-p', '--padding', type=float, default=0.5,
                       help='Padding in meters (default: 0.5)')
    parser.add_argument('--min-gcps', type=int, default=4,
                       help='Minimum GCPs required (default: 4)')
    parser.add_argument('--date', type=str, default=None,
                       help='Date for calibration file (YYYYMMDD, default: today)')
    parser.add_argument('--mode', type=str, choices=['pose-only', 'full'], default='pose-only',
                       help='Calibration mode: pose-only (R,t only, faster) or full (K,D,R,t, slower). Default: pose-only')
    parser.add_argument('--min-inlier-ratio', type=float, default=0.7,
                       help='Minimum inlier ratio for pose-only mode (default: 0.7, i.e., 70%%)')
    parser.add_argument('--max-rms', type=float, default=10.0,
                       help='Maximum RMS error for pose-only mode in pixels (default: 10.0)')

    args = parser.parse_args()

    success = recalibrate_single_camera(
        image_path=args.image,
        gcp_file=args.gcp_file,
        camera_id=args.camera_id,
        dem_path=args.dem,
        calibration_file=args.calibration,
        output_dir=args.output,
        resolution=args.resolution,
        padding_meters=args.padding,
        min_gcps=args.min_gcps,
        date=args.date,
        mode=args.mode,
        min_inlier_ratio=args.min_inlier_ratio,
        max_rms=args.max_rms
    )

    import sys
    sys.exit(0 if success else 1)