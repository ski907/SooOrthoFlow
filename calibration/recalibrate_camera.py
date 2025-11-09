import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import argparse

def load_gcp_targets(gcp_file, camera_id):
    """
    Load GCP world coordinates for a specific camera
    Returns list of GCP names and their X,Y,Z coordinates
    """
    gcp_data = pd.read_csv(gcp_file)
    camera_gcps = gcp_data[gcp_data['image_name'].str.contains(camera_id)]
    
    # Get unique GCP identifiers and their world coordinates
    gcps = []
    for idx, row in camera_gcps.iterrows():
        gcps.append({
            'name': f"GCP_{idx}",
            'X': row['X'],
            'Y': row['Y'],
            'Z': row['Z'],
            'original_col': row['col_sample'],
            'original_row': row['row_sample']
        })
    
    return gcps


class InteractiveGCPPicker:
    """
    Interactive tool to pick GCP locations in an image
    """
    def __init__(self, image_path, gcps):
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
            "Press 'q' to quit (saves progress)\n"
            "Close window to cancel"
        )
        self.text_display = self.fig.text(0.02, 0.98, instruction_text, 
                                          transform=self.fig.transFigure,
                                          verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Show original location hint if available
        self._show_hint()
        
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


def recalibrate_single_camera(image_path, gcp_file, camera_id, dem_path, 
                              calibration_file, output_dir='output',
                              resolution=0.005, padding_meters=0.5,
                              min_gcps=6):  # Lowered to 6 (absolute minimum for fisheye)
    """
    Interactively recalibrate a single camera
    """
    from undistort_and_orthorectify import (calibrate_fisheye_camera, create_orthorectification_params,
                          load_dem_from_tiff, create_ortho_lookup_tables_with_dem,
                          orthorectify_with_lookup, save_with_worldfile, undistort_fisheye)
    
    print("="*60)
    print(f"Interactive Recalibration for {camera_id}")
    print("="*60)
    
    # Load GCPs
    print(f"\nLoading GCP targets from {gcp_file}...")
    gcps = load_gcp_targets(gcp_file, camera_id)
    print(f"Found {len(gcps)} GCP targets")
    print(f"\nFor good calibration with limited targets:")
    print(f"  - Pick at least {min_gcps} points (ideally 8-10)")
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
        return
    
    if len(picked_points) < min_gcps:
        print(f"\n✗ Error: Need at least {min_gcps} GCPs for fisheye calibration")
        print(f"  Only have {len(picked_points)} points")
        print("\nAborting. Please run again and pick more points.")
        return
    
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
            return
    
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
            return
    
    if len(skipped_gcps) > 0:
        print(f"\nSkipped {len(skipped_gcps)} GCPs (clustered margin targets, etc)")
    
    # Create new GCP dataframe
    print(f"\nCreating updated GCP data with {len(picked_points)} points...")
    new_gcp_data = pd.DataFrame([{
        'image_name': f"{camera_id}_new",
        'X': p['gcp']['X'],
        'Y': p['gcp']['Y'],
        'Z': p['gcp']['Z'],
        'col_sample': p['col'],
        'row_sample': p['row']
    } for p in picked_points])
    
    # Save new GCP file for reference
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    new_gcp_file = output_path / f'GCP_{camera_id}_recalibrated.csv'
    new_gcp_data.to_csv(new_gcp_file, index=False)
    print(f"Saved new GCP file: {new_gcp_file}")
    
    # Perform calibration
    print(f"\nRecalibrating {camera_id}...")
    
    # Add camera_id column
    new_gcp_data['camera_id'] = camera_id
    
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
        return
    
    print(f"✓ Calibration complete - RMS: {rms:.4f} pixels")
    
    if rms > 5.0:
        print(f"⚠ WARNING: RMS error is high ({rms:.4f} > 5.0 pixels)")
        print("   This may indicate:")
        print("   - Inaccurate point picking (off by a few pixels)")
        print("   - Camera parameters changed significantly")
        print("   - Poor point distribution")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Try picking points more carefully.")
            return
    
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
    
    # Save outputs
    ortho_dir = output_path / 'orthorectified'
    ortho_dir.mkdir(exist_ok=True)
    ortho_path = ortho_dir / f"{camera_id}_recalibrated_ortho.tif"
    save_with_worldfile(ortho_img, geotransform, ortho_path)
    
    # Save undistorted for QC
    undistorted_dir = output_path / 'undistorted'
    undistorted_dir.mkdir(exist_ok=True)
    undistorted = undistort_fisheye(img, K, D)
    undist_path = undistorted_dir / f"{camera_id}_recalibrated_undistorted.tif"
    cv2.imwrite(str(undist_path), undistorted)
    print(f"Saved undistorted: {undist_path}")
    
    # Update calibration file
    print(f"\nUpdating calibration file...")
    with open(calibration_file, 'rb') as f:
        calibrations = pickle.load(f)
    
    # Backup old calibration
    backup_file = Path(str(calibration_file).replace('.pkl', f'_backup_{camera_id}.pkl'))
    with open(backup_file, 'wb') as f:
        pickle.dump({camera_id: calibrations[camera_id]}, f)
    print(f"Backed up old calibration: {backup_file}")
    
    # Update with new calibration
    calibrations[camera_id] = {
        'K': K,
        'D': D,
        'rvec': rvec,
        'tvec': tvec,
        'rms': rms,
        'image_size': image_size,
        'n_gcps': len(picked_points),
        'geotransform': geotransform,
        'dem_array': dem_array,
        'map_x': map_x,
        'map_y': map_y,
        'output_width': width,
        'output_height': height,
        'recalibrated': True,
        'gcps_skipped': len(skipped_gcps)
    }
    
    # Save updated calibration
    with open(calibration_file, 'wb') as f:
        pickle.dump(calibrations, f)
    
    print(f"✓ Updated calibration file: {calibration_file}")
    
    print("\n" + "="*60)
    print("Recalibration Complete!")
    print("="*60)
    print(f"Camera: {camera_id}")
    print(f"RMS error: {rms:.4f} pixels")
    print(f"GCPs used: {len(picked_points)}")
    print(f"GCPs skipped: {len(skipped_gcps)}")
    print(f"\nOutputs:")
    print(f"  - Updated calibration: {calibration_file}")
    print(f"  - Backup: {backup_file}")
    print(f"  - Test ortho: {ortho_path}")
    print(f"  - GCP file: {new_gcp_file}")
    print(f"\nNext steps:")
    print(f"  1. Load {ortho_path} in QGIS and verify alignment with GCPs")
    print(f"  2. If good, process remaining images with:")
    print(f"     python undistort_and_orthorectify.py process -i new_images/ -o new_ortho/")
    print(f"  3. If bad, restore backup and try again")


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
    parser.add_argument('-o', '--output', default='recalibration_output',
                       help='Output directory (default: recalibration_output)')
    parser.add_argument('-r', '--resolution', type=float, default=0.005,
                       help='Resolution in m/pixel (default: 0.005)')
    parser.add_argument('-p', '--padding', type=float, default=0.5,
                       help='Padding in meters (default: 0.5)')
    parser.add_argument('--min-gcps', type=int, default=4,
                       help='Minimum GCPs required (default: 4)')
    
    args = parser.parse_args()
    
    recalibrate_single_camera(
        image_path=args.image,
        gcp_file=args.gcp_file,
        camera_id=args.camera_id,
        dem_path=args.dem,
        calibration_file=args.calibration,
        output_dir=args.output,
        resolution=args.resolution,
        padding_meters=args.padding,
        min_gcps=args.min_gcps
    )