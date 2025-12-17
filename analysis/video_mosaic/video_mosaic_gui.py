#!/usr/bin/env python3
"""
Video Mosaic GUI

Graphical interface for creating mosaicked videos from multi-camera footage.
Can run standalone or be integrated into pipeline_gui.py as a popup.

Author: SooOrthoFlow Team
Version: 0.1.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
from pathlib import Path
import subprocess
import threading
from datetime import datetime
from typing import Optional
import sys

# Add parent directories to path for imports
script_dir = Path(__file__).parent
root_dir = script_dir.parent.parent
sys.path.insert(0, str(root_dir))


class VideoMosaicGUI:
    """GUI for video mosaic configuration and execution."""

    def __init__(self, root: Optional[tk.Toplevel] = None, standalone: bool = True):
        """
        Initialize video mosaic GUI.

        Parameters:
            root: Parent window (Toplevel for popup, or None for standalone)
            standalone: If True, create own Tk root window
        """
        if standalone and root is None:
            self.root = tk.Tk()
            self.root.title("Video Mosaic Generator")
        elif root is not None:
            self.root = root
            self.root.title("Video Mosaic Generator")
        else:
            raise ValueError("Must provide root or set standalone=True")

        self.root.geometry("900x800")

        # Use the globally defined paths
        self.script_dir = script_dir
        self.root_dir = root_dir

        # Load calibration to get available cameras
        self.calibration_file = self.root_dir / "calibration" / "camera_calibrations_20251203.csv"
        self.available_cameras = self._load_available_cameras()

        # Camera selection state
        self.camera_vars = {}  # camera_id -> BooleanVar

        # Configuration variables
        self.video_dir = tk.StringVar(value="test_videos/12-12-25/Test 13")
        self.start_time = tk.StringVar(value="2025-12-12 12:44:00")
        self.end_time = tk.StringVar(value="2025-12-12 12:47:00")
        self.interval_seconds = tk.DoubleVar(value=5.0)

        self.ortho_resolution = tk.DoubleVar(value=0.01)
        self.mosaic_method = tk.StringVar(value="zone_map")
        self.zone_map_shapefile = tk.StringVar(value="orthorectification/camera_zone_map/camera_zone_map.shp")
        self.rotation_angle = tk.DoubleVar(value=78.1)
        self.clip_shapefile = tk.StringVar(value="analysis/analysis_shapefiles/general_analysis_limits.shp")

        self.output_dir = tk.StringVar(value="output_data/video_mosaic")
        self.video_filename = tk.StringVar(value="mosaic_video.mp4")
        self.video_fps = tk.IntVar(value=5)
        self.video_codec = tk.StringVar(value="mp4v")

        # Processing state
        self.processing = False
        self.process = None

        # Build GUI
        self._create_widgets()

    def _load_available_cameras(self):
        """Load list of available cameras from calibration file."""
        try:
            from calibration.calibration_io import load_camera_calibrations
            calibrations = load_camera_calibrations(str(self.calibration_file))
            return sorted(calibrations.keys())
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not load cameras: {e}")
            return []

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with scrollbar
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Canvas + Scrollbar for scrolling
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === CAMERA SELECTION ===
        camera_frame = ttk.LabelFrame(scrollable_frame, text="Camera Selection", padding="10")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)

        # Select all/none buttons
        btn_frame = ttk.Frame(camera_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(btn_frame, text="Select All", command=self._select_all_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select None", command=self._select_none_cameras).pack(side=tk.LEFT, padx=5)

        # Camera checkboxes (in 3 columns)
        cam_grid = ttk.Frame(camera_frame)
        cam_grid.pack(fill=tk.X)

        for idx, camera_id in enumerate(self.available_cameras):
            var = tk.BooleanVar(value=False)
            self.camera_vars[camera_id] = var

            row = idx // 3
            col = idx % 3
            ttk.Checkbutton(cam_grid, text=camera_id, variable=var).grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=2
            )

        # === VIDEO INPUT ===
        input_frame = ttk.LabelFrame(scrollable_frame, text="Video Input", padding="10")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(input_frame, text="Video Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(input_frame, textvariable=self.video_dir, width=50).grid(row=0, column=1, pady=2)
        ttk.Button(input_frame, text="Browse", command=self._browse_video_dir).grid(row=0, column=2, padx=5)

        ttk.Label(input_frame, text="Start Time (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(input_frame, textvariable=self.start_time, width=50).grid(row=1, column=1, pady=2)

        ttk.Label(input_frame, text="End Time (YYYY-MM-DD HH:MM:SS):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(input_frame, textvariable=self.end_time, width=50).grid(row=2, column=1, pady=2)

        ttk.Label(input_frame, text="Frame Interval (seconds):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(input_frame, textvariable=self.interval_seconds, width=50).grid(row=3, column=1, pady=2)

        # === PROCESSING SETTINGS ===
        proc_frame = ttk.LabelFrame(scrollable_frame, text="Processing Settings", padding="10")
        proc_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(proc_frame, text="Ortho Resolution (m/pixel):").grid(row=0, column=0, sticky=tk.W, pady=2)
        res_combo = ttk.Combobox(proc_frame, textvariable=self.ortho_resolution, width=47,
                                 values=["0.0025", "0.005", "0.01"])
        res_combo.grid(row=0, column=1, pady=2)

        ttk.Label(proc_frame, text="Mosaic Method:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(proc_frame, textvariable=self.mosaic_method, width=47,
                     values=["zone_map", "center"]).grid(row=1, column=1, pady=2)

        ttk.Label(proc_frame, text="Zone Map Shapefile:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(proc_frame, textvariable=self.zone_map_shapefile, width=50).grid(row=2, column=1, pady=2)

        ttk.Label(proc_frame, text="Rotation Angle (degrees):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(proc_frame, textvariable=self.rotation_angle, width=50).grid(row=3, column=1, pady=2)

        ttk.Label(proc_frame, text="Clip Shapefile (optional):").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(proc_frame, textvariable=self.clip_shapefile, width=50).grid(row=4, column=1, pady=2)

        # === VIDEO OUTPUT ===
        output_frame = ttk.LabelFrame(scrollable_frame, text="Video Output", padding="10")
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, pady=2)

        ttk.Label(output_frame, text="Video Filename:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.video_filename, width=50).grid(row=1, column=1, pady=2)

        ttk.Label(output_frame, text="Video FPS:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.video_fps, width=50).grid(row=2, column=1, pady=2)

        ttk.Label(output_frame, text="Video Codec:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(output_frame, textvariable=self.video_codec, width=47,
                     values=["mp4v", "avc1", "XVID"]).grid(row=3, column=1, pady=2)

        # === CONTROL BUTTONS ===
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=10)

        self.run_btn = ttk.Button(control_frame, text="Generate Video", command=self._run_processing)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Save Config", command=self._save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Load Config", command=self._load_config).pack(side=tk.LEFT, padx=5)

        # === STATUS/LOG ===
        log_frame = ttk.LabelFrame(scrollable_frame, text="Status", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _select_all_cameras(self):
        """Select all cameras."""
        for var in self.camera_vars.values():
            var.set(True)

    def _select_none_cameras(self):
        """Deselect all cameras."""
        for var in self.camera_vars.values():
            var.set(False)

    def _browse_video_dir(self):
        """Browse for video directory."""
        directory = filedialog.askdirectory(initialdir=self.root_dir)
        if directory:
            # Make relative to root_dir
            try:
                rel_path = Path(directory).relative_to(self.root_dir)
                self.video_dir.set(str(rel_path))
            except ValueError:
                self.video_dir.set(directory)

    def _get_selected_cameras(self):
        """Get list of selected camera IDs."""
        return [cam_id for cam_id, var in self.camera_vars.items() if var.get()]

    def _create_config(self):
        """Create configuration dictionary from GUI inputs."""
        selected_cameras = self._get_selected_cameras()

        if not selected_cameras:
            raise ValueError("No cameras selected")

        config = {
            "input": {
                "video_dir": self.video_dir.get(),
                "camera_selection": {
                    "mode": "list",
                    "cameras": selected_cameras
                },
                "error_handling": {
                    "skip_missing_calibration": False,
                    "skip_missing_cache": False,
                    "skip_missing_videos": True,
                    "min_cameras_required": 1
                },
                "start_time": self.start_time.get(),
                "end_time": self.end_time.get(),
                "interval_seconds": self.interval_seconds.get()
            },
            "paths": {
                "calibration_file": str(self.calibration_file.relative_to(self.root_dir)),
                "dsm_file": "inputs/TLS_DTM_cropped_filled_utmNAD8319N.tif"
            },
            "processing": {
                "ortho_resolution": self.ortho_resolution.get(),
                "mosaic_method": self.mosaic_method.get(),
                "zone_map_shapefile": self.zone_map_shapefile.get(),
                "rotation_angle_deg": self.rotation_angle.get(),
                "clip_shapefile": self.clip_shapefile.get() if self.clip_shapefile.get() else None
            },
            "output": {
                "output_dir": self.output_dir.get(),
                "video_filename": self.video_filename.get(),
                "video_fps": self.video_fps.get(),
                "video_codec": self.video_codec.get()
            },
            "camera_time_offsets": {
                "NVR2": 0.0
            }
        }

        return config

    def _save_config(self):
        """Save current configuration to JSON file."""
        try:
            config = self._create_config()
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=self.script_dir
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                self._log(f"Configuration saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def _load_config(self):
        """Load configuration from JSON file."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.script_dir
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)

                # Update GUI from config
                self.video_dir.set(config['input'].get('video_dir', ''))
                self.start_time.set(config['input'].get('start_time', ''))
                self.end_time.set(config['input'].get('end_time', ''))
                self.interval_seconds.set(config['input'].get('interval_seconds', 5.0))

                self.ortho_resolution.set(config['processing'].get('ortho_resolution', 0.01))
                self.mosaic_method.set(config['processing'].get('mosaic_method', 'zone_map'))
                self.zone_map_shapefile.set(config['processing'].get('zone_map_shapefile', ''))
                self.rotation_angle.set(config['processing'].get('rotation_angle_deg', 0.0))
                self.clip_shapefile.set(config['processing'].get('clip_shapefile', ''))

                self.output_dir.set(config['output'].get('output_dir', ''))
                self.video_filename.set(config['output'].get('video_filename', ''))
                self.video_fps.set(config['output'].get('video_fps', 5))
                self.video_codec.set(config['output'].get('video_codec', 'mp4v'))

                # Update camera selection
                camera_selection = config['input'].get('camera_selection', {})
                if camera_selection.get('mode') == 'all':
                    self._select_all_cameras()
                elif camera_selection.get('mode') == 'list':
                    selected = set(camera_selection.get('cameras', []))
                    for cam_id, var in self.camera_vars.items():
                        var.set(cam_id in selected)

                self._log(f"Configuration loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {e}")

    def _log(self, message: str):
        """Add message to log window."""
        self.log_text.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _run_processing(self):
        """Run video mosaic processing in background thread."""
        if self.processing:
            messagebox.showwarning("Warning", "Processing already running")
            return

        try:
            # Validate and create config
            config = self._create_config()

            # Save temp config file
            temp_config_path = self.script_dir / "temp_mosaic_config.json"
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Update UI
            self.processing = True
            self.run_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self._log("Starting video mosaic processing...")
            self._log(f"Selected cameras: {len(self._get_selected_cameras())}")

            # Run in background thread
            thread = threading.Thread(target=self._process_worker, args=(temp_config_path,))
            thread.daemon = True
            thread.start()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.processing = False
            self.run_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

    def _process_worker(self, config_path: Path):
        """Worker thread for processing."""
        try:
            # Run video mosaic processor
            cmd = [
                "python",
                str(self.script_dir / "run_video_mosaic.py"),
                "--config",
                str(config_path)
            ]

            self.process = subprocess.Popen(
                cmd,
                cwd=self.root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to log
            for line in self.process.stdout:
                self.root.after(0, self._log, line.rstrip())

            self.process.wait()

            if self.process.returncode == 0:
                self.root.after(0, self._log, "Processing completed successfully!")
                self.root.after(0, messagebox.showinfo, "Success", "Video mosaic generated successfully!")
            else:
                self.root.after(0, self._log, f"Processing failed with code {self.process.returncode}")
                self.root.after(0, messagebox.showerror, "Error", "Processing failed. Check log for details.")

        except Exception as e:
            self.root.after(0, self._log, f"ERROR: {e}")
            self.root.after(0, messagebox.showerror, "Error", str(e))

        finally:
            self.process = None
            self.processing = False
            self.root.after(0, self.run_btn.configure, {'state': tk.NORMAL})
            self.root.after(0, self.stop_btn.configure, {'state': tk.DISABLED})

    def _stop_processing(self):
        """Stop running process."""
        if self.process:
            self.process.terminate()
            self._log("Processing stopped by user")
            self.processing = False
            self.run_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

    def run(self):
        """Run the GUI main loop (for standalone mode)."""
        self.root.mainloop()


def main():
    """Main entry point for standalone execution."""
    gui = VideoMosaicGUI(standalone=True)
    gui.run()


if __name__ == "__main__":
    main()
