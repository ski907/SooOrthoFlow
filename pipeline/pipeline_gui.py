#!/usr/bin/env python3
"""
Pipeline Control GUI

Graphical interface for configuring and running the soo locks model image processing pipeline.
Allows easy editing of master_control.json and launching pipeline operations.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
from pathlib import Path
import subprocess
import threading
import sys
import re
from datetime import datetime


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Soo Locks Model Image Processing Pipeline")
        self.root.geometry("800x700")

        # Config file path
        self.script_dir = Path(__file__).parent
        self.root_dir = self.script_dir.parent
        self.config_path = self.root_dir / 'master_control.json'

        # Variables for form fields
        self.video_folder = tk.StringVar()
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()
        self.interval = tk.StringVar(value="1min")
        self.time_mode = tk.StringVar(value="interval")  # "interval" or "endpoints"

        self.output_base = tk.StringVar(value="output_data")
        self.calibration_file = tk.StringVar(value="calibration/camera_calibrations.pkl")
        self.gcp_file = tk.StringVar(value="inputs/GCP_merged.csv")
        self.dsm_file = tk.StringVar(value="inputs/lidar_DSM_filled_cropped.tif")

        self.output_format = tk.StringVar(value="tiff")
        self.recursive = tk.BooleanVar(value=True)
        self.filename_pattern = tk.StringVar(value="CAMERA_DATETIME_DATETIME")
        self.mosaic_method = tk.StringVar(value="center")
        self.ortho_resolution = tk.StringVar(value="0.005")
        self.ortho_padding = tk.StringVar(value="0.5")

        # Light detection settings
        self.run_light_detection = tk.BooleanVar(value=True)
        self.light_detection_mask = tk.StringVar(value="analysis/ship_detection/area_to_review_ref.tif")

        # Coordinate transformation settings
        self.apply_world_transform = tk.BooleanVar(value=False)
        self.world_file_path = tk.StringVar(value="orthorectification/model_to_world.wld")

        # Pipeline process reference
        self.pipeline_process = None

        # Progress tracking
        self.total_timestamps = 0
        self.completed_orthos = 0
        self.completed_mosaics = 0
        self.current_phase = None  # 'extraction', 'ortho', 'mosaic'

        # Build UI
        self.create_widgets()

        # Try to load existing config
        if self.config_path.exists():
            self.load_config()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Create canvas and scrollbar
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="5")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Grid layout for canvas and scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Main frame is now the scrollable frame
        main_frame = scrollable_frame
        main_frame.columnconfigure(0, weight=1)

        # Bind mousewheel to scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Title
        title_label = ttk.Label(main_frame, text="Soo Locks Model Image Processing Pipeline",
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))

        # Config file management
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="5")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Config File:").grid(row=0, column=0, sticky=tk.W)
        config_label = ttk.Label(config_frame, textvariable=tk.StringVar(value=str(self.config_path)),
                                foreground="blue")
        config_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        ttk.Button(config_frame, text="Load", command=self.load_config).grid(row=0, column=2, padx=5)
        ttk.Button(config_frame, text="Save", command=self.save_config).grid(row=0, column=3, padx=5)

        # Input Parameters Section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="5")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        input_frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(input_frame, text="Video Folder:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.video_folder, width=80).grid(row=row, column=1,
                                                                               sticky=(tk.W, tk.E), padx=5)
        ttk.Button(input_frame, text="Browse...",
                  command=lambda: self.browse_directory(self.video_folder)).grid(row=row, column=2)

        row += 1
        ttk.Label(input_frame, text="Start Time:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(input_frame, textvariable=self.start_time, width=80).grid(row=row, column=1,
                                                                             sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        ttk.Button(input_frame, text="Auto-Detect Times",
                  command=self.auto_detect_video_times).grid(row=row, column=2, pady=(5, 0))

        row += 1
        ttk.Label(input_frame, text="End Time:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.end_time, width=80).grid(row=row, column=1,
                                                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Label(input_frame, text="Format: YYYY-MM-DD HH:MM:SS",
                 foreground="gray").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(input_frame, text="Time Mode:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        time_mode_frame = ttk.Frame(input_frame)
        time_mode_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(5, 0))
        ttk.Radiobutton(time_mode_frame, text="Use Interval", variable=self.time_mode,
                       value="interval", command=self.on_time_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(time_mode_frame, text="Start/End Only (2 frames)", variable=self.time_mode,
                       value="endpoints", command=self.on_time_mode_change).pack(side=tk.LEFT, padx=(20, 0))

        row += 1
        ttk.Label(input_frame, text="Interval:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        self.interval_frame = ttk.Frame(input_frame)
        self.interval_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        self.interval_entry = ttk.Entry(self.interval_frame, textvariable=self.interval, width=20)
        self.interval_entry.pack(side=tk.LEFT)
        self.interval_label = ttk.Label(self.interval_frame, text="Examples: 30s, 1min, 5min",
                 foreground="gray")
        self.interval_label.pack(side=tk.LEFT, padx=(10, 0))

        # Paths Section
        paths_frame = ttk.LabelFrame(main_frame, text="File Paths", padding="5")
        paths_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        paths_frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(paths_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(paths_frame, textvariable=self.output_base, width=70).grid(row=row, column=1,
                                                                              sticky=(tk.W, tk.E), padx=5)
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_directory(self.output_base)).grid(row=row, column=2)

        row += 1
        ttk.Label(paths_frame, text="Calibration File:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(paths_frame, textvariable=self.calibration_file, width=70).grid(row=row, column=1,
                                                                                   sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_file(self.calibration_file,
                                                   [("CSV Files", "*.csv"), ("Pickle Files", "*.pkl"), ("All Files", "*.*")])).grid(row=row, column=2, pady=(5, 0))

        row += 1
        ttk.Label(paths_frame, text="GCP File:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(paths_frame, textvariable=self.gcp_file, width=70).grid(row=row, column=1,
                                                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_file(self.gcp_file,
                                                   [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=row, column=2)

        row += 1
        ttk.Label(paths_frame, text="DSM File:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(paths_frame, textvariable=self.dsm_file, width=70).grid(row=row, column=1,
                                                                           sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_file(self.dsm_file,
                                                   [("GeoTIFF Files", "*.tif;*.tiff"), ("All Files", "*.*")])).grid(row=row, column=2, pady=(5, 0))

        # # Processing Options Section
        # proc_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="5")
        # proc_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        # proc_frame.columnconfigure(1, weight=1)

        # row = 0
        # ttk.Label(proc_frame, text="Output Format:").grid(row=row, column=0, sticky=tk.W)
        # format_combo = ttk.Combobox(proc_frame, textvariable=self.output_format,
        #                             values=["tiff", "png", "jpg"], state="readonly", width=15)
        # format_combo.grid(row=row, column=1, sticky=tk.W, padx=5)

        # ttk.Label(proc_frame, text="Mosaic Method:").grid(row=row, column=2, sticky=tk.W, padx=(20, 0))
        # mosaic_combo = ttk.Combobox(proc_frame, textvariable=self.mosaic_method,
        #                             values=["center", "average", "max", "min"], state="readonly", width=15)
        # mosaic_combo.grid(row=row, column=3, sticky=tk.W, padx=5)

        # row += 1
        # ttk.Checkbutton(proc_frame, text="Recursive video search",
        #                variable=self.recursive).grid(row=row, column=0, columnspan=2,
        #                                              sticky=tk.W, pady=(5, 0))

        # row += 1
        # ttk.Label(proc_frame, text="Filename Pattern:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        # ttk.Entry(proc_frame, textvariable=self.filename_pattern, width=40).grid(row=row, column=1,
        #                                                                           columnspan=3, sticky=(tk.W, tk.E),
        #                                                                           padx=5, pady=(5, 0))

        # row += 1
        # ttk.Label(proc_frame, text="Ortho Resolution (m/px):").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        # ttk.Entry(proc_frame, textvariable=self.ortho_resolution, width=15).grid(row=row, column=1,
        #                                                                           sticky=tk.W, padx=5, pady=(5, 0))

        # ttk.Label(proc_frame, text="Ortho Padding (m):").grid(row=row, column=2, sticky=tk.W,
        #                                                        padx=(20, 0), pady=(5, 0))
        # ttk.Entry(proc_frame, textvariable=self.ortho_padding, width=15).grid(row=row, column=3,
        #                                                                        sticky=tk.W, padx=5, pady=(5, 0))

        # Post-Processing Options Section
        postproc_frame = ttk.LabelFrame(main_frame, text="Post-Processing Options", padding="5")
        postproc_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        postproc_frame.columnconfigure(1, weight=1)

        # row = 0
        # ttk.Checkbutton(postproc_frame, text="Detect boat lights in mosaics",
        #                variable=self.run_light_detection).grid(row=row, column=0, columnspan=2,
        #                                                        sticky=tk.W)

        # row += 1
        # ttk.Label(postproc_frame, text="Light Mask File (optional):").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        # ttk.Entry(postproc_frame, textvariable=self.light_detection_mask, width=70).grid(row=row, column=1,
        #                                                                                   sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        # ttk.Button(postproc_frame, text="Browse...",
        #           command=lambda: self.browse_file(self.light_detection_mask,
        #                                            [("GeoTIFF Files", "*.tif;*.tiff"), ("All Files", "*.*")])).grid(row=row, column=2, pady=(5, 0))

        row = 0
        ttk.Checkbutton(postproc_frame, text="Transform mosaics to world coordinates",
                       variable=self.apply_world_transform).grid(row=row, column=0, columnspan=2,
                                                                 sticky=tk.W, pady=(5, 0))

        row += 1
        ttk.Label(postproc_frame, text="World File:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(postproc_frame, textvariable=self.world_file_path, width=70).grid(row=row, column=1,
                                                                                     sticky=(tk.W, tk.E), padx=5)
        ttk.Button(postproc_frame, text="Browse...",
                  command=lambda: self.browse_file(self.world_file_path,
                                                   [("World Files", "*.wld"), ("All Files", "*.*")])).grid(row=row, column=2)

        # # Calibration Settings Section
        # calib_frame = ttk.LabelFrame(main_frame, text="Calibration Settings", padding="5")
        # calib_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        # self.calib_mode = tk.StringVar(value='pose-only')
        # ttk.Label(calib_frame, text="Recalibration Mode:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        # ttk.Radiobutton(calib_frame, text="Pose-only (faster, for small camera shifts)",
        #                variable=self.calib_mode, value='pose-only').grid(row=0, column=1, sticky=tk.W)
        # ttk.Radiobutton(calib_frame, text="Full (complete recalibration)",
        #                variable=self.calib_mode, value='full').grid(row=1, column=1, sticky=tk.W)

        # # Initial Calibration Section
        # initial_calib_frame = ttk.LabelFrame(main_frame, text="Initial Calibration (from GCPs)", padding="5")
        # initial_calib_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 5))
        # initial_calib_frame.columnconfigure(1, weight=1)

        # # GCP file for calibration
        # self.calib_gcp_file = tk.StringVar(value="inputs/GCP_merged.csv")
        # row = 0
        # ttk.Label(initial_calib_frame, text="GCP File:").grid(row=row, column=0, sticky=tk.W, pady=2)
        # ttk.Entry(initial_calib_frame, textvariable=self.calib_gcp_file, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(5, 2))
        # ttk.Button(initial_calib_frame, text="Browse",
        #           command=lambda: self.browse_file(self.calib_gcp_file,
        #                                            [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=row, column=2)

        # # Image folder with concurrent images
        # self.calib_image_folder = tk.StringVar(value="inputs/IR_concurrent_with_lidar")
        # row += 1
        # ttk.Label(initial_calib_frame, text="Image Folder:").grid(row=row, column=0, sticky=tk.W, pady=2)
        # ttk.Entry(initial_calib_frame, textvariable=self.calib_image_folder, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(5, 2))
        # ttk.Button(initial_calib_frame, text="Browse",
        #           command=lambda: self.browse_folder(self.calib_image_folder)).grid(row=row, column=2)

        # # DEM file
        # self.calib_dem_file = tk.StringVar(value="inputs/lidar_DSM_filled_cropped.tif")
        # row += 1
        # ttk.Label(initial_calib_frame, text="DEM File:").grid(row=row, column=0, sticky=tk.W, pady=2)
        # ttk.Entry(initial_calib_frame, textvariable=self.calib_dem_file, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(5, 2))
        # ttk.Button(initial_calib_frame, text="Browse",
        #           command=lambda: self.browse_file(self.calib_dem_file,
        #                                            [("GeoTIFF Files", "*.tif *.tiff"), ("All Files", "*.*")])).grid(row=row, column=2)

        # # Output resolution
        # self.calib_resolution = tk.StringVar(value="0.0025")
        # row += 1
        # ttk.Label(initial_calib_frame, text="Resolution (m/pixel):").grid(row=row, column=0, sticky=tk.W, pady=2)
        # ttk.Entry(initial_calib_frame, textvariable=self.calib_resolution, width=15).grid(row=row, column=1, sticky=tk.W, padx=(5, 2))
        # ttk.Label(initial_calib_frame, text="(e.g., 0.0025 = 2.5mm/pixel)", foreground="gray").grid(row=row, column=1, sticky=tk.E)

        # # Output directory
        # self.calib_output_dir = tk.StringVar(value="output_calibration")
        # row += 1
        # ttk.Label(initial_calib_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=2)
        # ttk.Entry(initial_calib_frame, textvariable=self.calib_output_dir, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(5, 2))
        # ttk.Button(initial_calib_frame, text="Browse",
        #           command=lambda: self.browse_folder(self.calib_output_dir)).grid(row=row, column=2)

        # # Run button
        # row += 1
        # ttk.Button(initial_calib_frame, text="Run Initial Calibration",
        #           command=self.run_initial_calibration,
        #           style='Accent.TButton').grid(row=row, column=0, columnspan=3, pady=(10, 5))

        # ttk.Label(initial_calib_frame, text="Note: This will create a new camera_calibrations_YYYYMMDD.csv file",
        #          foreground="gray", font=('Arial', 8)).grid(row=row+1, column=0, columnspan=3, sticky=tk.W)

        # Progress Section
        progress_frame = ttk.LabelFrame(main_frame, text="Pipeline Progress", padding="5")
        progress_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        progress_frame.columnconfigure(0, weight=1)

        # Current step label
        self.current_step_var = tk.StringVar(value="Ready to run")
        ttk.Label(progress_frame, textvariable=self.current_step_var,
                 font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 2))

        # Overall progress bar
        ttk.Label(progress_frame, text="Overall:").grid(row=1, column=0, sticky=tk.W)
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate', length=600)
        self.overall_progress.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Detailed status text
        self.detail_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.detail_var,
                 foreground="black", font=('Arial', 9)).grid(row=3, column=0, sticky=tk.W)

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=(5, 0))

        self.run_button = ttk.Button(button_frame, text="Run Pipeline",
                                     command=self.run_pipeline, style='Accent.TButton')
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.recalibrate_button = ttk.Button(button_frame, text="Recalibrate Camera",
                                            command=self.recalibrate_camera)
        self.recalibrate_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Close", command=self.root.quit).pack(side=tk.LEFT, padx=5)

    def browse_directory(self, var):
        """Open directory browser"""
        initial_dir = var.get() if var.get() else str(self.root_dir)
        directory = filedialog.askdirectory(initialdir=initial_dir, title="Select Directory")
        if directory:
            # Convert to relative path if possible
            try:
                rel_path = Path(directory).relative_to(self.root_dir)
                var.set(str(rel_path))
            except ValueError:
                # Not relative to root_dir, use absolute path
                var.set(directory)

    def browse_file(self, var, filetypes):
        """Open file browser"""
        initial_file = var.get() if var.get() else ""
        initial_dir = Path(initial_file).parent if initial_file else str(self.root_dir)

        filename = filedialog.askopenfilename(initialdir=initial_dir, title="Select File",
                                             filetypes=filetypes)
        if filename:
            # Convert to relative path if possible
            try:
                rel_path = Path(filename).relative_to(self.root_dir)
                var.set(str(rel_path))
            except ValueError:
                # Not relative to root_dir, use absolute path
                var.set(filename)

    def auto_detect_video_times(self):
        """Scan all video files in the folder and detect earliest/latest times"""
        video_folder = self.video_folder.get()
        if not video_folder:
            messagebox.showwarning("No Folder", "Please select a video folder first")
            return

        video_path = Path(video_folder)
        if not video_path.exists():
            messagebox.showerror("Error", f"Folder does not exist: {video_folder}")
            return

        self.log_console("Scanning video files for timestamps...")

        try:
            import re

            # Find all video files
            video_extensions = ('.avi', '.mp4', '.mov', '.mkv', '.m4v')
            video_files = []
            for ext in video_extensions:
                video_files.extend(video_path.rglob(f'*{ext}'))

            if not video_files:
                messagebox.showwarning("No Videos", "No video files found in the selected folder")
                return

            self.log_console(f"Found {len(video_files)} video files")

            # Extract timestamps from filenames, grouped by camera
            # Key: (NVR folder, channel), Value: list of (start_time, end_time) tuples
            from collections import defaultdict
            camera_times = defaultdict(list)

            # Common timestamp patterns in filenames
            patterns = [
                r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_\s-]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})',  # YYYYMMDD_HHMMSS
                r'(\d{8})[T_-]?(\d{6})',  # YYYYMMDDHHMMSS
            ]

            for video_file in video_files:
                filename = video_file.stem
                parent_folder = video_file.parent.name  # e.g., "NVR1" or "NVR2"

                # Extract channel from filename (e.g., "ch1", "ch2", etc.)
                channel_match = re.search(r'ch(\d+)', filename, re.IGNORECASE)
                if not channel_match:
                    continue
                channel = f"ch{channel_match.group(1)}"

                # Extract all timestamps from filename (start and end times)
                found_times = []
                for pattern in patterns:
                    for match in re.finditer(pattern, filename):
                        try:
                            if len(match.groups()) == 6:
                                y, m, d, h, mins, s = match.groups()
                                dt = datetime(int(y), int(m), int(d), int(h), int(mins), int(s))
                            elif len(match.groups()) == 2:
                                date_part, time_part = match.groups()
                                y, m, d = date_part[:4], date_part[4:6], date_part[6:8]
                                h, mins, s = time_part[:2], time_part[2:4], time_part[4:6]
                                dt = datetime(int(y), int(m), int(d), int(h), int(mins), int(s))
                            else:
                                continue
                            found_times.append(dt)
                        except (ValueError, IndexError):
                            continue

                # Store start and end time for this video file
                if len(found_times) >= 2:
                    # Assume first timestamp is start, last is end
                    camera_key = (parent_folder, channel)
                    camera_times[camera_key].append((min(found_times), max(found_times)))
                elif len(found_times) == 1:
                    # Single timestamp - treat as both start and end
                    camera_key = (parent_folder, channel)
                    camera_times[camera_key].append((found_times[0], found_times[0]))

            if not camera_times:
                messagebox.showwarning("No Timestamps",
                    "Could not extract timestamps from video filenames.\n"
                    "Filenames should contain date/time in format like:\n"
                    "YYYYMMDD_HHMMSS or YYYY-MM-DD-HH-MM-SS")
                return

            # For each camera, find earliest start and latest end across all its files
            camera_ranges = {}
            for camera_key, time_list in camera_times.items():
                starts = [t[0] for t in time_list]
                ends = [t[1] for t in time_list]
                camera_ranges[camera_key] = (min(starts), max(ends))

            self.log_console(f"Found {len(camera_ranges)} cameras:")
            for camera_key, (start, end) in camera_ranges.items():
                nvr, ch = camera_key
                self.log_console(f"  {nvr}/{ch}: {start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')}")

            # Find the latest start time (when all cameras are ready)
            all_starts = [r[0] for r in camera_ranges.values()]
            all_ends = [r[1] for r in camera_ranges.values()]
            earliest = max(all_starts)  # Latest start = when all cameras have begun
            latest = min(all_ends)      # Earliest end = when first camera stops

            # Add 30 seconds to earliest and subtract 30 seconds from latest
            # to avoid first/last frames
            from datetime import timedelta
            earliest_adjusted = earliest + timedelta(seconds=30)
            latest_adjusted = latest - timedelta(seconds=30)

            # Round to nearest 30-second increment
            # Round down for start time
            earliest_seconds = earliest_adjusted.second
            if earliest_seconds % 30 != 0:
                earliest_seconds = (earliest_seconds // 30) * 30
                earliest_adjusted = earliest_adjusted.replace(second=earliest_seconds, microsecond=0)

            # Round up for end time
            latest_seconds = latest_adjusted.second
            if latest_seconds % 30 != 0:
                latest_seconds = ((latest_seconds // 30) + 1) * 30
                if latest_seconds >= 60:
                    latest_adjusted = latest_adjusted + timedelta(minutes=1)
                    latest_seconds = 0
                latest_adjusted = latest_adjusted.replace(second=latest_seconds, microsecond=0)

            # Update GUI fields
            self.start_time.set(earliest_adjusted.strftime("%Y-%m-%d %H:%M:%S"))
            self.end_time.set(latest_adjusted.strftime("%Y-%m-%d %H:%M:%S"))

            self.log_console(f"Detected time range:")
            self.log_console(f"  Raw earliest (latest start): {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_console(f"  Raw latest (earliest end): {latest.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_console(f"  Adjusted start (+30s, rounded): {earliest_adjusted.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_console(f"  Adjusted end (-30s, rounded): {latest_adjusted.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_console(f"  Duration: {latest_adjusted - earliest_adjusted}")

            messagebox.showinfo("Times Detected",
                f"Adjusted Start: {earliest_adjusted.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Adjusted End: {latest_adjusted.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"(+30s from latest start, -30s from earliest end, rounded to 30s intervals)\n"
                f"Videos scanned: {len(video_files)}\n"
                f"Cameras found: {len(camera_ranges)}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Error", f"Failed to detect video times:\n{e}\n\nFull traceback:\n{error_details}")
            self.log_console(f"ERROR: {e}")
            self.log_console(f"TRACEBACK:\n{error_details}")

    def on_time_mode_change(self):
        """Handle time mode radio button changes"""
        mode = self.time_mode.get()
        if mode == "endpoints":
            # Disable interval entry
            self.interval_entry.config(state='disabled')
            self.interval_label.config(foreground='lightgray')
            self.log_console("Time mode: Start/End only (will extract 2 frames)")
        else:
            # Enable interval entry
            self.interval_entry.config(state='normal')
            self.interval_label.config(foreground='gray')
            self.log_console(f"Time mode: Interval ({self.interval.get()})")

    def load_config(self):
        """Load configuration from master_control.json"""
        try:
            if not self.config_path.exists():
                self.log_console(f"Config file not found: {self.config_path}")
                return

            # Read file as text and fix backslashes before JSON parsing
            with open(self.config_path, 'r') as f:
                content = f.read()

            # Replace ALL backslashes with forward slashes
            # Windows paths should use forward slashes in JSON
            content = content.replace('\\', '/')

            config = json.loads(content)

            # Load input parameters
            self.video_folder.set(config.get('video_folder', ''))
            self.start_time.set(config.get('start_time', ''))
            self.end_time.set(config.get('end_time', ''))
            self.interval.set(config.get('interval', '1min'))
            self.time_mode.set(config.get('time_mode', 'interval'))
            self.on_time_mode_change()  # Update UI based on loaded mode

            # Load paths
            paths = config.get('paths', {})
            self.output_base.set(paths.get('output_base', 'output_data'))
            self.calibration_file.set(paths.get('calibration_file', 'calibration/camera_calibrations.pkl'))
            self.gcp_file.set(paths.get('gcp_file', 'inputs/GCP_merged.csv'))
            self.dsm_file.set(paths.get('dsm_file', 'inputs/lidar_DSM_filled_cropped.tif'))

            # Load processing options
            processing = config.get('processing', {})
            self.output_format.set(processing.get('output_format', 'tiff'))
            self.recursive.set(processing.get('recursive', True))
            self.filename_pattern.set(processing.get('filename_pattern', 'CAMERA_DATETIME_DATETIME'))
            self.mosaic_method.set(processing.get('mosaic_method', 'center'))
            self.ortho_resolution.set(str(processing.get('ortho_resolution', 0.005)))
            self.ortho_padding.set(str(processing.get('ortho_padding', 0.5)))
            self.run_light_detection.set(processing.get('run_light_detection', False))
            self.light_detection_mask.set(processing.get('light_detection_mask', ''))
            self.apply_world_transform.set(processing.get('apply_world_transform', False))
            self.world_file_path.set(processing.get('world_file_path', 'orthorectification/model_to_world.wld'))

            self.log_console(f"Loaded configuration from {self.config_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
            self.log_console(f"ERROR: {e}")

    def save_config(self):
        """Save configuration to master_control.json"""
        try:
            # Validate required fields
            if not self.video_folder.get():
                messagebox.showwarning("Validation Error", "Video folder is required")
                return

            if not self.start_time.get() or not self.end_time.get():
                messagebox.showwarning("Validation Error", "Start and end times are required")
                return

            # Load existing config to preserve extra fields (like camera_time_offsets)
            existing_config = {}
            if self.config_path.exists():
                try:
                    with open(self.config_path, 'r') as f:
                        content = f.read().replace('\\', '/')
                        existing_config = json.loads(content)
                except:
                    pass

            # Build config dictionary
            config = {
                "video_folder": self.video_folder.get(),
                "start_time": self.start_time.get(),
                "end_time": self.end_time.get(),
                "interval": self.interval.get(),
                "time_mode": self.time_mode.get(),
                "paths": {
                    "output_base": self.output_base.get(),
                    "calibration_file": self.calibration_file.get(),
                    "gcp_file": self.gcp_file.get(),
                    "dsm_file": self.dsm_file.get()
                },
                "processing": {
                    "output_format": self.output_format.get(),
                    "recursive": self.recursive.get(),
                    "filename_pattern": self.filename_pattern.get(),
                    "mosaic_method": self.mosaic_method.get(),
                    "ortho_resolution": float(self.ortho_resolution.get()),
                    "ortho_padding": float(self.ortho_padding.get()),
                    "run_light_detection": self.run_light_detection.get(),
                    "light_detection_mask": self.light_detection_mask.get(),
                    "apply_world_transform": self.apply_world_transform.get(),
                    "world_file_path": self.world_file_path.get()
                }
            }

            # Preserve extra fields from existing config (like camera_time_offsets)
            for key in existing_config:
                if key not in config:
                    config[key] = existing_config[key]

            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            self.log_console(f"Saved configuration to {self.config_path}")
            messagebox.showinfo("Success", "Configuration saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
            self.log_console(f"ERROR: {e}")

    def log_console(self, message):
        """Parse message and update progress bars"""
        # Update detail text (most recent message)
        self.detail_var.set(message[:80])  # Truncate long messages

        # Force GUI update
        self.root.update_idletasks()

        # Parse for progress updates
        import re

        # Starting frame extraction
        if "Starting frame extraction" in message:
            self.current_phase = 'extraction'
            self.current_step_var.set("Step 1/3: Extracting frames from videos")
            self.overall_progress['value'] = 0

        # Frame extraction complete
        elif "Frame extraction complete" in message:
            self.overall_progress['value'] = 33

        # Starting orthorectification
        elif "Starting orthorectification" in message:
            self.current_phase = 'ortho'
            self.current_step_var.set("Step 2/3: Orthorectifying images")
            self.overall_progress['value'] = 33
            self.completed_orthos = 0

            # Extract number of timestamps
            match = re.search(r'Processing (\d+) timestamps', message)
            if match:
                self.total_timestamps = int(match.group(1))

        # Ortho progress (individual timestamp completed)
        elif self.current_phase == 'ortho' and "files)" in message and ("✓" in message or "?" in message):
            self.completed_orthos += 1
            if self.total_timestamps > 0:
                progress = 33 + (self.completed_orthos / self.total_timestamps) * 33
                self.overall_progress['value'] = progress
                self.root.update_idletasks()

        # Orthorectification complete
        elif "Orthorectification complete" in message:
            self.overall_progress['value'] = 66

        # Starting mosaicking
        elif "Starting mosaicking" in message:
            self.current_phase = 'mosaic'
            self.current_step_var.set("Step 3/3: Creating mosaics")
            self.overall_progress['value'] = 66
            self.completed_mosaics = 0

            # Extract number of mosaics
            match = re.search(r'Creating (\d+) mosaics', message)
            if match:
                self.total_timestamps = int(match.group(1))

        # Mosaic progress (individual mosaic completed)
        elif self.current_phase == 'mosaic' and ("✓" in message or "?" in message) and message.strip().startswith("["):
            self.completed_mosaics += 1
            if self.total_timestamps > 0:
                progress = 66 + (self.completed_mosaics / self.total_timestamps) * 9
                self.overall_progress['value'] = progress
                self.root.update_idletasks()

        # Mosaicking complete
        elif "Mosaicking complete" in message:
            self.overall_progress['value'] = 75  # Leave room for optional light detection

        # Starting light detection
        elif "Starting light detection" in message:
            self.current_phase = 'lights'
            self.current_step_var.set("Step 4/4: Detecting boat lights (optional)")
            self.overall_progress['value'] = 75

        # Light detection progress (per-image)
        elif self.current_phase == 'lights' and " lights" in message and "mosaic_" in message:
            # Individual mosaic processed - just update detail, not progress bar
            # (we don't know total count ahead of time)
            pass

        # Light detection complete
        elif "Light detection complete" in message:
            self.overall_progress['value'] = 100

        # Pipeline complete
        elif "Pipeline completed in" in message:
            self.current_step_var.set("Pipeline complete!")
            self.overall_progress['value'] = 100

    def run_initial_calibration(self):
        """Run initial camera calibration from GCPs"""
        try:
            # Validate inputs
            gcp_file = Path(self.calib_gcp_file.get())
            image_folder = Path(self.calib_image_folder.get())
            dem_file = Path(self.calib_dem_file.get())

            if not gcp_file.exists():
                messagebox.showerror("Error", f"GCP file not found:\n{gcp_file}")
                return

            if not image_folder.exists():
                messagebox.showerror("Error", f"Image folder not found:\n{image_folder}")
                return

            if not dem_file.exists():
                messagebox.showerror("Error", f"DEM file not found:\n{dem_file}")
                return

            # Get resolution and output directory
            try:
                resolution = float(self.calib_resolution.get())
                if resolution <= 0:
                    raise ValueError("Resolution must be positive")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid resolution value:\n{e}")
                return

            output_dir = Path(self.calib_output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            self.log_console("=" * 60)
            self.log_console("Starting Initial Camera Calibration")
            self.log_console("=" * 60)
            self.log_console(f"GCP file: {gcp_file}")
            self.log_console(f"Image folder: {image_folder}")
            self.log_console(f"DEM file: {dem_file}")
            self.log_console(f"Resolution: {resolution} m/pixel")
            self.log_console(f"Output directory: {output_dir}")

            # Build command
            calibrate_script = self.root_dir / 'orthorectification' / 'undistort_and_orthorectify.py'
            cmd = [
                sys.executable,
                str(calibrate_script),
                'calibrate',
                '-g', str(gcp_file),
                '-i', str(image_folder),
                '-d', str(dem_file),
                '-r', str(resolution),
                '-o', str(output_dir)
            ]

            self.log_console(f"\nRunning command: {' '.join(cmd)}\n")

            # Run calibration in background thread
            thread = threading.Thread(
                target=self._run_calibration_thread,
                args=(cmd,),
                daemon=True
            )
            thread.start()

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            messagebox.showerror("Error", f"Failed to start calibration:\n{e}")
            self.log_console(f"ERROR: {e}")
            self.log_console(f"TRACEBACK:\n{error_details}")

    def _run_calibration_thread(self, cmd):
        """Run calibration in background thread"""
        try:
            # Set environment to force unbuffered output
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'

            # Run calibration with unbuffered output
            process = subprocess.Popen(
                cmd,
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env
            )

            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.root.after(0, self.log_console, line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.root.after(0, self.log_console, "\n" + "=" * 60)
                self.root.after(0, self.log_console, "Initial calibration completed successfully!")
                self.root.after(0, self.log_console, "=" * 60)
            else:
                self.root.after(0, self.log_console, f"\nCalibration failed with exit code {process.returncode}")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.root.after(0, self.log_console, f"ERROR: {e}")
            self.root.after(0, self.log_console, f"TRACEBACK:\n{error_details}")

    def run_pipeline(self):
        """Run the processing pipeline"""
        # Save config first
        self.save_config()

        if not self.config_path.exists():
            messagebox.showerror("Error", "Configuration file not found. Please save configuration first.")
            return

        # Disable buttons during run
        self.run_button.configure(state='disabled')
        self.recalibrate_button.configure(state='disabled')

        # Reset progress bars
        self.current_step_var.set("Initializing pipeline...")
        self.overall_progress['value'] = 0
        self.detail_var.set("Starting...")
        self.total_timestamps = 0
        self.completed_orthos = 0
        self.completed_mosaics = 0
        self.current_phase = None

        # Run pipeline in separate thread
        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()

    def _run_pipeline_thread(self):
        """Run pipeline in background thread"""
        try:
            # Change to pipeline directory
            run_script = self.script_dir / 'run.py'

            # Set environment to force unbuffered output
            import os
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'

            # Run pipeline with unbuffered output
            process = subprocess.Popen(
                [sys.executable, '-u', str(run_script)],  # -u for unbuffered
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                env=env
            )

            self.pipeline_process = process

            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.root.after(0, self.log_console, line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.root.after(0, self.log_console, "Pipeline completed successfully!")
            else:
                self.root.after(0, self.log_console, f"Pipeline failed with exit code {process.returncode}")

        except Exception as e:
            self.root.after(0, self.log_console, f"ERROR: {e}")

        finally:
            self.pipeline_process = None
            # Re-enable buttons
            self.root.after(0, self.run_button.configure, {'state': 'normal'})
            self.root.after(0, self.recalibrate_button.configure, {'state': 'normal'})

    def recalibrate_camera(self):
        """Launch recalibration interface"""
        try:
            # Save config first to ensure latest paths are available
            self.save_config()

            mode = self.calib_mode.get()
            self.log_console(f"Launching recalibration interface (preferred mode: {mode})...")
            self.log_console("Note: You will be prompted to confirm the mode in the terminal")

            # Launch recalibration script in a new terminal window
            recalibrate_script = self.script_dir / 'recalibrate.py'

            if sys.platform == 'win32':
                # Windows: Open new cmd window
                subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k',
                                sys.executable, str(recalibrate_script)])
            else:
                # Linux/Mac: Try to open in new terminal
                subprocess.Popen(['x-terminal-emulator', '-e',
                                sys.executable, str(recalibrate_script), '--interactive'])

            self.log_console("Recalibration interface launched in new window")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch recalibration:\n{e}")
            self.log_console(f"ERROR: {e}")


def main():
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use('clam')

    app = PipelineGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
