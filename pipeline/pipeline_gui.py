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
        self.root.geometry("900x800")

        # Config file path
        self.script_dir = Path(__file__).parent
        self.root_dir = self.script_dir.parent
        self.config_path = self.root_dir / 'master_control.json'

        # Variables for form fields
        self.video_folder = tk.StringVar()
        self.start_time = tk.StringVar()
        self.end_time = tk.StringVar()
        self.interval = tk.StringVar(value="1min")

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

        # Pipeline process reference
        self.pipeline_process = None

        # Build UI
        self.create_widgets()

        # Try to load existing config
        if self.config_path.exists():
            self.load_config()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main container with scrollbar
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Soo Locks Model Image Processing Pipeline",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Config file management
        config_frame = ttk.LabelFrame(main_frame, text="Configuration File", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Config File:").grid(row=0, column=0, sticky=tk.W)
        config_label = ttk.Label(config_frame, textvariable=tk.StringVar(value=str(self.config_path)),
                                foreground="blue")
        config_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))

        ttk.Button(config_frame, text="Load", command=self.load_config).grid(row=0, column=2, padx=5)
        ttk.Button(config_frame, text="Save", command=self.save_config).grid(row=0, column=3, padx=5)

        # Input Parameters Section
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(input_frame, text="Video Folder:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.video_folder, width=60).grid(row=row, column=1,
                                                                               sticky=(tk.W, tk.E), padx=5)
        ttk.Button(input_frame, text="Browse...",
                  command=lambda: self.browse_directory(self.video_folder)).grid(row=row, column=2)

        row += 1
        ttk.Label(input_frame, text="Start Time:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(input_frame, textvariable=self.start_time, width=60).grid(row=row, column=1,
                                                                             sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        ttk.Label(input_frame, text="Format: YYYY-MM-DD HH:MM:SS",
                 foreground="gray").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(input_frame, text="End Time:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.end_time, width=60).grid(row=row, column=1,
                                                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Label(input_frame, text="Format: YYYY-MM-DD HH:MM:SS",
                 foreground="gray").grid(row=row, column=2, sticky=tk.W)

        row += 1
        ttk.Label(input_frame, text="Interval:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        interval_frame = ttk.Frame(input_frame)
        interval_frame.grid(row=row, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        ttk.Entry(interval_frame, textvariable=self.interval, width=20).pack(side=tk.LEFT)
        ttk.Label(interval_frame, text="Examples: 30s, 1min, 5min",
                 foreground="gray").pack(side=tk.LEFT, padx=(10, 0))

        # Paths Section
        paths_frame = ttk.LabelFrame(main_frame, text="File Paths", padding="10")
        paths_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        paths_frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(paths_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(paths_frame, textvariable=self.output_base, width=50).grid(row=row, column=1,
                                                                              sticky=(tk.W, tk.E), padx=5)
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_directory(self.output_base)).grid(row=row, column=2)

        row += 1
        ttk.Label(paths_frame, text="Calibration File:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(paths_frame, textvariable=self.calibration_file, width=50).grid(row=row, column=1,
                                                                                   sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_file(self.calibration_file,
                                                   [("Pickle Files", "*.pkl"), ("All Files", "*.*")])).grid(row=row, column=2, pady=(5, 0))

        row += 1
        ttk.Label(paths_frame, text="GCP File:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(paths_frame, textvariable=self.gcp_file, width=50).grid(row=row, column=1,
                                                                           sticky=(tk.W, tk.E), padx=5)
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_file(self.gcp_file,
                                                   [("CSV Files", "*.csv"), ("All Files", "*.*")])).grid(row=row, column=2)

        row += 1
        ttk.Label(paths_frame, text="DSM File:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(paths_frame, textvariable=self.dsm_file, width=50).grid(row=row, column=1,
                                                                           sticky=(tk.W, tk.E), padx=5, pady=(5, 0))
        ttk.Button(paths_frame, text="Browse...",
                  command=lambda: self.browse_file(self.dsm_file,
                                                   [("GeoTIFF Files", "*.tif;*.tiff"), ("All Files", "*.*")])).grid(row=row, column=2, pady=(5, 0))

        # Processing Options Section
        proc_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        proc_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        proc_frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(proc_frame, text="Output Format:").grid(row=row, column=0, sticky=tk.W)
        format_combo = ttk.Combobox(proc_frame, textvariable=self.output_format,
                                    values=["tiff", "png", "jpg"], state="readonly", width=15)
        format_combo.grid(row=row, column=1, sticky=tk.W, padx=5)

        ttk.Label(proc_frame, text="Mosaic Method:").grid(row=row, column=2, sticky=tk.W, padx=(20, 0))
        mosaic_combo = ttk.Combobox(proc_frame, textvariable=self.mosaic_method,
                                    values=["center", "average", "max", "min"], state="readonly", width=15)
        mosaic_combo.grid(row=row, column=3, sticky=tk.W, padx=5)

        row += 1
        ttk.Checkbutton(proc_frame, text="Recursive video search",
                       variable=self.recursive).grid(row=row, column=0, columnspan=2,
                                                     sticky=tk.W, pady=(5, 0))

        row += 1
        ttk.Label(proc_frame, text="Filename Pattern:").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(proc_frame, textvariable=self.filename_pattern, width=40).grid(row=row, column=1,
                                                                                  columnspan=3, sticky=(tk.W, tk.E),
                                                                                  padx=5, pady=(5, 0))

        row += 1
        ttk.Label(proc_frame, text="Ortho Resolution (m/px):").grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        ttk.Entry(proc_frame, textvariable=self.ortho_resolution, width=15).grid(row=row, column=1,
                                                                                  sticky=tk.W, padx=5, pady=(5, 0))

        ttk.Label(proc_frame, text="Ortho Padding (m):").grid(row=row, column=2, sticky=tk.W,
                                                               padx=(20, 0), pady=(5, 0))
        ttk.Entry(proc_frame, textvariable=self.ortho_padding, width=15).grid(row=row, column=3,
                                                                               sticky=tk.W, padx=5, pady=(5, 0))

        # Console Output Section
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="10")
        console_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        self.console = scrolledtext.ScrolledText(console_frame, height=10, state='disabled',
                                                 bg='black', fg='white', font=('Courier', 9))
        self.console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Action Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=(10, 0))

        self.run_button = ttk.Button(button_frame, text="Run Pipeline",
                                     command=self.run_pipeline, style='Accent.TButton')
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.recalibrate_button = ttk.Button(button_frame, text="Recalibrate Camera",
                                            command=self.recalibrate_camera)
        self.recalibrate_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Close", command=self.root.quit).pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))

    def browse_directory(self, var):
        """Open directory browser"""
        initial_dir = var.get() if var.get() else str(self.root_dir)
        directory = filedialog.askdirectory(initialdir=initial_dir, title="Select Directory")
        if directory:
            var.set(directory)

    def browse_file(self, var, filetypes):
        """Open file browser"""
        initial_file = var.get() if var.get() else ""
        initial_dir = Path(initial_file).parent if initial_file else str(self.root_dir)

        filename = filedialog.askopenfilename(initialdir=initial_dir, title="Select File",
                                             filetypes=filetypes)
        if filename:
            var.set(filename)

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

            self.log_console(f"Loaded configuration from {self.config_path}")
            self.status_var.set("Configuration loaded")

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

            # Build config dictionary
            config = {
                "video_folder": self.video_folder.get(),
                "start_time": self.start_time.get(),
                "end_time": self.end_time.get(),
                "interval": self.interval.get(),
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
                    "ortho_padding": float(self.ortho_padding.get())
                }
            }

            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            self.log_console(f"Saved configuration to {self.config_path}")
            self.status_var.set("Configuration saved")
            messagebox.showinfo("Success", "Configuration saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
            self.log_console(f"ERROR: {e}")

    def log_console(self, message):
        """Write message to console output"""
        self.console.configure(state='normal')
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.console.insert(tk.END, f"[{timestamp}] {message}\n")
        self.console.see(tk.END)
        self.console.configure(state='disabled')

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

        self.log_console("="*60)
        self.log_console("Starting pipeline...")
        self.status_var.set("Pipeline running...")

        # Run pipeline in separate thread
        thread = threading.Thread(target=self._run_pipeline_thread, daemon=True)
        thread.start()

    def _run_pipeline_thread(self):
        """Run pipeline in background thread"""
        try:
            # Change to pipeline directory
            run_script = self.script_dir / 'run.py'

            # Run pipeline
            process = subprocess.Popen(
                [sys.executable, str(run_script)],
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )

            self.pipeline_process = process

            # Stream output
            for line in process.stdout:
                self.root.after(0, self.log_console, line.rstrip())

            process.wait()

            if process.returncode == 0:
                self.root.after(0, self.log_console, "Pipeline completed successfully!")
                self.root.after(0, self.status_var.set, "Pipeline completed successfully")
            else:
                self.root.after(0, self.log_console, f"Pipeline failed with exit code {process.returncode}")
                self.root.after(0, self.status_var.set, "Pipeline failed")

        except Exception as e:
            self.root.after(0, self.log_console, f"ERROR: {e}")
            self.root.after(0, self.status_var.set, "Pipeline error")

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

            self.log_console("Launching recalibration interface...")
            self.status_var.set("Recalibration in progress...")

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
            self.status_var.set("Ready")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch recalibration:\n{e}")
            self.log_console(f"ERROR: {e}")
            self.status_var.set("Ready")


def main():
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use('clam')

    app = PipelineGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
