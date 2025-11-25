# SooOrthoFlow - Soo Locks Image Processing Pipeline

Automated pipeline for processing fisheye camera videos into orthorectified and georeferenced mosaics for ice monitoring at the Soo Locks.

## Table of Contents
- [Installation](#installation)
- [Initial Setup](#initial-setup)
- [Initial Calibration](#initial-calibration)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline GUI Workflow](#pipeline-gui-workflow)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Git
- Anaconda or Miniconda

### Step 1: Install Git (if not already installed)

**Windows:**
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Verify installation by opening Command Prompt and typing:
   ```bash
   git --version
   ```

### Step 2: Install Conda (if not already installed)

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)

### Step 3: Clone the Repository

```bash
git clone https://github.com/ski907/SooOrthoFlow.git
cd SooOrthoFlow
```

### Step 4: Create Conda Environment

```bash
# Create the environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate sooorthoflow
```

**Note:** You'll need to activate this environment every time you want to use the pipeline:
```bash
conda activate sooorthoflow
```

---

## Working with Git and GitHub

### Updating Your Code

To get the latest updates from the GitHub repository:

**Check for updates:**
```bash
# Make sure you're in the SooOrthoFlow directory
cd SooOrthoFlow

# Check current status
git status

# Fetch latest changes (doesn't modify your files yet)
git fetch origin

# See what's new
git log HEAD..origin/main --oneline
```

**Update your code:**

If you have **no local changes**:
```bash
git pull origin main
```

If you have **local modifications**, choose one option:

*Option 1 - Stash changes temporarily:*
```bash
git stash                # Save your changes
git pull origin main     # Get updates
git stash pop            # Restore your changes
```

*Option 2 - Commit changes first:*
```bash
git add .
git commit -m "My local changes"
git pull origin main
```

### Handling Merge Conflicts

If `git pull` reports merge conflicts:
1. Open the conflicted files (Git shows which ones)
2. Look for conflict markers: `<<<<<<<`, `=======`, `>>>>>>>`
3. Edit the file to keep the version you want
4. Remove the conflict markers
5. Save and commit:
   ```bash
   git add .
   git commit -m "Resolved merge conflicts"
   ```

### Reverting to GitHub Version

**WARNING:** This discards ALL local changes!
```bash
git fetch origin
git reset --hard origin/main
```

### Viewing Changes

```bash
# See recent commits
git log --oneline -10

# See what changed in last commit
git diff HEAD~1

# View changes in specific file
git diff HEAD~1 path/to/file.py
```

---

## Initial Setup

### 1. Required Input Files

Before running the pipeline, ensure you have:

1. **Ground Control Points (GCPs)**: `inputs/GCP_merged.csv`
   - CSV file with columns: `point_id, x_model, y_model, elevation, x_world, y_world`
   - Model coordinates: Local coordinate system from your site survey
   - World coordinates: Real-world coordinates (e.g., UTM Zone 17N)

2. **Digital Surface Model (DSM)**: `inputs/lidar_DSM_filled_cropped.tif`
   - GeoTIFF file with elevation data for the area of interest
   - Should cover the entire monitoring area

3. **Video Files**: Organized by test/date
   ```
   test_videos/
   └── 20251119_Soo/
       ├── NVR1/
       │   ├── camera1_video1.avi
       │   └── camera1_video2.avi
       └── NVR2/
           ├── camera2_video1.avi
           └── camera2_video2.avi
   ```

### 2. Configure Master Control File

Copy the template and edit with your paths:

```bash
cp master_control.template.json master_control.json
```

Edit `master_control.json` to set your base paths (one-time setup):
```json
{
    "video_folder": "test_videos/20251119_Soo",
    "start_time": "2025-11-19 15:27:00",
    "end_time": "2025-11-19 15:37:00",
    "interval": "30s",
    "paths": {
        "output_base": "output_data",
        "calibration_file": "calibration/camera_calibrations.pkl",
        "gcp_file": "inputs/GCP_merged.csv",
        "dsm_file": "inputs/lidar_DSM_filled_cropped.tif"
    },
    "processing": {
        "output_format": "tiff",
        "recursive": true,
        "filename_pattern": "CAMERA_DATETIME_DATETIME",
        "mosaic_method": "center",
        "ortho_resolution": 0.0025,
        "ortho_padding": 0.5,
        "run_light_detection": false,
        "light_detection_mask": "analysis/ship_detection/area_to_review_ref.tif",
        "apply_world_transform": false,
        "world_file_path": "orthorectification/model_to_world.wld"
    }
}
```

---

## Initial Calibration

The first step is to calibrate your fisheye cameras. This determines the intrinsic parameters (lens distortion) and extrinsic parameters (camera position and orientation) for each camera.

### Prerequisites

**IMPORTANT:** Initial calibration requires a pre-made GCP CSV file that contains BOTH:
- World coordinates (X, Y, Z) in your coordinate system
- Image pixel coordinates (col_sample, row_sample) for each GCP in each camera image

This GCP file must be created before running calibration, typically using:
- QGIS or other GIS software for georeferencing
- Manual image annotation tools
- Provided by project lead who has already established GCP correspondences

**Required GCP CSV Format:**
```csv
image_name,channel,camera_name,X,Y,Z,col_sample,row_sample
camera1.jpg,1,NVR1_N910A6_ch1_main,10.5,20.3,100.2,1024,768
camera1.jpg,1,NVR1_N910A6_ch1_main,15.2,25.1,100.5,1234,890
...
```

Each camera should have 15-30 well-distributed GCPs for best results.

### Preparation

1. **GCP File**: Ensure `inputs/GCP_merged.csv` exists with world AND pixel coordinates

2. **Calibration Images**: One clear frame per camera showing the monitoring area
   ```
   calibration_images/
   ├── NVR1_N910A6_ch1_main.jpg
   ├── NVR1_N910A6_ch2_main.jpg
   └── ...
   ```

3. **DEM File**: `inputs/lidar_DSM_filled_cropped.tif`

### Running Initial Calibration

**Command Line (Automated):**
```bash
python orthorectification/undistort_and_orthorectify.py calibrate \
  -g inputs/GCP_merged.csv \
  -i calibration_images/ \
  -d inputs/lidar_DSM_filled_cropped.tif \
  -o calibration/
```

This will:
1. Read pre-defined GCP world coordinates and pixel coordinates from the CSV
2. Process each camera image found in the calibration images directory
3. Solve for camera parameters using OpenCV's fisheye calibration
4. Save calibration to `calibration/camera_calibrations_YYYYMMDD.pkl`

**No manual clicking required** - all GCP correspondences are pre-defined in the CSV file.

**Calibration Output:**
The calibration .pkl file contains for each camera:
- `K`: Camera intrinsic matrix (focal length, principal point)
- `D`: Distortion coefficients (fisheye model parameters)
- `rvec`: Rotation vector (camera orientation in 3D space)
- `tvec`: Translation vector (camera position in 3D space)
- `rms`: Reprojection error (lower is better, <2 pixels is good)

**Quality Check:**
After calibration, check the RMS values:
- < 2 pixels: Excellent
- 2-5 pixels: Good
- > 5 pixels: Review GCP pixel coordinates for accuracy

---

## Running the Pipeline

### Using the GUI (Recommended)

The easiest way to run the pipeline is using the graphical interface:

```bash
python pipeline/pipeline_gui.py
```

See [Pipeline GUI Workflow](#pipeline-gui-workflow) below for detailed instructions.

### Using Command Line

```bash
python pipeline/run.py
```

The pipeline will:
1. Extract frames from videos at specified timestamps
2. Undistort and orthorectify each frame
3. Create mosaics for each timestamp
4. Optionally detect boat lights (if enabled)
5. Optionally transform to world coordinates (if enabled)

### Partial Runs (for troubleshooting)

```bash
# Extract frames only
python pipeline/run.py --extract-only

# Process existing frames (skip extraction)
python pipeline/run.py --process-only

# Create mosaics only (skip extraction and orthorectification)
python pipeline/run.py --mosaic-only
```

---

## Pipeline GUI Workflow

### 1. Launch the GUI

```bash
python pipeline/pipeline_gui.py
```

### 2. Configure Input Parameters

**Video Folder:** Browse to select your video directory
- Should contain subdirectories with video files
- Example: `test_videos/20251119_Soo`

**Time Range:**
- **Start Time**: When to begin extracting frames (format: YYYY-MM-DD HH:MM:SS)
- **End Time**: When to stop extracting frames
- **Interval**: Time between frame extractions (e.g., "30s", "1min", "5min")

### 3. Set File Paths

**Output Directory**: Where to save processed data (default: `output_data`)

**Calibration File**: Path to camera calibration pickle file
- Created during initial calibration
- Example: `calibration/camera_calibrations_20251119.pkl`

**GCP File**: Ground control points CSV

**DSM File**: Digital surface model GeoTIFF

### 4. Processing Options

**Output Format**: Choose TIFF, PNG, or JPG

**Mosaic Method**:
- **Center**: Prefer pixels near image centers (recommended - less distortion)
- **Average**: Blend overlapping areas
- **Max/Min**: Take maximum or minimum values

**Recursive**: Enable to search subdirectories for videos

**Ortho Resolution**: Output resolution in meters per pixel (default: 0.005 = 5mm/pixel)

**Ortho Padding**: Extra area around monitoring region in meters (default: 0.5m)

### 5. Post-Processing Options

**Detect boat lights**: Analyze mosaics for bright spots (boats)
- Optional mask file to limit search area

**Transform mosaics to world coordinates**:
- Enable this to transform output mosaics from model to world coordinates
- Requires a world file created by georeferencing in QGIS
- World file path: `orthorectification/model_to_world.wld`

### 6. Run the Pipeline

1. Click **"Save"** to save your configuration
2. Click **"Run Pipeline"** to start processing
3. Monitor progress in the GUI:
   - Overall progress bar shows pipeline phase
   - Step progress shows current operation
   - Detailed messages appear at the bottom

### 7. Output Structure

Results are saved in subdirectories organized by test name and timestamp:

```
output_data/
└── 20251119_Soo_20251119_153000/
    ├── master_control.json         # Configuration used
    ├── time_config.json             # Frame extraction config
    ├── processing_log.txt           # Detailed log
    ├── frames/
    │   ├── 20251119_152700/
    │   │   ├── NVR1_camera1.tiff
    │   │   └── NVR2_camera2.tiff
    │   └── 20251119_152730/
    ├── orthos/
    │   ├── 20251119_152700/
    │   │   └── orthorectified/
    │   │       ├── NVR1_camera1_ortho.tif
    │   │       └── NVR2_camera2_ortho.tif
    │   └── 20251119_152730/
    └── mosaics/
        ├── mosaic_20251119_152700.tif
        └── mosaic_20251119_152730.tif
```

---

## Advanced Features

### Camera Time Offsets

If your video recording systems are not perfectly synchronized, you can specify time offsets in `master_control.json`:

```json
{
    "camera_time_offsets": {
        "NVR2": 80.0
    }
}
```

This will offset NVR2 videos by 80 seconds (extract frames 80 seconds later) to synchronize with other cameras.

### Periodic Recalibration

Cameras may shift position over time. Recalibrate using recent good frames:

**Option 1: GUI**
Click **"Recalibrate Camera"** button in the pipeline GUI

**Option 2: Command Line**
```bash
cd calibration
python recalibrate_camera.py
```

Choose calibration mode:
- **Pose-only** (faster): Updates camera position/orientation only
  - Use when camera was bumped or slightly moved
- **Full recalibration**: Re-solves all parameters
  - Use when significant changes occurred

The recalibration tool will:
1. Load the most recent calibration
2. Allow you to select a camera and test frame
3. Let you click GCPs in the frame
4. Update the calibration file with new parameters

### World Coordinate Transformation

To transform mosaics to real-world coordinates:

1. **Create a world file** using QGIS Georeferencer:
   - Load a reference mosaic in model coordinates
   - Add ground control points with known world coordinates
   - Generate the transformation
   - Save as `orthorectification/model_to_world.wld`

2. **Enable transformation** in the GUI or `master_control.json`:
   ```json
   "processing": {
       "apply_world_transform": true,
       "world_file_path": "orthorectification/model_to_world.wld"
   }
   ```

3. **Run pipeline**: Output mosaics will be in world coordinates

---

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Make sure environment is activated
conda activate sooorthoflow

# Reinstall problematic packages
conda install opencv numpy scipy pandas rasterio -c conda-forge
```

### Video File Not Found

- Check that video folder path is correct
- Ensure `recursive: true` if videos are in subdirectories
- Verify video file extensions are supported (.avi, .mp4, .mov, etc.)

### Poor Orthorectification Quality

- Check calibration RMS error (should be <2 pixels)
- Verify GCPs are accurate and well-distributed
- Ensure DSM covers the entire area
- Try recalibrating with more recent frames

### Mosaic Alignment Issues

- Check that all cameras use the same calibration date
- Verify camera time offsets are correct
- Ensure orthorectification parameters (resolution, padding) are appropriate

### Transformation Issues

- Verify world file exists and is formatted correctly
- Check that world file was created from an image with matching pixel dimensions
- Ensure coordinate reference system (CRS) is consistent

### GUI Issues

**Windows**: If GUI doesn't launch, tkinter may not be installed:
```bash
conda install tk -c conda-forge
```


### Performance Optimization

- Reduce `ortho_resolution` to process faster (larger pixels)
- Use fewer cores with environment variable: `export OMP_NUM_THREADS=4`
- Process timestamps in batches using start/end times

---

