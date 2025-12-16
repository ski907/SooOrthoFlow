# Video Mosaic - Streaming Multi-Camera Mosaic Video Processor

Efficiently creates mosaicked video from multiple synchronized camera sources using **single-pass processing**. No intermediate TIFF files are created - frames go directly from source videos → orthorectification → mosaicking → output video.

## Key Features

- **Single-pass processing**: Read → orthorectify → mosaic → write (no intermediate files)
- **6-12x faster** than TIFF-based approaches
- **10-60x less storage**: Compressed video vs individual TIFFs
- **Uses existing infrastructure**: Ortho caches, zone maps, calibrations
- **PIV-ready output**: Georeferenced video ready for velocity analysis

## Performance

**Example**: 2 cameras, 2-minute sequence @ 1-second intervals (120 frames)
- Processing time: ~10-15 seconds
- Output size: ~50-200 MB (vs 1-3 GB for TIFFs)
- Per-frame timing: ~80ms (20ms read + 40ms ortho + 10ms mosaic + 10ms write)

## Prerequisites

Before running the video mosaic processor, you must:

1. **Have camera calibrations** with orthorectification parameters
   - File: `calibration/camera_calibrations_YYYYMMDD.csv`

2. **Generate ortho caches** at your desired resolution
   - Run the orthorectification pipeline first
   - Caches are stored in `orthorectification/ortho_cache/`
   - Check cache directory: `ls orthorectification/ortho_cache/*_ortho_cache_*.pkl`

3. **Have zone map** (if using zone_map method)
   - File: `orthorectification/camera_zone_map/camera_zone_map.shp`

## Quick Start

### 1. Copy and Edit Configuration

```bash
# Copy template
cp analysis/video_mosaic/video_mosaic_config_template.json my_mosaic.json

# Edit configuration:
# - Set camera_ids (must match calibration file exactly)
# - Set start_time and end_time
# - Set video_dir path
# - Verify calibration_file path
# - Check ortho_resolution matches your cache
```

### 2. Run Processing

```bash
# Quick 30-second test
python analysis/video_mosaic/run_video_mosaic.py --config my_mosaic.json --quick

# Full processing
python analysis/video_mosaic/run_video_mosaic.py --config my_mosaic.json
```

### 3. Output

Outputs are saved to `output_dir` (default: `output_data/video_mosaic/`):
- `mosaic_video.mp4` - Mosaicked video
- `mosaic_metadata.json` - Geotransform and processing metadata

## Configuration

Key configuration parameters:

```json
{
  "input": {
    "video_dir": "test_videos/12-12-25/Test 13",
    "camera_ids": ["NVR1_N910A6_ch4_main", "NVR1_N910A6_ch5_main"],
    "start_time": "2025-12-12 12:44:00",
    "end_time": "2025-12-12 12:46:00",
    "interval_seconds": 1.0
  },
  "processing": {
    "ortho_resolution": 0.01,
    "mosaic_method": "zone_map"
  },
  "output": {
    "video_fps": 10,
    "video_codec": "mp4v"
  }
}
```

### Camera IDs

**IMPORTANT**: Camera IDs must match the calibration file exactly.

Check your calibration file:
```bash
head -1 calibration/camera_calibrations_20251203.csv
```

Example camera IDs:
- `NVR1_N910A6_ch4_main`
- `NVR1_N910A6_ch5_main`
- `NVR2_N910A6_ch1_main`

### Ortho Resolution

Must match the resolution of your ortho caches:
- `0.0025` = 2.5mm/pixel (high resolution)
- `0.005` = 5mm/pixel (medium resolution)
- `0.01` = 10mm/pixel (low resolution, recommended)

Check existing caches:
```bash
ls orthorectification/ortho_cache/ | grep NVR1_N910A6_ch4_main
```

### Mosaic Methods

**zone_map** (recommended for multi-camera):
- Uses shapefile zones to assign pixels to cameras
- Each camera gets a defined region
- Requires `zone_map_shapefile` parameter

**center**:
- Blends cameras based on distance from image center
- Good for 2-camera overlapping views
- No shapefile needed

## Command-Line Options

```bash
# Override parameters
python analysis/video_mosaic/run_video_mosaic.py \
    --config my_mosaic.json \
    --video-dir test_videos/different_test \
    --output-dir output_data/my_output \
    --start-time "2025-12-12 13:00:00" \
    --end-time "2025-12-12 13:05:00" \
    --fps 15

# Quick 30-second test
python analysis/video_mosaic/run_video_mosaic.py \
    --config my_mosaic.json \
    --quick
```

## Workflow Integration

### Use with PIV Analysis

After creating a mosaic video, you can:

1. **Extract frames for PIV**:
   ```bash
   # Extract frames from video at intervals
   ffmpeg -i mosaic_video.mp4 -vf fps=1 frame_%04d.png
   ```

2. **Run PIV analysis** on extracted frames using `analysis/run_piv_analysis.py`

3. **Or modify PIV tool** to read directly from video (future enhancement)

## Module Structure

```
analysis/video_mosaic/
├── __init__.py                         # Module exports
├── run_video_mosaic.py                 # CLI entry point
├── video_mosaic_processor.py           # Main orchestrator
├── camera_video_reader.py              # Frame synchronization
├── in_memory_mosaic.py                 # Mosaicking engine
├── video_mosaic_config_template.json   # Configuration template
└── README.md                           # This file
```

## Architecture: Single-Pass Processing

**Critical design**: Everything happens in ONE pass through source videos.

```
FOR EACH TIMESTAMP (12:44:00, 12:44:01, 12:44:02, ...):

  1. Synchronize & read raw frames
     ├─ NVR1_ch4.avi → seek to timestamp → read frame_ch4
     └─ NVR1_ch5.avi → seek to timestamp → read frame_ch5

  2. Orthorectify each (using pre-loaded caches)
     ├─ cv2.remap(frame_ch4, map_x_ch4, map_y_ch4) → ortho_ch4
     └─ cv2.remap(frame_ch5, map_x_ch5, map_y_ch5) → ortho_ch5

  3. Mosaic IN MEMORY (zone_map or center method)
     └─ mosaic_engine.combine(ortho_ch4, ortho_ch5) → mosaicked_frame

  4. Write to output video
     └─ video_writer.write(mosaicked_frame)

RESULT: Single mosaic_video.mp4 file
```

**No intermediate videos created!**

## Troubleshooting

### Error: "No ortho cache found for camera"

**Solution**: Run orthorectification first to generate caches at your desired resolution.

```bash
# Check if caches exist
ls orthorectification/ortho_cache/ | grep NVR1_N910A6_ch4_main

# If missing, run orthorectification to generate them
# (use your existing pipeline or targeted mosaic tool)
```

### Error: "No video found for camera at timestamp"

**Possible causes**:
1. Video files not in expected location
2. Video filename doesn't match expected pattern
3. Timestamp outside video time range

**Solution**:
- Check video directory structure (may need NVR subdirectories)
- Verify video filename format: `CAMERA_STARTTIME_ENDTIME.ext`
- Check video time range matches config

### Error: "Zone map cache not found"

**Solution**: Ensure zone map shapefile exists and is readable.

```bash
# Check shapefile
ls orthorectification/camera_zone_map/camera_zone_map.shp

# Or use 'center' method instead (no shapefile needed)
```

## Technical Details

### Video File Discovery

Videos are discovered using:
1. Camera ID parsing (e.g., `NVR1_N910A6_ch4_main` → NVR name: `NVR1`)
2. Search in NVR subdirectory first: `video_dir/NVR1/`
3. Fall back to root directory: `video_dir/`
4. Pattern matching on serial + channel + stream

### Time Synchronization

- Parses video filenames to extract start/end times
- Uses time-based seeking (`cv2.CAP_PROP_POS_MSEC`) for accuracy
- Supports camera-specific time offsets via `camera_time_offsets`

### Orthorectification

- Loads pre-computed lookup tables (map_x, map_y) from cache
- Uses `cv2.remap()` for fast transformation (~20ms per frame)
- Preserves geotransform for spatial reference

### Mosaicking

**Zone Map Method**:
- Pre-loads rasterized zone map (camera ID per pixel)
- Places pixels where zone matches camera
- Fast in-memory operation (~10ms per frame)

**Center Method**:
- Weights pixels by distance from image center
- Accumulates weighted pixels across all cameras
- Normalizes by total weight

## Example Scenarios

### 1. Quick Test (30 seconds)

```json
{
  "start_time": "2025-12-12 12:44:00",
  "end_time": "2025-12-12 12:44:30",
  "interval_seconds": 1.0,
  "video_fps": 10
}
```
Output: 30 frames, ~3-5 seconds processing

### 2. PIV Analysis (2 minutes)

```json
{
  "start_time": "2025-12-12 12:44:00",
  "end_time": "2025-12-12 12:46:00",
  "interval_seconds": 1.0,
  "video_fps": 10
}
```
Output: 120 frames, ~10-15 seconds processing

### 3. High Frequency (1 minute @ 0.5s)

```json
{
  "start_time": "2025-12-12 12:44:00",
  "end_time": "2025-12-12 12:45:00",
  "interval_seconds": 0.5,
  "video_fps": 15
}
```
Output: 120 frames, ~10-15 seconds processing

### 4. Single Camera

```json
{
  "camera_ids": ["NVR1_N910A6_ch4_main"],
  "mosaic_method": "center",
  "ortho_resolution": 0.0025
}
```
High-resolution single-camera view

## Version

- **Version**: 0.1.0
- **Author**: SooOrthoFlow Team
- **Date**: December 2025

## Future Enhancements

Potential improvements:
- Parallel frame pre-loading for even faster processing
- Direct PIV analysis from video (skip frame extraction)
- Multi-resolution output (low-res preview + high-res frames)
- Real-time preview during processing
- GPU acceleration for orthorectification
