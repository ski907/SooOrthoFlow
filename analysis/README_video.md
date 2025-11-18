# Mosaic to Video Converter

Converts a sequence of orthomosaic images into a video with automatic timestamp overlay, size management, and quality control.

## Features

- **Automatic Sorting**: Extracts timestamps from filenames and sorts chronologically
- **Size Management**: Downsample large mosaics to control file size
- **Timestamp Overlay**: Adds date/time stamps to each frame
- **Multiple Codecs**: H.264, H.265, Xvid, Motion JPEG
- **Quality Presets**: Easy quality vs file size tradeoffs
- **Progress Reporting**: Real-time encoding progress

## Requirements

```bash
# OpenCV and NumPy (already installed)
pip install opencv-python numpy
```

## Quick Start

### Basic Usage

Create 1 fps video with automatic sizing:

```bash
python mosaics_to_video.py -i ../output_data/20251016_Nav1/mosaics/ -o ship_navigation.mp4
```

### HD Video

Create HD resolution video:

```bash
python mosaics_to_video.py -i mosaics/ -o ship_hd.mp4 --max-width 1920 --max-height 1080 --fps 2
```

### Limit File Size

Constrain video to ~5 megapixels per frame (reduces file size):

```bash
python mosaics_to_video.py -i mosaics/ -o ship_small.mp4 --max-pixels 5000000
```

### High Quality

Use H.265 codec with high quality:

```bash
python mosaics_to_video.py -i mosaics/ -o ship_hq.mp4 --codec h265 --quality high
```

## Parameters

### Frame Rate

```bash
--fps 1          # 1 frame per second (default, good for slow events)
--fps 5          # 5 fps (smoother but larger file)
--fps 10         # 10 fps (very smooth)
--fps 0.5        # 1 frame every 2 seconds (time-lapse)
```

**Recommendations:**
- **Ship movement**: 1-2 fps (standard)
- **Fast events**: 5-10 fps
- **Long duration**: 0.5 fps (time-lapse)

### Resolution Control

Three ways to control resolution:

**1. Maximum Width**
```bash
--max-width 1920      # Limit width to 1920px, maintain aspect ratio
```

**2. Maximum Height**
```bash
--max-height 1080     # Limit height to 1080px, maintain aspect ratio
```

**3. Maximum Pixels** (recommended for file size control)
```bash
--max-pixels 2073600  # 1920x1080 = 2,073,600 pixels
--max-pixels 5000000  # ~2236x2236 or similar
```

**Common Resolutions:**
- 4K: `--max-pixels 8294400` (3840×2160)
- 1080p: `--max-pixels 2073600` (1920×1080)
- 720p: `--max-pixels 921600` (1280×720)
- Web: `--max-pixels 500000` (~707×707)

### Codecs

**H.264** (default) - Best compatibility
```bash
--codec h264
```
- ✓ Plays on all devices
- ✓ Good compression
- ✓ Fast encoding
- Best for: General use, sharing

**H.265/HEVC** - Best compression
```bash
--codec h265
```
- ✓ 50% smaller files than H.264
- ✗ Some devices can't play
- ✗ Slower encoding
- Best for: Storage, modern devices

**Xvid** - Good compatibility
```bash
--codec xvid
```
- ✓ Plays on most devices
- ~ Moderate compression
- Best for: Older systems

**Motion JPEG** - Highest quality
```bash
--codec mjpeg
```
- ✓ Maximum quality
- ✗ Very large files
- Best for: Editing, archival

### Quality Presets

```bash
--quality low        # Small files, acceptable quality
--quality medium     # Balanced (default)
--quality high       # High quality, larger files
--quality very_high  # Maximum quality, largest files
```

**Or set custom bitrate:**
```bash
--bitrate 5     # 5 Mbps
--bitrate 10    # 10 Mbps (high quality)
--bitrate 20    # 20 Mbps (very high quality)
```

### Timestamp Options

**Position:**
```bash
--timestamp-position top-left      # Default
--timestamp-position top-right
--timestamp-position bottom-left
--timestamp-position bottom-right
```

**Disable timestamps:**
```bash
--no-timestamps
```

## File Size Examples

For 100 orthomosaics at 1 fps (100 second video):

| Resolution | Quality | Codec | Estimated Size |
|------------|---------|-------|----------------|
| 1920×1080  | medium  | H.264 | ~250 MB        |
| 1920×1080  | high    | H.264 | ~375 MB        |
| 1920×1080  | medium  | H.265 | ~125 MB        |
| 3840×2160  | medium  | H.264 | ~1 GB          |
| 1280×720   | medium  | H.264 | ~110 MB        |

**To reduce file size:**
1. Use `--max-pixels` to limit resolution
2. Use `--codec h265` for better compression
3. Lower `--quality` preset
4. Reduce `--fps` if acceptable

## Usage Recipes

### Standard Ship Navigation Video

Good quality, reasonable size:

```bash
python mosaics_to_video.py \
  -i output_data/20251016_Nav1/mosaics/ \
  -o ship_navigation.mp4 \
  --fps 1 \
  --max-pixels 2073600 \
  --codec h264 \
  --quality medium
```

### High Quality Archive

Maximum quality for archival:

```bash
python mosaics_to_video.py \
  -i mosaics/ \
  -o ship_archive.mp4 \
  --fps 2 \
  --codec h265 \
  --quality very_high
```

### Web-Optimized Video

Small file for web sharing:

```bash
python mosaics_to_video.py \
  -i mosaics/ \
  -o ship_web.mp4 \
  --fps 1 \
  --max-width 1280 \
  --codec h264 \
  --quality medium
```

### Time-Lapse

Very long duration, slow playback:

```bash
python mosaics_to_video.py \
  -i mosaics/ \
  -o ship_timelapse.mp4 \
  --fps 0.5 \
  --max-pixels 2073600
```

### GIS Presentation

For presentations with clear timestamps:

```bash
python mosaics_to_video.py \
  -i mosaics/ \
  -o presentation.mp4 \
  --fps 2 \
  --max-width 1920 \
  --codec h264 \
  --quality high \
  --timestamp-position bottom-right
```

## Output Format

Video filename extensions determine container format:

- **`.mp4`** - MP4 container (recommended, universal compatibility)
- **`.avi`** - AVI container (older, larger files)
- **`.mkv`** - Matroska container (supports all codecs)
- **`.mov`** - QuickTime container

**Recommendation**: Use `.mp4` for best compatibility.

## Troubleshooting

### "No images found"
- Check input directory path
- Use `--recursive` if images are in subdirectories
- Verify image formats (.tif, .tiff, .png, .jpg)

### "Could not create video writer"
- Check output path is writable
- Ensure output directory exists
- Try different codec (e.g., `--codec h264`)

### Video is too large
1. Reduce resolution: `--max-pixels 2000000`
2. Lower quality: `--quality low`
3. Use H.265: `--codec h265`
4. Reduce frame rate: `--fps 0.5`

### Video is too small/poor quality
1. Increase quality: `--quality high` or `--quality very_high`
2. Use custom bitrate: `--bitrate 15`
3. Use Motion JPEG: `--codec mjpeg` (warning: very large)

### Timestamps wrong or missing
- Check filename has format: `YYYYMMDD_HHMMSS`
- Verify images sort correctly by filename
- Use `--no-timestamps` if timestamps aren't needed

### Video playback stutters
- Increase bitrate: `--bitrate 10`
- Use H.264: `--codec h264`
- Reduce resolution: `--max-width 1920`

## Advanced Usage

### Batch Processing Multiple Folders

Create videos for each test:

```bash
# Bash/Linux
for dir in output_data/*/mosaics; do
    test_name=$(basename $(dirname $dir))
    python mosaics_to_video.py -i "$dir" -o "videos/${test_name}.mp4" --fps 1 --max-pixels 2073600
done

# PowerShell/Windows
Get-ChildItem output_data\*\mosaics | ForEach-Object {
    $testName = $_.Parent.Name
    python mosaics_to_video.py -i $_.FullName -o "videos\$testName.mp4" --fps 1 --max-pixels 2073600
}
```

### Python API

```python
from mosaics_to_video import create_video_from_mosaics

success = create_video_from_mosaics(
    input_dir='mosaics/',
    output_file='output.mp4',
    fps=1,
    max_pixels=2073600,
    codec='h264',
    quality_preset='medium',
    add_timestamps=True,
    timestamp_position='top-left'
)
```

### Integration with Pipeline

Add to `master_control.json`:

```json
{
  "video_generation": {
    "enabled": true,
    "fps": 1,
    "max_pixels": 2073600,
    "codec": "h264",
    "quality": "medium"
  }
}
```

Add to `pipeline/run.py`:

```python
# After mosaicking
if master_config.get('video_generation', {}).get('enabled', False):
    video_params = master_config['video_generation']
    create_video_from_mosaics(
        input_dir=test_dir / 'mosaics',
        output_file=test_dir / f'{test_id}_video.mp4',
        fps=video_params.get('fps', 1),
        max_pixels=video_params.get('max_pixels'),
        codec=video_params.get('codec', 'h264'),
        quality_preset=video_params.get('quality', 'medium')
    )
```

## Performance Tips

1. **Encoding Speed**:
   - H.264 is fastest
   - Motion JPEG is very fast
   - H.265 is slowest

2. **Processing Large Mosaics**:
   - Always use `--max-pixels` to downsample
   - Start with `--max-pixels 2000000` and adjust

3. **RAM Usage**:
   - Each frame is loaded into memory
   - Large mosaics may need downsampling
   - Script processes one frame at a time (low memory)

## Comparison with Other Tools

**vs FFmpeg directly:**
- ✓ Automatic timestamp extraction and overlay
- ✓ Easier size management
- ✓ Progress reporting
- ✓ Built-in size estimation

**vs Video editing software:**
- ✓ Scriptable/automated
- ✓ Handles georeferenced images
- ✓ Consistent timestamping
- ✗ Less manual control

**When to use FFmpeg instead:**
If you need advanced features (filters, transitions, audio), use FFmpeg:

```bash
ffmpeg -framerate 1 -pattern_type glob -i 'mosaics/*.tif' \
  -c:v libx264 -crf 23 -pix_fmt yuv420p output.mp4
```
