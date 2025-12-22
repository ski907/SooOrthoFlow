# Frame Timestamp Extraction

Simple, lean utility to read burned-in timestamps from NVR video frames. Video file metadata timestamps are unreliable - this module reads the actual timestamp displayed in the video frame.

## Quick Start

```python
from frame_timestamp_extraction.timestamp_reader import read_timestamp

# Single video
timestamp = read_timestamp("path/to/NVR2_N910A6_ch4_main.avi")
print(f"Video starts at: {timestamp}")

# Batch processing
from pathlib import Path

video_dir = Path("test_videos/12-17-25/Test 23")
for video_path in video_dir.glob("*.avi"):
    ts = read_timestamp(video_path)
    print(f"{video_path.name}: {ts}")
```

## Setup (One-Time)

### 1. Templates Already Exist?

If templates are already generated for your NVR type, you're done! Just use `read_timestamp()`.

Check: `analysis/frame_timestamp_extraction/templates/NVR2_3/` (or `NVR1/`)

### 2. Generate Templates (if needed)

Run the interactive template generation tool:

```bash
cd analysis/frame_timestamp_extraction
python generate_templates.py --video path/to/reference/NVR2_video.avi --nvr NVR2_3
```

**Interactive workflow:**
1. Window opens showing timestamp ROI
2. Click on each character in order: `MM-DD-YYYY HH:MM:SS AP`
3. Templates saved to `templates/NVR2_3/`

**Tips:**
- Use a video with clear, high-quality timestamp
- Click center of each character
- Press `r` to reset if you make a mistake
- Press `q` when done

## How It Works

1. **Auto-detect NVR type** from filename (`NVR1_`, `NVR2_`, `NVR3_`)
2. **Extract timestamp ROI** (top 4-7%, right 20% of frame)
3. **Template matching** using OpenCV `cv2.matchTemplate`
4. **Parse timestamp** to Python datetime object

## Files

```
frame_timestamp_extraction/
├── timestamp_reader.py              # Main utility (use this)
├── generate_templates.py            # One-time template generator
├── config/
│   ├── slot_definitions_NVR1.json   # Character positions for NVR1
│   └── slot_definitions_NVR2_3.json # Character positions for NVR2/NVR3
└── templates/
    ├── NVR1/                        # Templates for NVR1
    └── NVR2_3/                      # Templates for NVR2 and NVR3
```

## Template Matching Details

- **Method**: Normalized cross-correlation (`TM_CCOEFF_NORMED`)
- **Template size**: 20×32 pixels
- **Threshold**: 0.65 (adjustable in `timestamp_reader.py`)
- **Preprocessing**: Grayscale, binary threshold at 200
- **Characters**: 0-9, A, M, P

## Troubleshooting

**"Cannot detect NVR type from filename"**
- Filename must contain `NVR1_`, `NVR2_`, or `NVR3_`

**"Template directory not found"**
- Run `generate_templates.py` to create templates

**"Failed to read timestamp" / returns None**
- Check that templates exist for your NVR type
- Verify video file can be opened
- Check that timestamp is visible in video frame

**Characters recognized as '?'**
- Template quality may be poor
- Regenerate templates from clearer reference video
- Check `MATCH_THRESHOLD` in `timestamp_reader.py` (try lowering to 0.5)

## Performance

- **Speed**: <100ms per video (with template caching)
- **Memory**: ~1 MB per NVR type (templates cached in memory)
- **Batch processing**: Reuses templates across multiple videos

## Example: Command-Line Usage

```bash
python timestamp_reader.py path/to/video.avi
```

Output:
```
Timestamp: 2025-12-17 12:00:01
```

## Notes

- NVR2 and NVR3 share the same timestamp layout (both use `NVR2_3` templates)
- Timestamps are read from frame 0 (first frame) by default
- Specify different frame: `read_timestamp(video_path, frame_index=30)`
- Returns `None` if parsing fails (check console output for warnings)
