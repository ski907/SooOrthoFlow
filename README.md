# Thermal Image Processing Pipeline

## Setup (One Time)

1. **Copy template config:**
   ```bash
   cp master_control.template.json master_control.json
   ```

2. **Edit `master_control.json`** - Set your paths in the "paths" section (do this once)

3. **Run initial calibration:**
   ```bash
   python calibrate.py
   ```
   Enter path to calibration images when prompted.

## Per Test Workflow

### 1. Organize Videos
Put videos in a folder named after your test:
```
test_videos/
└── test_20251016_1a/       ← Folder name becomes test ID
    ├── dac1/
    ├── dac2/
    └── dac3/
```

### 2. Edit Config
In `master_control.json`, update only these 4 fields:
```json
{
    "video_folder": "C:\\path\\to\\test_videos\\test_20251016_1a",
    "start_time": "2025-10-16 10:31:00",
    "end_time": "2025-10-16 11:45:00",
    "interval": "1min"
}
```
The test_id is automatically extracted from the folder name.

### 3. Run Pipeline
```bash
python run.py
```

That's it! Results will be in `data/test_20251016_1a/`

## Partial Runs (for troubleshooting)

```bash
python run.py --extract-only      # Just extract frames
python run.py --process-only      # Just orthorectify + mosaic (skip extraction)
python run.py --mosaic-only       # Just create mosaics (skip extraction + ortho)
```

## Periodic Recalibration

```bash
python recalibrate.py
```
Enter path to recent good frames when prompted.

## Output Structure

```
data/
└── test_20251016_1a/
    ├── master_control.json      # Config used
    ├── time_config.json         # Generated config
    ├── processing_log.txt       # What happened
    ├── frames/
    │   ├── 20251016103100/
    │   ├── 20251016103200/
    │   └── ...
    ├── orthos/
    │   ├── 20251016103100/
    │   └── ...
    └── mosaics/
        ├── mosaic_20251016103100.tif
        └── ...
```
