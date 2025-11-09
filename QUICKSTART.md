# QUICK REFERENCE - Thermal Image Processing

## During Experiment
✓ Note start time: _______________
✓ Note end time: _______________
✓ Choose test name: test_YYYYMMDD_##

## After Experiment

### Step 1: Download Videos
Create folder with test name and put videos inside:
```
test_videos/
└── test_20251016_1a/
    ├── dac1/
    ├── dac2/
    └── dac3/
```

### Step 2: Edit master_control.json
Change only these 4 lines:
```json
"video_folder": "C:\\full\\path\\to\\test_20251016_1a",
"start_time": "2025-10-16 10:31:00",
"end_time": "2025-10-16 11:45:00",
"interval": "1min"
```

### Step 3: Run
```bash
python run.py
```

### Step 4: Check Results
Look in: `data/test_20251016_1a/`
- `processing_log.txt` - Check if any errors
- `mosaics/` - Final output files

---

## If Something Goes Wrong

**Check the log:**
```
data/test_20251016_1a/processing_log.txt
```

**Re-run specific steps:**
```bash
python run.py --extract-only    # Just extract frames
python run.py --process-only    # Just ortho + mosaic
python run.py --mosaic-only     # Just mosaics
```

---

## First Time Setup (Skip if already done)

1. Copy template:
   ```bash
   cp master_control.template.json master_control.json
   ```

2. Edit paths section (one time only):
   ```json
   "paths": {
       "output_base": "data",
       "calibration_file": "calibration/camera_calibrations.pkl",
       ...
   }
   ```

3. Run calibration:
   ```bash
   python calibrate.py
   ```
