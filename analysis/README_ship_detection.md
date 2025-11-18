# Ship Detection and Tracking

Detects ships in georeferenced orthomosaic images and exports ship positions, shapes, and tracks for GIS analysis.

## Features

- **Accurate Detection**: Multiple detection methods (adaptive thresholding, Otsu, Canny edge)
- **Georeferenced Output**: Converts pixel coordinates to real-world coordinates using worldfiles
- **GIS-Ready Exports**: GeoJSON format compatible with QGIS, ArcGIS, etc.
- **Ship Metrics**: Length, beam, area, heading, centroid
- **Ship Tracking**: Creates LineString tracks showing ship movement over time
- **Visualization**: Optionally saves detection visualizations

## Requirements

```bash
pip install opencv-python numpy shapely
```

## Usage

### Single Mosaic

Detect ship in a single orthomosaic:

```bash
python ship_detection.py -i mosaics/20251016_103100/mosaic.tif -o output/ --visualize
```

### Batch Processing

Process all mosaics in a directory:

```bash
python ship_detection.py -d mosaics/ -o ship_analysis/ --method adaptive --visualize
```

### With Ship Track

Create a LineString showing ship movement:

```bash
python ship_detection.py -d output_data/20251016_Nav1/mosaics/ -o ship_analysis/ --track --visualize
```

## Detection Methods

### `adaptive` (Recommended)
- Adaptive thresholding - handles varying lighting conditions
- Best for thermal/infrared imagery
- Robust to brightness variations

### `otsu`
- Automatic threshold selection using Otsu's method
- Good for images with bimodal histograms
- Fast and simple

### `canny`
- Edge-based detection
- Good for high-contrast ship boundaries
- May need post-processing

## Parameters

### Area Constraints
Set minimum and maximum ship area to filter false detections:

```bash
python ship_detection.py -d mosaics/ -o output/ --min-area 100 --max-area 5000
```

- `--min-area`: Minimum ship area in m² (default: 50)
- `--max-area`: Maximum ship area in m² (default: 50000)

**How to determine ship area:**
- Typical lake freighter: 200-30,000 m² (e.g., 200m × 25m = 5,000 m²)
- Small vessel: 10-200 m²
- View your first detection to calibrate

### Feature Types

Export different geometric representations:

```bash
# Exact ship boundary (most accurate)
python ship_detection.py -d mosaics/ -o output/ --feature-type polygon

# Rotated bounding box (simplified)
python ship_detection.py -d mosaics/ -o output/ --feature-type bbox

# Centroid point only
python ship_detection.py -d mosaics/ -o output/ --feature-type point
```

## Output Files

### `ship_detections_polygon.geojson`
Individual ship detections with properties:
- **Geometry**: Polygon boundary in world coordinates
- **Properties**:
  - `length_m`: Ship length (m)
  - `beam_m`: Ship beam/width (m)
  - `area_m2`: Ship area (m²)
  - `heading_deg`: Ship heading (degrees, 0-180)
  - `centroid_x`, `centroid_y`: Centroid coordinates
  - `timestamp`: Image timestamp
  - `filename`: Source mosaic filename

### `ship_track.geojson`
LineString showing ship movement over time:
- **Geometry**: LineString connecting ship centroids
- **Properties**:
  - `n_points`: Number of detections
  - `total_distance_m`: Total distance traveled (m)
  - `start_time`, `end_time`: Time range

### `visualizations/`
Detection visualizations showing:
- Green: Ship boundary
- Blue: Rotated bounding box
- Red: Centroid
- Text: Dimensions and heading

### `detection_summary.json`
Processing summary with statistics

## GIS Workflow

### In QGIS

1. **Load GeoJSON**:
   - Drag `ship_detections_polygon.geojson` into QGIS
   - Or: Layer → Add Layer → Add Vector Layer

2. **Set CRS**:
   - Right-click layer → Set CRS
   - Choose UTM zone (e.g., EPSG:32616 for UTM Zone 16N)

3. **Symbolize**:
   - Right-click → Properties → Symbology
   - Fill with transparency to see orthomosaic underneath
   - Add labels showing ship dimensions

4. **Analyze Track**:
   - Load `ship_track.geojson`
   - Measure total distance
   - Calculate speed between points

5. **Overlay on Basemap**:
   - Add OpenStreetMap or satellite imagery
   - Verify ship positions

### Export to Shapefile

If needed for other GIS software:

```python
import geopandas as gpd

gdf = gpd.read_file('ship_detections_polygon.geojson')
gdf.to_file('ship_detections.shp')
```

## Accuracy Considerations

### Factors Affecting Accuracy

1. **Image Resolution**: Higher resolution → better boundary detection
   - Your mosaics: ~2.5mm/pixel is excellent for ship detection

2. **Contrast**: Ship must be distinguishable from water
   - Thermal/IR typically has good contrast
   - Adjust detection method if needed

3. **Georeference Accuracy**: Depends on camera calibration quality
   - RMS error < 5 pixels recommended
   - Check calibration before detection

### Improving Detection

**If ship not detected:**
- Lower `--min-area` threshold
- Try different `--method` (adaptive → otsu → canny)
- Check visualization to see what's being detected

**If false detections:**
- Raise `--min-area` threshold
- Lower `--max-area` to exclude large artifacts
- Use `adaptive` method for robustness

**If boundary inaccurate:**
- Check image quality and contrast
- Adjust morphological kernel sizes in code (lines 107-109)
- Increase resolution in orthorectification

### Expected Accuracy

With good calibration (RMS < 5px) and 2.5mm resolution:
- **Position accuracy**: ±1-2 cm (sub-pixel)
- **Dimension accuracy**: ±5-10 cm
- **Heading accuracy**: ±1-2 degrees
- **Suitable for**: Navigation analysis, traffic studies, clearance verification

## Troubleshooting

### "No contours found"
- Image may have poor contrast
- Try `--method otsu` or `--method canny`
- Check `--min-area` isn't too high

### "No valid contours found"
- Adjust `--min-area` and `--max-area`
- Check visualization to see what's being filtered

### "Worldfile not found"
- Ensure .tfw file exists alongside .tif
- Check that orthorectification saved worldfile

### Multiple ships detected
- Script returns largest valid contour
- Adjust area thresholds to exclude smaller objects
- For multi-ship detection, modify filtering logic

## Integration with Pipeline

Add to `pipeline/run.py` for automatic ship detection:

```python
# After mosaicking
if master_config.get('ship_detection', {}).get('enabled', False):
    ship_params = master_config['ship_detection']
    batch_process_mosaics(
        mosaic_dir=test_dir / 'mosaics',
        output_dir=test_dir / 'ship_analysis',
        method=ship_params.get('method', 'adaptive'),
        min_area=ship_params.get('min_area', 100),
        max_area=ship_params.get('max_area', 5000),
        create_track=True,
        save_visualizations=True
    )
```

Add to `master_control.json`:
```json
{
  "ship_detection": {
    "enabled": true,
    "method": "adaptive",
    "min_area": 100,
    "max_area": 5000
  }
}
```

## Advanced Usage

### Custom Processing

Modify the script for specific needs:

1. **Multiple ships**: Change line 148 to keep all valid contours
2. **Different shapes**: Fit ellipse, convex hull, or other shapes
3. **Additional metrics**: Calculate speed, acceleration, turn rate
4. **Classification**: Add ship type/size classification

### Python API

```python
from ship_detection import process_mosaic, export_to_geojson

# Process single mosaic
result = process_mosaic(
    'mosaic.tif',
    method='adaptive',
    min_area=100,
    max_area=5000,
    save_visualization=True,
    output_dir='output/'
)

# Export to GeoJSON
export_to_geojson([result], 'ships.geojson', feature_type='polygon')
```
