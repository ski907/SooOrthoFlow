#!/usr/bin/env python3
"""
Ship Tracking with Template Fitting

Detects ship position/orientation from thermal mosaics and fits a rigid template.
Outputs consistent ship shape across all frames (not amorphous blobs).

Two-phase approach:
1. Detect ship position from background subtraction (filters static ice)
2. Fit rigid template (rectangle or custom shape) to detected position/orientation

This filters out wake effects and produces uniform ship outlines for GIS analysis.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import re


def read_worldfile(tfw_path):
    """Read ESRI worldfile (.tfw) and return geotransform parameters"""
    with open(tfw_path, 'r') as f:
        lines = f.readlines()

    return {
        'pixel_width': float(lines[0].strip()),
        'rotation_y': float(lines[1].strip()),
        'rotation_x': float(lines[2].strip()),
        'pixel_height': float(lines[3].strip()),
        'x_origin': float(lines[4].strip()),
        'y_origin': float(lines[5].strip())
    }


def pixel_to_world(col, row, geotransform):
    """Convert pixel coordinates to world coordinates"""
    x = geotransform['x_origin'] + col * geotransform['pixel_width'] + row * geotransform['rotation_x']
    y = geotransform['y_origin'] + col * geotransform['rotation_y'] + row * geotransform['pixel_height']
    return x, y


def world_to_pixel(x, y, geotransform):
    """Convert world coordinates to pixel coordinates"""
    # Inverse transformation
    det = geotransform['pixel_width'] * geotransform['pixel_height'] - geotransform['rotation_x'] * geotransform['rotation_y']

    dx = x - geotransform['x_origin']
    dy = y - geotransform['y_origin']

    col = (geotransform['pixel_height'] * dx - geotransform['rotation_x'] * dy) / det
    row = (geotransform['pixel_width'] * dy - geotransform['rotation_y'] * dx) / det

    return col, row


def parse_timestamp_from_filename(filename):
    """Extract timestamp from filename"""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            pass
    return None


def create_background_model(mosaic_paths, sample_size=10, method='median'):
    """
    Create background model from multiple frames

    Args:
        mosaic_paths: List of mosaic file paths
        sample_size: Number of frames to use
        method: 'median' or 'mean'

    Returns:
        Background image (grayscale)
    """
    print(f"Creating background model from {sample_size} frames...")

    # Select evenly spaced frames
    if len(mosaic_paths) > sample_size:
        indices = np.linspace(0, len(mosaic_paths)-1, sample_size, dtype=int)
        sample_paths = [mosaic_paths[i] for i in indices]
    else:
        sample_paths = mosaic_paths

    # Load images
    images = []
    for path in sample_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            print(f"  Loaded {path.name}")

    if not images:
        return None

    # Stack and compute background
    stack = np.stack(images, axis=0)

    if method == 'median':
        background = np.median(stack, axis=0).astype(np.uint8)
    else:
        background = np.mean(stack, axis=0).astype(np.uint8)

    print(f"✓ Background model created ({method})")
    return background


def detect_ship_region(img, background, threshold=30, min_area_px=100, max_area_px=100000):
    """
    Detect ship region using background subtraction

    Args:
        img: Current frame (grayscale)
        background: Background model (grayscale)
        threshold: Difference threshold
        min_area_px: Minimum ship area in pixels
        max_area_px: Maximum ship area in pixels

    Returns:
        (centroid_px, orientation_deg, main_contour) or (None, None, None)
    """
    # Subtract background
    diff = cv2.absdiff(img, background)

    # Threshold
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to isolate ship hull from wake
    # Close small gaps in ship hull
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # Remove small noise and thin wake trails
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # Filter by area and get largest
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_px <= area <= max_area_px:
            valid_contours.append((area, contour))

    if not valid_contours:
        return None, None, None

    # Get largest valid contour (main ship body)
    valid_contours.sort(reverse=True)
    ship_area_px, ship_contour = valid_contours[0]

    # Get centroid
    M = cv2.moments(ship_contour)
    if M['m00'] == 0:
        return None, None, None

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    # Get orientation using minimum area rectangle
    rect = cv2.minAreaRect(ship_contour)
    (rect_cx, rect_cy), (width, height), angle = rect

    # Ensure angle represents heading (long axis direction)
    if width < height:
        angle = (angle + 90) % 180

    return (cx, cy), angle, ship_contour


def create_ship_template(length_m, beam_m, shape='rectangle', custom_points=None):
    """
    Create ship template in local coordinates (origin at center)

    Args:
        length_m: Ship length in meters
        beam_m: Ship beam in meters
        shape: 'rectangle' or 'ship'
        custom_points: Custom shape as list of (x, y) in meters

    Returns:
        Template points as numpy array (N, 2) in meters, centered at origin
    """
    if custom_points is not None:
        # Use custom shape
        template = np.array(custom_points, dtype=np.float64)
        # Center it
        centroid = template.mean(axis=0)
        template = template - centroid
        return template

    if shape == 'rectangle':
        # Simple rectangle
        half_length = length_m / 2.0
        half_beam = beam_m / 2.0
        template = np.array([
            [-half_length, -half_beam],  # Stern port
            [half_length, -half_beam],   # Bow port
            [half_length, half_beam],    # Bow starboard
            [-half_length, half_beam]    # Stern starboard
        ], dtype=np.float64)

    elif shape == 'ship':
        # Ship-shaped: pointed bow, flat stern
        half_beam = beam_m / 2.0
        stern_pos = -length_m / 2.0
        bow_pos = length_m / 2.0

        # Bow taper (20% of length)
        taper_start = bow_pos - 0.2 * length_m

        template = np.array([
            [stern_pos, -half_beam],      # Stern port corner
            [stern_pos, half_beam],       # Stern starboard corner
            [taper_start, half_beam],     # Start of bow taper starboard
            [bow_pos, 0],                 # Bow point (center)
            [taper_start, -half_beam],    # Start of bow taper port
        ], dtype=np.float64)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return template


def transform_template(template, centroid_world, angle_deg, geotransform):
    """
    Transform template to world coordinates

    Args:
        template: Template points (N, 2) in meters, centered at origin
        centroid_world: Ship centroid in world coordinates (x, y)
        angle_deg: Ship heading in degrees (0 = north, clockwise)
        geotransform: Worldfile parameters

    Returns:
        Transformed template in world coordinates
    """
    # Convert angle to radians (need to adjust for coordinate system)
    # In image space, angle=0 is horizontal (east), positive is clockwise
    # We want angle=0 to be north (up), so rotate by 90 degrees
    angle_rad = np.deg2rad(angle_deg - 90)

    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    # Rotate template
    rotated = template @ rotation.T

    # Translate to centroid
    cx, cy = centroid_world
    transformed = rotated + np.array([cx, cy])

    return transformed


def track_ships_with_template(mosaic_dir, output_dir,
                               ship_length_m, ship_beam_m,
                               template_shape='rectangle',
                               bg_sample_size=10, bg_method='median',
                               threshold=30, min_area=50, max_area=50000,
                               visualize=False):
    """
    Track ships and fit rigid template

    Args:
        mosaic_dir: Directory with mosaic TIF files
        output_dir: Output directory
        ship_length_m: Ship length in meters
        ship_beam_m: Ship beam in meters
        template_shape: 'rectangle' or 'ship'
        bg_sample_size: Frames for background model
        bg_method: 'median' or 'mean'
        threshold: Detection threshold
        min_area: Min ship area in m²
        max_area: Max ship area in m²
        visualize: Save visualization images

    Returns:
        List of detection results
    """
    mosaic_dir = Path(mosaic_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find mosaics
    mosaic_files = sorted(list(mosaic_dir.glob('*.tif')) + list(mosaic_dir.glob('*.tiff')))

    if not mosaic_files:
        print(f"No mosaics found in {mosaic_dir}")
        return []

    print(f"Found {len(mosaic_files)} mosaics")
    print(f"Ship template: {template_shape} ({ship_length_m}m × {ship_beam_m}m)")
    print(f"Detection threshold: {threshold}")
    print("="*60)

    # Create background model
    background = create_background_model(mosaic_files, bg_sample_size, bg_method)

    if background is None:
        print("Failed to create background model")
        return []

    # Save background for reference
    cv2.imwrite(str(output_dir / 'background_model.png'), background)
    print(f"Saved background model\n")

    # Create ship template (in meters, centered at origin)
    ship_template = create_ship_template(ship_length_m, ship_beam_m, template_shape)

    # Process each mosaic
    results = []

    for i, mosaic_path in enumerate(mosaic_files, 1):
        print(f"[{i}/{len(mosaic_files)}] {mosaic_path.name}")

        # Load image
        img = cv2.imread(str(mosaic_path))
        if img is None:
            print(f"  ✗ Could not load")
            continue

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Load worldfile
        tfw_path = mosaic_path.with_suffix('.tfw')
        if not tfw_path.exists():
            print(f"  ✗ No worldfile")
            continue

        geotransform = read_worldfile(tfw_path)

        # Calculate area thresholds in pixels
        pixel_area = abs(geotransform['pixel_width'] * geotransform['pixel_height'])
        min_area_px = int(min_area / pixel_area)
        max_area_px = int(max_area / pixel_area)

        # Detect ship region
        centroid_px, orientation_deg, ship_contour = detect_ship_region(
            gray, background, threshold, min_area_px, max_area_px
        )

        if centroid_px is None:
            print(f"  - No ship detected")
            continue

        # Convert centroid to world coordinates
        cx_world, cy_world = pixel_to_world(centroid_px[0], centroid_px[1], geotransform)

        # Transform template to world coordinates
        template_world = transform_template(
            ship_template, (cx_world, cy_world), orientation_deg, geotransform
        )

        # Convert template back to pixels for visualization
        template_px = np.array([
            world_to_pixel(pt[0], pt[1], geotransform) for pt in template_world
        ], dtype=np.float32)

        # Store result
        timestamp = parse_timestamp_from_filename(mosaic_path.name)

        result = {
            'filename': mosaic_path.name,
            'timestamp': timestamp.isoformat() if timestamp else None,
            'centroid_world': [float(cx_world), float(cy_world)],
            'centroid_pixel': [float(centroid_px[0]), float(centroid_px[1])],
            'heading_deg': float(orientation_deg),
            'template_world': template_world.tolist(),
            'template_pixel': template_px.tolist(),
            'ship_length_m': float(ship_length_m),
            'ship_beam_m': float(ship_beam_m),
            'geotransform': geotransform
        }

        results.append(result)

        print(f"  ✓ Ship tracked: position ({cx_world:.1f}, {cy_world:.1f}), heading {orientation_deg:.1f}°")

        # Visualization
        if visualize:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)

            # Create visualization
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Show difference from background
            diff = cv2.absdiff(gray, background)
            diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.6, diff_colored, 0.4, 0)

            # Draw detected contour (in red)
            if ship_contour is not None:
                cv2.drawContours(vis, [ship_contour], -1, (0, 0, 255), 2)

            # Draw template (in green, thick)
            template_px_int = template_px.astype(np.int32)
            cv2.polylines(vis, [template_px_int], isClosed=True, color=(0, 255, 0), thickness=4)

            # Draw centroid
            cv2.circle(vis, (int(centroid_px[0]), int(centroid_px[1])), 10, (255, 0, 255), -1)

            # Draw heading line
            heading_length = 50
            end_x = int(centroid_px[0] + heading_length * np.cos(np.deg2rad(orientation_deg)))
            end_y = int(centroid_px[1] + heading_length * np.sin(np.deg2rad(orientation_deg)))
            cv2.arrowedLine(vis, (int(centroid_px[0]), int(centroid_px[1])),
                          (end_x, end_y), (255, 255, 0), 3, tipLength=0.3)

            # Add text
            text = f"Heading: {orientation_deg:.1f}deg  |  {ship_length_m}m x {ship_beam_m}m"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imwrite(str(vis_dir / f"{mosaic_path.stem}_track.png"), vis)

    print("\n" + "="*60)
    print(f"Tracked ship in {len(results)}/{len(mosaic_files)} frames")
    print("="*60)

    return results


def export_to_geojson(results, output_path, feature_type='polygon'):
    """
    Export ship detections to GeoJSON

    Args:
        results: List of detection results
        output_path: Output GeoJSON file
        feature_type: 'polygon', 'point', or 'both'
    """
    features = []

    for result in results:
        template_coords = result['template_world']

        # Close polygon (first point = last point)
        if template_coords[0] != template_coords[-1]:
            template_coords.append(template_coords[0])

        if feature_type in ['polygon', 'both']:
            # Ship template as polygon
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [template_coords]
                },
                'properties': {
                    'filename': result['filename'],
                    'timestamp': result['timestamp'],
                    'heading_deg': result['heading_deg'],
                    'length_m': result['ship_length_m'],
                    'beam_m': result['ship_beam_m'],
                    'centroid_x': result['centroid_world'][0],
                    'centroid_y': result['centroid_world'][1]
                }
            }
            features.append(feature)

        if feature_type in ['point', 'both']:
            # Centroid as point
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': result['centroid_world']
                },
                'properties': {
                    'filename': result['filename'],
                    'timestamp': result['timestamp'],
                    'heading_deg': result['heading_deg'],
                    'length_m': result['ship_length_m'],
                    'beam_m': result['ship_beam_m']
                }
            }
            features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"✓ Exported {len(results)} detections to {output_path}")


def create_ship_track(results, output_path):
    """
    Create ship track as LineString

    Args:
        results: List of detection results
        output_path: Output GeoJSON file
    """
    if len(results) < 2:
        print("Need at least 2 detections to create track")
        return

    # Extract centroids in order
    coordinates = [result['centroid_world'] for result in results]

    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'LineString',
            'coordinates': coordinates
        },
        'properties': {
            'n_points': len(coordinates),
            'start_time': results[0]['timestamp'],
            'end_time': results[-1]['timestamp']
        }
    }

    geojson = {
        'type': 'FeatureCollection',
        'features': [feature]
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"✓ Exported ship track to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Ship tracking with rigid template fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking with rectangle template
  python ship_tracker_template.py -d mosaics/ -o output/ --length 100 --beam 20

  # Ship-shaped template
  python ship_tracker_template.py -d mosaics/ -o output/ --length 100 --beam 20 --shape ship

  # Adjust detection sensitivity
  python ship_tracker_template.py -d mosaics/ -o output/ --length 100 --beam 20 --threshold 20

  # Save visualizations
  python ship_tracker_template.py -d mosaics/ -o output/ --length 100 --beam 20 --visualize

Output files:
  - ship_positions.geojson: Template polygons at each detected position
  - ship_track.geojson: Ship path as LineString
  - background_model.png: Background model used for detection
  - visualizations/*.png: Detection visualizations (if --visualize)

The output is always a uniform ship template (not blob detection),
positioned and oriented to match the detected ship location.
        """
    )

    parser.add_argument('-d', '--directory', required=True,
                       help='Directory containing mosaic TIF files')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory')
    parser.add_argument('--length', type=float, required=True,
                       help='Ship length in meters')
    parser.add_argument('--beam', type=float, required=True,
                       help='Ship beam (width) in meters')
    parser.add_argument('--shape', choices=['rectangle', 'ship'],
                       default='rectangle',
                       help='Template shape (default: rectangle)')
    parser.add_argument('--threshold', type=int, default=30,
                       help='Detection threshold 0-255 (default: 30, lower=more sensitive)')
    parser.add_argument('--min-area', type=float, default=50,
                       help='Minimum ship area in m² (default: 50)')
    parser.add_argument('--max-area', type=float, default=50000,
                       help='Maximum ship area in m² (default: 50000)')
    parser.add_argument('--bg-samples', type=int, default=10,
                       help='Frames for background model (default: 10)')
    parser.add_argument('--bg-method', choices=['median', 'mean'],
                       default='median',
                       help='Background model method (default: median)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save detection visualizations')

    args = parser.parse_args()

    # Track ships
    results = track_ships_with_template(
        mosaic_dir=args.directory,
        output_dir=args.output,
        ship_length_m=args.length,
        ship_beam_m=args.beam,
        template_shape=args.shape,
        bg_sample_size=args.bg_samples,
        bg_method=args.bg_method,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        visualize=args.visualize
    )

    if not results:
        print("\nNo ships detected")
        return 1

    # Export results
    output_dir = Path(args.output)

    # Ship positions as polygons
    export_to_geojson(results, output_dir / 'ship_positions.geojson', feature_type='polygon')

    # Ship track as LineString
    if len(results) >= 2:
        create_ship_track(results, output_dir / 'ship_track.geojson')

    print("\n✓ Ship tracking complete!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
