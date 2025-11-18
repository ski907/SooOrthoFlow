#!/usr/bin/env python3
"""
Convert Orthomosaics to Video

Creates video from a sequence of orthomosaic images with timestamp overlay,
automatic size management, and quality control.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import re
from datetime import datetime
import json


def parse_timestamp_from_filename(filename):
    """
    Extract timestamp from filename
    Supports formats: YYYYMMDD_HHMMSS, YYYYMMDD_HHMMSS_*, etc.
    """
    patterns = [
        r'(\d{8}_\d{6})',  # YYYYMMDD_HHMMSS
        r'(\d{8})_(\d{6})',  # Separate date and time
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 1:
                timestamp_str = match.group(1).replace('_', '')
            else:
                timestamp_str = ''.join(match.groups())

            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                return timestamp
            except ValueError:
                continue

    return None


def get_image_files(input_dir, recursive=False):
    """Find all image files in directory"""
    input_dir = Path(input_dir)
    patterns = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']

    files = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))

    return sorted(set(files))


def sort_images_by_timestamp(image_files):
    """Sort images by timestamp extracted from filename"""
    timestamped_files = []

    for img_file in image_files:
        timestamp = parse_timestamp_from_filename(img_file.name)
        if timestamp:
            timestamped_files.append((timestamp, img_file))
        else:
            print(f"Warning: No timestamp found in {img_file.name}, using filename sort")
            timestamped_files.append((datetime.min, img_file))

    timestamped_files.sort(key=lambda x: x[0])
    return [f[1] for f in timestamped_files]


def calculate_target_size(original_size, max_width=None, max_height=None, max_pixels=None):
    """
    Calculate target size respecting constraints

    Args:
        original_size: (width, height)
        max_width: Maximum width (optional)
        max_height: Maximum height (optional)
        max_pixels: Maximum total pixels (optional)

    Returns:
        (target_width, target_height)
    """
    width, height = original_size
    aspect_ratio = width / height

    # Apply max pixels constraint
    if max_pixels and width * height > max_pixels:
        scale = np.sqrt(max_pixels / (width * height))
        width = int(width * scale)
        height = int(height * scale)

    # Apply max width constraint
    if max_width and width > max_width:
        width = max_width
        height = int(width / aspect_ratio)

    # Apply max height constraint
    if max_height and height > max_height:
        height = max_height
        width = int(height * aspect_ratio)

    # Ensure dimensions are even (required by many codecs)
    width = width - (width % 2)
    height = height - (height % 2)

    return width, height


def estimate_video_size(n_frames, resolution, fps, bitrate_mbps=None, quality_preset='medium'):
    """
    Estimate output video file size

    Args:
        n_frames: Number of frames
        resolution: (width, height)
        fps: Frames per second
        bitrate_mbps: Target bitrate in Mbps (optional)
        quality_preset: 'low', 'medium', 'high', 'very_high'

    Returns:
        Estimated size in MB
    """
    width, height = resolution
    duration_sec = n_frames / fps
    pixels_per_frame = width * height

    # Typical bitrates for H.264 (bits per pixel per frame)
    bpp_values = {
        'low': 0.05,
        'medium': 0.1,
        'high': 0.15,
        'very_high': 0.2
    }

    if bitrate_mbps:
        bitrate = bitrate_mbps * 1_000_000  # Convert to bps
    else:
        bpp = bpp_values.get(quality_preset, 0.1)
        bitrate = pixels_per_frame * fps * bpp

    size_mb = (bitrate * duration_sec) / (8 * 1024 * 1024)
    return size_mb


def add_timestamp_overlay(frame, timestamp, position='top-left',
                          font_scale=1.0, color=(255, 255, 255),
                          background=True):
    """
    Add timestamp overlay to frame

    Args:
        frame: Image frame
        timestamp: datetime object
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        font_scale: Text size multiplier
        color: Text color (BGR)
        background: Add semi-transparent background
    """
    if timestamp is None:
        return frame

    text = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * font_scale))

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate position
    margin = int(10 * font_scale)
    height, width = frame.shape[:2]

    if position == 'top-left':
        x, y = margin, margin + text_height
    elif position == 'top-right':
        x, y = width - text_width - margin, margin + text_height
    elif position == 'bottom-left':
        x, y = margin, height - margin
    elif position == 'bottom-right':
        x, y = width - text_width - margin, height - margin
    else:
        x, y = margin, margin + text_height

    # Add background rectangle
    if background:
        padding = int(5 * font_scale)
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Add text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def create_video_from_mosaics(input_dir, output_file,
                              fps=1, max_width=None, max_height=None,
                              max_pixels=None, codec='h264',
                              quality_preset='medium', bitrate_mbps=None,
                              add_timestamps=True, timestamp_position='top-left',
                              recursive=False):
    """
    Create video from mosaic images

    Args:
        input_dir: Directory containing mosaic images
        output_file: Output video file path
        fps: Frames per second
        max_width: Maximum video width
        max_height: Maximum video height
        max_pixels: Maximum pixels per frame (e.g., 1920*1080 = 2073600)
        codec: Video codec ('h264', 'h265', 'xvid', 'mjpeg')
        quality_preset: 'low', 'medium', 'high', 'very_high'
        bitrate_mbps: Target bitrate in Mbps (overrides quality_preset)
        add_timestamps: Add timestamp overlay
        timestamp_position: Position for timestamp overlay
        recursive: Search subdirectories

    Returns:
        True if successful
    """
    # Find and sort images
    image_files = get_image_files(input_dir, recursive)

    if not image_files:
        print(f"No images found in {input_dir}")
        return False

    print(f"Found {len(image_files)} image(s)")

    # Sort by timestamp
    image_files = sort_images_by_timestamp(image_files)

    # Load first image to determine size
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"Error: Could not load {image_files[0]}")
        return False

    original_height, original_width = first_img.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")

    # Calculate target size
    target_width, target_height = calculate_target_size(
        (original_width, original_height),
        max_width, max_height, max_pixels
    )

    print(f"Video size: {target_width}x{target_height}")
    print(f"Frame rate: {fps} fps")
    print(f"Duration: {len(image_files)/fps:.1f} seconds")

    # Estimate file size
    estimated_size = estimate_video_size(
        len(image_files), (target_width, target_height),
        fps, bitrate_mbps, quality_preset
    )
    print(f"Estimated output size: {estimated_size:.1f} MB")

    # Set up video writer with fallback options
    codec_map = {
        'h264': ['H264', 'X264', 'avc1', 'mp4v'],  # Multiple fallbacks for H.264
        'h265': ['hvc1', 'HEVC'],  # H.265/HEVC
        'xvid': ['XVID'],  # Xvid MPEG-4
        'mjpeg': ['MJPG']  # Motion JPEG
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try each codec variant until one works
    fourcc_options = codec_map.get(codec.lower(), ['mp4v'])
    writer = None

    for fourcc_str in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (target_width, target_height)
            )

            if writer.isOpened():
                print(f"Using codec: {fourcc_str}")
                break
            else:
                writer = None
        except Exception as e:
            continue

    if writer is None or not writer.isOpened():
        print(f"Error: Could not create video writer")
        print(f"Tried codecs: {', '.join(fourcc_options)}")
        print(f"\nTroubleshooting:")
        print(f"  1. Try --codec xvid (most compatible on Windows)")
        print(f"  2. Try --codec mjpeg (always works but large files)")
        print(f"  3. Install ffmpeg: 'pip install opencv-python-headless'")
        return False

    print(f"\nCreating video: {output_path.name}")
    print("="*60)

    # Process each image
    for i, img_file in enumerate(image_files, 1):
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not load {img_file.name}, skipping")
            continue

        # Resize if needed
        if img.shape[1] != target_width or img.shape[0] != target_height:
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Add timestamp overlay
        if add_timestamps:
            timestamp = parse_timestamp_from_filename(img_file.name)
            if timestamp:
                # Calculate font scale based on image size
                font_scale = target_width / 1920.0  # Scale for 1920px width baseline
                img = add_timestamp_overlay(img, timestamp,
                                          position=timestamp_position,
                                          font_scale=font_scale)

        # Write frame
        writer.write(img)

        # Progress
        if i % 10 == 0 or i == len(image_files):
            progress = (i / len(image_files)) * 100
            print(f"Progress: {i}/{len(image_files)} frames ({progress:.1f}%)")

    writer.release()

    # Get actual file size
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    print("="*60)
    print(f"✓ Video created: {output_path}")
    print(f"  Frames: {len(image_files)}")
    print(f"  Resolution: {target_width}x{target_height}")
    print(f"  Duration: {len(image_files)/fps:.1f}s")
    print(f"  File size: {actual_size_mb:.1f} MB")
    print(f"  Codec: {codec.upper()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create video from orthomosaic images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - 1 fps, auto-size
  python mosaics_to_video.py -i mosaics/ -o ship_video.mp4

  # 5 fps with HD resolution
  python mosaics_to_video.py -i mosaics/ -o output.mp4 --fps 5 --max-width 1920 --max-height 1080

  # Limit to 5 megapixels per frame (reduces file size)
  python mosaics_to_video.py -i mosaics/ -o output.mp4 --max-pixels 5000000

  # High quality H.265 codec
  python mosaics_to_video.py -i mosaics/ -o output.mp4 --codec h265 --quality high

  # Custom bitrate
  python mosaics_to_video.py -i mosaics/ -o output.mp4 --bitrate 10

  # No timestamps
  python mosaics_to_video.py -i mosaics/ -o output.mp4 --no-timestamps

Quality presets:
  low      - Small file size, lower quality (~0.05 bits/pixel)
  medium   - Balanced (default) (~0.1 bits/pixel)
  high     - High quality (~0.15 bits/pixel)
  very_high - Maximum quality (~0.2 bits/pixel)

Codec options:
  h264  - H.264/AVC (most compatible, good compression)
  h265  - H.265/HEVC (best compression, less compatible)
  xvid  - Xvid MPEG-4 (good compatibility)
  mjpeg - Motion JPEG (large files, highest quality)
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Input directory with mosaics')
    parser.add_argument('-o', '--output', required=True,
                       help='Output video file (.mp4, .avi, etc.)')
    parser.add_argument('--fps', type=float, default=1,
                       help='Frames per second (default: 1)')
    parser.add_argument('--max-width', type=int,
                       help='Maximum video width in pixels')
    parser.add_argument('--max-height', type=int,
                       help='Maximum video height in pixels')
    parser.add_argument('--max-pixels', type=int,
                       help='Maximum pixels per frame (e.g., 2073600 for 1920x1080)')
    parser.add_argument('--codec', choices=['h264', 'h265', 'xvid', 'mjpeg'],
                       default='h264',
                       help='Video codec (default: h264)')
    parser.add_argument('--quality', choices=['low', 'medium', 'high', 'very_high'],
                       default='medium',
                       help='Quality preset (default: medium)')
    parser.add_argument('--bitrate', type=float,
                       help='Target bitrate in Mbps (overrides quality preset)')
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Do not add timestamp overlay')
    parser.add_argument('--timestamp-position',
                       choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
                       default='top-left',
                       help='Timestamp position (default: top-left)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Search subdirectories for images')

    args = parser.parse_args()

    success = create_video_from_mosaics(
        input_dir=args.input,
        output_file=args.output,
        fps=args.fps,
        max_width=args.max_width,
        max_height=args.max_height,
        max_pixels=args.max_pixels,
        codec=args.codec,
        quality_preset=args.quality,
        bitrate_mbps=args.bitrate,
        add_timestamps=not args.no_timestamps,
        timestamp_position=args.timestamp_position,
        recursive=args.recursive
    )

    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
