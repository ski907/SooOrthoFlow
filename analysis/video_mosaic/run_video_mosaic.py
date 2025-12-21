#!/usr/bin/env python3
"""
Run Video Mosaic - CLI Entry Point

Streaming multi-camera mosaic video processor. Creates mosaicked video from
multiple synchronized camera sources using single-pass processing.

Usage:
    python analysis/video_mosaic/run_video_mosaic.py --config video_mosaic_config.json
    python analysis/video_mosaic/run_video_mosaic.py --config config.json --quick

Author: SooOrthoFlow Team
Version: 0.1.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.video_mosaic.video_mosaic_processor import VideoMosaicProcessor


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Streaming Multi-Camera Mosaic Video Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard processing
  python analysis/video_mosaic/run_video_mosaic.py --config video_mosaic_config.json

  # Quick test (30 seconds)
  python analysis/video_mosaic/run_video_mosaic.py --config video_mosaic_config.json --quick

  # Override parameters
  python analysis/video_mosaic/run_video_mosaic.py --config config.json \\
      --video-dir test_videos/12-12-25/Test_13 \\
      --output-dir output_data/my_mosaic

Configuration template:
  analysis/video_mosaic/video_mosaic_config_template.json

Features:
  - Single-pass processing (no intermediate files)
  - 6-12x faster than TIFF-based approaches
  - 10-60x less storage (compressed video vs TIFFs)
  - Uses pre-computed ortho caches for efficiency
  - Zone-map or center-weighted mosaicking
        """
    )

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--video-dir',
        type=str,
        help='Override video directory from config'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: process only 30 seconds for testing'
    )

    parser.add_argument(
        '--start-time',
        type=str,
        help='Override start time (YYYY-MM-DD HH:MM:SS)'
    )

    parser.add_argument(
        '--end-time',
        type=str,
        help='Override end time (YYYY-MM-DD HH:MM:SS)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        help='Override video FPS'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print(f"\nUse the template: analysis/video_mosaic/video_mosaic_config_template.json")
        sys.exit(1)

    print(f"Loading configuration: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Apply command-line overrides
    if args.video_dir:
        config['input']['video_dir'] = args.video_dir
        print(f"  Overriding video_dir: {args.video_dir}")

    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
        print(f"  Overriding output_dir: {args.output_dir}")

    if args.start_time:
        config['input']['start_time'] = args.start_time
        print(f"  Overriding start_time: {args.start_time}")

    if args.end_time:
        config['input']['end_time'] = args.end_time
        print(f"  Overriding end_time: {args.end_time}")

    if args.fps:
        config['output']['video_fps'] = args.fps
        print(f"  Overriding video_fps: {args.fps}")

    if args.quick:
        # Process only 30 seconds for quick test
        #from datetime import datetime, timedelta
        start = datetime.strptime(config['input']['start_time'], '%Y-%m-%d %H:%M:%S')
        end = start + timedelta(seconds=30)
        config['input']['end_time'] = end.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Quick mode: processing 30 seconds ({start} to {end})")

    # Create processor
    print(f"\nInitializing processor...")
    processor = VideoMosaicProcessor(config)

    # Run processing
    #start_time = 10
    start_time = datetime.now()
    success = processor.run()

    # Print elapsed time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nElapsed time: {elapsed:.1f} seconds")

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
