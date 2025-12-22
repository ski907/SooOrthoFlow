"""
Test script for timestamp reader - validates OCR on random videos.

Usage:
    python test_reader.py --dir path/to/video/directory --count 5
    python test_reader.py --dir path/to/video/directory --count 5 --show

Author: SooOrthoFlow Team
"""

import cv2
import argparse
import random
from pathlib import Path
from timestamp_reader import read_timestamp, _extract_frame_roi, _detect_nvr_type

def test_timestamp_reader(video_dir: str, count: int = 5, show_roi: bool = False):
    """
    Test timestamp reader on random videos from directory.

    Parameters:
        video_dir: Directory containing video files
        count: Number of random videos to test
        show_roi: If True, display timestamp ROI for visual verification
    """
    video_dir = Path(video_dir)

    # Find all video files
    video_files = list(video_dir.glob("**/*.avi"))
    if not video_files:
        print(f"No .avi files found in {video_dir}")
        return

    print("="*70)
    print(f"TIMESTAMP READER TEST")
    print("="*70)
    print(f"Video directory: {video_dir}")
    print(f"Total videos found: {len(video_files)}")
    print(f"Testing {min(count, len(video_files))} random videos")
    print()

    # Sample random videos
    sample_videos = random.sample(video_files, min(count, len(video_files)))

    results = []
    for i, video_path in enumerate(sample_videos, 1):
        print(f"[{i}/{len(sample_videos)}] {video_path.name}")

        # Read timestamp
        try:
            timestamp = read_timestamp(video_path)

            if timestamp:
                print(f"  ✓ Timestamp: {timestamp}")
                results.append({
                    'video': video_path.name,
                    'timestamp': timestamp,
                    'success': True
                })
            else:
                print(f"  ✗ Failed to parse timestamp")
                results.append({
                    'video': video_path.name,
                    'timestamp': None,
                    'success': False
                })

            # Optionally show ROI
            if show_roi:
                roi = _extract_frame_roi(video_path)
                if roi is not None:
                    # Scale up for visibility
                    roi_display = cv2.resize(roi, None, fx=3, fy=3,
                                            interpolation=cv2.INTER_NEAREST)

                    # Add timestamp text
                    if timestamp:
                        text = str(timestamp)
                        cv2.putText(roi_display, text, (10, roi_display.shape[0] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    cv2.imshow('Timestamp ROI - Press any key to continue', roi_display)
                    cv2.waitKey(0)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'video': video_path.name,
                'timestamp': None,
                'success': False,
                'error': str(e)
            })

        print()

    if show_roi:
        cv2.destroyAllWindows()

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    successful = sum(1 for r in results if r['success'])
    print(f"Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print()

    if successful < len(results):
        print("Failed videos:")
        for r in results:
            if not r['success']:
                error = r.get('error', 'Unknown')
                print(f"  - {r['video']}: {error}")
        print()

    # Show all timestamps
    print("All results:")
    for r in results:
        if r['success']:
            print(f"  ✓ {r['video']}: {r['timestamp']}")
        else:
            print(f"  ✗ {r['video']}: FAILED")


def main():
    parser = argparse.ArgumentParser(description='Test timestamp reader on random videos')
    parser.add_argument('--dir', required=True, help='Directory containing video files')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of random videos to test (default: 5)')
    parser.add_argument('--show', action='store_true',
                       help='Show timestamp ROI for visual verification')
    args = parser.parse_args()

    test_timestamp_reader(args.dir, args.count, args.show)


if __name__ == "__main__":
    main()
