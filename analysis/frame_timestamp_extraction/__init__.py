"""
Frame Timestamp Extraction - Read burned-in timestamps from NVR video frames.

Simple, lean utility to extract precise timestamps from video frames.
Video file metadata is unreliable - this reads the actual timestamp displayed in the frame.

Usage:
    from frame_timestamp_extraction.timestamp_reader import read_timestamp

    timestamp = read_timestamp("path/to/NVR2_ch4_main.avi")
    print(f"Video starts at: {timestamp}")

Author: SooOrthoFlow Team
"""

from .timestamp_reader import read_timestamp

__all__ = ['read_timestamp']
__version__ = '1.0.0'
