"""
Camera Video Reader - Frame Synchronization

Reads synchronized frames from multiple camera videos at specific timestamps.
Uses time-based seeking for accurate frame extraction from compressed video formats.

Author: SooOrthoFlow Team
Version: 0.1.0
"""

import cv2
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


class CameraVideoReader:
    """
    Reads synchronized frames from multiple camera videos.

    Handles:
    - Multiple video files per camera (time-segmented recordings)
    - Camera-specific time offsets
    - Time-based seeking for accurate frame extraction
    """

    def __init__(self, video_dir: Path, camera_ids: List[str], camera_time_offsets: Dict[str, float] = None):
        """
        Initialize camera video reader.

        Parameters:
            video_dir: Directory containing video files or NVR subdirectories
            camera_ids: List of camera IDs (e.g., ["NVR1_N910A6_ch4_main", "NVR1_N910A6_ch5_main"])
            camera_time_offsets: Dict mapping NVR names to time offsets in seconds
                                (e.g., {'NVR2': 15.0} means NVR2 videos are 15s ahead)
        """
        self.video_dir = Path(video_dir)
        self.camera_ids = camera_ids
        self.camera_time_offsets = camera_time_offsets or {}

        # Discover video files for each camera
        self.camera_videos = {}
        for camera_id in camera_ids:
            videos = self._find_videos_for_camera(camera_id)
            if videos:
                self.camera_videos[camera_id] = videos
            else:
                print(f"WARNING: No videos found for camera {camera_id}")

        # Currently opened video captures
        self.current_captures = {}
        self.current_video_info = {}

    def _find_videos_for_camera(self, camera_id: str) -> List[Dict]:
        """
        Find all video files for a camera.

        Parameters:
            camera_id: Camera ID (e.g., "NVR1_N910A6_ch4_main")

        Returns:
            List of video info dicts with {filename, start_time, end_time, camera_id}
        """
        # Parse camera_id to find search paths
        parts = camera_id.split('_')
        if len(parts) >= 2:
            nvr_name = parts[0]  # e.g., "NVR1"
            serial_channel_stream = '_'.join(parts[1:])  # e.g., "N910A6_ch4_main"
        else:
            nvr_name = None
            serial_channel_stream = camera_id

        # Search directories
        search_dirs = []
        if nvr_name and (self.video_dir / nvr_name).exists():
            search_dirs.append(self.video_dir / nvr_name)
        search_dirs.append(self.video_dir)

        # Video extensions
        video_exts = ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.m4v']

        videos = []
        for search_dir in search_dirs:
            for ext in video_exts:
                pattern = f"*{serial_channel_stream}*{ext}"
                for video_path in search_dir.glob(pattern):
                    video_info = self._parse_video_filename(video_path, camera_id)
                    if video_info:
                        videos.append(video_info)

        # Sort by start time
        videos.sort(key=lambda v: v['start_time'])

        return videos

    def _parse_video_filename(self, video_path: Path, camera_id: str) -> Optional[Dict]:
        """
        Parse video filename to extract start and end times.

        Expected format: CAMERA_STARTTIME_ENDTIME.ext
        where STARTTIME and ENDTIME are YYYYMMDDHHMMSS

        Parameters:
            video_path: Path to video file
            camera_id: Camera ID for this video

        Returns:
            Dict with {filename, start_time, end_time, camera_id} or None
        """
        # Look for pattern with two 14-digit datetime strings
        match = re.search(r'_(\d{14})_(\d{14})', video_path.stem)
        if not match:
            return None

        try:
            start_time_str = match.group(1)
            end_time_str = match.group(2)

            start_time = datetime.strptime(start_time_str, '%Y%m%d%H%M%S')
            end_time = datetime.strptime(end_time_str, '%Y%m%d%H%M%S')

            return {
                'filename': str(video_path),
                'start_time': start_time,
                'end_time': end_time,
                'camera_id': camera_id
            }
        except ValueError:
            return None

    def _find_video_for_timestamp(self, camera_id: str, timestamp: datetime) -> Optional[Dict]:
        """
        Find video file that contains the target timestamp.

        Parameters:
            camera_id: Camera ID
            timestamp: Target timestamp

        Returns:
            Video info dict or None
        """
        if camera_id not in self.camera_videos:
            return None

        for video_info in self.camera_videos[camera_id]:
            if video_info['start_time'] <= timestamp <= video_info['end_time']:
                return video_info

        return None

    def _get_camera_time_offset(self, camera_id: str) -> float:
        """
        Get time offset for camera based on NVR name.

        Parameters:
            camera_id: Camera ID (e.g., "NVR1_N910A6_ch4_main")

        Returns:
            Time offset in seconds
        """
        # Extract NVR name from camera_id
        nvr_name = camera_id.split('_')[0]
        return self.camera_time_offsets.get(nvr_name, 0.0)

    def get_frames_at_timestamp(self, timestamp: datetime) -> Optional[Dict[str, 'numpy.ndarray']]:
        """
        Extract synchronized frames from available cameras at specified timestamp.

        Skips cameras that are missing videos or frames (graceful degradation).

        Parameters:
            timestamp: Target timestamp

        Returns:
            Dict mapping camera_id to frame (numpy array), or None if NO cameras succeeded
        """
        frames = {}

        for camera_id in self.camera_ids:
            try:
                # Find video containing this timestamp
                video_info = self._find_video_for_timestamp(camera_id, timestamp)
                if not video_info:
                    # Skip this camera, continue with others
                    continue

                # Open video if not already open or if different video
                if (camera_id not in self.current_captures or
                    self.current_video_info.get(camera_id) != video_info):

                    # Close old capture if exists
                    if camera_id in self.current_captures:
                        self.current_captures[camera_id].release()

                    # Open new video
                    cap = cv2.VideoCapture(video_info['filename'])
                    if not cap.isOpened():
                        # Skip this camera, continue with others
                        continue

                    self.current_captures[camera_id] = cap
                    self.current_video_info[camera_id] = video_info
                else:
                    cap = self.current_captures[camera_id]

                # Calculate time offset from video start
                time_offset = (timestamp - video_info['start_time']).total_seconds()

                # Apply camera-specific time offset
                camera_offset = self._get_camera_time_offset(camera_id)
                time_offset -= camera_offset

                if time_offset < 0:
                    # Skip this camera, continue with others
                    continue

                # Seek to timestamp (use time-based seeking for accuracy)
                offset_ms = time_offset * 1000.0
                cap.set(cv2.CAP_PROP_POS_MSEC, offset_ms)

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    # Skip this camera, continue with others
                    continue

                frames[camera_id] = frame

            except Exception as e:
                # Skip this camera on any error, continue processing others
                continue

        # Return frames dict (even if partial), or None if NO cameras succeeded
        return frames if frames else None

    def close(self):
        """Release all open video captures."""
        for cap in self.current_captures.values():
            cap.release()
        self.current_captures.clear()
        self.current_video_info.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
