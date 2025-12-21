"""
Streaming Multi-Camera Mosaic Video Processor

This module provides efficient video processing for creating mosaicked video
output from multiple synchronized camera sources. It uses single-pass processing:
reading frames → orthorectifying → mosaicking in memory → writing to output video.

Key Features:
- Single-pass processing (no intermediate files)
- Uses pre-computed ortho caches for fast processing
- Zone-map based or center-based mosaicking
- 6-12x faster than TIFF-based approaches
- 10-60x less storage (compressed video vs TIFFs)

Author: SooOrthoFlow Team
Version: 0.1.0
"""

from .camera_video_reader import CameraVideoReader
from .in_memory_mosaic import InMemoryMosaicEngine
from .video_mosaic_processor import VideoMosaicProcessor

__all__ = [
    'CameraVideoReader',
    'InMemoryMosaicEngine',
    'VideoMosaicProcessor'
]

__version__ = '0.1.0'
