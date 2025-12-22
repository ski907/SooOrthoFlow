"""
Timestamp Reader - Simple utility to read burned-in timestamps from NVR video frames.

Usage:
    from frame_timestamp_extraction.timestamp_reader import read_timestamp

    timestamp = read_timestamp("path/to/video.avi")
    print(f"Video starts at: {timestamp}")

Author: SooOrthoFlow Team
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List

# Module-level caches
_TEMPLATE_CACHE = {}
_SLOT_CACHE = {}

# Default ROI coordinates (percentage of frame)
ROI_COORDS = {
    'top': 0.04,    # 4% from top
    'bottom': 0.07,  # 7% from top (height = 3%)
    'left': 0.80,    # 80% from left
    'right': 0.97    # 97% from left (width = 17%)
}

# Template matching parameters
TEMPLATE_SIZE = (20, 32)  # width, height
MATCH_THRESHOLD = 0.65
BINARY_THRESHOLD = 253  # Standard threshold for dark text on light background
WHITE_TEXT_THRESHOLD = 254  # Higher threshold for white text (keeps only very bright pixels)

# Get package directory
PACKAGE_DIR = Path(__file__).parent


def read_timestamp(video_path, frame_index=0) -> Optional[datetime]:
    """
    Read timestamp from video frame.

    Parameters:
        video_path: Path to video file
        frame_index: Frame number to read (default: 0 = first frame)

    Returns:
        datetime object if successful, None if failed

    Example:
        >>> timestamp = read_timestamp("NVR2_N910A6_ch4_main_20251217120000.avi")
        >>> print(timestamp)
        2025-12-17 12:00:01
    """
    try:
        # Detect NVR type from filename
        nvr_type = _detect_nvr_type(video_path)

        # Load templates and slot definitions for this NVR type
        templates = _load_templates(nvr_type)
        slots = _load_slot_definitions(nvr_type)

        # Extract timestamp ROI from video frame
        roi = _extract_frame_roi(video_path, frame_index)
        if roi is None:
            return None

        # Preprocess ROI
        roi_processed = _preprocess_roi(roi)

        # Recognize each character using template matching
        # Sort slots by x-position (left to right) instead of alphabetically
        sorted_slots = sorted(slots.items(), key=lambda item: item[1][0])  # Sort by x-coord

        # Read only digit/letter slots (skip separators - we'll add them in the format we expect)
        chars = []
        for slot_name, slot_coords in sorted_slots:
            # Skip separator slots
            if 'dash' in slot_name.lower() or 'colon' in slot_name.lower() or 'space' in slot_name.lower():
                continue

            char, confidence = _match_character(roi_processed, slot_coords, templates)
            chars.append(char)

        # Build timestamp string in expected format: MM-DD-YYYY HH:MM:SS AP
        # Expected: 2 month + 2 day + 4 year + 2 hour + 2 minute + 2 second + 1 A/P + 1 M = 16 chars
        if len(chars) != 16:
            print(f"Warning: Expected 16 characters, got {len(chars)}: {''.join(chars)}")
            return None

        ocr_string = (
            f"{chars[0]}{chars[1]}-"      # MM-
            f"{chars[2]}{chars[3]}-"      # DD-
            f"{chars[4]}{chars[5]}{chars[6]}{chars[7]} "  # YYYY
            f"{chars[8]}{chars[9]}:"      # HH:
            f"{chars[10]}{chars[11]}:"    # MM:
            f"{chars[12]}{chars[13]} "    # SS
            f"{chars[14]}M"     # AP
        )

        # Parse timestamp string to datetime
        timestamp = _parse_timestamp_string(ocr_string)

        return timestamp

    except Exception as e:
        print(f"Error reading timestamp from {video_path}: {e}")
        return None


def _detect_nvr_type(video_path) -> str:
    """
    Detect NVR type from video filename or path.

    Parameters:
        video_path: Path to video file (str or Path)

    Returns:
        "NVR1" or "NVR2_3"

    Raises:
        ValueError: If NVR type cannot be determined
    """
    path = Path(video_path)
    filename = path.name.upper()
    full_path = str(path).upper()

    # Check filename first (e.g., "NVR2_N910A6_ch4_main.avi")
    if "NVR1_" in filename:
        return "NVR1"
    elif "NVR2_" in filename or "NVR3_" in filename:
        return "NVR2_3"

    # Check directory path (e.g., "E:\12-17-25\Test 23\NVR2\...")
    if "\\NVR1\\" in full_path or "/NVR1/" in full_path:
        return "NVR1"
    elif "\\NVR2\\" in full_path or "/NVR2/" in full_path:
        return "NVR2_3"
    elif "\\NVR3\\" in full_path or "/NVR3/" in full_path:
        return "NVR2_3"

    raise ValueError(f"Cannot detect NVR type from path: {video_path}")


def _load_templates(nvr_type: str) -> Dict[str, np.ndarray]:
    """
    Load character templates for NVR type (with caching).

    Parameters:
        nvr_type: "NVR1" or "NVR2_3"

    Returns:
        Dictionary mapping character to template image
        e.g., {"0": <array>, "1": <array>, ..., "A": <array>, "M": <array>, "P": <array>}
    """
    # Check cache first
    if nvr_type in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[nvr_type]

    # Load templates from disk
    template_dir = PACKAGE_DIR / "templates" / nvr_type
    if not template_dir.exists():
        raise FileNotFoundError(
            f"Template directory not found: {template_dir}\n"
            f"Run generate_templates.py to create templates for {nvr_type}"
        )

    templates = {}
    # Load digit templates (0-9)
    for digit in "0123456789":
        template_path = template_dir / f"{digit}.png"
        if template_path.exists():
            templates[digit] = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: Missing template for '{digit}' in {nvr_type}")

    # Load letter templates (A, M, P for AM/PM)
    for letter in "AMP":
        template_path = template_dir / f"{letter}.png"
        if template_path.exists():
            templates[letter] = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: Missing template for '{letter}' in {nvr_type}")

    if not templates:
        raise FileNotFoundError(
            f"No templates found in {template_dir}\n"
            f"Run generate_templates.py to create templates for {nvr_type}"
        )

    # Cache for future use
    _TEMPLATE_CACHE[nvr_type] = templates

    return templates


def _load_slot_definitions(nvr_type: str) -> Dict[str, List[int]]:
    """
    Load slot definitions (character positions) for NVR type (with caching).

    Parameters:
        nvr_type: "NVR1" or "NVR2_3"

    Returns:
        Dictionary mapping slot name to [x, y, width, height]
        e.g., {"month1": [0, 6, 28, 44], "month2": [29, 6, 28, 44], ...}
    """
    # Check cache first
    if nvr_type in _SLOT_CACHE:
        return _SLOT_CACHE[nvr_type]

    # Load from JSON file
    slot_file = PACKAGE_DIR / "config" / f"slot_definitions_{nvr_type}.json"
    if not slot_file.exists():
        raise FileNotFoundError(f"Slot definition file not found: {slot_file}")

    with open(slot_file, 'r') as f:
        slots = json.load(f)

    # Cache for future use
    _SLOT_CACHE[nvr_type] = slots

    return slots


def _extract_frame_roi(video_path, frame_index: int = 0) -> Optional[np.ndarray]:
    """
    Extract timestamp ROI from video frame.

    Parameters:
        video_path: Path to video file
        frame_index: Frame number to extract

    Returns:
        ROI image (numpy array) or None if failed
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None

        # Seek to desired frame
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Cannot read frame {frame_index} from {video_path}")
            return None

        # Crop ROI
        h, w = frame.shape[:2]
        top = int(h * ROI_COORDS['top'])
        bottom = int(h * ROI_COORDS['bottom'])
        left = int(w * ROI_COORDS['left'])
        right = int(w * ROI_COORDS['right'])

        roi = frame[top:bottom, left:right]

        return roi

    except Exception as e:
        print(f"Error extracting frame ROI: {e}")
        return None


def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Preprocess ROI for template matching.

    Parameters:
        roi: ROI image (BGR or grayscale)

    Returns:
        Preprocessed image (grayscale, binary threshold)
    """
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Check if we likely have white text (bright pixels) or dark text
    # Look at the maximum brightness - if very high (>240), likely white text
    max_brightness = np.max(gray)
    has_white_text = max_brightness > 240

    if has_white_text:
        # Use higher threshold to keep only very bright pixels (white text)
        # This removes light-colored backgrounds that aren't pure white
        _, binary = cv2.threshold(gray, WHITE_TEXT_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Don't invert - keep white text on black background (matches templates)
    else:
        # Standard threshold for dark text on light background
        _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Check if we have black text on white background (needs inversion)
        if np.mean(binary) > 127:
            # Invert so text is white on black background (to match templates)
            binary = cv2.bitwise_not(binary)

    return binary


def _match_character(roi: np.ndarray, slot_coords: List[int],
                     templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    Match character in slot using template matching.

    Parameters:
        roi: Preprocessed ROI image
        slot_coords: [x, y, width, height] of character slot
        templates: Dictionary of character templates

    Returns:
        Tuple of (character, confidence)
        Returns ('?', 0.0) if no good match found
    """
    x, y, w, h = slot_coords

    # Extract character region from ROI
    char_img = roi[y:y+h, x:x+w]

    if char_img.size == 0:
        return ('?', 0.0)

    # Resize to template size
    char_img_resized = cv2.resize(char_img, TEMPLATE_SIZE)

    # Try matching against all templates
    best_char = '?'
    best_score = 0.0

    for char, template in templates.items():
        # Template matching
        result = cv2.matchTemplate(char_img_resized, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_char = char

    # Check if score exceeds threshold
    if best_score < MATCH_THRESHOLD:
        return ('?', best_score)

    return (best_char, best_score)


def _parse_timestamp_string(ocr_string: str) -> Optional[datetime]:
    """
    Parse OCR string to datetime object.

    Expected format: "MM-DD-YYYY HH:MM:SS AP"
    Example: "12-17-2025 03:16:01 PM"

    Parameters:
        ocr_string: Raw OCR string from template matching

    Returns:
        datetime object if successful, None if parsing failed
    """
    try:
        # Remove any '?' characters (failed matches)
        if '?' in ocr_string:
            print(f"Warning: OCR string contains uncertain characters: {ocr_string}")
            # For now, return None if any character failed
            return None

        # Parse datetime
        # Format: MM-DD-YYYY HH:MM:SS AM/PM
        dt = datetime.strptime(ocr_string, "%m-%d-%Y %I:%M:%S %p")

        return dt

    except ValueError as e:
        print(f"Cannot parse timestamp string '{ocr_string}': {e}")
        return None


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python timestamp_reader.py <video_path>")
        print("Example: python timestamp_reader.py test_videos/NVR2_N910A6_ch4_main.avi")
        sys.exit(1)

    video_path = sys.argv[1]
    timestamp = read_timestamp(video_path)

    if timestamp:
        print(f"Timestamp: {timestamp}")
    else:
        print("Failed to read timestamp")
