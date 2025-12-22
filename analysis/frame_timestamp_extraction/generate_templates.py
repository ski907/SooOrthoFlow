"""
Template Generation Tool - Interactive utility to create character templates for OCR.

This is a one-time setup tool. Run once per NVR type to create templates.

Usage:
    python generate_templates.py --video path/to/reference.avi --nvr NVR2_3
    python generate_templates.py --video path/to/reference.avi --nvr NVR2_3 --random

Author: SooOrthoFlow Team
"""

import cv2
import numpy as np
import json
import argparse
import random
from pathlib import Path

# Default ROI coordinates (same as timestamp_reader.py)
ROI_COORDS = {
    'top': 0.04,
    'bottom': 0.07,
    'left': 0.80,
    'right': 0.97
}

# Template size (width, height)
TEMPLATE_SIZE = (20, 32)

# Binary threshold for preprocessing
BINARY_THRESHOLD = 200
WHITE_TEXT_THRESHOLD = 220  # Higher threshold for white text


class TemplateGenerator:
    """Interactive template generation tool using slot definitions."""

    def __init__(self, video_path: str, nvr_type: str, use_random_frame: bool = False):
        """
        Initialize template generator.

        Parameters:
            video_path: Path to reference video file
            nvr_type: NVR type ("NVR1" or "NVR2_3")
            use_random_frame: If True, use random frame instead of first frame
        """
        self.video_path = video_path
        self.nvr_type = nvr_type
        self.use_random_frame = use_random_frame

        # Paths
        package_dir = Path(__file__).parent
        self.slot_file = package_dir / "config" / f"slot_definitions_{nvr_type}.json"
        self.output_dir = package_dir / "templates" / nvr_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load slot definitions
        with open(self.slot_file, 'r') as f:
            self.slots = json.load(f)

        # Character templates collected (stores list of images for averaging)
        self.template_images = {}  # char -> list of numpy arrays
        self.templates = {}  # char -> averaged template

    def run(self):
        """Run interactive template generation."""
        print("="*70)
        print("TEMPLATE GENERATION TOOL")
        print("="*70)
        print(f"Video: {self.video_path}")
        print(f"NVR Type: {self.nvr_type}")
        print(f"Random frame: {self.use_random_frame}")
        print(f"Output: {self.output_dir}")
        print()

        # Load existing templates (for averaging)
        self._load_existing_templates()

        # Extract and display ROI
        roi, frame_num = self._extract_roi()
        if roi is None:
            print("Failed to extract ROI from video")
            return

        if self.use_random_frame:
            print(f"Using frame: {frame_num}")
            print()

        # Preprocess ROI
        roi_processed = self._preprocess_roi(roi)

        print("Instructions:")
        print("  - Each character region will be shown")
        print("  - Type the character you see: 0-9, A, M, P")
        print("  - Press SPACE to skip (if don't want this instance)")
        print("  - Press 'r' to reload a different random frame")
        print("  - Press ESC to save and quit")
        print()

        # Process each slot
        self._process_slots(roi_processed)

    def _process_slots(self, roi_processed):
        """Process all character slots."""
        slot_names = sorted(self.slots.keys())

        for slot_name in slot_names:
            # Skip separator slots (dashes, colons, spaces)
            if 'dash' in slot_name or 'colon' in slot_name or 'space' in slot_name:
                continue

            slot_coords = self.slots[slot_name]
            x, y, w, h = slot_coords

            # Extract character region
            char_img = roi_processed[y:y+h, x:x+w]
            if char_img.size == 0:
                continue

            # Resize to template size
            char_resized = cv2.resize(char_img, TEMPLATE_SIZE)

            # Display character
            display = self._create_display(char_resized, slot_name)
            cv2.imshow('Template Generation - Type the character', display)

            # Wait for user input
            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == 27:  # ESC
                    print("\nSaving and quitting...")
                    cv2.destroyAllWindows()
                    self._save_averaged_templates()
                    self._print_summary()
                    return

                elif key == ord('r'):  # 'r' - reload random frame
                    print("\nReloading random frame...")
                    cv2.destroyAllWindows()
                    roi, frame_num = self._extract_roi()
                    if roi is not None:
                        print(f"Using frame: {frame_num}")
                        roi_processed = self._preprocess_roi(roi)
                        self._process_slots(roi_processed)
                    return

                elif key == 32:  # SPACE
                    print(f"  {slot_name}: Skipped")
                    break

                else:
                    # Convert key to character
                    char = chr(key).upper()

                    # Validate input
                    if char in '0123456789AMP':
                        # Add to template collection for averaging
                        if char not in self.template_images:
                            self.template_images[char] = []
                        self.template_images[char].append(char_resized.astype(np.float32))

                        count = len(self.template_images[char])
                        print(f"  {slot_name}: '{char}' → Added (count: {count})")
                        break
                    else:
                        print(f"  Invalid input. Press 0-9, A, M, P, SPACE to skip, 'r' for new frame, or ESC to quit")

        # Finished all slots - save and show summary
        cv2.destroyAllWindows()
        self._save_averaged_templates()
        self._print_summary()

    def _load_existing_templates(self):
        """Load existing templates for averaging."""
        for char in '0123456789AMP':
            template_path = self.output_dir / f"{char}.png"
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.template_images[char] = [template.astype(np.float32)]

    def _save_averaged_templates(self):
        """Average and save all collected templates."""
        for char, images in self.template_images.items():
            if len(images) == 0:
                continue

            # Average all images for this character
            averaged = np.mean(images, axis=0).astype(np.uint8)
            self.templates[char] = averaged

            # Save to file
            template_path = self.output_dir / f"{char}.png"
            cv2.imwrite(str(template_path), averaged)

    def _print_summary(self):
        """Print summary of templates created."""
        print()
        print("="*70)
        print("TEMPLATE GENERATION COMPLETE")
        print("="*70)
        print(f"Templates saved: {len(self.templates)}")
        print(f"Output directory: {self.output_dir}")
        print()
        print("Templates created (with sample counts):")
        for char in sorted(self.templates.keys()):
            count = len(self.template_images.get(char, []))
            print(f"  {char}.png (averaged from {count} sample(s))")

        # Check completeness
        required = set('0123456789AMP')
        missing = required - set(self.templates.keys())
        if missing:
            print()
            print("Missing templates (run again with --random to find more digits):")
            for char in sorted(missing):
                print(f"  {char}")
        else:
            print()
            print("All templates complete!")

    def _extract_roi(self) -> tuple:
        """Extract timestamp ROI from video frame.

        Returns:
            Tuple of (roi, frame_number)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None, 0

        frame_num = 0
        if self.use_random_frame:
            # Get total frames and pick random one
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 1:
                frame_num = random.randint(0, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, frame_num

        # Crop ROI
        h, w = frame.shape[:2]
        top = int(h * ROI_COORDS['top'])
        bottom = int(h * ROI_COORDS['bottom'])
        left = int(w * ROI_COORDS['left'])
        right = int(w * ROI_COORDS['right'])

        roi = frame[top:bottom, left:right]
        return roi, frame_num

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI (grayscale, binary threshold) - same as timestamp_reader.py."""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Check if we likely have white text (bright pixels) or dark text
        max_brightness = np.max(gray)
        has_white_text = max_brightness > 240

        if has_white_text:
            # Use higher threshold to keep only very bright pixels (white text)
            _, binary = cv2.threshold(gray, WHITE_TEXT_THRESHOLD, 255, cv2.THRESH_BINARY)
            # Don't invert - keep white text on black background
        else:
            # Standard threshold for dark text on light background
            _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
            # Invert if black text on white background
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)

        return binary

    def _create_display(self, char_img: np.ndarray, slot_name: str) -> np.ndarray:
        """Create display image showing character and prompt."""
        # Scale up character for better visibility
        scale = 10
        char_large = cv2.resize(char_img, (TEMPLATE_SIZE[0]*scale, TEMPLATE_SIZE[1]*scale),
                               interpolation=cv2.INTER_NEAREST)

        # Convert to color
        display = cv2.cvtColor(char_large, cv2.COLOR_GRAY2BGR)

        # Add border
        display = cv2.copyMakeBorder(display, 50, 100, 50, 50,
                                     cv2.BORDER_CONSTANT, value=(50, 50, 50))

        # Add text
        cv2.putText(display, f"Slot: {slot_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display, "Type: 0-9, A, M, P", (10, display.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display, "SPACE=Skip  R=NewFrame  ESC=SaveQuit", (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        return display


def main():
    parser = argparse.ArgumentParser(description='Generate character templates for timestamp OCR')
    parser.add_argument('--video', required=True, help='Path to reference video file')
    parser.add_argument('--nvr', required=True, choices=['NVR1', 'NVR2_3'],
                       help='NVR type (NVR1 or NVR2_3)')
    parser.add_argument('--random', action='store_true',
                       help='Use random frame instead of first frame (helpful for finding all digits)')
    args = parser.parse_args()

    generator = TemplateGenerator(args.video, args.nvr, args.random)
    generator.run()


if __name__ == "__main__":
    main()
