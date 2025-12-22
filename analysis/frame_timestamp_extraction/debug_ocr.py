"""
Debug OCR - Visualize what the OCR is seeing and how well templates match.

Usage:
    python debug_ocr.py --video path/to/video.avi

Author: SooOrthoFlow Team
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from timestamp_reader import (_detect_nvr_type, _load_templates, _load_slot_definitions,
                              _extract_frame_roi, _preprocess_roi, TEMPLATE_SIZE)

def debug_ocr(video_path: str):
    """
    Debug OCR on a single video - show ROI, preprocessed image, and template matching.

    Parameters:
        video_path: Path to video file
    """
    print("="*70)
    print("OCR DEBUG")
    print("="*70)
    print(f"Video: {video_path}")
    print()

    # Detect NVR type
    nvr_type = _detect_nvr_type(video_path)
    print(f"NVR type: {nvr_type}")

    # Load templates and slots
    templates = _load_templates(nvr_type)
    slots = _load_slot_definitions(nvr_type)

    print(f"Templates loaded: {len(templates)} characters")
    print(f"  Available: {sorted(templates.keys())}")
    print()

    # Extract ROI
    roi = _extract_frame_roi(video_path)
    if roi is None:
        print("Failed to extract ROI")
        return

    print(f"ROI shape: {roi.shape}")

    # Preprocess
    roi_processed = _preprocess_roi(roi)

    print(f"ROI mean brightness: {np.mean(roi_processed):.1f}")
    print(f"  (< 127 = inverted)")
    print()

    # Show original and preprocessed ROI
    roi_display = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    roi_proc_display = cv2.resize(roi_processed, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('1. Original ROI (3x)', roi_display)
    cv2.imshow('2. Preprocessed ROI (3x)', roi_proc_display)
    print("Showing ROI - press any key to continue...")
    cv2.waitKey(0)

    # Sort slots by position
    sorted_slots = sorted(slots.items(), key=lambda item: item[1][0])

    # Test each character slot
    print("\nTesting character recognition:")
    print("-" * 70)

    for slot_name, slot_coords in sorted_slots:
        # Skip separators
        if 'dash' in slot_name.lower() or 'colon' in slot_name.lower() or 'space' in slot_name.lower():
            continue

        x, y, w, h = slot_coords

        # Extract character
        char_img = roi_processed[y:y+h, x:x+w]
        if char_img.size == 0:
            continue

        # Resize to template size
        char_resized = cv2.resize(char_img, TEMPLATE_SIZE)

        # Try matching against all templates
        best_char = '?'
        best_score = 0.0
        scores = {}

        for char, template in templates.items():
            result = cv2.matchTemplate(char_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            scores[char] = max_val

            if max_val > best_score:
                best_score = max_val
                best_char = char

        # Display character and top matches
        char_display = cv2.resize(char_resized, (TEMPLATE_SIZE[0]*10, TEMPLATE_SIZE[1]*10),
                                 interpolation=cv2.INTER_NEAREST)
        char_display = cv2.cvtColor(char_display, cv2.COLOR_GRAY2BGR)

        # Add info
        cv2.putText(char_display, f"Slot: {slot_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(char_display, f"Best: '{best_char}' ({best_score:.3f})", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if best_score > 0.65 else (0, 0, 255), 1)

        # Top 3 matches
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (char, score) in enumerate(top_3):
            cv2.putText(char_display, f"{i+1}. '{char}': {score:.3f}", (10, 90 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow('Character Debug - Press any key for next', char_display)

        # Print to console
        threshold = 0.65
        status = "✓" if best_score >= threshold else "✗"
        print(f"{status} {slot_name:15s} → '{best_char}' (score: {best_score:.3f})")

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    print()
    print("="*70)
    print("Debug complete!")
    print()
    print("If scores are very low (<0.3):")
    print("  - Templates don't match the video style")
    print("  - Regenerate templates from this video with --random flag")
    print()
    print("If some characters work but others don't:")
    print("  - Need more template samples for those characters")
    print("  - Run generate_templates.py again on different videos")


def main():
    parser = argparse.ArgumentParser(description='Debug OCR template matching')
    parser.add_argument('--video', required=True, help='Path to video file')
    args = parser.parse_args()

    debug_ocr(args.video)


if __name__ == "__main__":
    main()
