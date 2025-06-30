#!/usr/bin/env python3
"""
Debug hood detection specifically
"""

import cv2
import numpy as np
from stage1_road_detection import (
    SmartHoodDetector, 
    ModernRoadDetector, 
    create_improved_accurate_config,
    apply_road_detection_fixes
)

def debug_hood_detection(video_path, frame_number=50):
    """Debug what hood detection is actually doing"""
    
    # Load frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to read frame {frame_number}")
        return
    
    print(f"Debugging hood detection on frame {frame_number}")
    print(f"Frame shape: {frame.shape}")
    
    # Test smart hood detector directly
    print("\n=== Testing SmartHoodDetector directly ===")
    hood_detector = SmartHoodDetector()
    hood_result = hood_detector.detect_hood(frame, frame_number)
    
    print(f"Hood detection result:")
    print(f"  Method: {hood_result.method_used}")
    print(f"  Confidence: {hood_result.confidence:.3f}")
    print(f"  Hood ratio: {hood_result.hood_ratio:.3f}")
    print(f"  Hood mask shape: {hood_result.hood_mask.shape}")
    print(f"  Hood pixels: {np.sum(hood_result.hood_mask > 0)}")
    print(f"  Hood percentage: {np.sum(hood_result.hood_mask > 0) / hood_result.hood_mask.size * 100:.1f}%")
    
    # Save hood mask visualization
    hood_vis = frame.copy()
    hood_vis[hood_result.hood_mask > 0] = [0, 0, 255]  # Red for hood areas
    cv2.imwrite("debug_hood_mask_overlay.jpg", hood_vis)
    
    # Save pure hood mask
    cv2.imwrite("debug_hood_mask_pure.jpg", hood_result.hood_mask)
    
    print(f"Saved hood visualizations: debug_hood_mask_overlay.jpg, debug_hood_mask_pure.jpg")
    
    # Test simple hood detection (15% ratio)
    print(f"\n=== Comparing with simple 15% hood detection ===")
    h, w = frame.shape[:2]
    simple_hood_height = int(h * 0.15)
    simple_hood_pixels = w * simple_hood_height
    smart_hood_pixels = np.sum(hood_result.hood_mask > 0)
    
    print(f"Simple hood (15%): {simple_hood_pixels} pixels")
    print(f"Smart hood: {smart_hood_pixels} pixels")
    print(f"Difference: {smart_hood_pixels - simple_hood_pixels} pixels")
    
    # Test road detection with and without hood
    print(f"\n=== Testing road detection with/without hood ===")
    
    # Without hood detection
    config = create_improved_accurate_config()
    detector_no_hood = ModernRoadDetector(config)
    road_mask_no_hood, conf_map = detector_no_hood.detect_road(frame)
    road_pixels_no_hood = np.sum(road_mask_no_hood > 0)
    
    # With our fixed hood detection
    detector_with_hood = ModernRoadDetector(config)
    detector_with_hood = apply_road_detection_fixes(detector_with_hood, use_smart_hood=True)
    road_mask_with_hood, conf_map_hood = detector_with_hood.detect_road(frame)
    road_pixels_with_hood = np.sum(road_mask_with_hood > 0)
    
    print(f"Road pixels without hood: {road_pixels_no_hood}")
    print(f"Road pixels with hood: {road_pixels_with_hood}")
    print(f"Excluded pixels: {road_pixels_no_hood - road_pixels_with_hood}")
    print(f"Exclusion percentage: {(road_pixels_no_hood - road_pixels_with_hood) / road_pixels_no_hood * 100:.1f}%")
    
    # Create comparison visualization
    comparison = np.hstack([
        detector_no_hood.visualize(frame, road_mask_no_hood),
        detector_with_hood.visualize(frame, road_mask_with_hood)
    ])
    cv2.imwrite("debug_hood_comparison.jpg", comparison)
    
    # Test manual hood exclusion
    print(f"\n=== Testing manual hood exclusion ===")
    road_mask_manual = road_mask_no_hood.copy()
    road_mask_manual[hood_result.hood_mask > 0] = 0
    road_pixels_manual = np.sum(road_mask_manual > 0)
    
    print(f"Manual exclusion pixels: {road_pixels_manual}")
    print(f"Manual vs automatic difference: {road_pixels_manual - road_pixels_with_hood}")
    
    # Save manual result
    manual_vis = detector_no_hood.visualize(frame, road_mask_manual)
    cv2.imwrite("debug_manual_hood_exclusion.jpg", manual_vis)
    
    print(f"\nSaved comparison: debug_hood_comparison.jpg")
    print(f"Saved manual exclusion: debug_manual_hood_exclusion.jpg")
    
    return {
        'hood_result': hood_result,
        'road_pixels_no_hood': road_pixels_no_hood,
        'road_pixels_with_hood': road_pixels_with_hood,
        'exclusion_working': road_pixels_with_hood < road_pixels_no_hood
    }

if __name__ == "__main__":
    result = debug_hood_detection("input_videos/10secondday.mp4", frame_number=50) 