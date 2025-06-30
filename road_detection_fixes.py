#!/usr/bin/env python3
"""
Updated road detection fixes with smart hood detection
This imports the smart hood detection from stage1_road_detection.py
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from stage1_road_detection import (
    RoadDetectionConfig, 
    SmartHoodDetector, 
    HoodDetectionResult,
    apply_smart_hood_exclusion,
    visualize_with_hood_detection,
    create_improved_accurate_config,
    improve_road_coverage,
    apply_road_detection_fixes
)

# Re-export the main functions for compatibility
__all__ = [
    'create_improved_accurate_config',
    'apply_road_detection_fixes', 
    'SmartHoodDetector',
    'apply_smart_hood_exclusion',
    'visualize_with_hood_detection',
    'improve_road_coverage',
    'test_hood_detection'
]

def test_hood_detection(video_path: str, output_path: str = "hood_detection_test.mp4"):
    """Test hood detection on a video"""
    from stage1_road_detection import ModernRoadDetector
    
    # Create detector with fixes
    config = create_improved_accurate_config()
    detector = ModernRoadDetector(config)
    detector = apply_road_detection_fixes(detector, use_smart_hood=True)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Testing smart hood detection on {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect road
        road_mask, _ = detector.detect_road(frame)
        
        # Visualize
        vis = detector.visualize(frame, road_mask, style='freespace')
        out.write(vis)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
            # Print hood detection info
            if hasattr(detector, '_last_hood_result'):
                hood_result = detector._last_hood_result
                print(f"  Hood detection: {hood_result.method_used} "
                      f"({hood_result.confidence:.2f} confidence, "
                      f"{hood_result.hood_ratio:.1%} coverage)")
    
    cap.release()
    out.release()
    print(f"✓ Test complete! Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test" and len(sys.argv) > 2:
            # Test smart hood detection
            video_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else "smart_hood_test.mp4"
            test_hood_detection(video_file, output_file)
        else:
            # Process single video with smart hood detection
            video_file = sys.argv[1]
            test_hood_detection(video_file)
    else:
        # Demo setup
        print("=== Smart Hood Detection (via road_detection_fixes.py) ===")
        
        from stage1_road_detection import ModernRoadDetector
        
        # Create improved detector
        config = create_improved_accurate_config()
        detector = ModernRoadDetector(config)
        
        # Apply smart hood detection fixes
        detector = apply_road_detection_fixes(detector, use_smart_hood=True)
        
        print("✓ Detector ready with smart hood detection!")
        print("  - Imports from stage1_road_detection.py")
        print("  - Color consistency detection")
        print("  - Edge pattern analysis") 
        print("  - Geometric shape detection")
        print("  - Adaptive calibration")
        print("  - Improved road coverage")
        print("\nUsage examples:")
        print("  python road_detection_fixes.py test input_videos/Road_Lane.mp4")
        print("  python road_detection_fixes.py input_videos/Road_Lane.mp4") 