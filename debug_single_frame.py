#!/usr/bin/env python3
"""
Debug script to test road detection on a single frame
"""

import cv2
import numpy as np
from stage1_road_detection import (
    ModernRoadDetector, 
    RoadDetectionConfig, 
    create_improved_accurate_config,
    apply_road_detection_fixes
)

def test_different_configs(video_path, frame_number=100):
    """Test different configurations on a single frame"""
    
    # Load the frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to read frame {frame_number}")
        return
    
    print(f"Testing frame {frame_number} from {video_path}")
    print(f"Frame shape: {frame.shape}")
    
    # Test different configurations
    configs = {
        "very_simple": RoadDetectionConfig(
            model_type="transformers",
            conf_threshold=0.3,  # Very low threshold
            temporal_smooth=False,
            edge_refinement=False,  # No edge refinement
            advanced_edge_refinement=False,  # No advanced refinement
            multi_class_awareness=False,  # No multiclass filtering
            geometric_filtering=False,
            bilateral_filter=False,
            perspective_correction=False,
            multi_scale=False,
            debug_mode=True
        ),
        
        "low_threshold": RoadDetectionConfig(
            model_type="transformers", 
            conf_threshold=0.2,  # Even lower threshold
            temporal_smooth=False,
            edge_refinement=True,
            advanced_edge_refinement=False,  # Disable advanced
            multi_class_awareness=False,  # Disable multiclass
            debug_mode=True
        ),
        
        "current_fixed": create_improved_accurate_config(),
        
        "with_hood_detection": create_improved_accurate_config(),  # Will apply hood fixes below
        
        "original_pytorch": RoadDetectionConfig(
            model_type="pytorch",
            conf_threshold=0.5,
            debug_mode=True
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n=== Testing {config_name} ===")
        try:
            detector = ModernRoadDetector(config)
            
            # Apply hood detection fixes for specific configs
            if config_name == "with_hood_detection":
                print("  Applying smart hood detection fixes...")
                detector = apply_road_detection_fixes(detector, use_smart_hood=True)
            
            road_mask, confidence_map = detector.detect_road(frame)
            
            # Calculate stats
            total_pixels = road_mask.shape[0] * road_mask.shape[1]
            road_pixels = np.sum(road_mask > 0)
            road_percentage = (road_pixels / total_pixels) * 100
            
            print(f"Road coverage: {road_percentage:.1f}%")
            print(f"Road pixels: {road_pixels}/{total_pixels}")
            
            # Save visualization
            vis = detector.visualize(frame, road_mask)
            output_name = f"debug_{config_name}_result.jpg"
            cv2.imwrite(output_name, vis)
            
            # Save mask
            mask_name = f"debug_{config_name}_mask.jpg" 
            cv2.imwrite(mask_name, road_mask)
            
            results[config_name] = {
                'road_percentage': road_percentage,
                'road_pixels': road_pixels,
                'output_file': output_name,
                'mask_file': mask_name
            }
            
            print(f"Saved: {output_name} and {mask_name}")
            
        except Exception as e:
            print(f"Error with {config_name}: {e}")
            results[config_name] = {'error': str(e)}
    
    # Summary
    print(f"\n=== SUMMARY ===")
    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name}: ERROR - {result['error']}")
        else:
            print(f"{config_name}: {result['road_percentage']:.1f}% coverage")
    
    return results

if __name__ == "__main__":
    # Test on 10secondday.mp4
    results = test_different_configs("input_videos/10secondday.mp4", frame_number=50) 