#!/usr/bin/env python3
"""
Road Detection Parameter Tuning Script
=====================================

Easily test different configurations to reduce car bleeding and improve edge precision.
"""

from stage1_road_detection import (
    ModernRoadDetector, 
    create_accurate_fast_config,
    create_conservative_accurate_config, 
    create_ultra_precise_config,
    create_edge_tuned_config
)
import cv2
import numpy as np
import argparse

def test_config(video_path, config, config_name, test_frames=20):
    """Test a configuration on video frames"""
    print(f"\n🧪 Testing {config_name}:")
    print(f"   Confidence threshold: {config.conf_threshold}")
    print(f"   Edge threshold: {config.confidence_edge_threshold}")
    print(f"   Car filtering: {config.multi_class_awareness}")
    print(f"   Advanced edge refinement: {config.advanced_edge_refinement}")
    
    detector = ModernRoadDetector(config)
    cap = cv2.VideoCapture(video_path)
    
    coverages = []
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        road_mask, _ = detector.detect_road(frame)
        coverage = (np.sum(road_mask > 0) / (road_mask.shape[0] * road_mask.shape[1])) * 100
        coverages.append(coverage)
    
    cap.release()
    avg_coverage = np.mean(coverages)
    
    print(f"   ✅ Average coverage: {avg_coverage:.1f}%")
    
    # Assess car bleeding risk
    risk = "HIGH" if config.conf_threshold < 0.8 else \
           "MEDIUM" if config.conf_threshold < 0.85 else \
           "LOW" if config.conf_threshold < 0.9 else "MINIMAL"
    print(f"   🚗 Car bleeding risk: {risk}")
    
    return avg_coverage

def main():
    parser = argparse.ArgumentParser(description="Tune road detection parameters")
    parser.add_argument("video_path", help="Path to test video")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to test")
    parser.add_argument("--config", choices=['current', 'conservative', 'ultra', 'edge', 'all'], 
                       default='all', help="Configuration to test")
    
    args = parser.parse_args()
    
    print("🎯 ROAD DETECTION PARAMETER TUNING")
    print("=" * 50)
    print(f"Video: {args.video_path}")
    print(f"Test frames: {args.frames}")
    
    configs = {
        'current': (create_accurate_fast_config(), "Current (0.77)"),
        'conservative': (create_conservative_accurate_config(), "Conservative (0.85)"),
        'ultra': (create_ultra_precise_config(), "Ultra Precise (0.9)"),
        'edge': (create_edge_tuned_config(), "Edge Tuned (0.8/0.95)")
    }
    
    if args.config == 'all':
        test_configs = configs
    else:
        test_configs = {args.config: configs[args.config]}
    
    results = []
    for key, (config, name) in test_configs.items():
        # Ensure geometric filtering is disabled (it was causing issues)
        config.geometric_filtering = False
        config.debug_mode = False
        
        coverage = test_config(args.video_path, config, name, args.frames)
        results.append((name, coverage, config.conf_threshold, config.confidence_edge_threshold))
    
    print(f"\n📊 TUNING RESULTS SUMMARY:")
    print(f"Configuration      | Coverage | Conf | Edge | Recommended For")
    print(f"-" * 65)
    for name, coverage, conf, edge in results:
        use_case = "Car bleeding issues" if conf >= 0.85 else \
                  "Edge precision" if edge >= 0.95 else \
                  "Balanced use"
        print(f"{name:<17} | {coverage:>6.1f}% | {conf:.2f} | {edge:.2f} | {use_case}")
    
    print(f"\n🔧 PARAMETER GUIDE:")
    print(f"• conf_threshold (0.77-0.9): Higher = less car bleeding, slightly lower coverage")
    print(f"• confidence_edge_threshold (0.85-0.95): Higher = cleaner road boundaries")
    print(f"• multi_class_awareness: Essential for removing cars/people")
    print(f"• advanced_edge_refinement: Aggressive car/person removal")
    print(f"• geometric_filtering: DISABLED (was filtering all pixels)")

if __name__ == "__main__":
    main() 