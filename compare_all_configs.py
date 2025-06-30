#!/usr/bin/env python3
"""
Compare All Parameter Configurations
===================================

Process one video through all 4 configs to compare car bleeding and edge precision.
"""

import os
import time
from stage1_road_detection import (
    ModernRoadDetector, 
    create_accurate_fast_config,
    create_conservative_accurate_config, 
    create_ultra_precise_config,
    create_edge_tuned_config
)
import cv2
import numpy as np

def process_video_with_config(input_path, output_dir, config, config_name):
    """Process video with specific configuration"""
    print(f"\n🎬 Processing with {config_name} config...")
    print(f"   Confidence: {config.conf_threshold} | Edge: {config.confidence_edge_threshold}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup detector
    detector = ModernRoadDetector(config)
    
    # Setup video capture and writer
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output files
    output_video = os.path.join(output_dir, f"road_detection_{config_name.lower().replace(' ', '_')}.mp4")
    output_masks = os.path.join(output_dir, f"road_masks_{config_name.lower().replace(' ', '_')}.mp4")
    
    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_overlay = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    writer_masks = cv2.VideoWriter(output_masks, fourcc, fps, (width, height))
    
    frame_count = 0
    total_coverage = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            road_mask, confidence_map = detector.detect_road(frame)
            
            # Calculate coverage
            coverage = (np.sum(road_mask > 0) / (road_mask.shape[0] * road_mask.shape[1])) * 100
            total_coverage += coverage
            
            # Create overlay
            overlay = frame.copy()
            overlay[road_mask > 0] = [0, 255, 0]  # Green road overlay
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Add text info
            info_text = f"{config_name} | Frame: {frame_count+1}/{total_frames} | Coverage: {coverage:.1f}%"
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create mask visualization
            mask_vis = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
            
            # Write frames
            writer_overlay.write(result)
            writer_masks.write(mask_vis)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"   Frame {frame_count}/{total_frames} | {fps_current:.1f} FPS | Avg coverage: {total_coverage/frame_count:.1f}%")
    
    finally:
        cap.release()
        writer_overlay.release()
        writer_masks.release()
    
    avg_coverage = total_coverage / frame_count if frame_count > 0 else 0
    processing_time = time.time() - start_time
    
    print(f"   ✅ Complete! Average coverage: {avg_coverage:.1f}% | Time: {processing_time:.1f}s")
    
    return avg_coverage, processing_time

def main():
    # Configuration
    input_video = "input_videos/80397-572395744_small.mp4"
    base_output_dir = "config_comparison"
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Define all configurations
    configs = [
        (create_accurate_fast_config(), "Current_0.77", "current"),
        (create_conservative_accurate_config(), "Conservative_0.85", "conservative"), 
        (create_ultra_precise_config(), "Ultra_Precise_0.9", "ultra"),
        (create_edge_tuned_config(), "Edge_Tuned_0.8", "edge")
    ]
    
    print("🎯 PROCESSING VIDEO WITH ALL PARAMETER CONFIGURATIONS")
    print("=" * 60)
    print(f"Input video: {input_video}")
    print(f"Output directory: {base_output_dir}/")
    print(f"Testing {len(configs)} configurations...")
    
    results = []
    
    for config, display_name, dir_name in configs:
        # Disable problematic settings
        config.geometric_filtering = False
        config.debug_mode = False
        
        # Create output directory for this config
        output_dir = os.path.join(base_output_dir, dir_name)
        
        # Process video
        avg_coverage, processing_time = process_video_with_config(
            input_video, output_dir, config, display_name
        )
        
        results.append((display_name, avg_coverage, processing_time, config.conf_threshold, config.confidence_edge_threshold))
    
    # Print comparison summary
    print(f"\n📊 CONFIGURATION COMPARISON RESULTS:")
    print(f"Configuration      | Coverage | Time | Conf | Edge | Car Bleeding Risk")
    print(f"-" * 75)
    
    for name, coverage, time_taken, conf, edge in results:
        risk = "HIGH" if conf < 0.8 else "MEDIUM" if conf < 0.85 else "LOW" if conf < 0.9 else "MINIMAL"
        print(f"{name:<17} | {coverage:>6.1f}% | {time_taken:>3.0f}s | {conf:.2f} | {edge:.2f} | {risk}")
    
    print(f"\n📁 OUTPUT STRUCTURE:")
    print(f"config_comparison/")
    print(f"├── current/          # Current (0.77) - baseline")
    print(f"├── conservative/     # Conservative (0.85) - reduced car bleeding") 
    print(f"├── ultra/           # Ultra Precise (0.9) - minimal car bleeding")
    print(f"└── edge/            # Edge Tuned (0.8/0.95) - better edge precision")
    
    print(f"\n📥 DOWNLOAD COMMAND:")
    print(f"scp -r root@82.221.170.242:/root/dashcam-pipeline/{base_output_dir} ./")

if __name__ == "__main__":
    main() 