#!/usr/bin/env python3
"""
Simple video processing test with robust video writing
"""

import cv2
import numpy as np
from stage1_road_detection import ModernRoadDetector, create_improved_accurate_config

def process_video_simple(input_path, output_path, max_frames=100):
    """Process video with simple, robust video writing"""
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    # Initialize detector with our fixed config
    config = create_improved_accurate_config()
    detector = ModernRoadDetector(config)
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Limit frames for testing
    process_frames = min(max_frames, total_frames)
    
    # Simple video writer settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Simple codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Cannot create output video")
        cap.release()
        return
    
    frame_count = 0
    road_percentages = []
    
    print(f"Processing {process_frames} frames...")
    
    while frame_count < process_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect road
        road_mask, confidence_map = detector.detect_road(frame)
        
        # Calculate coverage
        total_pixels = road_mask.shape[0] * road_mask.shape[1]
        road_pixels = np.sum(road_mask > 0)
        road_percentage = (road_pixels / total_pixels) * 100
        road_percentages.append(road_percentage)
        
        # Create visualization
        vis = detector.visualize(frame, road_mask)
        
        # Add stats text
        text = f"Frame {frame_count+1}: {road_percentage:.1f}% road"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(vis)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count}/{process_frames} frames, avg coverage: {np.mean(road_percentages):.1f}%")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Stats
    avg_coverage = np.mean(road_percentages) if road_percentages else 0
    print(f"\n✅ Processing complete!")
    print(f"Processed: {frame_count} frames")
    print(f"Average road coverage: {avg_coverage:.1f}%")
    print(f"Coverage range: {min(road_percentages):.1f}% - {max(road_percentages):.1f}%")
    print(f"Output saved: {output_path}")
    
    return {
        'frames_processed': frame_count,
        'avg_coverage': avg_coverage,
        'coverage_range': (min(road_percentages), max(road_percentages)),
        'output_path': output_path
    }

if __name__ == "__main__":
    # Test with 10secondday.mp4 - process first 100 frames
    result = process_video_simple(
        "input_videos/10secondday.mp4", 
        "simple_road_detection_test.mp4",
        max_frames=100
    ) 