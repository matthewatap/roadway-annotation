#!/usr/bin/env python3
"""
Benchmark pipeline to find bottlenecks
"""

import time
import cv2
import numpy as np
from stage1_road_detection import (
    ModernRoadDetector, 
    create_improved_accurate_config,
    apply_road_detection_fixes
)

def benchmark_pipeline_steps(video_path, num_frames=50):
    """Benchmark each step of the pipeline"""
    
    print(f"Benchmarking {num_frames} frames from {video_path}")
    
    # Setup
    config = create_improved_accurate_config()
    
    # Test 1: Basic detector (no fixes)
    print("\n=== Testing Basic Detector (No Fixes) ===")
    detector_basic = ModernRoadDetector(config)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    
    # Benchmark basic detection
    start_time = time.time()
    for i, frame in enumerate(frames):
        road_mask, conf_map = detector_basic.detect_road(frame)
        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(frames)}")
    basic_time = time.time() - start_time
    basic_fps = len(frames) / basic_time
    print(f"Basic detection: {basic_time:.1f}s, {basic_fps:.1f} FPS")
    
    # Test 2: With fixes applied
    print("\n=== Testing With All Fixes ===")
    detector_fixed = ModernRoadDetector(config)
    detector_fixed = apply_road_detection_fixes(detector_fixed, use_smart_hood=True)
    
    start_time = time.time()
    for i, frame in enumerate(frames):
        road_mask, conf_map = detector_fixed.detect_road(frame)
        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(frames)}")
    fixed_time = time.time() - start_time
    fixed_fps = len(frames) / fixed_time
    print(f"With fixes: {fixed_time:.1f}s, {fixed_fps:.1f} FPS")
    
    # Test 3: Individual components
    print("\n=== Testing Individual Components ===")
    
    # Just coverage improvement
    from stage1_road_detection import improve_road_coverage
    
    # Get a sample road mask and confidence map
    sample_frame = frames[0]
    sample_road_mask, sample_conf_map = detector_basic.detect_road(sample_frame)
    
    # Benchmark coverage improvement
    start_time = time.time()
    for _ in range(10):  # Test 10 times
        improved_mask = improve_road_coverage(sample_road_mask, sample_conf_map)
    coverage_time = (time.time() - start_time) / 10
    print(f"Coverage improvement per frame: {coverage_time*1000:.1f}ms")
    
    # Benchmark hood detection
    from stage1_road_detection import SmartHoodDetector
    hood_detector = SmartHoodDetector()
    
    start_time = time.time()
    for _ in range(10):  # Test 10 times
        hood_result = hood_detector.detect_hood(sample_frame)
    hood_time = (time.time() - start_time) / 10
    print(f"Hood detection per frame: {hood_time*1000:.1f}ms")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Basic detection: {basic_fps:.1f} FPS")
    print(f"With all fixes: {fixed_fps:.1f} FPS")
    print(f"Performance drop: {((basic_fps - fixed_fps) / basic_fps * 100):.1f}%")
    print(f"Coverage improvement: {coverage_time*1000:.1f}ms/frame")
    print(f"Hood detection: {hood_time*1000:.1f}ms/frame")
    
    return {
        'basic_fps': basic_fps,
        'fixed_fps': fixed_fps,
        'coverage_time_ms': coverage_time * 1000,
        'hood_time_ms': hood_time * 1000
    }

if __name__ == "__main__":
    result = benchmark_pipeline_steps("input_videos/10secondday.mp4", num_frames=30) 