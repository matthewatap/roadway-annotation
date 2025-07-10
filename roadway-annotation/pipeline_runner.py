#!/usr/bin/env python3
"""
Multi-Video Pipeline Runner
Process 1, 2, or all videos through the staged annotation pipeline
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import json
import time
import numpy as np
from stage1_road_detection import RoadDetectionConfig, process_video_stage1, process_video

def get_available_videos(input_dir: str = "input_videos") -> list:
    """Get list of available videos in input directory"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    videos = []
    
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(input_dir, ext)))
    
    return sorted(videos)

def create_output_structure(video_name: str, base_output_dir: str = "outputs") -> dict:
    """Create organized output directory structure for a video"""
    video_stem = Path(video_name).stem
    
    structure = {
        'base_dir': os.path.join(base_output_dir, video_stem),
        'stage1_dir': os.path.join(base_output_dir, video_stem, 'stage1_road_detection'),
        'stage2_dir': os.path.join(base_output_dir, video_stem, 'stage2_lane_detection'),
        'stage3_dir': os.path.join(base_output_dir, video_stem, 'stage3_lane_classification'),
        'stage4_dir': os.path.join(base_output_dir, video_stem, 'stage4_special_markings'),
        'stage5_dir': os.path.join(base_output_dir, video_stem, 'stage5_road_structure'),
        'final_dir': os.path.join(base_output_dir, video_stem, 'final_output')
    }
    
    # Create directories
    for dir_path in structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return structure

def process_single_video(video_path: str, config: RoadDetectionConfig, stages: list = [1]) -> dict:
    """Process a single video through specified stages"""
    print(f"\n{'='*60}")
    print(f"🎬 Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # Create output structure
    output_struct = create_output_structure(os.path.basename(video_path))
    
    results = {
        'video_path': video_path,
        'video_name': os.path.basename(video_path),
        'output_structure': output_struct,
        'stages_completed': [],
        'stage_outputs': {},
        'total_processing_time': 0
    }
    
    start_time = time.time()
    
    # Stage 1: Modern Road Detection
    if 1 in stages:
        print(f"\n🛣️  Stage 1: Modern Road Detection")
        stage1_output = os.path.join(output_struct['stage1_dir'], 'road_detection.mp4')
        
        try:
            # Import the functions from stage1_road_detection directly
            from stage1_road_detection import (
                ModernRoadDetector, 
                RoadDetectionValidator,
                create_improved_accurate_config,
                apply_road_detection_fixes
            )
            import cv2
            import numpy as np
            
            # Use our improved config that trusts the model's semantic understanding
            if config.model_type == "transformers":
                # Always use the improved accurate config - it's not slow anymore!
                improved_config = create_improved_accurate_config()
                print("Using IMPROVED ACCURATE config (trusts SegFormer-B5 semantic understanding)")
                print(f"  - Multi-class awareness: {improved_config.multi_class_awareness}")
                print(f"  - Conf threshold: {improved_config.conf_threshold}")
                print(f"  - Trusts model instead of hardcoded hood values")
            else:
                improved_config = config
            
            # Initialize detector
            detector = ModernRoadDetector(improved_config)
            
            # Apply minimal enhancements that trust the model's understanding
            detector = apply_road_detection_fixes(detector, use_smart_hood=True)
            print("Applied minimal enhancements - trusting SegFormer-B5 semantic understanding")
            
            validator = RoadDetectionValidator()
            
            # Open video
            print(f"Opening video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            print(f"Video opened: {cap.isOpened()}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            if total_frames == 0:
                print("ERROR: Total frames is 0!")
                cap.release()
                raise ValueError("Could not read video properties")
            
            # Output video with robust error handling
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(stage1_output, fourcc, fps, (width, height))
            
            # Mask output
            mask_output = stage1_output.replace('.mp4', '_masks.mp4')
            mask_out = cv2.VideoWriter(mask_output, fourcc, fps, (width, height), isColor=False)
            
            if not out.isOpened() or not mask_out.isOpened():
                cap.release()
                if out.isOpened():
                    out.release()
                if mask_out.isOpened():
                    mask_out.release()
                raise RuntimeError("Failed to create video writers")
            
            print(f"Processing {total_frames} frames...")
            
            frame_count = 0
            processing_start_time = time.time()
            all_metrics = []
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect road with adaptive calibration
                    road_mask, confidence_map = detector.detect_road(frame)
                    
                    # Create metadata for compatibility
                    road_pixels = np.sum(road_mask > 0)
                    total_pixels = road_mask.shape[0] * road_mask.shape[1]
                    road_percentage = (road_pixels / total_pixels) * 100
                    
                    metadata = {
                        'road_percentage': road_percentage,
                        'frame_shape': frame.shape[:2],
                        'model_type': detector.config.model_type,
                        'hood_exclusion': True,
                        'coverage_improved': True,
                        'confidence_threshold': detector.config.conf_threshold,
                        'frame_count': frame_count
                    }
                    
                    # Validate
                    metrics = validator.calculate_metrics(road_mask)
                    metadata.update(metrics)
                    all_metrics.append(metadata)
                    
                    # Visualize with freespace style
                    result = detector.visualize(frame, road_mask, style='freespace')
                    
                    # Write outputs
                    out.write(result)
                    mask_out.write(road_mask)
                    
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        elapsed = time.time() - processing_start_time
                        fps_actual = frame_count / elapsed
                        eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                        print(f"Progress: {frame_count}/{total_frames} ({fps_actual:.1f} FPS, ETA: {eta:.0f}s)")
            
            finally:
                # CRITICAL: Always release video resources in finally block to prevent corruption
                print("🔄 Releasing video resources...")
                cap.release()
                out.release()
                mask_out.release()
                print("✅ Video resources released successfully")
            
            total_time = time.time() - processing_start_time
            
            # Save statistics
            stats = {
                'total_frames': frame_count,
                'total_time': total_time,
                'avg_fps': frame_count / total_time if total_time > 0 else 0,
                'avg_road_percentage': np.mean([m['road_percentage'] for m in all_metrics]) if all_metrics else 0,
                'model_type': detector.model_type,
                'quality_metrics': {
                    'avg_regions': np.mean([m['num_regions'] for m in all_metrics]) if all_metrics else 0,
                    'avg_largest_region_ratio': np.mean([m['largest_region_ratio'] for m in all_metrics]) if all_metrics else 0,
                    'avg_edge_smoothness': np.mean([m['edge_smoothness'] for m in all_metrics]) if all_metrics else 0
                }
            }
            
            stats_file = stage1_output.replace('.mp4', '_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            results['stages_completed'].append(1)
            results['stage_outputs']['stage1'] = {
                'visualization': stage1_output,
                'masks': mask_output,
                'stats': stats_file
            }
            
            print(f"✓ Stage 1 completed successfully")
            print(f"  - Processed {frame_count} frames in {total_time:.1f}s")
            print(f"  - Average FPS: {frame_count/total_time:.1f}")
            print(f"  - Avg Road Coverage: {stats['avg_road_percentage']:.1f}%")
            print(f"  - Model: {stats['model_type']}")
            
        except Exception as e:
            print(f"❌ Stage 1 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['stage_outputs']['stage1'] = {'error': str(e)}
    
    # Stage 2: Lane Detection
    if 2 in stages:
        print(f"\n🛣️  Stage 2: Advanced Lane Detection")
        stage2_output = os.path.join(output_struct['stage2_dir'], 'lane_detection.mp4')
        
        try:
            # Import improved lane detection
            from lane_detection import AdvancedLaneDetector, LaneDetectionConfig, LaneModel
            import cv2
            
            # Configure lane detection
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            lane_config = LaneDetectionConfig(
                model=LaneModel.ULTRA_FAST,
                device=device,
                confidence_threshold=0.5,
                max_lanes=4,
                use_fp16=True if device == "cuda" else False,
                visualize=True
            )
            
            # Initialize detector
            lane_detector = AdvancedLaneDetector(lane_config)
            
            # Open input video (use original, not road detection output)
            print(f"Processing lanes in: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Lane detection on {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(stage2_output, fourcc, fps, (width, height))
            
            if not out.isOpened():
                cap.release()
                raise RuntimeError("Failed to create video writer for stage 2")
            
            # Process frames
            frame_count = 0
            processing_start_time = time.time()
            all_lane_results = []
            total_lanes_detected = 0
            
            print(f"Processing {total_frames} frames for lane detection...")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect lanes
                    result = lane_detector.detector.detect_lanes(frame)
                    lanes = result['lanes']
                    confidences = result['confidence']
                    
                    # Store results
                    frame_result = {
                        'frame': frame_count + 1,
                        'lanes': [lane.tolist() for lane in lanes],
                        'confidence': confidences,
                        'timestamp': frame_count / fps
                    }
                    all_lane_results.append(frame_result)
                    total_lanes_detected += len(lanes)
                    
                    # Visualize
                    if lane_config.visualize:
                        vis_frame = lane_detector._visualize_lanes(frame, lanes, confidences)
                        
                        # Add info text
                        info_text = f"Frame {frame_count+1}/{total_frames} | Lanes: {len(lanes)} | Model: {lane_config.model.value}"
                        cv2.putText(vis_frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        out.write(vis_frame)
                    else:
                        out.write(frame)
                    
                    frame_count += 1
                    
                    # Progress update
                    if frame_count % 30 == 0:
                        elapsed = time.time() - processing_start_time
                        fps_actual = frame_count / elapsed
                        eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                        print(f"Lane Progress: {frame_count}/{total_frames} ({fps_actual:.1f} FPS, ETA: {eta:.0f}s)")
            
            finally:
                # CRITICAL: Always release video resources to prevent corruption
                cap.release()
                out.release()
            
            total_time = time.time() - processing_start_time
            avg_lanes = total_lanes_detected / frame_count if frame_count > 0 else 0
            
            # Save lane detection results
            lane_results = {
                'video_info': {
                    'input_path': video_path,
                    'output_path': stage2_output,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': total_frames
                },
                'detection_info': {
                    'model': lane_config.model.value,
                    'device': lane_config.device,
                    'processing_time': total_time,
                    'fps_processed': frame_count / total_time,
                    'total_lanes_detected': total_lanes_detected,
                    'average_lanes_per_frame': avg_lanes,
                    'confidence_threshold': lane_config.confidence_threshold
                },
                'frame_results': all_lane_results
            }
            
            # Save JSON results
            json_path = stage2_output.replace('.mp4', '_lanes.json')
            with open(json_path, 'w') as f:
                json.dump(lane_results, f, indent=2)
            
            results['stages_completed'].append(2)
            results['stage_outputs']['stage2'] = {
                'visualization': stage2_output,
                'lane_data': json_path,
                'stats': lane_results['detection_info']
            }
            
            print(f"✓ Stage 2 completed successfully")
            print(f"  - Processed {frame_count} frames in {total_time:.1f}s")
            print(f"  - Average FPS: {frame_count/total_time:.1f}")
            print(f"  - Total lanes detected: {total_lanes_detected}")
            print(f"  - Average lanes per frame: {avg_lanes:.2f}")
            print(f"  - Model: {lane_config.model.value}")
            print(f"  - Output: {stage2_output}")
            print(f"  - Lane data: {json_path}")
            
        except Exception as e:
            print(f"❌ Stage 2 failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results['stage_outputs']['stage2'] = {'error': str(e)}
    
    # Stage 3: Lane Classification (Coming Next)
    if 3 in stages:
        print(f"\n🏷️  Stage 3: Lane Classification (Coming Soon)")
        # TODO: Implement Stage 3
        print("   - Classify lane types (solid, dashed, yellow, white)")
        print("   - Consider lane position context")
    
    # Stage 4: Special Markings (Coming Next)
    if 4 in stages:
        print(f"\n🚦 Stage 4: Special Markings (Coming Soon)")
        # TODO: Implement Stage 4
        print("   - Detect crosswalks, stop lines, arrows")
        print("   - Use specialized marking detection models")
    
    # Stage 5: Road Structure (Coming Next)
    if 5 in stages:
        print(f"\n🏗️  Stage 5: Road Structure (Coming Soon)")
        # TODO: Implement Stage 5
        print("   - Detect medians, shoulders, barriers")
        print("   - Complete road scene understanding")
    
    results['total_processing_time'] = time.time() - start_time
    
    # Save processing results
    results_file = os.path.join(output_struct['base_dir'], 'processing_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Video processing complete!")
    print(f"   📁 Results saved to: {output_struct['base_dir']}")
    print(f"   ⏱️  Total time: {results['total_processing_time']:.1f}s")
    
    return results

def main():
    """Main pipeline runner"""
    parser = argparse.ArgumentParser(description='Multi-Video Dashcam Annotation Pipeline')
    parser.add_argument('--videos', type=str, nargs='*', 
                       help='Specific video names to process (e.g., Road_Lane.mp4)')
    parser.add_argument('--all', action='store_true',
                       help='Process all videos in input_videos folder')
    parser.add_argument('--list', action='store_true',
                       help='List available videos')
    parser.add_argument('--stages', type=int, nargs='*', default=[1, 2],
                       help='Stages to run (1-5), default: [1, 2]')
    parser.add_argument('--config', type=str, choices=['fast', 'balanced', 'accurate'], 
                       default='accurate', help='Processing configuration')
    
    args = parser.parse_args()
    
    # Get available videos
    available_videos = get_available_videos()
    
    if not available_videos:
        print("❌ No videos found in input_videos/ directory")
        return
    
    # List available videos
    if args.list:
        print("\n📹 Available videos:")
        for i, video in enumerate(available_videos, 1):
            video_name = os.path.basename(video)
            file_size = os.path.getsize(video) / (1024*1024)  # MB
            print(f"   {i}. {video_name} ({file_size:.1f} MB)")
        return
    
    # Configure processing settings
    config_presets = {
        'fast': RoadDetectionConfig(
            model_type="pytorch",
            conf_threshold=0.6,
            temporal_smooth=False,
            edge_refinement=False,
            multi_scale=False,
            scales=[1.0]
        ),
        'balanced': RoadDetectionConfig(
            model_type="pytorch",  # Use pytorch instead of transformers for better compatibility
            conf_threshold=0.6,
            temporal_smooth=True,
            edge_refinement=True,
            perspective_correction=True,
            multi_scale=False,
            scales=[1.0]
        ),
        'accurate': RoadDetectionConfig(
            model_type="transformers",  # SegFormer-B5 for best accuracy
            conf_threshold=0.77,
            temporal_smooth=True,
            edge_refinement=True,
            advanced_edge_refinement=True,  # NEW: Enhanced edge refinement
            confidence_edge_threshold=0.85,  # NEW: Higher confidence at edges
            multi_class_awareness=True,      # NEW: Use other classes to constrain roads
            geometric_filtering=True,        # NEW: Geometric constraints
            bilateral_filter=True,           # NEW: Edge-preserving smoothing
            perspective_correction=True,
            multi_scale=True,
            scales=[0.75, 1.0, 1.25]
        )
    }
    
    config = config_presets[args.config]
    
    print(f"\n🚀 Dashcam Annotation Pipeline")
    print(f"   📋 Configuration: {args.config}")
    print(f"   🎯 Stages: {args.stages}")
    print(f"   🎬 Videos found: {len(available_videos)}")
    
    # Determine which videos to process
    videos_to_process = []
    
    if args.all:
        videos_to_process = available_videos
        print(f"   ▶️  Processing ALL videos")
    elif args.videos:
        # Process specific videos
        for video_name in args.videos:
            # Check if it's a full path or just a filename
            if os.path.exists(video_name):
                videos_to_process.append(video_name)
            else:
                # Look for it in available videos
                matching_videos = [v for v in available_videos if os.path.basename(v) == video_name]
                if matching_videos:
                    videos_to_process.extend(matching_videos)
                else:
                    print(f"   ⚠️  Video not found: {video_name}")
        
        if videos_to_process:
            print(f"   ▶️  Processing specific videos: {[os.path.basename(v) for v in videos_to_process]}")
    else:
        # Interactive selection
        print(f"\n📹 Available videos:")
        for i, video in enumerate(available_videos, 1):
            video_name = os.path.basename(video)
            file_size = os.path.getsize(video) / (1024*1024)  # MB
            print(f"   {i}. {video_name} ({file_size:.1f} MB)")
        
        while True:
            try:
                choice = input(f"\nSelect video number (1-{len(available_videos)}) or 'all': ").strip()
                if choice.lower() == 'all':
                    videos_to_process = available_videos
                    break
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_videos):
                        videos_to_process = [available_videos[idx]]
                        break
                    else:
                        print(f"Invalid choice. Please enter 1-{len(available_videos)} or 'all'")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return
    
    if not videos_to_process:
        print("❌ No videos selected for processing")
        return
    
    # Process videos
    all_results = []
    total_start_time = time.time()
    
    for i, video_path in enumerate(videos_to_process, 1):
        print(f"\n🎬 Processing video {i}/{len(videos_to_process)}")
        
        try:
            result = process_single_video(video_path, config, args.stages)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(video_path)}: {str(e)}")
            all_results.append({
                'video_path': video_path,
                'video_name': os.path.basename(video_path),
                'error': str(e)
            })
    
    # Summary
    total_time = time.time() - total_start_time
    successful = len([r for r in all_results if 'error' not in r])
    failed = len(all_results) - successful
    
    print(f"\n{'='*60}")
    print(f"🏁 PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📁 Results in: outputs/")
    
    # Save overall summary
    summary = {
        'total_videos': len(videos_to_process),
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'config': args.config,
        'stages': args.stages,
        'results': all_results
    }
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    with open('outputs/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📊 Summary saved to: outputs/pipeline_summary.json")

if __name__ == "__main__":
    main() 