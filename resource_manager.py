#!/usr/bin/env python3
"""
Resource Management for Runpod Safety
Monitors GPU/CPU/Memory usage and prevents system overload
"""

import psutil
import torch
import gc
import time
import threading
from typing import Dict, Optional, Callable
import warnings
import os

class ResourceMonitor:
    """Monitor system resources and prevent overload"""
    
    def __init__(self, 
                 max_gpu_memory_gb: float = 6.0,
                 max_ram_percent: float = 85.0,
                 max_cpu_percent: float = 90.0,
                 check_interval: float = 2.0):
        
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.max_ram_percent = max_ram_percent
        self.max_cpu_percent = max_cpu_percent
        self.check_interval = check_interval
        
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        
        print(f"🔧 Resource Monitor initialized:")
        print(f"   GPU Memory Limit: {max_gpu_memory_gb:.1f} GB")
        print(f"   RAM Limit: {max_ram_percent:.1f}%")
        print(f"   CPU Limit: {max_cpu_percent:.1f}%")
        print(f"   CUDA Available: {self.cuda_available}")
    
    def get_gpu_usage(self) -> Dict[str, float]:
        """Get current GPU usage"""
        if not self.cuda_available:
            return {'memory_used_gb': 0.0, 'memory_percent': 0.0}
        
        try:
            memory_used = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_percent = (memory_used / memory_total) * 100
            
            return {
                'memory_used_gb': memory_used,
                'memory_total_gb': memory_total,
                'memory_percent': memory_percent
            }
        except:
            return {'memory_used_gb': 0.0, 'memory_percent': 0.0}
    
    def get_system_usage(self) -> Dict[str, float]:
        """Get current system usage"""
        # RAM usage
        ram = psutil.virtual_memory()
        
        # CPU usage (average over 1 second)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'ram_used_gb': ram.used / (1024**3),
            'ram_total_gb': ram.total / (1024**3),
            'ram_percent': ram.percent,
            'cpu_percent': cpu_percent,
            'available_ram_gb': ram.available / (1024**3)
        }
    
    def get_all_usage(self) -> Dict[str, float]:
        """Get comprehensive resource usage"""
        gpu_usage = self.get_gpu_usage()
        system_usage = self.get_system_usage()
        
        return {**gpu_usage, **system_usage}
    
    def is_safe_to_continue(self) -> tuple[bool, str]:
        """Check if it's safe to continue processing"""
        usage = self.get_all_usage()
        
        # Check GPU memory
        if self.cuda_available and usage.get('memory_used_gb', 0) > self.max_gpu_memory_gb:
            return False, f"GPU memory too high: {usage['memory_used_gb']:.1f}GB > {self.max_gpu_memory_gb}GB"
        
        # Check RAM
        if usage['ram_percent'] > self.max_ram_percent:
            return False, f"RAM usage too high: {usage['ram_percent']:.1f}% > {self.max_ram_percent}%"
        
        # Check CPU
        if usage['cpu_percent'] > self.max_cpu_percent:
            return False, f"CPU usage too high: {usage['cpu_percent']:.1f}% > {self.max_cpu_percent}%"
        
        return True, "Resources OK"
    
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        print("🧹 Cleaning up memory...")
        
        # Python garbage collection
        gc.collect()
        
        # PyTorch GPU memory cleanup
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("   ✅ Memory cleanup complete")
    
    def wait_for_safe_resources(self, timeout: float = 60.0) -> bool:
        """Wait until resources are safe, with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            safe, reason = self.is_safe_to_continue()
            if safe:
                return True
            
            print(f"⚠️  Waiting for resources: {reason}")
            self.cleanup_memory()
            time.sleep(self.check_interval)
        
        return False
    
    def print_usage(self):
        """Print current resource usage"""
        usage = self.get_all_usage()
        
        print(f"\n📊 Resource Usage:")
        if self.cuda_available:
            print(f"   🎮 GPU Memory: {usage.get('memory_used_gb', 0):.1f}GB / {usage.get('memory_total_gb', 0):.1f}GB ({usage.get('memory_percent', 0):.1f}%)")
        print(f"   💾 RAM: {usage['ram_used_gb']:.1f}GB / {usage['ram_total_gb']:.1f}GB ({usage['ram_percent']:.1f}%)")
        print(f"   🔥 CPU: {usage['cpu_percent']:.1f}%")
        print(f"   ✅ Available RAM: {usage['available_ram_gb']:.1f}GB")
    
    def start_monitoring(self, callback: Optional[Callable] = None):
        """Start background resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        if callback:
            self.callbacks.append(callback)
        
        def monitor_loop():
            while self.monitoring:
                safe, reason = self.is_safe_to_continue()
                if not safe:
                    print(f"🚨 Resource Alert: {reason}")
                    self.cleanup_memory()
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        try:
                            callback(reason)
                        except:
                            pass
                
                time.sleep(self.check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Background resource monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Resource monitoring stopped")

class SafePipelineRunner:
    """Resource-safe pipeline runner for runpod"""
    
    def __init__(self, conservative_mode: bool = True, gpu_type: str = "auto"):
        self.conservative_mode = conservative_mode
        self.gpu_type = gpu_type
        
        # Auto-detect GPU or use specified type
        if gpu_type == "auto":
            gpu_type = self._detect_gpu_type()
        
        # GPU-specific resource limits
        if gpu_type == "l4":
            # NVIDIA L4 has 24GB VRAM - can be more aggressive
            if conservative_mode:
                self.monitor = ResourceMonitor(
                    max_gpu_memory_gb=16.0,  # Use up to 16GB of 24GB VRAM
                    max_ram_percent=80.0,    # L4 instances usually have good RAM
                    max_cpu_percent=85.0,    # Good CPU on L4 instances
                    check_interval=2.0       # Check every 2 seconds
                )
                print("🚀 L4 CONSERVATIVE mode: 16GB GPU, 80% RAM limit")
            else:
                self.monitor = ResourceMonitor(
                    max_gpu_memory_gb=20.0,  # Use up to 20GB of 24GB VRAM
                    max_ram_percent=85.0,
                    max_cpu_percent=90.0,
                    check_interval=2.0
                )
                print("⚡ L4 PERFORMANCE mode: 20GB GPU, 85% RAM limit")
        else:
            # Generic/smaller GPU settings
            if conservative_mode:
                self.monitor = ResourceMonitor(
                    max_gpu_memory_gb=4.0,   # Conservative for unknown GPU
                    max_ram_percent=75.0,
                    max_cpu_percent=80.0,
                    check_interval=3.0
                )
                print("🛡️  GENERIC CONSERVATIVE mode: 4GB GPU limit")
            else:
                self.monitor = ResourceMonitor()
                print("⚖️  GENERIC BALANCED mode")
        
        # Start background monitoring
        self.monitor.start_monitoring(self._resource_alert_callback)
    
    def _detect_gpu_type(self) -> str:
        """Auto-detect GPU type"""
        if not torch.cuda.is_available():
            return "cpu"
        
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "l4" in gpu_name:
                return "l4"
            elif "a100" in gpu_name:
                return "a100"
            elif "v100" in gpu_name:
                return "v100"
            elif "rtx" in gpu_name:
                return "rtx"
            else:
                return "generic"
        except:
            return "generic"
    
    def _resource_alert_callback(self, reason: str):
        """Handle resource alerts"""
        print(f"⚠️  Resource management: {reason}")
        print("   Taking 10 second break...")
        time.sleep(10)
    
    def safe_process_video(self, video_path: str, output_path: str, config):
        """Process video with resource safety"""
        print(f"\n🛡️  Safe processing: {video_path}")
        
        # Initial resource check
        self.monitor.print_usage()
        safe, reason = self.monitor.is_safe_to_continue()
        
        if not safe:
            print(f"❌ Cannot start processing: {reason}")
            return False
        
        try:
            # Import here to avoid loading models unnecessarily
            from stage1_road_detection import ModernRoadDetector, RoadDetectionValidator
            import cv2
            import numpy as np
            
            # Initialize with conservative settings
            print("🔧 Initializing detector...")
            detector = ModernRoadDetector(config)
            validator = RoadDetectionValidator()
            
            # Check resources after model loading
            self.monitor.print_usage()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
            
            # Setup output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            # Process frames with safety checks
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resource safety check every 10 frames
                if frame_count % 10 == 0:
                    if not self.monitor.wait_for_safe_resources(timeout=30):
                        print("❌ Resource timeout - stopping processing")
                        break
                
                # Process frame
                road_mask, confidence_map = detector.detect_road(frame)
                result = detector.visualize(frame, road_mask, style='freespace')
                out.write(result)
                
                frame_count += 1
                
                # Progress and resource update every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                    
                    print(f"Progress: {frame_count}/{total_frames} ({fps_actual:.1f} FPS, ETA: {eta:.0f}s)")
                    self.monitor.print_usage()
                    
                    # Memory cleanup every 60 frames
                    if frame_count % 60 == 0:
                        self.monitor.cleanup_memory()
            
            # Cleanup
            cap.release()
            out.release()
            
            total_time = time.time() - start_time
            print(f"✅ Processing complete: {total_time:.1f}s, {frame_count} frames")
            
            # Final cleanup
            self.monitor.cleanup_memory()
            
            return True
            
        except Exception as e:
            print(f"❌ Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.monitor.stop_monitoring()
        self.monitor.cleanup_memory()

# Convenience function for easy use
def run_safe_pipeline(video_name: str = None, conservative: bool = True, gpu_type: str = "auto"):
    """Run pipeline safely on runpod"""
    
    # Get available videos
    import glob
    videos = glob.glob("input_videos/*.mp4")
    
    if not videos:
        print("❌ No videos found in input_videos/")
        return
    
    # Select video
    if video_name:
        video_path = f"input_videos/{video_name}"
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
    else:
        # Use smallest video for testing
        video_sizes = [(v, os.path.getsize(v)) for v in videos]
        video_path = min(video_sizes, key=lambda x: x[1])[0]
        print(f"🎯 Auto-selected smallest video: {os.path.basename(video_path)}")
    
    # Setup safe configuration
    from stage1_road_detection import RoadDetectionConfig
    
    # L4-optimized configurations
    if gpu_type == "l4" or (gpu_type == "auto" and torch.cuda.is_available() and "l4" in torch.cuda.get_device_name(0).lower()):
        if conservative:
            config = RoadDetectionConfig(
                model_type="pytorch",      # Start with PyTorch, can upgrade to transformers
                conf_threshold=0.6,
                temporal_smooth=True,      # L4 can handle this
                edge_refinement=True,      # L4 can handle this
                multi_scale=False,
                scales=[1.0]
            )
            print("🚀 L4 CONSERVATIVE: Enhanced PyTorch with temporal smoothing")
        else:
            config = RoadDetectionConfig(
                model_type="transformers",  # L4 can handle SegFormer
                conf_threshold=0.5,
                temporal_smooth=True,
                edge_refinement=True,
                advanced_edge_refinement=True,
                multi_scale=False,
                scales=[1.0]
            )
            print("⚡ L4 PERFORMANCE: SegFormer with advanced features")
    else:
        # Generic/conservative settings for other GPUs
        if conservative:
            config = RoadDetectionConfig(
                model_type="pytorch",
                conf_threshold=0.7,
                temporal_smooth=False,
                edge_refinement=False,
                multi_scale=False,
                scales=[1.0]
            )
            print("🛡️  CONSERVATIVE settings for unknown GPU")
        else:
            config = RoadDetectionConfig(
                model_type="pytorch",
                conf_threshold=0.6,
                temporal_smooth=True,
                edge_refinement=True,
                multi_scale=False,
                scales=[1.0]
            )
            print("⚖️  BALANCED settings")
    
    # Create output path
    video_name = os.path.basename(video_path)
    output_dir = f"outputs/{os.path.splitext(video_name)[0]}/stage1_road_detection"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/road_detection_safe.mp4"
    
    # Run safely
    runner = SafePipelineRunner(conservative_mode=conservative, gpu_type=gpu_type)
    
    try:
        success = runner.safe_process_video(video_path, output_path, config)
        
        if success:
            print(f"\n🎉 SUCCESS!")
            print(f"   📁 Output: {output_path}")
            print(f"   📊 Check resource usage logs above")
        else:
            print(f"\n❌ FAILED - check logs above")
    
    finally:
        runner.cleanup()

if __name__ == "__main__":
    print("🚀 Safe Pipeline Runner for Runpod L4")
    print("=" * 50)
    
    # Auto-detect L4 and run optimized settings
    run_safe_pipeline(conservative=True, gpu_type="auto") 