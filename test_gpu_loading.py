#!/usr/bin/env python3
"""
Test GPU loading for our models
"""

import torch
import numpy as np
from stage1_road_detection import ModernRoadDetector, create_improved_accurate_config

def test_gpu_loading():
    """Test if our models are actually loading on GPU"""
    
    print("=== GPU Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    print("\n=== Testing ModernRoadDetector ===")
    config = create_improved_accurate_config()
    print(f"Config model type: {config.model_type}")
    
    detector = ModernRoadDetector(config)
    print(f"Detector device: {detector.device}")
    print(f"Model type: {detector.model_type}")
    
    # Check if model is on GPU
    if hasattr(detector, 'model') and detector.model is not None:
        if hasattr(detector.model, 'parameters'):
            model_device = next(detector.model.parameters()).device
            print(f"Model device: {model_device}")
        else:
            print("Model has no parameters to check device")
    else:
        print("No model found")
    
    print(f"Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Test with a small dummy frame
    print("\n=== Testing Detection on Dummy Frame ===")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Monitor GPU usage during detection
    torch.cuda.synchronize()  # Wait for any pending operations
    start_memory = torch.cuda.memory_allocated(0)
    
    road_mask, confidence_map = detector.detect_road(dummy_frame)
    
    torch.cuda.synchronize()  # Wait for operations to complete
    end_memory = torch.cuda.memory_allocated(0)
    
    print(f"Memory used during detection: {(end_memory - start_memory) / 1024**3:.2f} GB")
    print(f"Total GPU memory now: {end_memory / 1024**3:.2f} GB")
    print(f"Detection result shape: {road_mask.shape}")
    
    return detector

if __name__ == "__main__":
    detector = test_gpu_loading() 