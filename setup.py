#!/usr/bin/env python3
"""
Setup script for Dashcam Pipeline
Installs dependencies and prepares environment
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'input_videos',
        'outputs',
        'weights',
        'weights/ultra_fast'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def test_pytorch():
    """Test PyTorch installation"""
    print("🔧 Testing PyTorch...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    
    return True

def test_opencv():
    """Test OpenCV installation"""
    print("🔧 Testing OpenCV...")
    
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("   ❌ OpenCV not installed")
        return False
    
    return True

def download_test_model():
    """Download a small test model to verify torch hub works"""
    print("🔧 Testing model download...")
    
    try:
        import torch
        # Try to load a small model to test torch hub
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        print("   ✅ Torch hub working")
        del model  # Clean up
    except Exception as e:
        print(f"   ⚠️  Torch hub test failed: {e}")
        print("   This might cause issues with model loading")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Dashcam Pipeline")
    print("=" * 50)
    
    success = True
    
    # Install dependencies
    if not install_requirements():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Test installations
    if not test_pytorch():
        success = False
    
    if not test_opencv():
        success = False
    
    if not download_test_model():
        print("   ⚠️  Model download test failed - you may need internet for first run")
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Download videos: gdown --folder 1EQ9o4gpCkAhABGtlYSp8Aqi_nNIaFVYf -O input_videos/")
        print("2. List videos: python pipeline_runner.py --list")
        print("3. Run pipeline: python pipeline_runner.py")
    else:
        print("❌ Setup had some issues - check errors above")
    
    return success

if __name__ == "__main__":
    main() 