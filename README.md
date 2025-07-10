# 🚗 Dashcam Annotation Pipeline

A comprehensive computer vision pipeline for automatically annotating dashcam footage with road scene understanding, including road detection, lane detection, and advanced road markings analysis.

## 🎯 Features

### 🛣️ **Stage 1: Road Detection**

* **Multiple Models**: PyTorch DeepLabV3 & HuggingFace SegFormer
* **Advanced Edge Refinement**: Prevents bleeding onto sidewalks/buildings
* **Temporal Smoothing**: Consistent detection across frames
* **Multi-class Awareness**: Uses semantic context for better boundaries

### 🛤️ **Stage 2: Lane Detection**

* **State-of-the-Art Models**: Ultra-Fast-Lane-Detection, YOLOP, LaneATT
* **GPU Acceleration**: CUDA support with FP16 optimization
* **Real-time Processing**: Optimized for video streams
* **Confidence Scoring**: Per-lane detection confidence

### 🏷️ **Stages 3-5: Coming Soon**

* Lane Classification (solid, dashed, yellow, white)
* Special Markings (crosswalks, stop lines, arrows)
* Road Structure (medians, shoulders, barriers)

## 🚀 Quick Start

### 1\. Setup Environment

```bash
# Clone the repository
git clone https://github.com/matthewatap/roadway-annotation.git
cd roadway-annotation

# Install dependencies and setup
python setup.py
```

### 2\. Download Model Weights

#### a. DeepLabV3 (PyTorch)
```bash
mkdir -p weights/deeplabv3
wget -O weights/deeplabv3/deeplabv3_resnet101_cityscapes.pth https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
```

#### b. SegFormer (HuggingFace)
```bash
mkdir -p weights/segformer
wget -O weights/segformer/segformer_b5.pth https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024/resolve/main/pytorch_model.bin
```

#### c. Ultra-Fast-Lane-Detection
```bash
mkdir -p weights/ultra_fast
wget -O weights/ultra_fast/culane_18.pth https://github.com/cfzd/Ultra-Fast-Lane-Detection/releases/download/v1.0/culane_18.pth
```

#### d. YOLOP
```bash
mkdir -p weights/YoloP/yolop
wget -O weights/YoloP/yolop/yolop.pth https://github.com/hustvl/YOLOP/releases/download/v1.0/yolop.pth
wget -O weights/YoloP/End-to-end.pth https://github.com/hustvl/YOLOP/releases/download/v1.0/End-to-end.pth
wget -O weights/YoloP/yolop-320-320.onnx https://github.com/hustvl/YOLOP/releases/download/v1.0/yolop-320-320.onnx
wget -O weights/YoloP/yolop-640-640.onnx https://github.com/hustvl/YOLOP/releases/download/v1.0/yolop-640-640.onnx
wget -O weights/YoloP/yolop-1280-1280.onnx https://github.com/hustvl/YOLOP/releases/download/v1.0/yolop-1280-1280.onnx
```

#### e. LaneATT
```bash
mkdir -p weights/laneatt
wget -O weights/laneatt/tusimple_18.pth https://github.com/harryhan618/LaneATT/releases/download/v1.0/tusimple_18.pth
```

### 3\. Download Test Videos

```bash
# Create input videos directory
mkdir -p input_videos

# Download sample dashcam videos (optional)
# You can add your own videos to the input_videos directory
```

### 4\. Run Pipeline

```bash
# List available videos
python pipeline_runner.py --list

# Process single video (fast)
python pipeline_runner.py --videos "Road_Lane.mp4" --config fast

# Process all videos (accurate quality - default)
python pipeline_runner.py --all

# High accuracy mode
python pipeline_runner.py --all --config accurate --stages 1 2
```

## 📁 Project Structure

```
roadway-annotation/
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 📄 setup.py                  # Setup and installation script
├── 📄 .gitignore               # Git ignore rules
├── 🐍 pipeline_runner.py       # Main pipeline orchestrator
├── 🐍 stage1_road_detection.py # Road segmentation models
├── 🐍 lane_detection.py        # Lane detection models
├── 📁 input_videos/            # Input dashcam videos (ignored by git)
├── 📁 outputs/                 # Processing results (ignored by git)
└── 📁 weights/                 # Model weights cache (ignored by git)
```

## 🔧 Configuration Modes

### **Fast Mode** ⚡

* PyTorch DeepLabV3 (smaller model)
* Minimal post-processing
* \~15-20 FPS processing speed
* Good for quick previews

### **Balanced Mode** ⚖️

* PyTorch DeepLabV3 with enhancements
* Temporal smoothing + edge refinement
* \~8-12 FPS processing speed
* Good quality/speed tradeoff

### **Accurate Mode** 🎯 (Default)

* HuggingFace SegFormer-B5 (transformer model)
* Multi-scale processing (0.75x, 1.0x, 1.25x)
* Advanced edge refinement + geometric filtering
* Multi-class awareness + perspective correction
* \~3-5 FPS processing speed
* Highest quality results

## 📊 Output Files

For each processed video, the pipeline generates:

```
outputs/video_name/
├── stage1_road_detection/
│   ├── road_detection.mp4      # Visualization video
│   ├── road_detection_masks.mp4 # Binary masks
│   └── road_detection_stats.json # Processing statistics
├── stage2_lane_detection/       # (When implemented)
├── stage3_lane_classification/  # (Future)
├── stage4_special_markings/     # (Future)
├── stage5_road_structure/       # (Future)
└── processing_results.json     # Complete metadata
```

## 📋 Requirements

### Hardware

* **Minimum**: 8GB RAM, 4GB GPU VRAM
* **Recommended**: 16GB RAM, 8GB GPU VRAM
* **CUDA**: Optional but highly recommended for GPU acceleration

### Software

* Python 3.8+
* PyTorch 1.9+
* OpenCV 4.5+
* CUDA 11.0+ (optional)

See `requirements.txt` for complete dependency list. 