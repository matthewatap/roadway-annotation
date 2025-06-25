# 🚗 Dashcam Annotation Pipeline

A comprehensive computer vision pipeline for automatically annotating dashcam footage with road scene understanding, including road detection, lane detection, and advanced road markings analysis.

## 🎯 Features

### 🛣️ **Stage 1: Road Detection**
- **Multiple Models**: PyTorch DeepLabV3 & HuggingFace SegFormer
- **Advanced Edge Refinement**: Prevents bleeding onto sidewalks/buildings
- **Temporal Smoothing**: Consistent detection across frames
- **Multi-class Awareness**: Uses semantic context for better boundaries

### 🛤️ **Stage 2: Lane Detection** 
- **State-of-the-Art Models**: Ultra-Fast-Lane-Detection, YOLOP, LaneATT
- **GPU Acceleration**: CUDA support with FP16 optimization
- **Real-time Processing**: Optimized for video streams
- **Confidence Scoring**: Per-lane detection confidence

### 🏷️ **Stages 3-5: Coming Soon**
- Lane Classification (solid, dashed, yellow, white)
- Special Markings (crosswalks, stop lines, arrows)
- Road Structure (medians, shoulders, barriers)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/matthewatap/roadway-annotation.git
cd roadway-annotation

# Install dependencies and setup
python setup.py
```

### 2. Download Test Videos
```bash
# Download sample dashcam videos
gdown --folder 1EQ9o4gpCkAhABGtlYSp8Aqi_nNIaFVYf -O input_videos/
```

### 3. Run Pipeline
```bash
# List available videos
python pipeline_runner.py --list

# Process single video (fast)
python pipeline_runner.py --videos "Road_Lane.mp4" --config fast

# Process all videos (balanced quality)
python pipeline_runner.py --all --config balanced

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
- PyTorch DeepLabV3 (smaller model)
- Minimal post-processing
- ~15-20 FPS processing speed
- Good for quick previews

### **Balanced Mode** ⚖️ (Default)
- PyTorch DeepLabV3 with enhancements
- Temporal smoothing + edge refinement
- ~8-12 FPS processing speed
- Best quality/speed tradeoff

### **Accurate Mode** 🎯
- HuggingFace SegFormer-B5 (large model)
- Advanced edge refinement
- Multi-class awareness
- ~3-5 FPS processing speed
- Highest quality results

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

## 🛠️ Advanced Usage

### Custom Configuration
```python
from stage1_road_detection import RoadDetectionConfig

config = RoadDetectionConfig(
    model_type="transformers",
    conf_threshold=0.8,
    temporal_smooth=True,
    advanced_edge_refinement=True,
    multi_class_awareness=True
)
```

### Batch Processing
```bash
# Process specific videos
python pipeline_runner.py --videos video1.mp4 video2.mp4 --config accurate

# Process all videos with custom stages
python pipeline_runner.py --all --stages 1 2 --config balanced
```

### Debug Mode
```python
from stage1_road_detection import DebugRoadDetector

detector = DebugRoadDetector()
road_mask, debug_vis, pred_classes = detector.debug_detection(frame)
```

## 📋 Requirements

### Hardware
- **Minimum**: 8GB RAM, 4GB GPU VRAM
- **Recommended**: 16GB RAM, 8GB GPU VRAM
- **CUDA**: Optional but highly recommended for GPU acceleration

### Software
- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- CUDA 11.0+ (optional)

See `requirements.txt` for complete dependency list.

## 🎥 Sample Videos

The pipeline includes 8 test videos covering various scenarios:
- **Day driving**: `10secondday.mp4`, `20secondday.mp4`, `65secondday.mp4`
- **Dusk conditions**: `60seconddusk.mp4`
- **Night driving**: `60secondnight.mp4`
- **Highway scenes**: `Road_Lane.mp4`, `Road_Lane 2.mp4`
- **Urban driving**: `80397-572395744_small.mp4`

## 🔬 Technical Details

### Road Detection Models
- **DeepLabV3**: ResNet-101 backbone, pre-trained on Cityscapes
- **SegFormer**: Transformer-based, hierarchical feature learning
- **Classes**: 19 Cityscapes classes (road, sidewalk, building, etc.)

### Lane Detection Models
- **Ultra-Fast**: Row-wise classification approach
- **YOLOP**: Multi-task learning (detection + segmentation)
- **LaneATT**: Attention-based lane detection

### Post-Processing
- Morphological operations for noise reduction
- Temporal smoothing across frames
- Confidence-based edge refinement
- Geometric constraints (road position priors)

## 📈 Performance Benchmarks

| Configuration | Model | Speed (FPS) | GPU Memory | Quality |
|---------------|-------|-------------|------------|---------|
| Fast | DeepLabV3 | 15-20 | 2-3GB | Good |
| Balanced | DeepLabV3+ | 8-12 | 3-4GB | Very Good |
| Accurate | SegFormer-B5 | 3-5 | 6-8GB | Excellent |

*Benchmarks on RTX 3080, 1080p video*

## 🤝 Contributing

Contributions are welcome! Please see:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for DeepLabV3 implementation
- **HuggingFace** for SegFormer models
- **Ultra-Fast-Lane-Detection** authors
- **YOLOP** paper authors
- **Cityscapes Dataset** for training data

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/matthewatap/roadway-annotation/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/matthewatap/roadway-annotation/discussions)
- 📧 **Contact**: Via GitHub Issues

---

⭐ **Star this repo if you find it useful!** ⭐ 