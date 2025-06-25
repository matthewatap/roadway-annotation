#!/usr/bin/env python3
"""
Stage 1: Road Detection for Dashcam Annotation Pipeline
Integrated with debug capabilities and fixed detection methods
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import os

@dataclass
class RoadDetectionConfig:
    """Configuration for road detection"""
    model_type: str = "pytorch"  # "pytorch", "transformers", or "debug"
    conf_threshold: float = 0.7
    temporal_smooth: bool = True
    edge_refinement: bool = True
    perspective_correction: bool = False
    multi_scale: bool = False
    scales: list = None
    debug_mode: bool = False
    
    # Enhanced edge refinement parameters
    advanced_edge_refinement: bool = False
    confidence_edge_threshold: float = 0.8  # Higher confidence required at edges
    geometric_filtering: bool = False       # Filter based on road geometry
    multi_class_awareness: bool = False     # Use other classes to constrain roads
    bilateral_filter: bool = False          # Edge-preserving smoothing
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1.0]

class ModernRoadDetector:
    """Modern road detector with multiple backends"""
    
    def __init__(self, config: Optional[RoadDetectionConfig] = None):
        self.config = config or RoadDetectionConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.prev_mask = None
        
        # Cityscapes classes for reference
        self.cityscapes_classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the appropriate model based on config"""
        if self.config.model_type in ["pytorch", "debug"]:
            print(f"Loading PyTorch DeepLabV3 model on {self.device}")
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            self.model_type = "pytorch"
        elif self.config.model_type == "transformers":
            print("Loading SegFormer model...")
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            self.model.to(self.device)
            self.model.eval()
            self.model_type = "transformers"
            print(f"Using transformers for road detection")
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def detect_road(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Main detection method - returns (road_mask, confidence_map)"""
        if self.config.model_type in ["pytorch", "debug"]:
            return self._detect_pytorch(frame)
        elif self.config.model_type == "transformers":
            return self._detect_transformers(frame)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _detect_pytorch(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """PyTorch DeepLabV3 detection with fixed method"""
        original_size = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess_pytorch(frame)
        
        # Get model output
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        
        # Resize to original
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        # Get road probability directly (fixed method)
        probs = F.softmax(output, dim=1)
        road_prob = probs[0, 0].cpu().numpy()  # Class 0 is road
        confidence_map = road_prob.copy()
        
        # Use higher threshold to be more selective (fixed approach)
        road_mask = (road_prob > self.config.conf_threshold).astype(np.uint8) * 255
        
        # Minimal post-processing to preserve accuracy
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove very small components
        if self.config.edge_refinement:
            road_mask = self._clean_small_components(road_mask)
        
        # Temporal smoothing
        if self.config.temporal_smooth and self.prev_mask is not None:
            road_mask = cv2.addWeighted(road_mask, 0.8, self.prev_mask, 0.2, 0)
        
        self.prev_mask = road_mask.copy()
        
        return road_mask, confidence_map
    
    def _detect_transformers(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SegFormer transformers detection"""
        original_size = frame.shape[:2]
        
        # Preprocess with transformers
        inputs = self.processor(images=frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Resize to original
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
        
        # Get road probability
        probs = F.softmax(logits, dim=1)
        road_prob = probs[0, 0].cpu().numpy()
        confidence_map = road_prob.copy()
        
        # Create mask
        road_mask = (road_prob > self.config.conf_threshold).astype(np.uint8) * 255
        
        # Enhanced edge refinement
        if self.config.advanced_edge_refinement:
            road_mask = self._advanced_edge_refinement(frame, road_mask, confidence_map, probs)
        elif self.config.edge_refinement:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            road_mask = self._clean_small_components(road_mask)
        
        # Temporal smoothing
        if self.config.temporal_smooth and self.prev_mask is not None:
            road_mask = cv2.addWeighted(road_mask, 0.8, self.prev_mask, 0.2, 0)
        
        self.prev_mask = road_mask.copy()
        
        return road_mask, confidence_map
    
    def _advanced_edge_refinement(self, frame: np.ndarray, road_mask: np.ndarray, 
                                confidence_map: np.ndarray, all_probs: torch.Tensor) -> np.ndarray:
        """Advanced edge refinement to prevent bleeding onto walls/sidewalks"""
        refined_mask = road_mask.copy()
        
        # 1. Confidence-based edge refinement
        if self.config.confidence_edge_threshold > self.config.conf_threshold:
            refined_mask = self._confidence_edge_refinement(refined_mask, confidence_map)
        
        # 2. Multi-class awareness (prevent bleeding onto sidewalks/buildings)
        if self.config.multi_class_awareness:
            refined_mask = self._multiclass_edge_refinement(refined_mask, all_probs)
        
        # 3. Geometric filtering (roads are typically horizontal)
        if self.config.geometric_filtering:
            refined_mask = self._geometric_edge_filtering(refined_mask, frame)
        
        # 4. Edge-preserving bilateral filtering
        if self.config.bilateral_filter:
            refined_mask = self._bilateral_edge_filter(refined_mask, frame)
        
        # 5. Clean up small components
        refined_mask = self._clean_small_components(refined_mask)
        
        return refined_mask
    
    def _confidence_edge_refinement(self, road_mask: np.ndarray, confidence_map: np.ndarray) -> np.ndarray:
        """Require higher confidence at road edges to prevent bleeding"""
        # Find edge pixels
        edges = cv2.Canny(road_mask, 50, 150)
        
        # Dilate edges to create edge zone
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_zone = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply higher threshold in edge zones
        refined_mask = road_mask.copy()
        edge_pixels = edge_zone > 0
        high_conf_edges = confidence_map > self.config.confidence_edge_threshold
        
        # Keep edge pixels only if they have high confidence
        refined_mask[edge_pixels & ~high_conf_edges] = 0
        
        return refined_mask
    
    def _multiclass_edge_refinement(self, road_mask: np.ndarray, all_probs: torch.Tensor) -> np.ndarray:
        """Use other class predictions to constrain road boundaries"""
        # Get all class probabilities
        probs_np = all_probs[0].cpu().numpy()  # Shape: [num_classes, H, W]
        
        # Classes that should NOT be road (sidewalk=1, building=2, wall=3, fence=4)
        non_road_classes = [1, 2, 3, 4, 8]  # sidewalk, building, wall, fence, vegetation
        
        # Create mask of areas strongly predicted as non-road
        non_road_mask = np.zeros(road_mask.shape[:2], dtype=bool)
        for cls in non_road_classes:
            if cls < probs_np.shape[0]:
                # Areas with >60% confidence of being this non-road class
                non_road_mask |= probs_np[cls] > 0.6
        
        # Remove road pixels that are strongly predicted as non-road
        refined_mask = road_mask.copy()
        refined_mask[non_road_mask] = 0
        
        return refined_mask
    
    def _geometric_edge_filtering(self, road_mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Filter edges based on road geometry (roads are typically horizontal)"""
        h, w = road_mask.shape
        
        # Create gravity mask - lower part of image more likely to be road
        y_coords = np.arange(h).reshape(-1, 1)
        gravity_weight = (y_coords / h) ** 0.5  # Stronger weight towards bottom
        
        # Find contours
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        refined_mask = np.zeros_like(road_mask)
        
        for contour in contours:
            # Calculate contour's center of mass
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                
                # Check if contour is in reasonable road position (not too high)
                if cy > h * 0.3:  # Road shouldn't be in top 30% of image
                    cv2.drawContours(refined_mask, [contour], -1, 255, -1)
        
        return refined_mask
    
    def _bilateral_edge_filter(self, road_mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Apply edge-preserving bilateral filter to smooth boundaries"""
        # Convert to float for bilateral filtering
        mask_float = road_mask.astype(np.float32) / 255.0
        
        # Apply bilateral filter using original image as guide
        filtered = cv2.bilateralFilter(mask_float, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Convert back to binary mask
        refined_mask = (filtered > 0.5).astype(np.uint8) * 255
        
        return refined_mask
    
    def _preprocess_pytorch(self, image: np.ndarray) -> torch.Tensor:
        """Standard PyTorch preprocessing"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def _clean_small_components(self, mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected components"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only significant components
            min_area = mask.shape[0] * mask.shape[1] * 0.001  # 0.1% of image
            mask_clean = np.zeros_like(mask)
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    cv2.drawContours(mask_clean, [contour], -1, 255, -1)
            return mask_clean
        return mask
    
    def visualize(self, frame: np.ndarray, road_mask: np.ndarray, style: str = "freespace") -> np.ndarray:
        """Create visualization of road detection"""
        if style == "freespace":
            return self._visualize_freespace(frame, road_mask)
        elif style == "debug":
            return self._visualize_debug(frame, road_mask)
        else:
            return self._visualize_overlay(frame, road_mask)
    
    def _visualize_freespace(self, frame: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        """Clean freespace-style visualization"""
        overlay = frame.copy()
        
        # Semi-transparent blue overlay for road
        mask_bool = road_mask > 0
        blue_color = np.array([255, 100, 0])  # BGR - orange-blue
        overlay[mask_bool] = (overlay[mask_bool] * 0.6 + blue_color * 0.4).astype(np.uint8)
        
        # Add orange boundaries
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 165, 255), 2)
        
        # Add info
        coverage = np.sum(road_mask > 0) / (road_mask.shape[0] * road_mask.shape[1]) * 100
        cv2.putText(overlay, f"Road Detection ({self.model_type})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Coverage: {coverage:.1f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay
    
    def _visualize_overlay(self, frame: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        """Simple overlay visualization"""
        overlay = frame.copy()
        mask_bool = road_mask > 0
        overlay[mask_bool] = [0, 255, 0]  # Green overlay
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def _visualize_debug(self, frame: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
        """Debug visualization with detailed info"""
        # For debug mode, we need to run the full debug detection
        if hasattr(self, '_last_debug_vis'):
            return self._last_debug_vis
        else:
            return self._visualize_freespace(frame, road_mask)

class DebugRoadDetector(ModernRoadDetector):
    """Extended detector with debug capabilities"""
    
    def __init__(self, config: Optional[RoadDetectionConfig] = None):
        if config is None:
            config = RoadDetectionConfig(model_type="debug", debug_mode=True)
        super().__init__(config)
    
    def debug_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Debug what the model is actually detecting"""
        original_size = frame.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess_pytorch(frame)
        
        # Get model output
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        
        # Resize to original
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        # Get predictions for each class
        probs = F.softmax(output, dim=1)[0]  # Shape: [num_classes, H, W]
        
        # Get the predicted class for each pixel
        pred_classes = torch.argmax(probs, dim=0).cpu().numpy()
        
        # Debug: Show what classes are being predicted
        unique_classes, counts = np.unique(pred_classes, return_counts=True)
        print("\nDetected classes in frame:")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / pred_classes.size) * 100
            if cls < len(self.cityscapes_classes):
                print(f"  Class {cls} ({self.cityscapes_classes[cls]}): {percentage:.1f}%")
            else:
                print(f"  Class {cls} (unknown): {percentage:.1f}%")
        
        # Create visualizations for debugging
        debug_vis = self._create_debug_visualization(frame, pred_classes, probs)
        
        # Create proper road mask
        road_mask = self._create_proper_road_mask(pred_classes, probs)
        
        self._last_debug_vis = debug_vis
        
        return road_mask, debug_vis, pred_classes
    
    def _create_debug_visualization(self, frame: np.ndarray, pred_classes: np.ndarray, probs: torch.Tensor) -> np.ndarray:
        """Create visualization showing what model sees"""
        h, w = frame.shape[:2]
        
        # Create a color map for different classes
        color_map = {
            0: [128, 64, 128],    # road - purple
            1: [244, 35, 232],    # sidewalk - pink
            2: [70, 70, 70],      # building - gray
            8: [107, 142, 35],    # vegetation - green
            10: [70, 130, 180],   # sky - blue
            13: [0, 0, 142],      # car - dark blue
            # Add more as needed
        }
        
        # Create colored segmentation
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in color_map.items():
            seg_colored[pred_classes == cls] = color
        
        # Overlay on original
        overlay = cv2.addWeighted(frame, 0.7, seg_colored, 0.3, 0)
        
        # Add class labels
        y_pos = 30
        for cls in np.unique(pred_classes):
            if cls in color_map and cls < len(self.cityscapes_classes):
                color = color_map[cls]
                cv2.rectangle(overlay, (10, y_pos-20), (30, y_pos-5), color, -1)
                cv2.putText(overlay, self.cityscapes_classes[cls], (35, y_pos-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25
        
        return overlay
    
    def _create_proper_road_mask(self, pred_classes: np.ndarray, probs: torch.Tensor) -> np.ndarray:
        """Create road mask using only actual road pixels"""
        # Method 1: Use only pixels classified as road (class 0)
        road_mask = (pred_classes == 0).astype(np.uint8) * 255
        
        # Clean up with minimal post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small disconnected components
        road_mask = self._clean_small_components(road_mask)
        
        return road_mask

class RoadDetectionValidator:
    """Validator for road detection quality"""
    
    def calculate_metrics(self, road_mask: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for road detection"""
        # Basic coverage
        total_pixels = road_mask.shape[0] * road_mask.shape[1]
        road_pixels = np.sum(road_mask > 0)
        coverage = (road_pixels / total_pixels) * 100
        
        # Region analysis
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_regions = len(contours)
        
        # Largest region ratio
        largest_region_ratio = 0
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            largest_area = max(areas)
            largest_region_ratio = (largest_area / road_pixels) * 100 if road_pixels > 0 else 0
        
        # Edge smoothness (simplified)
        edges = cv2.Canny(road_mask, 50, 150)
        edge_pixels = np.sum(edges > 0)
        edge_smoothness = 100 - min(100, (edge_pixels / road_pixels * 100)) if road_pixels > 0 else 0
        
        return {
            'coverage': coverage,
            'num_regions': num_regions,
            'largest_region_ratio': largest_region_ratio,
            'edge_smoothness': edge_smoothness
        }

# Legacy support functions for pipeline compatibility
def process_video_stage1(input_path: str, output_path: str, config: RoadDetectionConfig) -> str:
    """Legacy function for pipeline compatibility"""
    detector = ModernRoadDetector(config)
    validator = RoadDetectionValidator()
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Mask output
    mask_output = output_path.replace('.mp4', '_masks.mp4')
    mask_out = cv2.VideoWriter(mask_output, fourcc, fps, (width, height), isColor=False)
    
    print(f"Processing {total_frames} frames...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect road
        road_mask, confidence_map = detector.detect_road(frame)
        
        # Visualize
        result = detector.visualize(frame, road_mask, style='freespace')
        
        # Write outputs
        out.write(result)
        mask_out.write(road_mask)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
            print(f"Progress: {frame_count}/{total_frames} ({fps_actual:.1f} FPS, ETA: {eta:.0f}s)")
    
    cap.release()
    out.release()
    mask_out.release()
    
    total_time = time.time() - start_time
    print(f"✓ Stage 1 completed in {total_time:.1f}s")
    
    return mask_output

def process_video(input_path: str, output_path: str, config: Optional[RoadDetectionConfig] = None) -> str:
    """Modern process_video function"""
    if config is None:
        config = RoadDetectionConfig()
    return process_video_stage1(input_path, output_path, config)

# Debug functions
def debug_single_frame(video_path: str, frame_number: int = 100):
    """Debug a single frame to understand the issue"""
    print("=== Debugging Road Detection ===")
    
    # Initialize detector
    detector = DebugRoadDetector()
    
    # Get specific frame
    if not os.path.exists(video_path):
        video_path = f"input_videos/{video_path}"
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame")
        return
    
    # Debug detection
    print(f"\nAnalyzing frame {frame_number}...")
    road_mask, debug_vis, pred_classes = detector.debug_detection(frame)
    
    # Try alternative method
    road_mask_alt, _ = detector.detect_road(frame)
    
    # Create visualizations
    road_vis = detector.visualize(frame, road_mask, style='freespace')
    road_vis_alt = detector.visualize(frame, road_mask_alt, style='freespace')
    
    # Save debug outputs
    cv2.imwrite("debug_original.jpg", frame)
    cv2.imwrite("debug_all_classes.jpg", debug_vis)
    cv2.imwrite("debug_road_only.jpg", road_vis)
    cv2.imwrite("debug_road_alternative.jpg", road_vis_alt)
    cv2.imwrite("debug_road_mask.jpg", road_mask)
    
    print("\nDebug images saved:")
    print("  - debug_original.jpg: Original frame")
    print("  - debug_all_classes.jpg: All detected classes")
    print("  - debug_road_only.jpg: Road detection (method 1)")
    print("  - debug_road_alternative.jpg: Road detection (method 2)")
    print("  - debug_road_mask.jpg: Binary road mask")

def process_video_fixed(input_path: str, output_path: str):
    """Process video with fixed road detection"""
    print("=== Fixed Road Detection ===")
    
    config = RoadDetectionConfig(model_type="pytorch", conf_threshold=0.7, temporal_smooth=True)
    detector = ModernRoadDetector(config)
    
    if not os.path.exists(input_path):
        input_path = f"input_videos/{input_path}"
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use fixed detection method
        road_mask, _ = detector.detect_road(frame)
        
        # Visualize
        result = detector.visualize(frame, road_mask, style='freespace')
        
        out.write(result)
        frame_count += 1
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"Processed {frame_count}/{total_frames} frames ({fps_actual:.1f} FPS)")
    
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"✓ Complete! Output: {output_path} ({total_time:.1f}s)")

if __name__ == "__main__":
    # First, debug a single frame to understand the issue
    debug_single_frame("Road_Lane.mp4", frame_number=100)
    
    # Then process the full video with fixed detection
    process_video_fixed("Road_Lane.mp4", "road_detection_fixed.mp4") 