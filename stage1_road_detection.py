#!/usr/bin/env python3
"""
Stage 1: Road Detection for Dashcam Annotation Pipeline
Integrated with debug capabilities, fixed detection methods, and smart hood detection
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
class HoodDetectionResult:
    """Results from hood detection"""
    hood_mask: np.ndarray
    hood_ratio: float
    confidence: float
    method_used: str
    hood_polygon: Optional[np.ndarray] = None

class SmartHoodDetector:
    """Intelligent hood detection that adapts to different vehicles and camera positions"""
    
    def __init__(self):
        self.calibration_frames = []
        self.detected_hood_ratio = None
        self.hood_template = None
        self.is_calibrated = False
        
    def detect_hood(self, frame: np.ndarray, frame_number: int = 0) -> HoodDetectionResult:
        """
        Detect vehicle hood using multiple methods
        Returns a mask where hood pixels are 255, rest are 0
        """
        h, w = frame.shape[:2]
        
        # Try multiple detection methods in order of reliability
        methods = [
            self._detect_hood_by_color_consistency,
            self._detect_hood_by_edge_patterns,
            self._detect_hood_by_geometry
        ]
        
        best_result = None
        best_confidence = 0
        
        for method in methods:
            try:
                result = method(frame)
                if result and result.confidence > best_confidence:
                    best_result = result
                    best_confidence = result.confidence
                    
                # If we have high confidence, use it
                if best_confidence > 0.8:
                    break
            except Exception as e:
                continue
        
        # Fallback to calibrated value or conservative estimate
        if best_result is None or best_confidence < 0.3:
            best_result = self._fallback_hood_detection(frame)
        
        # Update calibration
        if frame_number < 30 and best_confidence > 0.6:
            self._update_calibration(best_result)
        
        return best_result
    
    def _detect_hood_by_color_consistency(self, frame: np.ndarray) -> Optional[HoodDetectionResult]:
        """
        Detect hood by looking for consistent dark region at bottom
        Hoods are typically uniform in color and darker than road
        """
        h, w = frame.shape[:2]
        
        # Analyze bottom 40% of image
        bottom_region = frame[int(h * 0.6):, :]
        
        # Convert to LAB for better color analysis
        lab = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Find dark, uniform regions
        mask = np.zeros((bottom_region.shape[0], w), dtype=np.uint8)
        
        # Scan from bottom up
        for y in range(bottom_region.shape[0] - 1, -1, -1):
            row = l_channel[y, :]
            
            # Check if row is dark and uniform
            mean_val = np.mean(row)
            std_val = np.std(row)
            
            # Hood characteristics: dark (low L) and uniform (low std)
            if mean_val < 50 and std_val < 15:
                mask[y:, :] = 255
            else:
                # Check if we've found enough hood
                if np.sum(mask) > 0:
                    break
        
        # Validate the detected region
        hood_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        hood_ratio = hood_pixels / (h * w)
        
        if 0.05 < hood_ratio < 0.3:  # Reasonable hood size
            # Create full image mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[int(h * 0.6):, :] = mask
            
            # Refine edges
            full_mask = self._refine_hood_edges(full_mask, frame)
            
            return HoodDetectionResult(
                hood_mask=full_mask,
                hood_ratio=hood_ratio,
                confidence=0.7,
                method_used="color_consistency"
            )
        
        return None
    
    def _detect_hood_by_edge_patterns(self, frame: np.ndarray) -> Optional[HoodDetectionResult]:
        """
        Detect hood by characteristic edge patterns
        Hood-windshield boundary often has strong horizontal edge
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Focus on bottom half
        bottom_half = edges[h//2:, :]
        
        # Detect horizontal lines using Hough transform
        lines = cv2.HoughLinesP(bottom_half, 1, np.pi/180, 100, 
                               minLineLength=w//3, maxLineGap=50)
        
        if lines is not None:
            # Find strong horizontal lines
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Nearly horizontal (within 10 degrees)
                if angle < 10 or angle > 170:
                    # Adjust y coordinates to full image
                    y1 += h//2
                    y2 += h//2
                    horizontal_lines.append((y1 + y2) // 2)
            
            if horizontal_lines:
                # Find the most prominent horizontal edge
                horizontal_lines.sort()
                
                # Look for hood boundary (usually in bottom 30%)
                for hood_y in horizontal_lines:
                    if hood_y > h * 0.7:
                        # Create hood mask
                        hood_mask = np.zeros((h, w), dtype=np.uint8)
                        hood_mask[hood_y:, :] = 255
                        
                        # Validate using color consistency
                        hood_region = frame[hood_y:, :]
                        mean_color = np.mean(hood_region.reshape(-1, 3), axis=0)
                        color_std = np.std(hood_region.reshape(-1, 3))
                        
                        if color_std < 30:  # Uniform color
                            hood_ratio = (h - hood_y) / h
                            
                            return HoodDetectionResult(
                                hood_mask=hood_mask,
                                hood_ratio=hood_ratio,
                                confidence=0.8,
                                method_used="edge_patterns"
                            )
        
        return None
    
    def _detect_hood_by_geometry(self, frame: np.ndarray) -> Optional[HoodDetectionResult]:
        """
        Detect hood using geometric assumptions
        Hood often appears as a dark trapezoid at bottom
        """
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for dark regions in bottom third
        bottom_third = gray[2*h//3:, :]
        
        # Threshold to find dark areas
        _, binary = cv2.threshold(bottom_third, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Check if it's significant
            if area > (w * h * 0.05):  # At least 5% of image
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # Check if it's roughly trapezoidal (4-6 vertices)
                if 4 <= len(approx) <= 6:
                    # Create mask
                    mask = np.zeros((bottom_third.shape[0], w), dtype=np.uint8)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                    
                    # Full image mask
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[2*h//3:, :] = mask
                    
                    hood_ratio = np.sum(full_mask > 0) / (h * w)
                    
                    # Adjust polygon coordinates to full image
                    adjusted_polygon = approx.copy()
                    adjusted_polygon[:, :, 1] += 2*h//3
                    
                    return HoodDetectionResult(
                        hood_mask=full_mask,
                        hood_ratio=hood_ratio,
                        confidence=0.6,
                        method_used="geometry",
                        hood_polygon=adjusted_polygon
                    )
        
        return None
    
    def _fallback_hood_detection(self, frame: np.ndarray) -> HoodDetectionResult:
        """
        Conservative fallback when other methods fail
        Uses calibrated value or minimal exclusion
        """
        h, w = frame.shape[:2]
        
        # Use calibrated ratio if available
        if self.detected_hood_ratio:
            hood_height = int(h * self.detected_hood_ratio)
        else:
            # Conservative estimate - just exclude very bottom
            hood_height = int(h * 0.08)
        
        hood_mask = np.zeros((h, w), dtype=np.uint8)
        hood_mask[-hood_height:, :] = 255
        
        return HoodDetectionResult(
            hood_mask=hood_mask,
            hood_ratio=hood_height / h,
            confidence=0.3,
            method_used="fallback"
        )
    
    def _refine_hood_edges(self, hood_mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Refine hood mask edges using image gradients
        """
        h, w = frame.shape[:2]
        
        # Find the top edge of current mask
        hood_top = np.where(hood_mask > 0)[0].min() if np.any(hood_mask) else h
        
        # Look for strong gradients near the edge
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Search region around current edge
        search_start = max(0, hood_top - 30)
        search_end = min(h, hood_top + 30)
        
        if search_start < search_end:
            search_region = np.abs(grad_y[search_start:search_end, :])
            
            # Find strongest horizontal edge
            edge_strength = np.mean(search_region, axis=1)
            best_edge = np.argmax(edge_strength) + search_start
            
            # Update mask
            refined_mask = np.zeros_like(hood_mask)
            refined_mask[best_edge:, :] = 255
            
            # Smooth the edge
            kernel = np.ones((5, w), np.float32) / (5 * w)
            refined_mask = cv2.filter2D(refined_mask, -1, kernel)
            refined_mask = (refined_mask > 127).astype(np.uint8) * 255
            
            return refined_mask
        
        return hood_mask
    
    def _update_calibration(self, result: HoodDetectionResult):
        """Update calibration with high-confidence detections"""
        if result.confidence > 0.6:
            if self.detected_hood_ratio is None:
                self.detected_hood_ratio = result.hood_ratio
            else:
                # Moving average
                self.detected_hood_ratio = 0.9 * self.detected_hood_ratio + 0.1 * result.hood_ratio
            
            if len(self.calibration_frames) >= 10:
                self.is_calibrated = True

def apply_smart_hood_exclusion(detector, hood_detector: Optional[SmartHoodDetector] = None):
    """
    Apply smart hood exclusion to any road detector
    """
    if hood_detector is None:
        hood_detector = SmartHoodDetector()
    
    original_detect = detector.detect_road
    frame_count = 0
    
    def detect_with_smart_hood_exclusion(frame):
        nonlocal frame_count
        
        # Detect hood
        hood_result = hood_detector.detect_hood(frame, frame_count)
        frame_count += 1
        
        # Get road detection
        road_mask, confidence_map = original_detect(frame)
        
        # Apply hood mask (inverse - where hood is NOT)
        hood_exclusion_mask = cv2.bitwise_not(hood_result.hood_mask)
        road_mask = cv2.bitwise_and(road_mask, hood_exclusion_mask)
        
        # Also zero out confidence in hood area
        confidence_map = confidence_map * (hood_exclusion_mask / 255.0)
        
        # Store hood info for visualization
        detector._last_hood_result = hood_result
        
        return road_mask, confidence_map
    
    detector.detect_road = detect_with_smart_hood_exclusion
    return detector

def visualize_with_hood_detection(frame: np.ndarray, road_mask: np.ndarray, 
                                 hood_result: HoodDetectionResult) -> np.ndarray:
    """
    Visualize both road and hood detection
    """
    vis = frame.copy()
    h, w = frame.shape[:2]
    
    # Show road in blue
    road_overlay = np.zeros_like(vis)
    road_overlay[road_mask > 0] = [255, 100, 0]  # Orange-blue
    vis = cv2.addWeighted(vis, 0.7, road_overlay, 0.3, 0)
    
    # Show hood boundary
    hood_edge = cv2.Canny(hood_result.hood_mask, 50, 150)
    vis[hood_edge > 0] = [0, 0, 255]  # Red edge
    
    # If we have hood polygon, draw it
    if hood_result.hood_polygon is not None:
        cv2.drawContours(vis, [hood_result.hood_polygon], -1, (0, 255, 255), 2)
    
    # Add text info
    info_y = 30
    cv2.putText(vis, f"Hood Detection: {hood_result.method_used}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 30
    cv2.putText(vis, f"Hood Coverage: {hood_result.hood_ratio:.1%}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += 30
    cv2.putText(vis, f"Confidence: {hood_result.confidence:.2f}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show road coverage (excluding hood area)
    road_pixels = np.sum(road_mask > 0)
    non_hood_pixels = np.sum(hood_result.hood_mask == 0)
    road_coverage = road_pixels / non_hood_pixels * 100 if non_hood_pixels > 0 else 0
    info_y += 30
    cv2.putText(vis, f"Road Coverage: {road_coverage:.1f}%", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis

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
    
    # NEW: Fast video processing options
    max_resolution: int = 1024  # Resize large images for speed
    fast_processing: bool = True  # Enable fast video mode
    
    def __post_init__(self):
        if self.scales is None:
            self.scales = [1.0]

def create_fast_video_config():
    """Create a fast configuration optimized for video processing"""
    return RoadDetectionConfig(
        model_type="transformers",
        
        # Fast video processing settings
        conf_threshold=0.4,  # Still capture good road areas
        
        temporal_smooth=True,  # Keep this, it's fast
        edge_refinement=True,  # Basic cleanup
        
        # DISABLE expensive processing for speed
        advanced_edge_refinement=False,  # This is the bottleneck!
        multi_class_awareness=False,    # Skip expensive person detection
        geometric_filtering=False,      # Skip geometric analysis
        bilateral_filter=False,         # Skip bilateral filtering
        perspective_correction=False,
        
        multi_scale=False,
        scales=[1.0],
        debug_mode=False,  # No verbose logging
        
        # NEW: Fast video processing options
        max_resolution=1024,  # Resize large images for speed
        fast_processing=True  # Enable fast video mode
    )

def create_improved_accurate_config():
    """Create the unified accurate configuration - trust the model's semantic understanding"""
    return RoadDetectionConfig(
        model_type="transformers",       # SegFormer-B5 - trained on Cityscapes, knows road vs non-road
        conf_threshold=0.5,             # Trust the model - reasonable threshold
        temporal_smooth=True,
        edge_refinement=True,
        advanced_edge_refinement=False,  # Let the model do its job
        confidence_edge_threshold=0.6,   # Reasonable edge confidence
        multi_class_awareness=True,      # Use semantic classes properly (not for aggressive filtering)
        geometric_filtering=False,       # Trust the model's spatial understanding
        bilateral_filter=False,          # Trust the model's edge detection
        perspective_correction=False,    # Trust the model's perspective understanding
        multi_scale=True,               # Keep multi-scale for robustness
        scales=[0.75, 1.0, 1.25]       # Multi-scale helps with different road sizes
    )

def create_github_exact_accurate_config():
    """Create the EXACT GitHub 'accurate' configuration (very aggressive, precision-focused)"""
    return RoadDetectionConfig(
        model_type="transformers",  # SegFormer-B5 for best accuracy
        conf_threshold=0.77,        # GitHub exact: very high precision
        temporal_smooth=True,
        edge_refinement=True,
        advanced_edge_refinement=True,  # Enhanced edge refinement
        confidence_edge_threshold=0.85,  # GitHub exact: very strict edges
        multi_class_awareness=True,      # Use other classes to constrain roads + person safety
        geometric_filtering=True,        # Geometric constraints
        bilateral_filter=True,           # Edge-preserving smoothing
        perspective_correction=True,
        multi_scale=True,
        scales=[0.75, 1.0, 1.25]
    )

def improve_road_coverage(road_mask, confidence_map, expand_ratio=1.2):
    """
    Post-process to improve road coverage with adaptive dilation
    Can be applied after detection
    """
    h, w = road_mask.shape
    
    # 1. Adaptive dilation based on distance from camera
    # More expansion for distant road (top of image)
    expanded = road_mask.copy()
    
    for y in range(0, h, 10):
        # Calculate expansion factor based on position
        # More expansion at top (distant road), less at bottom
        position_factor = 1.0 - (y / h) * 0.5  # 1.0 at top, 0.5 at bottom
        local_kernel_size = int(7 * expand_ratio * position_factor)
        
        if local_kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (local_kernel_size, local_kernel_size))
            
            # Process a strip of the image
            strip_start = y
            strip_end = min(y + 10, h)
            
            strip = road_mask[strip_start:strip_end, :]
            strip_expanded = cv2.dilate(strip, kernel, iterations=1)
            expanded[strip_start:strip_end, :] = strip_expanded
    
    # 2. Only keep expanded areas with reasonable confidence
    conf_threshold = 0.3  # Lower threshold for expanded areas
    valid_expansion = (confidence_map > conf_threshold).astype(np.uint8) * 255
    expanded = cv2.bitwise_and(expanded, valid_expansion)
    
    # 3. Combine with original
    improved = cv2.bitwise_or(road_mask, expanded)
    
    # 4. Fill gaps horizontally (roads are continuous)
    for y in range(int(h * 0.3), int(h * 0.9)):
        row = improved[y, :]
        
        # Find road segments
        changes = np.diff(np.concatenate([[0], row, [0]]))
        starts = np.where(changes > 0)[0]
        ends = np.where(changes < 0)[0]
        
        # Ensure starts and ends are properly paired
        if len(starts) >= 2 and len(ends) >= len(starts):
            # Fill small gaps between segments
            for i in range(len(starts) - 1):
                if i < len(ends):  # Safety check
                    gap_start = ends[i]
                    gap_end = starts[i + 1]
                    gap_size = gap_end - gap_start
                    
                    # Fill if gap is small relative to image width
                    if gap_size < w * 0.15:  # Max 15% of width
                        improved[y, gap_start:gap_end] = 255
    
    # 5. Morphological closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    improved = cv2.morphologyEx(improved, cv2.MORPH_CLOSE, kernel_close)
    
    # 6. Smooth boundaries
    improved = cv2.medianBlur(improved, 5)
    
    return improved

def apply_road_detection_fixes(detector, use_smart_hood=True):
    """
    Apply minimal safety enhancements to trust the model's semantic understanding.
    SegFormer-B5 is trained on Cityscapes and knows road vs hood vs dashboard.
    
    Args:
        detector: The road detector instance
        use_smart_hood: If True, apply minimal safety hood detection only in extreme cases
    """
    
    # FIXED: Don't apply hood detection here, we'll do it after coverage improvement
    # Store the original detect method
    original_detect = detector.detect_road
    
    def enhanced_detect_with_hood_respect(frame):
        # Get the model's detection - trust the semantic segmentation
        road_mask, confidence_map = original_detect(frame)
        
        # Only apply minimal coverage improvement if needed
        road_mask = improve_road_coverage(road_mask, confidence_map)
        
        # TRUST THE MODEL - it knows what's road vs hood vs dashboard
        # Only apply minimal safety hood detection in extreme cases
        if use_smart_hood:
            hood_detector = SmartHoodDetector()
            hood_result = hood_detector.detect_hood(frame)
            detector._last_hood_result = hood_result
            
            # Only exclude hood if it's VERY obvious and the model missed it
            h, w = road_mask.shape
            smart_hood_percentage = np.sum(hood_result.hood_mask > 0) / (h * w)
            
            # Only intervene if hood detection is very confident AND significant
            if smart_hood_percentage > 0.15 and hood_result.confidence > 0.8:
                print(f"  High-confidence hood detected ({smart_hood_percentage:.1%}), applying minimal exclusion")
                road_mask[hood_result.hood_mask > 0] = 0
                confidence_map[hood_result.hood_mask > 0] = 0
            else:
                print(f"  Trusting model's semantic understanding (hood: {smart_hood_percentage:.1%})")
        
        return road_mask, confidence_map
    
    # Replace the detector's method completely
    detector.detect_road = enhanced_detect_with_hood_respect
    
    print(f"Applied enhancements: minimal coverage improvement + trust model's semantic understanding")
    
    # Add enhanced visualization if using smart hood
    if use_smart_hood:
        original_visualize = detector.visualize
        
        def enhanced_visualize(frame, road_mask, style="freespace"):
            # Get base visualization
            vis = original_visualize(frame, road_mask, style)
            
            # Add hood detection info if available
            if hasattr(detector, '_last_hood_result'):
                hood_result = detector._last_hood_result
                
                # Draw hood boundary in red
                hood_edge = cv2.Canny(hood_result.hood_mask, 50, 150)
                vis[hood_edge > 0] = [0, 0, 255]
                
                # Add hood detection info
                info_text = f"Hood: {hood_result.method_used} ({hood_result.confidence:.2f})"
                cv2.putText(vis, info_text, (10, vis.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return vis
        
        detector.visualize = enhanced_visualize
    
    return detector

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
        """
        FIXED: Advanced edge refinement that doesn't remove all road pixels
        """
        refined_mask = road_mask.copy()
        
        # FIXED: Safety check - if mask is already small, skip aggressive refinement
        road_pixels = np.sum(road_mask > 0)
        total_pixels = road_mask.shape[0] * road_mask.shape[1]
        road_coverage = road_pixels / total_pixels
        
        if road_coverage < 0.05:  # Less than 5% coverage
            print(f"  Skipping aggressive refinement (coverage: {road_coverage:.1%})")
            return road_mask
        
        print(f"  Starting refinement with {road_pixels} road pixels ({road_coverage:.1%} coverage)")
        
        # 1. Confidence-based edge refinement
        if self.config.confidence_edge_threshold > self.config.conf_threshold:
            refined_mask = self._confidence_edge_refinement(refined_mask, confidence_map)
            remaining_after_confidence = np.sum(refined_mask > 0)
            print(f"  After confidence refinement: {remaining_after_confidence} pixels")
        
        # 2. FIXED: Multi-class awareness with safety checks
        if self.config.multi_class_awareness:
            before_multiclass = np.sum(refined_mask > 0)
            refined_mask = self._multiclass_edge_refinement(refined_mask, all_probs)
            after_multiclass = np.sum(refined_mask > 0)
            print(f"  After multiclass refinement: {after_multiclass} pixels (removed {before_multiclass - after_multiclass})")
        
        # 3. Geometric filtering (keep this, it's usually helpful)
        if self.config.geometric_filtering:
            before_geometric = np.sum(refined_mask > 0)
            refined_mask = self._geometric_edge_filtering(refined_mask, frame)
            after_geometric = np.sum(refined_mask > 0)
            print(f"  After geometric filtering: {after_geometric} pixels")
        
        # 4. Edge-preserving bilateral filtering
        if self.config.bilateral_filter:
            refined_mask = self._bilateral_edge_filter(refined_mask, frame)
        
        # 5. Clean up small components
        refined_mask = self._clean_small_components(refined_mask)
        
        final_pixels = np.sum(refined_mask > 0)
        print(f"  Final refined mask: {final_pixels} pixels")
        
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
        """
        FIXED: Enhanced multi-class refinement with proper person exclusion
        The original version was too aggressive and removed all road pixels
        """
        # Get all class probabilities
        probs_np = all_probs[0].cpu().numpy()  # Shape: [num_classes, H, W]
        
        # FIXED: Much more conservative thresholds
        non_road_classes = {
            # Static infrastructure - higher thresholds
            1: 0.8,   # sidewalk - was 0.6, now 0.8
            2: 0.8,   # building - was 0.6, now 0.8
            3: 0.8,   # wall - was 0.6, now 0.8
            4: 0.8,   # fence - was 0.6, now 0.8
            8: 0.7,   # vegetation - was 0.5, now 0.7
            9: 0.7,   # terrain - was 0.5, now 0.7
            10: 0.9,  # sky - was 0.7, now 0.9
            
            # FIXED: People and vehicles - still low but not extreme
            11: 0.3,  # person - was 0.15, now 0.3
            12: 0.3,  # rider - was 0.15, now 0.3
            13: 0.5,  # car - was 0.3, now 0.5
            14: 0.5,  # truck - was 0.3, now 0.5
            15: 0.5,  # bus - was 0.3, now 0.5
            16: 0.5,  # train - was 0.3, now 0.5
            17: 0.4,  # motorcycle - was 0.2, now 0.4
            18: 0.4   # bicycle - was 0.2, now 0.4
        }
        
        # Create mask of areas strongly predicted as non-road
        non_road_mask = np.zeros(road_mask.shape[:2], dtype=bool)
        
        # FIXED: Only process if we have enough classes
        if probs_np.shape[0] <= 1:
            print("  Warning: Not enough classes for multiclass refinement")
            return road_mask
        
        # FIXED: Track what we're removing for debugging
        removal_stats = {}
        
        for cls, threshold in non_road_classes.items():
            if cls < probs_np.shape[0]:
                class_prob = probs_np[cls]
                
                # Special handling for person class
                if cls == 11:  # person
                    person_mask = class_prob > threshold
                    
                    # FIXED: Much less aggressive dilation
                    if np.any(person_mask):
                        # Smaller dilation kernel
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # was (25, 25)
                        person_mask_dilated = cv2.dilate(person_mask.astype(np.uint8), kernel, iterations=1)  # was 2
                        
                        # FIXED: Remove the aggressive vertical blob detection
                        # person_mask_dilated = self._detect_vertical_blobs(person_mask_dilated, class_prob)
                        
                        non_road_mask |= person_mask_dilated.astype(bool)
                        
                        # Debug info
                        person_pixels = np.sum(person_mask)
                        removal_stats[f'person_{cls}'] = person_pixels
                        if person_pixels > 0:
                            print(f"  Person detected: {person_pixels} pixels (threshold: {threshold})")
                else:
                    class_mask = class_prob > threshold
                    if np.any(class_mask):
                        non_road_mask |= class_mask
                        removal_stats[f'class_{cls}'] = np.sum(class_mask)
        
        # FIXED: Remove the overly aggressive pattern detection
        # potential_people = self._detect_potential_people_by_pattern(road_mask.shape, all_probs)
        # non_road_mask |= potential_people
        
        # FIXED: Safety check - don't remove more than 50% of road pixels
        road_pixels_original = np.sum(road_mask > 0)
        pixels_to_remove = np.sum(non_road_mask & (road_mask > 0))
        removal_ratio = pixels_to_remove / road_pixels_original if road_pixels_original > 0 else 0
        
        if removal_ratio > 0.5:  # If we're removing more than 50% of road
            print(f"  WARNING: Would remove {removal_ratio:.1%} of road pixels - limiting removal")
            
            # FIXED: Only remove the highest confidence non-road areas
            # Create a more conservative mask
            conservative_non_road_mask = np.zeros_like(non_road_mask)
            
            for cls, threshold in non_road_classes.items():
                if cls < probs_np.shape[0]:
                    # Use much higher threshold for conservative removal
                    conservative_threshold = min(0.9, threshold + 0.3)
                    class_prob = probs_np[cls]
                    conservative_non_road_mask |= class_prob > conservative_threshold
            
            non_road_mask = conservative_non_road_mask
        
        # Remove road pixels that are in non-road areas
        refined_mask = road_mask.copy()
        
        # Count how many road pixels we're removing
        removed_pixels = np.sum(refined_mask[non_road_mask] > 0)
        if removed_pixels > 0:
            print(f"  Removing {removed_pixels} road pixels due to non-road classes")
            print(f"  Removal ratio: {removed_pixels/road_pixels_original:.1%}")
        
        refined_mask[non_road_mask] = 0
        
        # FIXED: Final safety check - if we removed too much, return original
        remaining_pixels = np.sum(refined_mask > 0)
        if remaining_pixels < road_pixels_original * 0.1:  # Less than 10% remaining
            print(f"  ERROR: Too much road removed ({remaining_pixels} pixels remaining)")
            print(f"  Returning original mask with {road_pixels_original} pixels")
            return road_mask  # Return original mask
        
        return refined_mask
    
    def _detect_vertical_blobs(self, person_mask: np.ndarray, person_prob: np.ndarray) -> np.ndarray:
        """Detect vertical blob-like structures that might be people"""
        h, w = person_mask.shape
        enhanced_mask = person_mask.copy()
        
        # Look for vertical continuous regions (people are typically vertical)
        for x in range(w):
            column = person_prob[:, x]
            
            # Find continuous high-probability regions
            high_prob_regions = []
            in_region = False
            start_y = 0
            
            for y in range(h):
                if column[y] > 0.1:  # Very low threshold
                    if not in_region:
                        in_region = True
                        start_y = y
                else:
                    if in_region:
                        # Check if region is tall enough to be a person
                        if y - start_y > h * 0.05:  # At least 5% of image height
                            high_prob_regions.append((start_y, y))
                        in_region = False
            
            # Mark vertical regions as potential people
            for start_y, end_y in high_prob_regions:
                # Check aspect ratio (people are tall and narrow)
                region_height = end_y - start_y
                if region_height > h * 0.05:  # Reasonable height
                    # Expand horizontally around this column
                    x_start = max(0, x - 10)
                    x_end = min(w, x + 10)
                    enhanced_mask[start_y:end_y, x_start:x_end] = 1
        
        return enhanced_mask
    
    def _detect_potential_people_by_pattern(self, shape: Tuple[int, int], all_probs: torch.Tensor) -> np.ndarray:
        """Detect potential people using pattern analysis"""
        h, w = shape
        potential_people = np.zeros((h, w), dtype=bool)
        
        # Get probabilities for relevant classes
        probs_np = all_probs[0].cpu().numpy()
        
        # Look for areas that are:
        # 1. Not strongly any other class
        # 2. In expected pedestrian zones
        # 3. Have certain shape characteristics
        
        # Calculate "unknown" areas (low confidence for all classes)
        max_class_prob = np.max(probs_np, axis=0)
        uncertain_areas = max_class_prob < 0.5
        
        # Focus on lower part of image where people typically appear
        pedestrian_zone = np.zeros((h, w), dtype=bool)
        pedestrian_zone[int(h * 0.3):, :] = True  # Bottom 70% of image
        
        # Combine: uncertain areas in pedestrian zones might be people
        potential_people = uncertain_areas & pedestrian_zone
        
        # Also check for vertical edge patterns (people have strong vertical edges)
        gray_prob = np.mean(probs_np[:3], axis=0)  # Average of first few classes
        edges_y = cv2.Sobel(gray_prob, cv2.CV_64F, 0, 1, ksize=3)
        vertical_edges = np.abs(edges_y) > 0.1
        
        # Vertical edges in pedestrian zones
        potential_people |= (vertical_edges & pedestrian_zone)
        
        # Dilate to be safe
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        potential_people = cv2.dilate(potential_people.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        return potential_people
    
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
                
                # Check if contour is in reasonable road position and has proper shape
                if cy > h * 0.25:  # Road shouldn't be in top 25% of image
                    # Additional check: contour should be reasonably wide (roads are typically wide)
                    x, y, w, h_contour = cv2.boundingRect(contour)
                    aspect_ratio = w / h_contour if h_contour > 0 else 0
                    
                    # Only accept contours that look like road segments (wide, not too vertical)
                    if aspect_ratio > 0.5 and h_contour < h * 0.7:
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

def test_hood_detection(video_path: str, output_path: str = "hood_detection_test.mp4"):
    """Test hood detection on a video"""
    # Create detector with fixes
    config = create_improved_accurate_config()
    detector = ModernRoadDetector(config)
    detector = apply_road_detection_fixes(detector, use_smart_hood=True)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Testing smart hood detection on {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect road
        road_mask, _ = detector.detect_road(frame)
        
        # Visualize
        vis = detector.visualize(frame, road_mask, style='freespace')
        out.write(vis)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
            # Print hood detection info
            if hasattr(detector, '_last_hood_result'):
                hood_result = detector._last_hood_result
                print(f"  Hood detection: {hood_result.method_used} "
                      f"({hood_result.confidence:.2f} confidence, "
                      f"{hood_result.hood_ratio:.1%} coverage)")
    
    cap.release()
    out.release()
    print(f"✓ Test complete! Output saved to {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "debug" and len(sys.argv) > 2:
            # Debug a single frame to understand the issue
            video_file = sys.argv[2]
            frame_num = int(sys.argv[3]) if len(sys.argv) > 3 else 100
            debug_single_frame(video_file, frame_num)
            
        elif command == "test" and len(sys.argv) > 2:
            # Test smart hood detection
            video_file = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else "smart_hood_test.mp4"
            test_hood_detection(video_file, output_file)
            
        elif command == "process" and len(sys.argv) > 3:
            # Process with fixed detection
            input_file = sys.argv[2]
            output_file = sys.argv[3]
            process_video_fixed(input_file, output_file)
            
        else:
            print("Usage:")
            print("  python stage1_road_detection.py debug <video> [frame_num]")
            print("  python stage1_road_detection.py test <video> [output]")
            print("  python stage1_road_detection.py process <input> <output>")
    else:
        # Demo smart hood detection setup
        print("=== Smart Hood Detection Demo ===")
        
        # Create improved detector
        config = create_improved_accurate_config()
        detector = ModernRoadDetector(config)
        
        # Apply smart hood detection fixes
        detector = apply_road_detection_fixes(detector, use_smart_hood=True)
        
        print("✓ Detector ready with smart hood detection!")
        print("  - Color consistency detection")
        print("  - Edge pattern analysis")
        print("  - Geometric shape detection")
        print("  - Adaptive calibration")
        print("  - Improved road coverage")
        print("\nUsage examples:")
        print("  python stage1_road_detection.py test input_videos/Road_Lane.mp4")
        print("  python stage1_road_detection.py debug input_videos/Road_Lane.mp4 100") 