#!/usr/bin/env python3
"""
Fixed Lane Detection using Ultra-Fast-Lane-Detection approach
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import json
import time
from enum import Enum

class LaneModel(Enum):
    """Available lane detection models"""
    ULTRA_FAST = "ultra_fast"
    SCNN_LIKE = "scnn_like"
    
@dataclass
class LaneDetectionConfig:
    """Configuration for lane detection"""
    model: LaneModel = LaneModel.ULTRA_FAST
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.5
    max_lanes: int = 4
    input_size: Tuple[int, int] = (800, 288)  # width, height
    row_anchors: List[float] = None  # Will be initialized in __post_init__
    griding_num: int = 200  # Number of horizontal grids
    use_fp16: bool = False
    visualize: bool = True
    
    def __post_init__(self):
        if self.row_anchors is None:
            # Create row anchors from top to bottom of the image
            # Start from 0.4 (40% from top) to 1.0 (bottom)
            self.row_anchors = np.linspace(0.4, 1.0, 56).tolist()

class FixedUltraFastLaneNet(nn.Module):
    """Fixed Ultra-Fast Lane Detection Network"""
    
    def __init__(self, num_lanes=4, num_row_anchors=56, num_cols=201, backbone='resnet34'):
        super().__init__()
        
        self.num_lanes = num_lanes
        self.num_row_anchors = num_row_anchors
        self.num_cols = num_cols  # 200 columns + 1 for background
        
        # Use ResNet backbone
        if backbone == 'resnet34':
            import torchvision.models as models
            resnet = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_channels = 512
        else:
            # Simple CNN backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            backbone_channels = 512
        
        # Feature aggregation
        self.conv1 = nn.Conv2d(backbone_channels, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Global features
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Additional local features
        self.local_conv = nn.Conv2d(backbone_channels, 128, 1)
        self.local_bn = nn.BatchNorm2d(128)
        self.local_relu = nn.ReLU(inplace=True)
        self.local_pool = nn.AdaptiveAvgPool2d((4, 8))
        
        # Classifier heads
        feature_size = 256 + 128 * 4 * 8
        
        # Main classification head (which column each row anchor belongs to)
        self.cls_head = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, num_lanes * num_row_anchors * num_cols)
        )
        
        # Auxiliary head for lane existence prediction
        self.aux_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_lanes)
        )
        
    def forward(self, x):
        # Backbone features
        feat = self.backbone(x)
        
        # Global features
        global_feat = self.conv1(feat)
        global_feat = self.bn1(global_feat)
        global_feat = self.relu1(global_feat)
        global_feat = self.gap(global_feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        
        # Local features
        local_feat = self.local_conv(feat)
        local_feat = self.local_bn(local_feat)
        local_feat = self.local_relu(local_feat)
        local_feat = self.local_pool(local_feat)
        local_feat = local_feat.view(local_feat.size(0), -1)
        
        # Concatenate features
        feat_concat = torch.cat([global_feat, local_feat], dim=1)
        
        # Classification output
        cls_out = self.cls_head(feat_concat)
        cls_out = cls_out.view(-1, self.num_lanes, self.num_row_anchors, self.num_cols)
        
        # Auxiliary output (lane existence)
        aux_out = self.aux_head(feat_concat)
        aux_out = torch.sigmoid(aux_out)
        
        return {'cls_out': cls_out, 'aux_out': aux_out}

class FixedUltraFastLaneDetector:
    """Fixed Ultra-Fast Lane Detector"""
    
    def __init__(self, config: LaneDetectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._create_model()
        
    def _create_model(self):
        """Create and initialize the model"""
        print("Creating Ultra-Fast Lane Detection model...")
        
        model = FixedUltraFastLaneNet(
            num_lanes=self.config.max_lanes,
            num_row_anchors=len(self.config.row_anchors),
            num_cols=self.config.griding_num + 1,  # +1 for background
            backbone='resnet34'
        )
        
        model.to(self.device)
        model.eval()
        
        if self.config.use_fp16 and self.config.device == "cuda":
            model.half()
            
        return model
    
    def detect_lanes(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect lanes in image"""
        h, w = image.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Inference
        with torch.no_grad():
            if self.config.use_fp16 and self.config.device == "cuda":
                input_tensor = input_tensor.half()
            
            output = self.model(input_tensor)
        
        # Postprocess
        lanes = self._postprocess(output, (h, w))
        
        # Extract confidence from auxiliary output
        lane_probs = output['aux_out'].cpu().numpy()[0]
        confidences = [float(prob) for prob, lane in zip(lane_probs, lanes) if len(lane) > 0]
        
        return {
            'lanes': lanes,
            'confidence': confidences,
            'raw_output': output if hasattr(self.config, 'save_raw_output') and self.config.save_raw_output else None
        }
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize
        resized = cv2.resize(image, self.config.input_size)
        
        # Convert to RGB if needed
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # To tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _postprocess(self, output: Dict, original_shape: Tuple) -> List[np.ndarray]:
        """Convert model output to lane coordinates"""
        h, w = original_shape
        
        # Get classification output
        cls_out = output['cls_out'][0]  # Shape: [num_lanes, num_row_anchors, num_cols]
        aux_out = output['aux_out'][0]  # Shape: [num_lanes]
        
        # Process each lane
        lanes = []
        
        for lane_idx in range(self.config.max_lanes):
            # Check if lane exists (confidence > threshold)
            if aux_out[lane_idx] < self.config.confidence_threshold:
                continue
            
            # Get predicted columns for each row anchor
            lane_cls = cls_out[lane_idx]  # Shape: [num_row_anchors, num_cols]
            
            # Apply softmax to get probabilities
            lane_probs = F.softmax(lane_cls, dim=1)
            
            # Get most likely column for each row
            _, predicted_cols = lane_probs.max(dim=1)
            predicted_cols = predicted_cols.cpu().numpy()
            
            # Also get the confidence
            max_probs = lane_probs.max(dim=1)[0].cpu().numpy()
            
            # Convert to image coordinates
            lane_points = []
            for row_idx, (col_idx, prob) in enumerate(zip(predicted_cols, max_probs)):
                # Skip background class (index 0) or low confidence predictions
                if col_idx == 0 or prob < 0.5:
                    continue
                
                # Convert row anchor to y coordinate
                y = int(self.config.row_anchors[row_idx] * h)
                
                # Convert column index to x coordinate
                # col_idx ranges from 1 to griding_num (0 is background)
                x = int((col_idx - 1) * w / self.config.griding_num)
                
                lane_points.append([x, y])
            
            # Only add lane if it has enough points
            if len(lane_points) >= 2:
                # Sort points by y coordinate (top to bottom)
                lane_points = sorted(lane_points, key=lambda p: p[1])
                
                # Smooth the lane using polynomial fitting
                lane_array = np.array(lane_points)
                if len(lane_points) >= 4:
                    lane_array = self._smooth_lane(lane_array)
                
                lanes.append(lane_array)
        
        return lanes
    
    def _smooth_lane(self, lane_points: np.ndarray, degree: int = 3) -> np.ndarray:
        """Smooth lane points using polynomial fitting"""
        if len(lane_points) < degree + 1:
            return lane_points
        
        # Extract x and y coordinates
        x = lane_points[:, 0]
        y = lane_points[:, 1]
        
        try:
            # Fit polynomial
            coeffs = np.polyfit(y, x, degree)
            poly = np.poly1d(coeffs)
            
            # Generate smooth points
            y_smooth = np.linspace(y.min(), y.max(), max(len(y), 20))
            x_smooth = poly(y_smooth)
            
            # Ensure x coordinates are within image bounds
            x_smooth = np.clip(x_smooth, 0, lane_points[:, 0].max() + 50)
            
            smooth_lane = np.column_stack([x_smooth, y_smooth])
            return smooth_lane.astype(np.int32)
        except:
            # If fitting fails, return original points
            return lane_points

class LaneDetectorSCNN:
    """Alternative SCNN-like detector for comparison"""
    
    def __init__(self, config: LaneDetectionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    def detect_lanes(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect lanes using traditional CV + CNN approach"""
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Region of Interest
        roi_mask = np.zeros_like(edges)
        roi_vertices = np.array([
            [(0, h), (w * 0.4, h * 0.6), (w * 0.6, h * 0.6), (w, h)]
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=2,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        # Group lines into lanes
        lanes = self._group_lines_into_lanes(lines, image.shape)
        
        # Generate confidence scores
        confidences = [0.7] * len(lanes)  # Fixed confidence for traditional method
        
        return {
            'lanes': lanes,
            'confidence': confidences
        }
    
    def _group_lines_into_lanes(self, lines, img_shape) -> List[np.ndarray]:
        """Group detected lines into lanes"""
        if lines is None:
            return []
        
        h, w = img_shape[:2]
        lanes = []
        
        # Separate left and right lanes based on slope
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope
            if abs(slope) < 0.5:  # Nearly horizontal
                continue
            
            if slope < 0:  # Left lane
                left_lines.append(line[0])
            else:  # Right lane
                right_lines.append(line[0])
        
        # Fit lines to create lanes
        for lines_group in [left_lines, right_lines]:
            if len(lines_group) < 2:
                continue
            
            # Extract all points
            points = []
            for x1, y1, x2, y2 in lines_group:
                points.extend([(x1, y1), (x2, y2)])
            
            if len(points) < 2:
                continue
            
            points = np.array(points)
            
            # Fit a line through the points
            vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Generate lane points
            lane_points = []
            for y in np.linspace(h * 0.6, h, 20):
                if vy[0] != 0:
                    x = int(x0[0] + (y - y0[0]) * vx[0] / vy[0])
                    if 0 <= x < w:
                        lane_points.append([x, int(y)])
            
            if len(lane_points) >= 2:
                lanes.append(np.array(lane_points))
        
        return lanes

class AdvancedLaneDetector:
    """Main class that can use different lane detection models"""
    
    def __init__(self, config: LaneDetectionConfig):
        self.config = config
        self.detector = self._create_detector()
        
    def _create_detector(self):
        """Create the appropriate detector based on config"""
        if self.config.model == LaneModel.ULTRA_FAST:
            return FixedUltraFastLaneDetector(self.config)
        elif self.config.model == LaneModel.SCNN_LIKE:
            return LaneDetectorSCNN(self.config)
        else:
            return FixedUltraFastLaneDetector(self.config)
    
    def process_video(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Process entire video for lane detection"""
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise RuntimeError("Failed to create video writer")
        
        # Statistics
        all_results = []
        frame_count = 0
        total_lanes_detected = 0
        start_time = time.time()
        
        print(f"Processing video with {self.config.model.value} model...")
        print(f"Total frames: {total_frames}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect lanes
                result = self.detector.detect_lanes(frame)
                lanes = result['lanes']
                confidences = result['confidence']
                
                # Store results
                frame_result = {
                    'frame': frame_count,
                    'lanes': [lane.tolist() for lane in lanes],
                    'confidence': confidences,
                    'timestamp': frame_count / fps
                }
                all_results.append(frame_result)
                
                total_lanes_detected += len(lanes)
                
                # Visualize
                if self.config.visualize:
                    vis_frame = self._visualize_lanes(frame, lanes, confidences)
                    
                    # Add info text
                    info_text = f"Frame {frame_count}/{total_frames} | Lanes: {len(lanes)} | Model: {self.config.model.value}"
                    cv2.putText(vis_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    out.write(vis_frame)
                else:
                    out.write(frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = frame_count / elapsed
                    eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
                    print(f"Progress: {frame_count}/{total_frames} ({fps_proc:.1f} FPS, ETA: {eta:.0f}s)")
        
        finally:
            # CRITICAL: Always release video resources to prevent corruption
            cap.release()
            out.release()
        
        # Save results
        processing_time = time.time() - start_time
        avg_lanes = total_lanes_detected / frame_count if frame_count > 0 else 0
        
        results = {
            'video_info': {
                'input_path': input_path,
                'output_path': output_path,
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            },
            'detection_info': {
                'model': self.config.model.value,
                'device': self.config.device,
                'processing_time': processing_time,
                'fps_processed': frame_count / processing_time,
                'total_lanes_detected': total_lanes_detected,
                'average_lanes_per_frame': avg_lanes
            },
            'frame_results': all_results
        }
        
        # Save JSON results
        json_path = output_path.replace('.mp4', '_lanes.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Lane detection completed!")
        print(f"  Model: {self.config.model.value}")
        print(f"  Processing time: {processing_time:.1f}s ({frame_count/processing_time:.1f} FPS)")
        print(f"  Average lanes per frame: {avg_lanes:.2f}")
        print(f"  Output video: {output_path}")
        print(f"  Results JSON: {json_path}")
        
        return results
    
    def _visualize_lanes(self, frame: np.ndarray, lanes: List[np.ndarray], confidences: List[float]) -> np.ndarray:
        """Visualize detected lanes on frame"""
        vis_frame = frame.copy()
        
        # Color palette for lanes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw each lane
        for i, (lane, conf) in enumerate(zip(lanes, confidences)):
            if len(lane) < 2:
                continue
            
            color = colors[i % len(colors)]
            
            # Draw lane as thick polyline
            pts = lane.astype(np.int32)
            cv2.polylines(vis_frame, [pts], False, color, thickness=5)
            
            # Draw points
            for pt in pts[::5]:  # Every 5th point
                cv2.circle(vis_frame, tuple(pt), 5, color, -1)
                cv2.circle(vis_frame, tuple(pt), 7, (255, 255, 255), 2)
            
            # Add confidence label at the bottom of the lane
            if len(lane) > 0:
                # Find bottom point
                bottom_pt = lane[np.argmax(lane[:, 1])]
                label_pos = (int(bottom_pt[0]), int(bottom_pt[1]) - 10)
                label = f"L{i+1}: {conf:.2f}"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, 
                            (label_pos[0] - 5, label_pos[1] - text_height - 5),
                            (label_pos[0] + text_width + 5, label_pos[1] + 5),
                            (0, 0, 0), -1)
                
                cv2.putText(vis_frame, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame

def main():
    """Example usage and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Lane Detection')
    parser.add_argument('input_video', help='Path to input dashcam video')
    parser.add_argument('output_video', help='Path to output video')
    parser.add_argument('--model', type=str, default='ultra_fast', 
                       choices=['ultra_fast', 'scnn_like'], 
                       help='Lane detection model to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--max-lanes', type=int, default=4,
                       help='Maximum number of lanes to detect')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Configure detector
    config = LaneDetectionConfig(
        model=LaneModel(args.model),
        device=args.device,
        confidence_threshold=args.confidence,
        max_lanes=args.max_lanes,
        visualize=not args.no_visualize
    )
    
    # Run detection
    detector = AdvancedLaneDetector(config)
    results = detector.process_video(args.input_video, args.output_video)
    
    print("\nDone!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Testing lane detection with fixed model...")
        
        # Test configuration
        config = LaneDetectionConfig(
            model=LaneModel.ULTRA_FAST,
            device="cuda" if torch.cuda.is_available() else "cpu",
            visualize=True
        )
        
        detector = AdvancedLaneDetector(config)
        print(f"Detector initialized with {config.model.value} model on {config.device}")
        
        # Create a test image
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Draw some fake lane lines
        cv2.line(test_img, (400, 720), (500, 400), (255, 255, 255), 5)
        cv2.line(test_img, (880, 720), (780, 400), (255, 255, 255), 5)
        
        # Add some noise
        noise = np.random.randint(0, 50, test_img.shape, dtype=np.uint8)
        test_img = cv2.add(test_img, noise)
        
        # Test detection
        result = detector.detector.detect_lanes(test_img)
        print(f"Test detection: Found {len(result['lanes'])} lanes")
        
        # Visualize
        if result['lanes']:
            vis = detector._visualize_lanes(test_img, result['lanes'], result['confidence'])
            cv2.imwrite("test_lane_detection.jpg", vis)
            print("Test visualization saved to test_lane_detection.jpg")
    else:
        main() 