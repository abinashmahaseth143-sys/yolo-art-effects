# src/core/detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

@dataclass
class ArtObject:
    """Data class for detected objects"""
    label: str
    category: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized 0-1
    center: Tuple[float, float]
    area: float

class ArtDetector:
    """Main detector class using YOLOv8"""
    
    def __init__(self, model_size='n', device='cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
            device: 'cpu' or 'cuda' or '0' for GPU 0
        """
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.device = device
        
        # Map COCO classes to art effect categories
        self.category_map = self._create_category_map()
        
        # Performance tracking
        self.inference_times = []
        
        print("Model loaded successfully!")
    
    def _create_category_map(self) -> Dict[str, str]:
        """Create mapping from COCO labels to art categories"""
        return {
            # Human elements
            'person': 'human',
            
            # Animals
            'bird': 'animal', 'cat': 'animal', 'dog': 'animal',
            'horse': 'animal', 'sheep': 'animal', 'cow': 'animal',
            'elephant': 'animal', 'bear': 'animal', 'zebra': 'animal',
            'giraffe': 'animal',
            
            # Vehicles
            'bicycle': 'vehicle', 'car': 'vehicle', 'motorcycle': 'vehicle',
            'airplane': 'vehicle', 'bus': 'vehicle', 'train': 'vehicle',
            'truck': 'vehicle', 'boat': 'vehicle',
            
            # Buildings/Structures
            'building': 'structure', 'house': 'structure',
            'skyscraper': 'structure', 'bridge': 'structure',
            'traffic light': 'structure', 'fire hydrant': 'structure',
            
            # Nature
            'tree': 'nature', 'plant': 'nature', 'flower': 'nature',
            'mountain': 'nature', 'sky': 'nature',
            
            # Furniture/Objects
            'chair': 'object', 'couch': 'object', 'bed': 'object',
            'dining table': 'object', 'toilet': 'object',
            
            # Default
            '__default__': 'other'
        }
    
    def detect(self, image_input, confidence_threshold=0.25):
        """
        Detect objects in image
        
        Args:
            image_input: Can be file path, numpy array, or PIL Image
            confidence_threshold: Minimum confidence score
            
        Returns:
            List of ArtObject instances
        """
        # Load image if path is provided
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Cannot read image: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            # Assume PIL Image
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        
        # Store original for reference
        self.original_image = image.copy()
        self.image_height, self.image_width = image.shape[:2]
        
        # Run detection with timing
        start_time = time.time()
        
        results = self.model(
            image,
            conf=confidence_threshold,
            imgsz=640,
            device=self.device,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Normalize bbox to 0-1 range
                bbox_normalized = [
                    bbox[0] / self.image_width,    # x1
                    bbox[1] / self.image_height,   # y1
                    bbox[2] / self.image_width,    # x2
                    bbox[3] / self.image_height    # y2
                ]
                
                # Calculate center
                center_x = (bbox_normalized[0] + bbox_normalized[2]) / 2
                center_y = (bbox_normalized[1] + bbox_normalized[3]) / 2
                
                # Calculate area
                width = bbox_normalized[2] - bbox_normalized[0]
                height = bbox_normalized[3] - bbox_normalized[1]
                area = width * height
                
                # Categorize object
                category = self.category_map.get(label, 'other')
                
                # Create ArtObject
                art_obj = ArtObject(
                    label=label,
                    category=category,
                    confidence=confidence,
                    bbox=bbox_normalized,
                    center=(center_x, center_y),
                    area=area
                )
                
                detections.append(art_obj)
        
        print(f"Detected {len(detections)} objects in {inference_time:.3f}s")
        return detections
    
    def visualize(self, image_path=None, detections=None, save_path=None):
        """
        Visualize detections on image
        
        Args:
            image_path: Path to image (optional if already loaded)
            detections: List of ArtObject (optional, will detect if not provided)
            save_path: Path to save visualization
            
        Returns:
            Visualization image as numpy array
        """
        if image_path:
            image = cv2.imread(image_path)
            if detections is None:
                detections = self.detect(image_path)
        else:
            image = self.original_image.copy()
            if detections is None:
                detections = self.detect(image)
        
        # Color map for categories
        colors = {
            'human': (0, 255, 0),      # Green
            'animal': (255, 165, 0),   # Orange
            'vehicle': (0, 0, 255),    # Red
            'structure': (255, 255, 0),# Cyan
            'nature': (0, 255, 255),   # Yellow
            'object': (255, 0, 255),   # Magenta
            'other': (128, 128, 128)   # Gray
        }
        
        vis_image = image.copy()
        
        for obj in detections:
            # Convert normalized bbox to pixels
            x1 = int(obj.bbox[0] * self.image_width)
            y1 = int(obj.bbox[1] * self.image_height)
            x2 = int(obj.bbox[2] * self.image_width)
            y2 = int(obj.bbox[3] * self.image_height)
            
            # Get color for category
            color = colors.get(obj.category, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{obj.label}: {obj.confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                2
            )
            
            # Draw center point
            center_x = int(obj.center[0] * self.image_width)
            center_y = int(obj.center[1] * self.image_height)
            cv2.circle(vis_image, (center_x, center_y), 4, color, -1)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"Visualization saved to {save_path}")
        
        return vis_image
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'total_detections': len(self.inference_times),
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times)
        }