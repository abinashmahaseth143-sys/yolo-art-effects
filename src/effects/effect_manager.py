# src/effects/effect_manager.py
import cv2
import numpy as np
from typing import List, Dict
import json
from datetime import datetime
from .effect_library import ArtEffectLibrary
from .animation_engine import AnimationEngine

class EffectManager:
    """Manages application of effects to detected objects"""
    
    def __init__(self):
        self.effect_lib = ArtEffectLibrary()
        self.animator = AnimationEngine(fps=10, duration=3.0)
        
        # Effect history for undo/redo
        self.effect_history = []
        self.max_history = 10
    
    def apply_effects(self, image: np.ndarray, detections: List, 
                     effect_strategy: str = 'category_based') -> np.ndarray:
        """
        Apply effects to image based on detections
        
        Args:
            image: Input image
            detections: List of detected objects
            effect_strategy: How to choose effects
                - 'category_based': Different effects per category
                - 'uniform': Same effect for all
                - 'size_based': Effects based on object size
                
        Returns:
            Processed image
        """
        processed = image.copy()
        
        if not detections:
            print("No detections to apply effects to")
            return processed
        
        for detection in detections:
            # Choose effect based on strategy
            if effect_strategy == 'category_based':
                effect_type = self._choose_effect_by_category(detection.category)
            elif effect_strategy == 'uniform':
                effect_type = 'glow'  # Default uniform effect
            elif effect_strategy == 'size_based':
                effect_type = self._choose_effect_by_size(detection.area)
            else:
                effect_type = 'glow'
            
            # Apply effect
            processed = self.effect_lib.apply_effect(
                processed, detection.bbox, effect_type, intensity=1.0
            )
        
        # Save to history
        self._save_to_history(image, processed, effect_strategy)
        
        return processed
    
    def _choose_effect_by_category(self, category: str) -> str:
        """Choose appropriate effect for object category"""
        effect_map = {
            'human': 'glow',
            'animal': 'particles',
            'vehicle': 'speed_lines',
            'structure': 'float',
            'nature': 'wind_effect',
            'object': 'pulse',
            'other': 'glow'
        }
        return effect_map.get(category, 'glow')
    
    def _choose_effect_by_size(self, area: float) -> str:
        """Choose effect based on object size"""
        if area > 0.3:  # Large objects (>30% of image)
            return 'float'
        elif area > 0.1:  # Medium objects (10-30%)
            return 'glow'
        else:  # Small objects (<10%)
            return 'particles'
    
    def create_animated_version(self, image: np.ndarray, detections: List,
                               effect_type: str = 'pulse') -> List[np.ndarray]:
        """
        Create animated version of image with effects
        
        Args:
            image: Base image
            detections: List of detected objects
            effect_type: Type of effect to animate
            
        Returns:
            List of animation frames
        """
        return self.animator.create_animation(
            image, detections, effect_type, self.effect_lib
        )
    
    def process_image_sequence(self, image_paths: List[str], 
                              output_format: str = 'gif') -> str:
        """
        Process multiple images and create sequence
        
        Args:
            image_paths: List of image paths
            output_format: 'gif' or 'video'
            
        Returns:
            Path to output file
        """
        all_frames = []
        
        for img_path in image_paths:
            # Load and process each image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # For demo, create simple animated effect
            # In real use, you'd detect objects first
            frames = self.animator.create_animation(
                image, [], 'pulse', self.effect_lib
            )
            all_frames.extend(frames)
        
        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_format == 'gif':
            output_path = f'outputs/animated/sequence_{timestamp}.gif'
            self.animator.save_gif(all_frames[:30], output_path)  # Limit to 30 frames
        else:
            output_path = f'outputs/animated/sequence_{timestamp}.mp4'
            self.animator.save_video(all_frames, output_path)
        
        return output_path
    
    def _save_to_history(self, original, processed, effect_strategy):
        """Save processing step to history"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy': effect_strategy,
            'original_shape': original.shape,
            'processed_shape': processed.shape
        }
        
        self.effect_history.append(history_entry)
        
        # Limit history size
        if len(self.effect_history) > self.max_history:
            self.effect_history.pop(0)
    
    def get_statistics(self):
        """Get effect application statistics"""
        return {
            'total_processed': len(self.effect_history),
            'recent_strategies': [h['strategy'] for h in self.effect_history[-3:]]
        }

# Test the effect system
def test_effects():
    """Test the effect system"""
    from src.core.detector import ArtDetector
    
    # Initialize
    detector = ArtDetector()
    effect_manager = EffectManager()
    
    # Test image
    test_image = 'data/images/classical_paintings/classical_paintings_000.jpg'
    
    # Detect objects
    image = cv2.imread(test_image)
    detections = detector.detect(test_image)
    
    print(f"Detected {len(detections)} objects")
    
    # Apply effects with different strategies
    strategies = ['category_based', 'uniform', 'size_based']
    
    for strategy in strategies:
        print(f"\nApplying effects with strategy: {strategy}")
        
        # Apply effects
        result = effect_manager.apply_effects(image.copy(), detections, strategy)
        
        # Save result
        output_path = f'outputs/static/effect_{strategy}.jpg'
        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")
        
        # Create animation
        if len(detections) > 0:
            frames = effect_manager.create_animated_version(
                image.copy(), detections, 'pulse'
            )
            
            # Save animation
            anim_path = f'outputs/animated/animation_{strategy}.gif'
            effect_manager.animator.save_gif(frames[:20], anim_path)
            print(f"Animation saved: {anim_path}")
    
    # Print statistics
    stats = effect_manager.get_statistics()
    print(f"\nEffect Statistics: {stats}")

if __name__ == "__main__":
    test_effects()