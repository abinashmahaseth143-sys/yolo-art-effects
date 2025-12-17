# src/effects/animation_engine.py
import cv2
import numpy as np
from typing import List, Dict
import imageio
from dataclasses import dataclass
import math

@dataclass
class AnimationFrame:
    frame: np.ndarray
    timestamp: float
    effect_params: Dict

class AnimationEngine:
    """Engine for creating animated effects"""
    
    def __init__(self, fps=10, duration=3.0):
        self.fps = fps
        self.duration = duration
        self.total_frames = int(fps * duration)
    
    def create_animation(self, base_image: np.ndarray, 
                        detections: List, 
                        effect_type: str,
                        effect_library) -> List[np.ndarray]:
        """
        Create animated sequence
        
        Args:
            base_image: Base image to animate
            detections: List of detected objects
            effect_type: Type of effect to animate
            effect_library: EffectLibrary instance
            
        Returns:
            List of animation frames
        """
        frames = []
        
        for frame_num in range(self.total_frames):
            # Start with base image
            frame = base_image.copy()
            
            # Calculate animation progress
            progress = frame_num / self.total_frames
            
            # Apply animated effect to each detection
            for det in detections:
                frame = self._apply_animated_effect(
                    frame, det, effect_type, progress, effect_library
                )
            
            frames.append(frame)
        
        return frames
    
    def _apply_animated_effect(self, image, detection, effect_type, 
                              progress, effect_library):
        """Apply effect with animation progress"""
        
        # Different animations based on effect type
        if effect_type == "pulse":
            return self._animate_pulse(image, detection, progress, effect_library)
        elif effect_type == "float":
            return self._animate_float(image, detection, progress, effect_library)
        elif effect_type == "particles":
            return self._animate_particles(image, detection, progress, effect_library)
        elif effect_type == "glow":
            return self._animate_glow(image, detection, progress, effect_library)
        else:
            # Default: vary intensity
            intensity = 0.3 + 0.2 * math.sin(progress * 4 * math.pi)
            return effect_library.apply_effect(
                image, detection.bbox, effect_type, intensity
            )
    
    def _animate_pulse(self, image, detection, progress, effect_library):
        """Animated pulsing effect"""
        # Sinusoidal intensity variation
        pulse_speed = 2.0
        intensity = 0.3 + 0.2 * math.sin(progress * pulse_speed * 2 * math.pi)
        
        return effect_library.apply_effect(
            image, detection.bbox, 'glow', intensity
        )
    
    def _animate_float(self, image, detection, progress, effect_library):
        """Animated floating effect"""
        # Vertical oscillation
        original_bbox = detection.bbox.copy()
        
        # Calculate vertical offset
        float_range = 0.02  # 2% of image height
        offset = float_range * math.sin(progress * 2 * math.pi)
        
        # Apply offset to bbox
        animated_bbox = [
            original_bbox[0],
            original_bbox[1] + offset,
            original_bbox[2],
            original_bbox[3] + offset
        ]
        
        return effect_library.apply_effect(
            image, animated_bbox, 'float', 1.0
        )
    
    def _animate_particles(self, image, detection, progress, effect_library):
        """Animated particle effect"""
        # Vary particle count with time
        intensity = 0.5 + 0.3 * math.sin(progress * 2 * math.pi)
        
        return effect_library.apply_effect(
            image, detection.bbox, 'particles', intensity
        )
    
    def _animate_glow(self, image, detection, progress, effect_library):
        """Animated glowing effect"""
        # Color cycling glow
        intensity = 0.4 + 0.3 * abs(math.sin(progress * 3 * math.pi))
        
        return effect_library.apply_effect(
            image, detection.bbox, 'glow', intensity
        )
    
    def save_gif(self, frames: List[np.ndarray], output_path: str):
        """Save frames as animated GIF"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Convert BGR to RGB
        rgb_frames = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)
        
        # Save as GIF
        imageio.mimsave(output_path, rgb_frames, fps=self.fps, loop=0)
        print(f"✓ Saved animated GIF: {output_path}")
    
    def save_video(self, frames: List[np.ndarray], output_path: str):
        """Save frames as MP4 video"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, self.fps, (width, height)
        )
        
        # Write frames
        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"✓ Saved video: {output_path}")