# src/effects/effect_library.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from typing import List, Tuple, Dict
import math

class ArtEffectLibrary:
    """Library of artistic effects for different object categories"""
    
    def __init__(self):
        # Map categories to available effects
        self.category_effects = {
            'human': ['glow', 'halo', 'aura', 'sparkle_trail'],
            'animal': ['particles', 'fur_effect', 'motion_blur', 'magic_trail'],
            'vehicle': ['speed_lines', 'light_trail', 'neon_outline', 'motion'],
            'structure': ['float', 'pulse', 'neon_glow', 'breathing'],
            'nature': ['wind_effect', 'growth', 'particle_field', 'magic_dust'],
            'object': ['highlight', 'pixelate', 'color_shift', 'vibrate'],
            'other': ['highlight', 'simple_glow']
        }
        
        # Effect parameters
        self.effect_params = {
            'glow': {'intensity': 0.3, 'blur_size': 51},
            'particles': {'count': 50, 'size_range': (2, 6)},
            'speed_lines': {'line_count': 20, 'length_range': (30, 100)},
            'float': {'offset': 10, 'opacity': 0.3},
            'wind_effect': {'strength': 15, 'direction': 'right'},
            'pulse': {'speed': 2.0, 'max_intensity': 0.5}
        }
    
    def apply_effect(self, image: np.ndarray, bbox: List[float], 
                    effect_type: str, intensity: float = 1.0) -> np.ndarray:
        """
        Apply specified effect to region
        
        Args:
            image: Input image (BGR format)
            bbox: Normalized bounding box [x1, y1, x2, y2]
            effect_type: Type of effect to apply
            intensity: Effect intensity multiplier
            
        Returns:
            Modified image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self._bbox_to_pixels(bbox, w, h)
        
        # Ensure ROI is valid
        if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            return image
        
        # Apply effect
        if effect_type == 'glow':
            return self._apply_glow(image, (x1, y1, x2, y2), intensity)
        elif effect_type == 'particles':
            return self._apply_particles(image, (x1, y1, x2, y2), intensity)
        elif effect_type == 'speed_lines':
            return self._apply_speed_lines(image, (x1, y1, x2, y2), intensity)
        elif effect_type == 'float':
            return self._apply_float(image, (x1, y1, x2, y2), intensity)
        elif effect_type == 'wind_effect':
            return self._apply_wind(image, (x1, y1, x2, y2), intensity)
        elif effect_type == 'pulse':
            return self._apply_pulse(image, (x1, y1, x2, y2), intensity)
        else:
            return self._apply_default_effect(image, (x1, y1, x2, y2), intensity)
    
    def _apply_glow(self, image, bbox, intensity):
        """Apply glowing aura effect"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Extract ROI with padding for glow
        pad = 20
        roi_x1 = max(0, x1 - pad)
        roi_y1 = max(0, y1 - pad)
        roi_x2 = min(w, x2 + pad)
        roi_y2 = min(h, y2 + pad)
        
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return image
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
        
        # Create mask for the object
        obj_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        obj_x1 = x1 - roi_x1
        obj_y1 = y1 - roi_y1
        obj_x2 = x2 - roi_x1
        obj_y2 = y2 - roi_y1
        
        if obj_x2 > obj_x1 and obj_y2 > obj_y1:
            cv2.rectangle(obj_mask, 
                         (max(0, obj_x1), max(0, obj_y1)),
                         (min(roi.shape[1], obj_x2), min(roi.shape[0], obj_y2)),
                         255, -1)
        
        # Apply Gaussian blur for glow
        blur_size = int(25 * intensity)
        if blur_size % 2 == 0:
            blur_size += 1
        
        blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
        
        # Create glow mask (dilated version of object)
        kernel = np.ones((15, 15), np.uint8)
        glow_mask = cv2.dilate(obj_mask, kernel, iterations=2)
        glow_mask = cv2.GaussianBlur(glow_mask, (31, 31), 0)
        glow_mask = glow_mask.astype(float) / 255.0
        
        # Apply glow
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - glow_mask) + blurred[:, :, c] * glow_mask
        
        # Put back into image
        image[roi_y1:roi_y2, roi_x1:roi_x2] = roi
        
        return image
    
    def _apply_particles(self, image, bbox, intensity):
        """Add magical particle effects"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Convert to PIL for drawing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Calculate center and size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size = max((x2 - x1), (y2 - y1)) // 2
        
        # Particle colors
        colors = [
            (255, 255, 255),  # White
            (255, 255, 0),    # Yellow
            (0, 255, 255),    # Cyan
            (255, 0, 255),    # Magenta
            (255, 200, 0),    # Gold
        ]
        
        # Generate particles
        particle_count = int(50 * intensity)
        for _ in range(particle_count):
            # Random position in circular area
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(size * 0.8, size * 1.5)
            
            px = int(center_x + distance * math.cos(angle))
            py = int(center_y + distance * math.sin(angle))
            
            # Random size
            particle_size = random.randint(2, int(6 * intensity))
            
            # Random color
            color = random.choice(colors)
            
            # Draw particle
            draw.ellipse([
                px - particle_size, py - particle_size,
                px + particle_size, py + particle_size
            ], fill=color)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _apply_speed_lines(self, image, bbox, intensity):
        """Add motion/speed lines"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Determine direction (assume moving right)
        line_count = int(20 * intensity)
        
        for i in range(line_count):
            # Start from random point on left side
            start_x = random.randint(x1, x1 + (x2 - x1) // 4)
            start_y = random.randint(y1, y2)
            
            # End point further right
            length = random.randint(30, int(100 * intensity))
            end_x = min(w - 1, start_x + length)
            
            # Slight vertical variation
            end_y = start_y + random.randint(-20, 20)
            end_y = max(0, min(h - 1, end_y))
            
            # Draw line with transparency
            line_width = random.randint(1, 3)
            alpha = random.randint(128, 255)
            color = (255, 255, 255, alpha)
            
            draw.line([start_x, start_y, end_x, end_y], 
                     fill=color, width=line_width)
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def _apply_float(self, image, bbox, intensity):
        """Create floating/hovering effect"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Extract object
        obj_height = y2 - y1
        obj_width = x2 - x1
        
        if obj_height <= 0 or obj_width <= 0:
            return image
        
        # Create shadow below
        shadow_y1 = y2 + 5
        shadow_y2 = shadow_y1 + int(10 * intensity)
        
        if shadow_y2 < h:
            shadow_roi = image[shadow_y1:shadow_y2, x1:x2].copy()
            
            # Darken for shadow
            shadow_dark = cv2.addWeighted(
                shadow_roi, 0.3,
                np.zeros_like(shadow_roi), 0.7,
                0
            )
            
            # Blur shadow
            shadow_blur = cv2.GaussianBlur(shadow_dark, (21, 21), 0)
            image[shadow_y1:shadow_y2, x1:x2] = shadow_blur
        
        # Create floating duplicate above
        float_offset = int(15 * intensity)
        float_y1 = max(0, y1 - float_offset)
        float_y2 = float_y1 + obj_height
        
        if float_y2 < h:
            # Copy and make semi-transparent
            float_obj = image[y1:y2, x1:x2].copy()
            float_transparent = cv2.addWeighted(
                float_obj, 0.4,
                np.zeros_like(float_obj), 0.6,
                0
            )
            
            # Place above original
            image[float_y1:float_y2, x1:x2] = float_transparent
        
        return image
    
    def _apply_wind(self, image, bbox, intensity):
        """Apply wind/motion distortion effect"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Create wind distortion map
        wind_strength = int(15 * intensity)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]
        
        if roi_h == 0 or roi_w == 0:
            return image
        
        # Create displacement map
        map_x = np.zeros((roi_h, roi_w), dtype=np.float32)
        map_y = np.zeros((roi_h, roi_w), dtype=np.float32)
        
        for i in range(roi_h):
            # Horizontal displacement increases from top to bottom
            displacement = wind_strength * (i / roi_h)
            map_x[i, :] = np.arange(roi_w) + displacement
            map_y[i, :] = i
        
        # Apply remap for wind effect
        wind_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        # Blend with original
        alpha = 0.7
        blended_roi = cv2.addWeighted(roi, alpha, wind_roi, 1-alpha, 0)
        
        # Put back
        image[y1:y2, x1:x2] = blended_roi
        
        return image
    
    def _apply_pulse(self, image, bbox, intensity):
        """Apply pulsing effect (for demonstration)"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Simple highlight effect
        roi = image[y1:y2, x1:x2].copy()
        
        # Increase brightness
        brightness = int(50 * intensity)
        roi = cv2.add(roi, (brightness, brightness, brightness, 0))
        
        # Put back
        image[y1:y2, x1:x2] = roi
        
        return image
    
    def _apply_default_effect(self, image, bbox, intensity):
        """Default highlight effect"""
        x1, y1, x2, y2 = bbox
        
        # Draw outline
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        return image
    
    def _bbox_to_pixels(self, bbox_norm, width, height):
        """Convert normalized bbox to pixel coordinates"""
        return (
            int(bbox_norm[0] * width),
            int(bbox_norm[1] * height),
            int(bbox_norm[2] * width),
            int(bbox_norm[3] * height)
        )