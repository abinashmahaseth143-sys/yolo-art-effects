# src/ui/realtime_processor.py
import cv2
import numpy as np
import threading
import time
from queue import Queue
import pygame
from pygame.locals import *

class RealTimeProcessor:
    """Real-time processing with PyGame interface"""
    
    def __init__(self, detector, effect_manager):
        self.detector = detector
        self.effect_manager = effect_manager
        self.running = False
        self.current_frame = None
        self.processed_frame = None
        self.frame_queue = Queue(maxsize=1)
        self.effect_params = {
            'effect_type': 'glow',
            'intensity': 1.0,
            'show_boxes': True
        }
        
        # PyGame setup
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Real-Time Art Effect Processor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def start_webcam(self, camera_index=0):
        """Start real-time webcam processing"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.start()
        
        # Start PyGame loop
        self._run_pygame_loop()
        
        return True
    
    def _process_frames(self):
        """Process frames in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Store current frame
            self.current_frame = frame.copy()
            
            # Process frame
            processed = self._process_single_frame(frame)
            
            # Put in queue for display
            if self.frame_queue.empty():
                self.frame_queue.put(processed)
            
            # Control processing rate
            time.sleep(0.033)  # ~30 FPS
    
    def _process_single_frame(self, frame):
        """Process a single frame"""
        # Detect objects
        detections = self.detector.detect(frame, confidence_threshold=0.25)
        
        # Apply effects
        processed = self.effect_manager.apply_effects(
            frame.copy(), detections, 'category_based'
        )
        
        # Add info overlay
        info_text = f"Objects: {len(detections)} | FPS: {self.clock.get_fps():.1f}"
        cv2.putText(processed, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return processed
    
    def _run_pygame_loop(self):
        """Main PyGame display loop"""
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_1:
                        self.effect_params['effect_type'] = 'glow'
                    elif event.key == K_2:
                        self.effect_params['effect_type'] = 'particles'
                    elif event.key == K_UP:
                        self.effect_params['intensity'] = min(
                            2.0, self.effect_params['intensity'] + 0.1
                        )
                    elif event.key == K_DOWN:
                        self.effect_params['intensity'] = max(
                            0.1, self.effect_params['intensity'] - 0.1
                        )
            
            # Get latest processed frame
            if not self.frame_queue.empty():
                self.processed_frame = self.frame_queue.get()
            
            # Display frame
            if self.processed_frame is not None:
                # Convert to PyGame surface
                frame_rgb = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(
                    np.rot90(frame_rgb)
                )
                
                # Scale to fit screen
                scaled = pygame.transform.scale(frame_surface, self.screen.get_size())
                self.screen.blit(scaled, (0, 0))
                
                # Add UI overlay
                self._draw_ui_overlay()
            
            pygame.display.flip()
            self.clock.tick(30)
        
        # Cleanup
        self.cap.release()
        pygame.quit()
    
    def _draw_ui_overlay(self):
        """Draw UI overlay on PyGame screen"""
        # Instructions
        instructions = [
            "CONTROLS:",
            "1/2 - Change effect",
            "UP/DOWN - Adjust intensity",
            "ESC - Exit"
        ]
        
        y_offset = 50
        for line in instructions:
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset))
            y_offset += 40
        
        # Current settings
        settings_text = f"Effect: {self.effect_params['effect_type']} | "
        settings_text += f"Intensity: {self.effect_params['intensity']:.1f}"
        settings_surf = self.font.render(settings_text, True, (255, 255, 0))
        self.screen.blit(settings_surf, (10, self.screen.get_height() - 50))

# Usage
def run_realtime():
    """Run real-time processing"""
    from src.core.detector import ArtDetector
    from src.effects.effect_manager import EffectManager
    
    print("🚀 Starting Real-Time Art Effect Processor...")
    print("Press ESC to exit")
    
    detector = ArtDetector(model_size='n')
    effect_manager = EffectManager()
    
    processor = RealTimeProcessor(detector, effect_manager)
    processor.start_webcam()

if __name__ == "__main__":
    run_realtime()