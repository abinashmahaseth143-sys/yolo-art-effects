# src/ui/web_interface.py - COMPLETE CORRECTED VERSION
import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

from src.core.detector import ArtDetector
from src.effects.effect_manager import EffectManager

class ArtEffectWebApp:
    """Web interface for the Art Effect System"""
    
    def __init__(self):
        self.detector = ArtDetector(model_size="n")
        self.effect_manager = EffectManager()
        
        # Create output directories
        os.makedirs("outputs/web_results", exist_ok=True)
        os.makedirs("outputs/web_animations", exist_ok=True)
        
        # State
        self.last_detections = []
        self.last_image = None
    
    def process_image(self, image, confidence, effect_strategy, 
                     create_animation, animation_effect):
        """
        Main processing function for Gradio
        
        Args:
            image: Uploaded image
            confidence: Detection confidence threshold
            effect_strategy: How to apply effects
            create_animation: Whether to create animation
            animation_effect: Effect to animate
            
        Returns:
            Processed image, animation path, detection info
        """
        if image is None:
            return None, None, "Please upload an image"
        
        # Convert PIL to OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.last_image = image_cv.copy()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Detect objects
            print(f"Detecting objects with confidence: {confidence}")
            self.last_detections = self.detector.detect(
                image_cv, confidence_threshold=confidence
            )
            
            if not self.last_detections:
                # Try with lower confidence as fallback
                print("No detections, trying with confidence 0.05...")
                self.last_detections = self.detector.detect(
                    image_cv, confidence_threshold=0.05
                )
            
            detection_info = self._format_detection_info(self.last_detections)
            
            # Only apply effects if we have detections
            if self.last_detections:
                print(f"Applying effects with strategy: {effect_strategy}")
                processed = self.effect_manager.apply_effects(
                    image_cv.copy(), self.last_detections, effect_strategy
                )
                
                # Save processed image
                processed_path = f"outputs/web_results/processed_{timestamp}.jpg"
                cv2.imwrite(processed_path, processed)
                
                # Convert back to RGB for display
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
                # Create animation if requested
                animation_path = None
                if create_animation and self.last_detections:
                    print(f"Creating animation with effect: {animation_effect}")
                    frames = self.effect_manager.create_animated_version(
                        image_cv.copy(), self.last_detections, animation_effect
                    )
                    
                    animation_path = f"outputs/web_animations/animation_{timestamp}.gif"
                    self.effect_manager.animator.save_gif(frames, animation_path)
                
                return processed_rgb, animation_path, detection_info
            else:
                # Return original image with helpful message
                original_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                return original_rgb, None, detection_info + "\n\n⚠️ Try lowering confidence to 0.05-0.1"
                
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            return None, None, error_msg
    
    def process_webcam(self, webcam_image):
        """Process image from webcam"""
        if webcam_image is None:
            return None, "No webcam image received"
        
        # Simple processing for webcam
        image_cv = cv2.cvtColor(np.array(webcam_image), cv2.COLOR_RGB2BGR)
        
        # Detect with lower confidence for speed
        detections = self.detector.detect(image_cv, confidence_threshold=0.15)
        
        # Apply simple glow effect
        processed = self.effect_manager.apply_effects(
            image_cv, detections, "uniform"
        )
        
        # Convert back to RGB
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        return processed_rgb, f"Detected {len(detections)} objects"
    
    def batch_process(self, image_files, effect_strategy):
        """Process multiple images at once"""
        if not image_files:
            return "No files selected"
        
        results = []
        for img_file in image_files:
            # Read image
            image = cv2.imread(img_file.name)
            if image is None:
                continue
            
            # Detect and process
            detections = self.detector.detect(image, confidence_threshold=0.15)
            processed = self.effect_manager.apply_effects(
                image, detections, effect_strategy
            )
            
            # Save result
            filename = Path(img_file.name).stem
            output_path = f"outputs/batch/{filename}_processed.jpg"
            cv2.imwrite(output_path, processed)
            results.append(output_path)
        
        return f"Processed {len(results)} images. Results saved to outputs/batch/"
    
    def _format_detection_info(self, detections):
        """Format detection results for display"""
        if not detections:
            return "No objects detected. Try:\n• Lower confidence setting (0.05-0.1)\n• Clearer images with people/cars/animals\n• Check image isn't too dark or blurry"  # FIXED: isn't
        
        info = f"**Detected {len(detections)} objects:**\n\n"
        info += "| Object | Category | Confidence | Size |\n"
        info += "|--------|----------|------------|------|\n"
        
        for det in detections:
            size_percent = det.area * 100
            info += f"| {det.label} | {det.category} | {det.confidence:.2f} | {size_percent:.1f}% |\n"
        
        return info

def create_interface():
    """Create and configure Gradio interface"""
    app = ArtEffectWebApp()
    
    # Define tabs
    with gr.Blocks(title="🎨 AI Art Effect Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 AI Art Effect Generator")
        gr.Markdown("Transform your artworks with AI-powered dynamic effects using YOLO object detection")
        
        with gr.Tabs():
            # Tab 1: Single Image Processing
            with gr.TabItem("🖼️ Single Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Artwork",
                            type="pil",
                            height=300
                        )
                        
                        with gr.Accordion("⚙️ Settings", open=False):
                            confidence_slider = gr.Slider(
                                minimum=0.05, maximum=0.9, value=0.15,
                                label="Detection Confidence",
                                info="Start with 0.15 for best results. Lower = more detections"
                            )
                            
                            strategy_dropdown = gr.Dropdown(
                                choices=["category_based", "uniform", "size_based"],
                                value="category_based",
                                label="Effect Strategy",
                                info="How to choose effects for different objects"
                            )
                            
                            animate_checkbox = gr.Checkbox(
                                label="Create Animation (GIF)",
                                value=False
                            )
                            
                            animation_effect = gr.Dropdown(
                                choices=["pulse", "float", "glow", "particles"],
                                value="pulse",
                                label="Animation Effect",
                                visible=True
                            )
                        
                        process_btn = gr.Button("✨ Apply Effects", variant="primary")
                    
                    with gr.Column():
                        image_output = gr.Image(
                            label="Processed Result",
                            height=300
                        )
                        
                        animation_output = gr.File(
                            label="Download Animation",
                            visible=True
                        )
                        
                        detection_output = gr.Markdown(
                            label="Detection Results"
                        )
                
                # Connect components
                process_btn.click(
                    fn=app.process_image,
                    inputs=[image_input, confidence_slider, 
                           strategy_dropdown, animate_checkbox,
                           animation_effect],
                    outputs=[image_output, animation_output, detection_output]
                )
                
                # Show/hide animation effect based on checkbox
                animate_checkbox.change(
                    fn=lambda x: gr.Dropdown(visible=x),
                    inputs=[animate_checkbox],
                    outputs=[animation_effect]
                )
            
            # Tab 2: Webcam Processing
            with gr.TabItem("📷 Webcam"):
                with gr.Row():
                    with gr.Column():
                        webcam_input = gr.Image(
                            label="Upload Photo (from camera or file)",
                            type="pil",
                            height=300
                        )
                        webcam_btn = gr.Button("🎥 Process Image", variant="primary")
                    
                    with gr.Column():
                        webcam_output = gr.Image(
                            label="Processed Result",
                            height=300
                        )
                        webcam_info = gr.Textbox(
                            label="Detection Info"
                        )
                
                webcam_btn.click(
                    fn=app.process_webcam,
                    inputs=[webcam_input],
                    outputs=[webcam_output, webcam_info]
                )
            
            # Tab 3: Batch Processing
            with gr.TabItem("📚 Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        batch_files = gr.Files(
                            label="Select Images",
                            file_types=["image"],
                            file_count="multiple"
                        )
                        
                        batch_strategy = gr.Dropdown(
                            choices=["category_based", "uniform", "size_based"],
                            value="category_based",
                            label="Effect Strategy"
                        )
                        
                        batch_btn = gr.Button("🚀 Process All", variant="primary")
                    
                    with gr.Column():
                        batch_output = gr.Textbox(
                            label="Processing Results",
                            lines=5
                        )
                
                batch_btn.click(
                    fn=app.batch_process,
                    inputs=[batch_files, batch_strategy],
                    outputs=[batch_output]
                )
            
                        # Tab 4: Examples Gallery - SHOW 12+ EXAMPLES
            with gr.TabItem("🏛️ Examples Gallery"):
                gr.Markdown("## 🎯 One-Click Examples Gallery")
                gr.Markdown("Click any button to instantly load and process that image")
                
                # Get ALL images from your personal folder
                import os
                personal_images_dir = "data/images/personal"
                
                # List ALL 20 image files
                image_files = []
                if os.path.exists(personal_images_dir):
                    all_files = sorted(os.listdir(personal_images_dir))
                    for file in all_files:
                        if any(file.lower().endswith(ext) for ext in [".jfif", ".jpg", ".png", ".jpeg"]):
                            full_path = os.path.join(personal_images_dir, file)
                            gradio_path = full_path.replace("\\", "/")
                            if os.path.exists(full_path):
                                image_files.append((gradio_path, file))
                
                print(f"🎯 Loaded {len(image_files)} images for examples gallery")
                
                if image_files:
                    # Create 3 ROWS of examples (4 examples per row)
                    gr.Markdown("### 📸 Row 1: Quick Tests (Click Any)")
                    examples_row1 = []
                    for i in range(min(4, len(image_files))):
                        img_path, img_name = image_files[i]
                        examples_row1.append([
                            img_path, 0.15, "category_based", False, "pulse"
                        ])
                    
                    gr.Examples(
                        examples=examples_row1,
                        inputs=[image_input, confidence_slider, 
                               strategy_dropdown, animate_checkbox,
                               animation_effect],
                        outputs=[image_output, animation_output, detection_output],
                        fn=app.process_image,
                        cache_examples=False,
                        label=""
                    )
                    
                    # Row 2
                    if len(image_files) >= 8:
                        gr.Markdown("### 📸 Row 2: Different Strategies")
                        examples_row2 = []
                        for i in range(4, min(8, len(image_files))):
                            img_path, img_name = image_files[i]
                            strategy = ["uniform", "size_based", "category_based", "uniform"][i-4]
                            effect = ["glow", "float", "particles", "pulse"][i-4]
                            examples_row2.append([
                                img_path, 0.1, strategy, False, effect
                            ])
                        
                        gr.Examples(
                            examples=examples_row2,
                            inputs=[image_input, confidence_slider, 
                                   strategy_dropdown, animate_checkbox,
                                   animation_effect],
                            outputs=[image_output, animation_output, detection_output],
                            fn=app.process_image,
                            cache_examples=False,
                            label=""
                        )
                    
                    # Row 3
                    if len(image_files) >= 12:
                        gr.Markdown("### 📸 Row 3: With Animations")
                        examples_row3 = []
                        for i in range(8, min(12, len(image_files))):
                            img_path, img_name = image_files[i]
                            create_anim = [True, False, True, False][i-8]
                            examples_row3.append([
                                img_path, 0.15, "category_based", create_anim, "pulse"
                            ])
                        
                        gr.Examples(
                            examples=examples_row3,
                            inputs=[image_input, confidence_slider, 
                                   strategy_dropdown, animate_checkbox,
                                   animation_effect],
                            outputs=[image_output, animation_output, detection_output],
                            fn=app.process_image,
                            cache_examples=False,
                            label=""
                        )
                    
                    # Show thumbnail preview of ALL images
                    gr.Markdown(f"### 🖼️ All {len(image_files)} Available Images")
                    gr.Markdown("**Click 'Browse All Images' tab to see full collection**")
                    
                    # Quick access buttons for remaining images
                    if len(image_files) > 12:
                        gr.Markdown(f"### 🔄 More Images Available")
                        gr.Markdown(f"**Total images in system: {len(image_files)}**")
                        gr.Markdown("Use 'Browse All Images' tab or upload manually in 'Single Image' tab")
                    
                else:
                    gr.Markdown("### No images found")
                    
                    
                                # Tab 5: Browse All 20 Images
            with gr.TabItem("📂 Browse All Images"):
                gr.Markdown("## 📂 All 20 Test Images")
                gr.Markdown("All images from `data/images/personal/` folder")
                
                import os
                
                # Get all 20 images
                image_dir = "data/images/personal"
                images = []
                if os.path.exists(image_dir):
                    images = sorted([f for f in os.listdir(image_dir) 
                                   if f.lower().endswith('.jfif')])
                
                gr.Markdown(f"### ✅ Found {len(images)} images:")
                
                # Show in 5-column grid (4 images per row)
                rows = 5  # 5 rows × 4 columns = 20 images
                cols = 4
                
                for row in range(rows):
                    with gr.Row():
                        for col in range(cols):
                            idx = row * cols + col
                            if idx < len(images):
                                img_name = images[idx]
                                img_path = os.path.join(image_dir, img_name).replace("\\", "/")
                                
                                with gr.Column(min_width=200):
                                    # Image with number
                                    gr.Image(
                                        value=img_path,
                                        label=f"#{idx+1}: {img_name}",
                                        height=120,
                                        show_label=True
                                    )
                                    
                                    # Quick info
                                    gr.Markdown(f"**Image {idx+1}**")
                                    gr.Markdown(f"`{img_name}`")
                
                # Usage instructions
                gr.Markdown("### 🎯 How to Use:")
                gr.Markdown("""
                1. **Note the image number** you want to test
                2. Go to **"🖼️ Single Image"** tab
                3. Click **"Upload Artwork"**
                4. Navigate to `data/images/personal/`
                5. Select the image (e.g., `project1.jfif`)
                6. Click **"✨ Apply Effects"**
                """)
                
                # Quick test buttons for first 3 images
                gr.Markdown("### 🚀 Quick Test (First 3 Images):")
                with gr.Row():
                    for i in range(min(3, len(images))):
                        with gr.Column():
                            img_name = images[i]
                            img_path = os.path.join(image_dir, img_name).replace("\\", "/")
                            gr.Markdown(f"**Test Image {i+1}:**")
                            gr.Markdown(f"`{img_name}`")
                            test_btn = gr.Button(f"Test #{i+1}", variant="secondary")
                            
                            # Function to load this image
                            def load_image(img_path=img_path, img_name=img_name):
                                return img_path, f"Loaded: {img_name}"
                            
                            test_output = gr.Textbox(visible=False)
                            test_btn.click(fn=load_image, outputs=test_output)
            
            # Tab 6: About & Documentation
            with gr.TabItem("📖 About"):
                gr.Markdown("""
                ## About This Project
                
                **AI Art Effect Generator** uses YOLO object detection to identify elements in artworks
                and apply dynamic visual effects in real-time.
                
                ### Features:
                - Real-time object detection with YOLOv8
                - Category-aware effect application
                - Animated GIF generation
                - Webcam and batch processing
                - Customizable effect parameters
                
                ### How It Works:
                1. **Detection**: YOLO identifies objects in your image
                2. **Categorization**: Objects are grouped (human, animal, vehicle, etc.)
                3. **Effect Selection**: Appropriate effects chosen based on category
                4. **Application**: Effects applied to detected regions
                5. **Animation**: Optional animated versions created
                
                ### Educational Value:
                This project demonstrates practical applications of:
                - Computer Vision in creative contexts
                - Real-time AI processing
                - Interactive art systems
                - AI-assisted creativity tools
                """)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown(
            "**Academic Project** • AI for Creativity • "
            "Built with YOLOv8, OpenCV, and Gradio"
        )
    
    return interface

def launch_app(share=False, server_port=7860):
    """Launch the web application"""
    interface = create_interface()
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="127.0.0.1",  # Changed from 0.0.0.0 to 127.0.0.1
        show_error=True
    )

if __name__ == "__main__":
    # Create output directories
    os.makedirs("outputs/web_results", exist_ok=True)
    os.makedirs("outputs/web_animations", exist_ok=True)
    os.makedirs("outputs/batch", exist_ok=True)
    
    # Launch the app
    print("🚀 Launching Art Effect Generator...")
    print("🌐 Open http://localhost:7860 in your browser")
    launch_app(share=False) 