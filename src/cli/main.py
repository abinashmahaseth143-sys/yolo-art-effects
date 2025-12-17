# src/cli/main.py
import argparse
import cv2
import os
import json
from datetime import datetime
from pathlib import Path

from src.core.detector import ArtDetector
from src.effects.effect_manager import EffectManager

def process_single_image(args):
    """Process a single image"""
    print(f"Processing: {args.input}")
    
    # Initialize components
    detector = ArtDetector(model_size=args.model_size, device=args.device)
    effect_manager = EffectManager()
    
    # Read image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image {args.input}")
        return
    
    # Detect objects
    print("Detecting objects...")
    detections = detector.detect(image, confidence_threshold=args.confidence)
    print(f"Found {len(detections)} objects")
    
    # Apply effects
    print(f"Applying effects with strategy: {args.strategy}")
    processed = effect_manager.apply_effects(
        image.copy(), detections, args.strategy
    )
    
    # Save output
    input_stem = Path(args.input).stem
    output_path = args.output or f"{input_stem}_processed.jpg"
    
    cv2.imwrite(output_path, processed)
    print(f"✓ Saved to: {output_path}")
    
    # Create animation if requested
    if args.animate:
        print("Creating animation...")
        frames = effect_manager.create_animated_version(
            image.copy(), detections, args.animation_effect
        )
        
        anim_path = f"{input_stem}_animated.gif"
        effect_manager.animator.save_gif(frames, anim_path)
        print(f"✓ Animation saved to: {anim_path}")
    
    # Save detection report
    if args.report:
        report = {
            'input_file': args.input,
            'output_file': output_path,
            'detection_count': len(detections),
            'detections': [
                {
                    'label': det.label,
                    'category': det.category,
                    'confidence': det.confidence,
                    'bbox': det.bbox,
                    'area': det.area
                }
                for det in detections
            ],
            'processing_time': detector.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = f"{input_stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to: {report_path}")

def process_directory(args):
    """Process all images in a directory"""
    print(f"Processing directory: {args.input}")
    
    detector = ArtDetector(model_size=args.model_size, device=args.device)
    effect_manager = EffectManager()
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(args.input).glob(f'*{ext}'))
        image_files.extend(Path(args.input).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for img_path in image_files:
        try:
            print(f"\nProcessing: {img_path.name}")
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Detect and process
            detections = detector.detect(image, args.confidence)
            processed = effect_manager.apply_effects(
                image.copy(), detections, args.strategy
            )
            
            # Save
            output_path = Path(args.output) / f"{img_path.stem}_processed{img_path.suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(output_path), processed)
            print(f"  ✓ Saved: {output_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Processed {len(image_files)} images")

def create_demo_gallery(args):
    """Create a demo gallery with examples"""
    print("Creating demo gallery...")
    
    # This would create HTML gallery - simplified version
    gallery_html = """
    <html>
    <head>
        <title>AI Art Effect Gallery</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .gallery { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
            .item { border: 1px solid #ddd; padding: 10px; text-align: center; }
            img { max-width: 100%; height: auto; }
            .caption { margin-top: 10px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>AI Art Effect Gallery</h1>
        <div class="gallery">
    """
    
    # Add your examples here
    examples = [
        ("example1.jpg", "Glow effect on classical painting"),
        ("example2.jpg", "Particle effect on portrait"),
        ("example3.jpg", "Float effect on cityscape"),
    ]
    
    for img, caption in examples:
        gallery_html += f"""
        <div class="item">
            <img src="{img}" alt="{caption}">
            <div class="caption">{caption}</div>
        </div>
        """
    
    gallery_html += """
        </div>
    </body>
    </html>
    """
    
    with open('gallery.html', 'w') as f:
        f.write(gallery_html)
    
    print("✓ Gallery created: gallery.html")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI Art Effect Generator - Apply dynamic effects to artworks using YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python -m src.cli.main single --input artwork.jpg --strategy category_based
  
  # Process with animation
  python -m src.cli.main single --input photo.jpg --animate --animation-effect pulse
  
  # Process directory of images
  python -m src.cli.main batch --input ./artworks/ --output ./processed/
  
  # Use GPU for faster processing
  python -m src.cli.main single --input large_image.jpg --device cuda --model-size m
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Single image parser
    single_parser = subparsers.add_parser('single', help='Process single image')
    single_parser.add_argument('--input', '-i', required=True, help='Input image path')
    single_parser.add_argument('--output', '-o', help='Output image path')
    single_parser.add_argument('--confidence', '-c', type=float, default=0.3,
                              help='Detection confidence threshold (0.1-0.9)')
    single_parser.add_argument('--strategy', '-s', default='category_based',
                              choices=['category_based', 'uniform', 'size_based'],
                              help='Effect application strategy')
    single_parser.add_argument('--model-size', default='n',
                              choices=['n', 's', 'm', 'l'],
                              help='YOLO model size')
    single_parser.add_argument('--device', default='cpu',
                              help='Processing device (cpu or cuda)')
    single_parser.add_argument('--animate', action='store_true',
                              help='Create animated GIF')
    single_parser.add_argument('--animation-effect', default='pulse',
                              choices=['pulse', 'float', 'glow', 'particles'],
                              help='Effect to animate')
    single_parser.add_argument('--report', action='store_true',
                              help='Generate JSON report')
    
    # Batch processing parser
    batch_parser = subparsers.add_parser('batch', help='Process directory of images')
    batch_parser.add_argument('--input', '-i', required=True, help='Input directory')
    batch_parser.add_argument('--output', '-o', default='./processed/',
                             help='Output directory')
    batch_parser.add_argument('--confidence', '-c', type=float, default=0.3)
    batch_parser.add_argument('--strategy', '-s', default='category_based')
    batch_parser.add_argument('--model-size', default='n')
    batch_parser.add_argument('--device', default='cpu')
    
    # Gallery parser
    gallery_parser = subparsers.add_parser('gallery', help='Create demo gallery')
    
    # Web UI parser
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=7860,
                           help='Port for web interface')
    web_parser.add_argument('--share', action='store_true',
                           help='Create public share link')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        process_single_image(args)
    elif args.command == 'batch':
        process_directory(args)
    elif args.command == 'gallery':
        create_demo_gallery(args)
    elif args.command == 'web':
        from src.ui.web_interface import launch_app
        launch_app(share=args.share, server_port=args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()