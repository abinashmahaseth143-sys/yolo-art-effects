# test_detector.py
from src.core.detector import ArtDetector
import cv2

def test_basic_detection():
    """Test the detector with sample images"""
    detector = ArtDetector(model_size='n', device='cpu')
    
    # Test images
    test_images = [
        'data/images/classical_paintings/classical_paintings_000.jpg',
        'data/images/personal/your_photo.jpg'  # Replace with actual
    ]
    
    for img_path in test_images:
        print(f"\n{'='*50}")
        print(f"Processing: {img_path}")
        
        try:
            # Detect objects
            detections = detector.detect(img_path)
            
            # Print results
            print(f"Found {len(detections)} objects:")
            for obj in detections[:5]:  # Show first 5
                print(f"  - {obj.label} ({obj.category}): {obj.confidence:.2f}")
            
            # Visualize
            output_path = f'outputs/static/detected_{img_path.split("/")[-1]}'
            detector.visualize(img_path, detections, output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Print performance stats
    stats = detector.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_basic_detection()