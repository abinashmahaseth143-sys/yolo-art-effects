# src/utils/dataset_manager.py
import os
import json
from PIL import Image

class DatasetManager:
    def __init__(self, data_root='data/images'):
        self.data_root = data_root
        self.manifest = {}
        
    def scan_dataset(self):
        """Scan and catalog all images"""
        manifest = {}
        
        for category in os.listdir(self.data_root):
            category_path = os.path.join(self.data_root, category)
            if os.path.isdir(category_path):
                manifest[category] = []
                
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(category_path, img_file)
                        
                        # Get image metadata
                        try:
                            with Image.open(img_path) as img:
                                width, height = img.size
                                manifest[category].append({
                                    'filename': img_file,
                                    'path': img_path,
                                    'dimensions': (width, height),
                                    'size_kb': os.path.getsize(img_path) // 1024
                                })
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")
        
        self.manifest = manifest
        return manifest
    
    def save_manifest(self, path='data/dataset_manifest.json'):
        """Save dataset information"""
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        print(f"Manifest saved to {path}")
        
    def get_statistics(self):
        """Print dataset statistics"""
        total_images = sum(len(imgs) for imgs in self.manifest.values())
        print(f"Total images: {total_images}")
        for category, images in self.manifest.items():
            print(f"  {category}: {len(images)} images")

# Run it
if __name__ == "__main__":
    dm = DatasetManager()
    dm.scan_dataset()
    dm.save_manifest()
    dm.get_statistics()