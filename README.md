# 🎨 AI Art Effect Generator with YOLO

An intelligent system that transforms artworks by applying dynamic visual effects to detected objects using YOLOv8 real-time object detection.

## 📋 Features

- **Real-time object detection** using YOLOv8
- **Category-aware visual effects** (glow, particles, speed lines, float, wind, pulse)
- **Animated GIF generation** from static images
- **Interactive web interface** with Gradio
- **20+ test images** included
- **Multiple processing modes**: Single image, batch, webcam, examples

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Windows/Mac/Linux
- 4GB+ RAM
- Webcam (optional, for live processing)

### Installation

1. **Clone/Download** the project:
   ```bash
   git clone https://github.com/yourusername/yolo_art_effects.git
   cd yolo_art_effects
 
Create virtual environment (Windows):
   python -m venv venv
venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt

Running the Application
python -m src.cli.main web --port 7862

