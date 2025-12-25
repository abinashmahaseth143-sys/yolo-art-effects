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


You can download zip from github then unzip it 
Open vs code and open folder you unzip then open terminal

### Installation

1. **Clone/Download** the project:
   ```bash
   git clone https://github.com/abinashmahaseth143-sys/yolo-art-effects.git
   cd yolo_art_effects
 
Create virtual environment (Windows):
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Running the Application
python -m src.cli.main web --port 7862


After running the application model will load sucessfully you need to click on link then gradio will open after gradio is open you can download image from browse all images and upload and for webcam you can just click on bottom it will open camera and detect and give results
Running on local URL:  http://127.0.0.1:7862

To exit type Ctrl+C


