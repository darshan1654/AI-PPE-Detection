# 🛡️ AI-Powered PPE Detection System

![Project Banner](https://via.placeholder.com/1200x400?text=AI+PPE+Detection+System)

Real-time safety compliance monitoring using computer vision and YOLOv8.

## ✨ Features
- Detect helmets, vests, masks, and gloves
- Multiple input sources:
  - 📷 Browser webcam (photos)
  - 🎥 Local webcam (video)
  - 📹 RTSP/IP cameras
  - 📂 File uploads
- Violation logging with timestamps
- Interactive Streamlit dashboard

## 🛠️ Tech Stack

- Python 3.9+
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- Pandas

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/darshan1654/AI-PPE-Detection.git
cd AI-PPE-Detection
```
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/MacOS
 # venv\Scripts\activate   # Windows
```
```bash
# Install dependencies
pip install -r requirements.txt
```
```bash
# Running the Application
streamlit run app.py
```
