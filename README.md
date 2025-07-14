# 🛡️ AI-Powered PPE Detection & Industrial Surveillance System

![Project Banner](https://via.placeholder.com/1200x400?text=AI+PPE+Detection+and+Industrial+Surveillance)

An advanced computer vision system for real-time safety compliance monitoring in industrial environments, developed in collaboration with Jyoti CNC Automation.

## 🌟 Key Features

### 👷 PPE Detection System
- Real-time detection of safety equipment:
  - Hard hats 🪖
  - Safety vests 🦺 
  - Face masks 😷
  - Protective gloves 🧤
- Multiple input sources:
  - 📷 Browser webcam (photo capture)
  - 🎥 OpenCV webcam (local live video)
  - 📹 RTSP/IP camera streams
  - 📂 Video/image file uploads

### 🏭 Industrial Surveillance
- 24/7 anomaly detection in CCTV feeds
- Automatic violation logging with timestamps
- Real-time alerts for safety breaches

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3.9+-blue) | Core programming language |
| ![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-brightgreen) | Real-time object detection |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-orange) | Image/video processing |
| ![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red) | Interactive web dashboard |
| ![Pandas](https://img.shields.io/badge/Pandas-1.3+-yellow) | Data logging & analysis |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git

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

🖥️ Usage Guide
Select Input Source:

Browser Webcam: For single photo analysis

OpenCV Webcam: For live video (local only)

RTSP Stream: For IP camera integration

File Upload: For analyzing stored media

View Results:

Real-time detection overlay

Processing metrics

Violation alerts

Access Logs:

Timestamped violation records

Statistical analysis

Export functionality

👥 Development Team
Group ID: G00171
Institution: SVIT Vasad
Industry Partner: Intel AI for Manufacturing

Name	Role	Contact
Kushal A. Parekh	Team Lead	22ce113@svitvasad.ac.in
Darshan Pardeshi	CV Engineer	darshanpardeshi1654@gmail.com
Param V. Jani	Backend Developer	janiparam61@gmail.com
Darshan Panchal	UI/UX Designer	mpdarshanpanchal001031@gmail.com
Jaymin Raval	Testing Engineer	ravaljaymin2908@gmail.com
📜 License
This project is licensed for academic and research purposes under the MIT License.

🤝 Acknowledgements
We extend our gratitude to:

Ultralytics for YOLOv8

Jyoti CNC Automation for industry guidance

SVIT Vasad faculty for technical support

<div align="center"> <a href="https://github.com/darshan1654/AI-PPE-Detection"> <img src="https://img.shields.io/github/stars/darshan1654/AI-PPE-Detection?style=social" alt="GitHub Stars"> </a> <a href="https://github.com/darshan1654/AI-PPE-Detection/fork"> <img src="https://img.shields.io/github/forks/darshan1654/AI-PPE-Detection?style=social" alt="GitHub Forks"> </a> </div> ```
