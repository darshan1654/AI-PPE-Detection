
# 🛡️📹 AI-Powered Industrial Safety Surveillance System

**Organization:** Jyoti CNC Automation, Rajkot  
**Category:** Industry Defined Problem

![AI CCTV Surveillance · Streamlit_page-0001](https://github.com/user-attachments/assets/58a5799f-b7a4-47f7-b165-b4078429a1ea) ![AI CCTV Surveillance · Streamlit2_page-0001](https://github.com/user-attachments/assets/452a0632-cbe2-4a3b-b6ee-3c876327804f)


---

## 📄 Project Overview

This project integrates **AI-powered Personal Protective Equipment (PPE) detection** and **CCTV-based anomaly surveillance** into a single real-time system. It is designed to improve **workplace safety**, **regulatory compliance**, and **operational efficiency** in industrial environments.

Using advanced **deep learning models** and **computer vision**, the system ensures that workers adhere to safety protocols and that unusual or unsafe activities are promptly flagged.

---

## ✅ Key Features

### 👷 PPE Detection Module
- Real-time detection of:
  - 🪖 Helmets  
  - 😷 Face Masks  
  - 👷 Safety Vests  
  - 🧤 Gloves  
- Supports live feed from **Webcam/IP Camera**
- Upload and analyze **video/image files**
- Detection logs with timestamps and confidence scores

### 📹 CCTV Anomaly Detection Module
- Real-time detection of:
  - 🚫 Entry into restricted zones  
  - ⚠️ Safety violations (e.g., no helmet, improper behavior)  
  - 🚷 Suspicious or unsafe movements  
- Continuous monitoring via CCTV/IP camera
- Alert generation on detection

### 📊 Dashboard (Built with Streamlit)
- Live status feed with detection results
- Real-time preview of camera feed
- Violation and compliance logs
- Summary statistics and compliance reports

---

## 📦 Tech Stack

| Component       | Technology         |
|----------------|--------------------|
| 💡 AI Model     | YOLOv8 (Ultralytics) |
| 🧠 Backend       | Python, OpenCV     |
| 🌐 Frontend/UI  | Streamlit          |
| 🎥 Video Input  | Webcam/IP Camera   |
| 📊 Data Logging | Pandas, CSV Logs   |

---

## 🏭 Industrial Benefits

- ✅ Automated compliance with PPE policies  
- 🔍 Real-time safety monitoring  
- 📉 Reduced accident risk and manual supervision  
- 📊 Actionable insights from safety data  

---

## 📸 Sample Outputs

- 📷 Detected image with PPE boxes  
- 🧾 Logs with timestamp and violation type  
- 📈 Streamlit dashboard with real-time updates  

---

## 🔧 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/darshan1654/AI-PPE-Detection.git

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py               # Streamlit app
├── best.pt              # YOLOv8 model files
├── yolov8n.pt           # YOLOv8 model files
├── violation_logs.csv   # Detection logs CSV file
├── requirements.txt     # Dependencies
├── packages.txt         # packages
└── runtime.txt          # For deployment (e.g., Heroku)
```

---

## 👨‍💻 Team Details

**Group ID:** G00171

| Name             | Email                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Kushal A. Parekh | [22ce113@svitvasad.ac.in](mailto:22ce113@svitvasad.ac.in)                   |
| Darshan Pardeshi | [darshanpardeshi1654@gmail.com](mailto:darshanpardeshi1654@gmail.com)       |
| Param V. Jani    | [janiparam61@gmail.com](mailto:janiparam61@gmail.com)                       |
| Darshan Panchal  | [mpdarshanpanchal001031@gmail.com](mailto:mpdarshanpanchal001031@gmail.com) |
| Jaymin Raval     | [ravaljaymin2908@gmail.com](mailto:ravaljaymin2908@gmail.com)               |

---

## 🔮 Future Scope

* 🔔 Voice/Email/SMS alert system
* 🔗 Integration with ERP systems
* 📊 Admin dashboard with analytics
* 📤 Auto-upload violation clips
* 📡 Multi-location camera support

---

## 🏆 Achievements

* ✅ Used in **real industrial setup** at Jyoti CNC
* 🎓 Presented at **college-level expo**
* 📡 Successfully tested with **live camera feeds**

---

## 📜 License

For **Academic and Research use** only.

---

## 🙏 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org)
* [Streamlit](https://streamlit.io)
* Jyoti CNC Automation Pvt. Ltd.
* SVIT Vasad

---

## 📬 Contact

📧 [janiparam61@gmail.com](mailto:janiparam61@gmail.com)
📧 [darshanpardeshi1654@gmail.com](mailto:darshanpardeshi1654@gmail.com)
📧 [22ce113@svitvasad.ac.in](mailto:22ce113@svitvasad.ac.in)
📧 [mpdarshanpanchal001031@gmail.com](mailto:mpdarshanpanchal001031@gmail.com)
📧 [ravaljaymin2908@gmail.com](mailto:ravaljaymin2908@gmail.com)

<img width="709" height="186" alt="image" src="https://github.com/user-attachments/assets/9afae834-e0bd-4a0f-9b6a-919b491df72d" />

---
