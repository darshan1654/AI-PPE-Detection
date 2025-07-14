import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os
import time
import pandas as pd
from datetime import datetime

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

LOG_FILE = "violation_logs.csv"
st.set_page_config(page_title="AI CCTV Surveillance", layout="wide")
st.title("🎯 AI-Powered CCTV Surveillance System for Jyoti CNC")
st.markdown("Detects anomalies, safety breaches, and human presence using webcam, video file, image, or IP CCTV camera (RTSP stream).")

source_type = st.sidebar.radio(
    "Select Input Source",
    ['Webcam', 'Upload Video', 'Upload Image', 'RTSP IP Camera']
)

temp_dir = tempfile.mkdtemp()

# Violation logger
def log_violation(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, class_name, round(confidence, 2)]], columns=["Timestamp", "Violation", "Confidence"])
    if os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        entry.to_csv(LOG_FILE, mode='w', header=True, index=False)

# Frame processor
def process_frame(frame):
    results = model(frame)[0]
    annotated_frame = results.plot()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        confidence = float(box.conf[0])
        if "NO" in class_name.upper():
            log_violation(class_name, confidence)
    return annotated_frame, results

# Stream handler (Webcam, RTSP, or Video File)
def display_video(video_source):
    cap = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    stop_button = st.button("🛑 Stop Stream")
    fail_count = 0

    while cap.isOpened():
        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 10:
                st.warning("⚠️ Stream lost or ended.")
                break
            continue
        fail_count = 0

        annotated_frame, _ = process_frame(frame)
        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        time.sleep(0.03)

    cap.release()

# Image handler
def process_image(image_path):
    frame = cv2.imread(image_path)
    annotated_frame, _ = process_frame(frame)
    return annotated_frame

# Logic for different sources
if source_type == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded. Processing...")
        display_video(temp_video_path)

elif source_type == 'Webcam':
    if st.button("Start Webcam"):
        display_video(0)

elif source_type == 'Upload Image':
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        temp_image_path = os.path.join(temp_dir, uploaded_image.name)
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_image.read())
        st.success("Image uploaded. Processing...")
        annotated_image = process_image(temp_image_path)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

elif source_type == 'RTSP IP Camera':
    rtsp_url = st.text_input("Enter RTSP Stream URL", placeholder="rtsp://username:password@192.168.1.100:554/stream1")
    if rtsp_url:
        if st.button("Start RTSP Stream"):
            try:
                display_video(rtsp_url)
            except Exception as e:
                st.error(f"Unable to open RTSP stream: {e}")

# Violation log viewer
st.markdown("## 📄 Violation Logs")
if os.path.exists(LOG_FILE):
    df_logs = pd.read_csv(LOG_FILE)
    st.dataframe(df_logs.tail(10))
    st.download_button("📥 Download Full Log", data=df_logs.to_csv(index=False), file_name="violation_logs.csv", mime="text/csv")
else:
    st.info("No violations logged yet.")
