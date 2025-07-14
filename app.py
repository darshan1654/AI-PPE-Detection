import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os
import time
import pandas as pd
from datetime import datetime
import numpy as np  
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container

# Load YOLOv8 model
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")
    st.stop()

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

LOG_FILE = "violation_logs.csv"

# Set page config with improved styling
st.set_page_config(
    page_title="AI CCTV Surveillance",
    layout="wide",
    page_icon="👷",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stSelectbox, .stTextInput, .stRadio>div {
            background-color: white;
            border-radius: 8px;
            padding: 8px;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stDataFrame {
            border-radius: 8px;
        }
        .css-1v3fvcr {
            padding: 1rem;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .css-1y4p8pa {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .violation-card {
            border-left: 4px solid #ff4b4b;
            padding: 12px;
            margin-bottom: 8px;
            background-color: #fff8f8;
            border-radius: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# Main header with colored header
colored_header(
    label="👷 AI CCTV Surveillance System",
    description="An AI-powered CCTV surveillance system for real-time detection of PPE compliance, including helmet and mask violations, using YOLO and computer vision.",
    color_name="blue-70",
)

# Sidebar with improved layout
with st.sidebar:
    st.image("home.jpeg", use_container_width=True)
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-top: 20px;">
        <span style="font-size: 1.5em;">⚙️</span>
        <span style="font-weight: bold; font-size: 1.1rem;">Configuration</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Updated source options with browser webcam
    source_type = st.radio(
        "Select Input Source",
        ['Browser Webcam (Photo)', 'Upload Video', 'Upload Image', 'RTSP IP Camera', 'OpenCV Webcam (Local Only)'],
        index=0,
        help="Choose the source for surveillance feed"
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Model", "YOLOv8", "Active")
    status_col2.metric("FPS", "30", "Live")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This AI surveillance system detects:
    - PPE violations
    - Safety breaches
    - Unauthorized access
    - Other anomalies
    """)

# Violation logger with enhanced functionality
def log_violation(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, class_name, round(confidence, 2)]], 
                        columns=["Timestamp", "Violation", "Confidence"])
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, index=False)
    else:
        try:
            # Read existing file to check format
            existing = pd.read_csv(LOG_FILE)
            if not all(col in existing.columns for col in ["Timestamp", "Violation", "Confidence"]):
                # If columns don't match, recreate file
                entry.to_csv(LOG_FILE, index=False)
            else:
                # Append without header
                entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
        except:
            # If file is corrupted, recreate it
            entry.to_csv(LOG_FILE, index=False)
    
    # Display real-time alert for violation
    with st.container():
        st.markdown(f"""
        <div class="violation-card">
            <strong>⚠️ New Violation Detected</strong><br>
            Type: {class_name}<br>
            Confidence: {round(confidence, 2)}%<br>
            Time: {timestamp}
        </div>
        """, unsafe_allow_html=True)

# Frame processor with additional metrics
def process_frame(frame):
    results = model(frame)[0]
    annotated_frame = results.plot()
    
    # Calculate metrics
    violation_count = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        confidence = float(box.conf[0])
        if "NO" in class_name.upper():
            log_violation(class_name, confidence)
            violation_count += 1
    
    return annotated_frame, results, violation_count

# Enhanced stream handler with metrics display
def display_video(video_source):
    cap = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    stop_button = st.button("🛑 Stop Stream")
    fail_count = 0
    
    # Create metrics row
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
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

        start_time = time.time()
        annotated_frame, results, violation_count = process_frame(frame)
        processing_time = time.time() - start_time
        
        # Update metrics
        with metrics_col1:
            st.metric("Processing Time", f"{processing_time*1000:.1f} ms")
        with metrics_col2:
            st.metric("Objects Detected", len(results.boxes))
        with metrics_col3:
            st.metric("Violations", violation_count, delta_color="inverse")
        
        # Style the metric cards (updated parameters)
        style_metric_cards(
            background_color="#f8f9fa",
            border_left_color="#0068c9",
            box_shadow="0 2px 8px rgba(0,0,0,0.1)"
        )
        
        # Display frame
        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), 
                      channels="RGB", use_container_width=True)
        time.sleep(0.03)

    cap.release()

# Enhanced image handler
def process_image(image_path):
    frame = cv2.imread(image_path)
    start_time = time.time()
    annotated_frame, results, violation_count = process_frame(frame)
    processing_time = time.time() - start_time
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Processing Time", f"{processing_time*1000:.1f} ms")
    col2.metric("Violations Detected", violation_count)
    
    return annotated_frame

# Main content area with improved layout
tab1, tab2 = st.tabs(["Live Monitoring", "Violation Logs"])

with tab1:
    # Browser Webcam Option
    if source_type == 'Browser Webcam (Photo)':
        st.info("ℹ️ Captures single photos (browser permission required)")
        
        captured_image = st.camera_input("Take a photo for PPE detection")
        
        if captured_image:
            # Convert to OpenCV format
            file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process the image
            annotated_frame, results, violation_count = process_frame(frame)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption="Processed Image",
                        use_container_width=True)
            with col2:
                st.metric("Processing Time", "N/A (single image)")
                st.metric("Violations Detected", violation_count)

    elif source_type == 'OpenCV Webcam (Local Only)':
        if os.environ.get('IS_STREAMLIT_CLOUD'):
            st.error("OpenCV webcam only works when running locally!")
            st.info("Tip: Use 'Browser Webcam' for photo capture in the cloud")
        else:
            if st.button("🎥 Start Webcam"):
                display_video(0)

    elif source_type == 'Upload Image':
        with stylable_container(
            key="image_container",
            css_styles="""
                {
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                }
            """
        ):
            uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                temp_image_path = os.path.join(tempfile.gettempdir(), uploaded_image.name)
                with open(temp_image_path, 'wb') as f:
                    f.write(uploaded_image.read())
                st.success("✅ Image uploaded successfully. Processing...")
                annotated_image = process_image(temp_image_path)
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 
                        caption="Processed Image with PPE Detection", 
                        use_container_width=True)

    elif source_type == 'RTSP IP Camera':
        with stylable_container(
            key="rtsp_container",
            css_styles="""
                {
                    border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                }
            """
        ):
            rtsp_url = st.text_input(
                "Enter RTSP Stream URL",
                placeholder="rtsp://username:password@192.168.1.100:554/stream1",
                help="Enter the RTSP URL of your IP camera"
            )
            if rtsp_url:
                if st.button("📡 Start RTSP Stream", type="primary"):
                    try:
                        display_video(rtsp_url)
                    except Exception as e:
                        st.error(f"❌ Unable to open RTSP stream: {e}")

with tab2:
    # Enhanced violation log viewer
    st.markdown("### 📄 Recent Violation Logs")
    if os.path.exists(LOG_FILE):
        try:
            df_logs = pd.read_csv(LOG_FILE)
            if not df_logs.empty:
                # Ensure required columns exist
                if all(col in df_logs.columns for col in ["Timestamp", "Violation", "Confidence"]):
                    st.dataframe(
                        df_logs.tail(10).sort_values("Timestamp", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add download and clear buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Full Log", 
                            data=df_logs.to_csv(index=False), 
                            file_name="violation_logs.csv", 
                            mime="text/csv"
                        )
                    with col2:
                        if st.button("🗑️ Clear Logs", type="secondary"):
                            if os.path.exists(LOG_FILE):
                                os.remove(LOG_FILE)
                                st.success("Logs cleared successfully!")
                                st.rerun()
                    
                    # Show violation statistics
                    st.markdown("### 📊 Violation Statistics")
                    violations_by_type = df_logs["Violation"].value_counts()
                    st.bar_chart(violations_by_type)
                else:
                    st.warning("Log file format is incorrect - recreating it")
                    os.remove(LOG_FILE)
                    # Create new empty log file with correct format
                    pd.DataFrame(columns=["Timestamp", "Violation", "Confidence"]).to_csv(LOG_FILE, index=False)
        except Exception as e:
            st.error(f"Error reading log file: {str(e)}")
            # Attempt to recreate the log file
            pd.DataFrame(columns=["Timestamp", "Violation", "Confidence"]).to_csv(LOG_FILE, index=False)
    else:
        st.info("ℹ️ No violations logged yet.")
