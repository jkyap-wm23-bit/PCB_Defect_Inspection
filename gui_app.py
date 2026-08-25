import os
import tempfile
import io
import zipfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportImage

# ============================================================
# 1. CONFIGURATION
# ============================================================
# Using relative path for GitHub/Cloud deployment compatibility
MODEL_PATH = "best.pt" 

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}

# Preprocessing Constants (Must match training data)
TARGET_WIDTH = 640
TARGET_HEIGHT = 640
MEDIAN_KERNEL = 5
USE_CONTRAST_STRETCH = False
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="PCB Defect Inspection System", page_icon="🔧", layout="wide")

# ============================================================
# 3. LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"YOLO model was not found: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ============================================================
# 4. IMAGE CALIBRATION
# ============================================================
def calibrate_image(image, target_width=640, target_height=640):
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    calibrated = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    calibrated[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    return calibrated

# ============================================================
# 5. PREPROCESSING
# ============================================================
def preprocess_image(image):
    # 1. Calibration (Keep color for final UI display)
    calibrated_color = calibrate_image(image, TARGET_WIDTH, TARGET_HEIGHT)
    
    # 2. Grayscale & Blur for AI
    gray = cv2.cvtColor(calibrated_color, cv2.COLOR_BGR2GRAY)
    processed = cv2.medianBlur(gray, MEDIAN_KERNEL)
    
    if USE_CONTRAST_STRETCH:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        
    # 3. CLAHE Enhancement
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
        processed = clahe.apply(processed)
        
    # 4. Format for YOLO
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    return calibrated_color, processed_bgr

# ============================================================
# 6. DETECT PCB DEFECTS
# ============================================================
def detect_defects(image, confidence, iou_threshold):
    calibrated_color, processed_bgr = preprocess_image(image)
    
    # Run prediction on the enhanced image
    results = model.predict(source=processed_bgr, conf=confidence, iou=iou_threshold, verbose=False)
    result = results[0]
    
    # Draw results on the original color image
    annotated = result.plot(img=calibrated_color)
    
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence_score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            detections.append({
                "Defect": CLASS_NAMES.get(class_id, str(class_id)),
                "Confidence": confidence_score,
                "X1": int(x1), "Y1": int(y1), "X2": int(x2), "Y2": int(y2)
            })
            
    return calibrated_color, annotated, detections

# ============================================================
# 7. STATISTICS & STATUS LOGIC
# ============================================================
def calculate_statistics(detections):
    if not detections:
        return pd.DataFrame(columns=["Defect", "Count"])
    df = pd.DataFrame(detections)
    stats = df["Defect"].value_counts().reset_index()
    stats.columns = ["Defect", "Count"]
    return stats

def determine_status(total_defects):
    if total_defects == 0:
        return "PASS"
    elif total_defects <= 2:
        return "WARNING"
    else:
        return "FAIL"

# ============================================================
# 8. PDF REPORT GENERATOR
# ============================================================
def generate_pdf_report(image_name, detections, statistics, annotated_image, confidence, iou_threshold):
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, "detection.jpg")
    pdf_path = os.path.join(temp_dir, "PCB_Inspection_Report.pdf")
    
    cv2.imwrite(image_path, annotated_image)
    
    total_defects = len(detections)
    status = determine_status(total_defects)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("PCB Defect Inspection Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Image:</b> {image_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Model:</b> YOLO11s Hybrid", styles["Normal"]))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>Total Detected Defects:</b> {total_defects}", styles["Heading2"]))
    story.append(Paragraph(f"<b>Inspection Status:</b> {status}", styles["Heading2"]))
    story.append(Spacer(1, 15))
    
    # Stats Table
    table_data = [["Defect Type", "Count"]]
    for _, row in statistics.iterrows():
        table_data.append([row["Defect"], str(row["Count"])])
        
    table = Table(table_data, colWidths=[250, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("Detection Result", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(ReportImage(image_path, width=6.5 * inch, height=6.5 * inch))
    
    doc.build(story)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

# ============================================================
# 9. SIDEBAR CONTROL PANEL
# ============================================================
st.sidebar.title("⚙️ Control Panel")

confidence = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("### Preprocessing Profile")
st.sidebar.write(f"- **Size:** {TARGET_WIDTH} × {TARGET_HEIGHT}")
st.sidebar.write(f"- **Median Blur:** {MEDIAN_KERNEL} × {MEDIAN_KERNEL}")
st.sidebar.write(f"- **CLAHE:** {'ON' if USE_CLAHE else 'OFF'}")
st.sidebar.write(f"- **CLAHE Clip Limit:** {CLAHE_CLIP_LIMIT}")
st.sidebar.write(f"- **CLAHE Tile:** {CLAHE_TILE_SIZE} × {CLAHE_TILE_SIZE}")

# ============================================================
# 10. MAIN GUI TABS
# ============================================================
st.title("🔬 Automated PCB Defect Inspection")
st.write("YOLO11s-based visual inspection with real-time CLAHE enhancement and reporting.")

tab_single, tab_batch, tab_video = st.tabs(["🖼️ Single Image", "📦 Bulk Images", "🎥 Video Stream"])

# ------------------------------------------------------------
# TAB 1: SINGLE IMAGE
# ------------------------------------------------------------
with tab_single:
    uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"], key="single")
    
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        calib_color, annotated, detections = detect_defects(image, confidence, iou_threshold)
        statistics = calculate_statistics(detections)
        status = determine_status(len(detections))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original (Calibrated)")
            st.image(cv2.cvtColor(calib_color, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("YOLO11s Detection")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        st.subheader("Inspection Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Defects", len(detections))
        m2.metric("Detection Types", len(statistics))
        m3.metric("Status", status)
        
        if not statistics.empty:
            st.dataframe(statistics, use_container_width=True)
            
        pdf_bytes = generate_pdf_report(uploaded_file.name, detections, statistics, annotated, confidence, iou_threshold)
        st.download_button(
            label="📄 Download PDF Inspection Report",
            data=pdf_bytes,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_report.pdf",
            mime="application/pdf"
        )

# ------------------------------------------------------------
# TAB 2: BULK IMAGES
# ------------------------------------------------------------
with tab_batch:
    uploaded_files = st.file_uploader("Upload multiple PCB images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    
    if uploaded_files:
        st.write(f"Images selected: {len(uploaded_files)}")
        
        if st.button("Process All Images"):
            batch_results = []
            zip_buffer = io.BytesIO()
            progress = st.progress(0)
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for index, uploaded in enumerate(uploaded_files):
                    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None: continue
                    
                    calib_color, annotated, detections = detect_defects(image, confidence, iou_threshold)
                    total_defects = len(detections)
                    status = determine_status(total_defects)
                    
                    batch_results.append({
                        "Image": uploaded.name,
                        "Defects": total_defects,
                        "Status": status
                    })
                    
                    # Write to ZIP
                    is_success, buffer = cv2.imencode(".jpg", annotated)
                    if is_success:
                        zip_file.writestr(f"annotated_{uploaded.name}", buffer.tobytes())
                        
                    # Show Preview
                    with st.expander(f"Preview: {uploaded.name} | Status: {status} | Defects: {total_defects}"):
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                    progress.progress((index + 1) / len(uploaded_files))
                    
            results_df = pd.DataFrame(batch_results)
            st.subheader("Batch Inspection Results")
            st.dataframe(results_df, use_container_width=True)
            
            if not results_df.empty:
                st.write("### Batch Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Processed", len(results_df))
                c2.metric("PASS", len(results_df[results_df["Status"] == "PASS"]))
                c3.metric("WARNING", len(results_df[results_df["Status"] == "WARNING"]))
                c4.metric("FAIL", len(results_df[results_df["Status"] == "FAIL"]))
                
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("📊 Download CSV Report", data=results_df.to_csv(index=False), file_name="batch_results.csv", mime="text/csv")
                with dl2:
                    st.download_button("🗂️ Download Annotated Images (ZIP)", data=zip_buffer.getvalue(), file_name="annotated_batch.zip", mime="application/zip")

# ------------------------------------------------------------
# TAB 3: VIDEO PROCESSING
# ------------------------------------------------------------
with tab_video:
    uploaded_video = st.file_uploader("Upload PCB inspection video", type=["mp4", "avi", "mov"], key="video")
    
    if uploaded_video is not None:
        if st.button("Process Video"):
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_video.read())
            temp_input.close()
            
            cap = cv2.VideoCapture(temp_input.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_video.close()
            writer = cv2.VideoWriter(output_video.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            
            progress = st.progress(0)
            frame_counter = 0
            total_detection_count = 0
            preview_placeholder = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                calib_color, annotated, detections = detect_defects(frame, confidence, iou_threshold)
                total_detection_count += len(detections)
                
                annotated_frame = cv2.resize(annotated, (width, height))
                writer.write(annotated_frame)
                
                frame_counter += 1
                if total_frames > 0: progress.progress(min(frame_counter / total_frames, 1.0))
                
                if frame_counter % 5 == 0:
                    preview_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Processing Frame {frame_counter}")
                    
            cap.release()
            writer.release()
            progress.progress(1.0)
            
            st.success("✅ Video processing completed.")
            st.write(f"**Frames processed:** {frame_counter} | **Total defects tracked:** {total_detection_count}")
            
            with open(output_video.name, "rb") as f:
                st.download_button("🎬 Download Processed Video", data=f.read(), file_name="pcb_inspection_video.mp4", mime="video/mp4")
