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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportImage, PageBreak

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_PATH = "best.pt" 

CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}

TARGET_WIDTH = 640
TARGET_HEIGHT = 640

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
# 5. PREPROCESSING (DYNAMIC)
# ============================================================
def preprocess_image(image, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile):
    calibrated_color = calibrate_image(image, TARGET_WIDTH, TARGET_HEIGHT)

    gray = cv2.cvtColor(calibrated_color, cv2.COLOR_BGR2GRAY)

    # Replaced Median filter with Average Blur
    processed = cv2.blur(gray, (avg_kernel, avg_kernel))

    if use_contrast:
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        processed = clahe.apply(processed)

    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    return calibrated_color, processed_bgr

# ============================================================
# 6. DETECT PCB DEFECTS
# ============================================================
def detect_defects(image, confidence, iou_threshold, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile):
    calibrated_color, processed_bgr = preprocess_image(
        image, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile
    )

    results = model.predict(source=processed_bgr, conf=confidence, iou=iou_threshold, verbose=False)
    result = results[0]

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
# 8. PDF REPORT GENERATORS
# ============================================================
def generate_single_pdf_report(image_name, detections, statistics, annotated_image, confidence, iou_threshold):
    with tempfile.TemporaryDirectory() as temp_dir:
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
        story.append(Paragraph(f"<b>Confidence Threshold:</b> {confidence:.2f} | <b>IoU Threshold:</b> {iou_threshold:.2f}", styles["Normal"]))
        story.append(Spacer(1, 15))
        story.append(Paragraph(f"<b>Total Detected Defects:</b> {total_defects}", styles["Heading2"]))
        story.append(Paragraph(f"<b>Inspection Status:</b> {status}", styles["Heading2"]))
        story.append(Spacer(1, 15))

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

def generate_batch_pdf_report(batch_results, summary_df):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "PCB_Batch_Report.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        # Cover Page
        story.append(Paragraph("Batch PCB Defect Inspection Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Total Images Processed:</b> {len(summary_df)}", styles["Normal"]))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Batch Summary", styles["Heading2"]))
        summary_table_data = [["Image Name", "Status", "Total Defects"]]
        for _, row in summary_df.iterrows():
            summary_table_data.append([row["Image"], row["Status"], str(row["Defects"])])

        t = Table(summary_table_data, colWidths=[250, 100, 100])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (1, 0), (-1, -1), "CENTER")
        ]))
        story.append(t)
        story.append(PageBreak())

        # Individual Pages
        for index, item in enumerate(batch_results):
            img_path = os.path.join(temp_dir, f"temp_{index}.jpg")
            cv2.imwrite(img_path, item["Annotated"])

            story.append(Paragraph(f"Image: {item['Image']}", styles["Heading2"]))
            story.append(Paragraph(f"<b>Status:</b> {item['Status']} | <b>Defects:</b> {item['Defects']}", styles["Normal"]))
            story.append(Spacer(1, 10))
            story.append(ReportImage(img_path, width=6.5 * inch, height=6.5 * inch))

            if not item["Stats"].empty:
                story.append(Spacer(1, 10))
                stat_data = [["Defect Type", "Count"]]
                for _, row in item["Stats"].iterrows():
                    stat_data.append([row["Defect"], str(row["Count"])])

                stat_t = Table(stat_data, colWidths=[200, 100])
                stat_t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER")
                ]))
                story.append(stat_t)

            story.append(PageBreak())

        doc.build(story)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes

# ============================================================
# 9. SIDEBAR CONTROL PANEL
# ============================================================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.write("### AI Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("### Preprocessing Tuning")
avg_kernel = st.sidebar.slider("Average Blur Kernel Size", 1, 15, 5, 2)
use_contrast = st.sidebar.checkbox("Apply Contrast Stretching", value=False)
use_clahe = st.sidebar.checkbox("Apply CLAHE", value=True)

if use_clahe:
    clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)
    clahe_tile = st.sidebar.slider("CLAHE Tile Size", 2, 16, 8, 2)
else:
    clahe_clip, clahe_tile = 2.0, 8

# ============================================================
# 10. MAIN GUI
# ============================================================
st.title("🔬 Automated PCB Defect Inspection")
st.write("Tune your preprocessing settings in the sidebar. The image updates in real-time.")

tab_single, tab_batch = st.tabs(["🖼️ Single Image", "📦 Bulk Images"])

# ------------------------------------------------------------
# TAB 1: SINGLE IMAGE
# ------------------------------------------------------------
with tab_single:
    uploaded_file = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Real-time preview generation
        _, processed_bgr = preprocess_image(image, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile)

        st.subheader("Live Preprocessing Preview")
        st.image(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")

        if st.button("🚀 Run Inspection", type="primary", key="btn_single"):
            with st.spinner("Analyzing board..."):
                calib_color, annotated, detections = detect_defects(
                    image, confidence, iou_threshold, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile
                )
                statistics = calculate_statistics(detections)
                status = determine_status(len(detections))

                st.subheader("YOLO11s Detection Result")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.subheader("Inspection Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Defects", len(detections))
                m2.metric("Detection Types", len(statistics))
                m3.metric("Status", status)

                if not statistics.empty:
                    st.dataframe(statistics, use_container_width=True)

                pdf_bytes = generate_single_pdf_report(
                    uploaded_file.name, detections, statistics, annotated, confidence, iou_threshold
                )
                st.download_button(
                    label="📄 Download PDF Report",
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

        # Real-time preview of the FIRST image in the batch
        st.write("### Tune settings using the first image:")
        first_file_bytes = np.frombuffer(uploaded_files[0].getvalue(), np.uint8)
        first_image = cv2.imdecode(first_file_bytes, cv2.IMREAD_COLOR)
        _, first_processed_bgr = preprocess_image(first_image, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile)
        st.image(cv2.cvtColor(first_processed_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")

        if st.button("🚀 Process Batch", type="primary", key="btn_batch"):
            batch_results = []
            summary_data = []
            zip_buffer = io.BytesIO()
            progress = st.progress(0)

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for index, uploaded in enumerate(uploaded_files):
                    file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    if img is None: continue

                    calib_color, annotated, detections = detect_defects(
                        img, confidence, iou_threshold, avg_kernel, use_contrast, use_clahe, clahe_clip, clahe_tile
                    )
                    total_defects = len(detections)
                    status = determine_status(total_defects)
                    stats = calculate_statistics(detections)

                    summary_data.append({
                        "Image": uploaded.name,
                        "Defects": total_defects,
                        "Status": status
                    })

                    batch_results.append({
                        "Image": uploaded.name,
                        "Defects": total_defects,
                        "Status": status,
                        "Annotated": annotated,
                        "Stats": stats
                    })

                    # Write annotated image to ZIP
                    is_success, buffer = cv2.imencode(".jpg", annotated)
                    if is_success:
                        zip_file.writestr(f"annotated_{uploaded.name}", buffer.tobytes())

                    progress.progress((index + 1) / len(uploaded_files))

            summary_df = pd.DataFrame(summary_data)
            st.subheader("Batch Inspection Completed")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Processed", len(summary_df))
            c2.metric("PASS", len(summary_df[summary_df["Status"] == "PASS"]))
            c3.metric("WARNING", len(summary_df[summary_df["Status"] == "WARNING"]))
            c4.metric("FAIL", len(summary_df[summary_df["Status"] == "FAIL"]))

            st.dataframe(summary_df, use_container_width=True)

            # Generate Batch PDF Report
            batch_pdf_bytes = generate_batch_pdf_report(batch_results, summary_df)

            st.write("### Download Results")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="📄 Download Full Batch PDF Report", 
                    data=batch_pdf_bytes, 
                    file_name="batch_inspection_report.pdf", 
                    mime="application/pdf"
                )
            with dl2:
                st.download_button(
                    label="🗂️ Download Annotated Images (ZIP)", 
                    data=zip_buffer.getvalue(), 
                    file_name="annotated_batch.zip", 
                    mime="application/zip"
                )
