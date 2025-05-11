import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
import io

# Placeholder functions for GitHub integration
def save_label_to_github(label, filename):
    st.warning("GitHub upload not implemented. Label would be saved as: " + filename)
    # Implement GitHub API call here (e.g., using PyGithub or requests + personal access token)

def save_image_to_github(image_array, filename):
    st.warning("GitHub upload not implemented. Image would be saved as: " + filename)
    # Convert BGR to RGB and then save using PIL or cv2.imencode if needed

# Load YOLOv8 model once
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Title & Description
st.set_page_config(page_title="Bone Fracture Detector", layout="wide")
st.title("🦴 Bone Fracture Detection using YOLOv8")
st.write("Upload an X-ray image to detect possible bone fractures using a YOLOv8 model.")

# Sidebar Upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

# Detection function
def detect_bone_fracture(image_array, model):
    results = model(image_array)
    detections = []

    for result in results:
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = f'Fracture: {confidence:.2f}'
                detections.append((x1, y1, x2, y2, label))

                # Draw bounding box
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_array, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if not detections:
        cv2.putText(image_array, "No fractures detected.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), detections

# Main logic
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.subheader("🔍 Detection In Progress...")
    with st.spinner("Running YOLOv8 model..."):
        result_img, detections = detect_bone_fracture(img_bgr.copy(), model)

    st.success("Detection complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(result_img, caption="Detection Result", use_column_width=True)

    # Detection summary
    st.subheader("📊 Detection Summary")
    if detections:
        for i, (x1, y1, x2, y2, label) in enumerate(detections):
            st.write(f"🔸 Detection {i+1}: `{label}` at ({x1},{y1}) → ({x2},{y2})")
    else:
        st.info("✅ No fractures detected.")

    # Save checkbox + button
    st.markdown("---")
    fracture = st.checkbox("✅ Mark this image as containing a fracture", key="fracture_checkbox")
    if st.button("💾 Save Result to GitHub"):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        label_filename = f"label_{timestamp}.txt"
        image_filename = f"image_{timestamp}.jpg"
        label = 1 if fracture else 0

        # Convert RGB to BGR before saving if needed
        save_label_to_github(str(label), label_filename)
        save_image_to_github(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR), image_filename)

else:
    st.info("📤 Upload an image from the sidebar to get started.")
