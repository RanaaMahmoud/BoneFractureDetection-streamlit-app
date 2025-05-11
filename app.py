import streamlit as st
import cv2
import numpy as np
import base64
import requests
import logging
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# Load YOLOv8 model
model = YOLO('best.pt') 

def detect_bone_fracture(image_array, model):
    """Detect bone fracture in an image using the YOLOv8 model."""
    results = model(image_array)
    no_detections = True

    # Get bounding box coordinates and labels from predictions
    for result in results:
        if len(result.boxes) > 0:
            no_detections = False  # Detection found
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]                  # Confidence score
                label = f'Fracture: {confidence:.2f}'     # Label with confidence

                # Draw bounding box
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label to the image
                cv2.putText(image_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # If no detections were found, display a message on the image
    if no_detections:
        message = "No fractures detected."
        cv2.putText(image_array, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # Convert the image from BGR to RGB for displaying
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_rgb

# Streamlit UI
st.title("Bone Fracture Detection with YOLOv8")
st.write("Upload an image to detect fractures using the YOLOv8 model.")

# File uploader to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the PIL image to a NumPy array (for YOLOv8)
    img_array = np.array(image)

    # YOLOv8 expects BGR format, so convert RGB (PIL format) to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform detection using the YOLOv8 model
    result_img = detect_bone_fracture(img_bgr, model)

    # Display the result image with bounding boxes
    st.image(result_img, caption='Model Prediction', use_column_width=True)

    # Handle label saving and upload to GitHub
    fracture = st.checkbox("Mark as fracture for this image", key="fracture_checkbox")  # Unique key added
    if st.button("Save and Upload", key="save_upload_button"):  # Unique key for button
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        label = 1 if fracture else 0
        label_filename = f"label_{timestamp}.txt"
        image_filename = f"image_{timestamp}.jpg"
        
        # Save label and image to GitHub
        save_label_to_github(str(label), label_filename)
        save_image_to_github(result_img, image_filename)

else:
    st.write("Please upload an image.")
