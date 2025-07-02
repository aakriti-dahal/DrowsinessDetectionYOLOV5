import streamlit as st
import os
from PIL import Image
import subprocess
import uuid
import shutil

st.title("🛌 Drowsiness Detection with YOLOv5")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

detect_dir = "runs/detect"

if uploaded_file is not None:
    # Save uploaded file
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully.")

    # Run YOLOv5 detect.py
    st.info("Running detection...")
    result = subprocess.run(
        ["python", "detect.py", "--weights", "best.pt", "--source", file_path, "--conf", "0.25", "--save-txt", "--save-conf"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode == 0:
        st.success("Detection complete.")
        # Find latest run folder
        dirs = sorted([os.path.join(detect_dir, d) for d in os.listdir(detect_dir)], key=os.path.getmtime, reverse=True)
        if dirs:
            result_folder = dirs[0]
            result_files = os.listdir(result_folder)
            for file in result_files:
                if file.endswith((".jpg", ".png")):
                    st.image(os.path.join(result_folder, file), caption="Detected Output", use_column_width=True)
                elif file.endswith(".mp4"):
                    st.video(os.path.join(result_folder, file))
    else:
        st.error("Detection failed.")
        st.text(result.stderr)
import streamlit as st
import cv2
import torch
import tempfile
import numpy as np

st.title("📹 Live Drowsiness Detection with YOLOv5")

run_live = st.button("Start Live Detection")

if run_live:
    stframe = st.empty()

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to read from webcam.")
            break

        # Run detection
        results = model(frame)
        annotated_frame = np.squeeze(results.render())  # render() returns list of images

        # Show the frame in Streamlit
        stframe.image(annotated_frame, channels="BGR")

    cap.release()
