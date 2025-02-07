import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model YOLOv8 yang sudah dilatih
MODEL_PATH = "yolov8_best_model.pt"  # Sesuaikan dengan path model Anda
model = YOLO(MODEL_PATH)

# Fungsi untuk mendeteksi kebakaran hutan pada gambar
def detect_fire(image):
    results = model(image)  # Lakukan deteksi
    detections = results[0].boxes.data.cpu().numpy()  # Ambil bounding box

    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.5:  # Confidence threshold 50%
            label = f"WILDFIRE {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# Konfigurasi UI Streamlit
st.title("🔥 Deteksi Kebakaran Hutan Real-Time dengan YOLOv8")
st.write("Gunakan kamera atau unggah gambar untuk mendeteksi kebakaran hutan.")

# Pilih sumber input: Kamera atau Upload Gambar
option = st.radio("Pilih Sumber Input:", ("Kamera", "Upload Gambar"))

if option == "Kamera":
    st.write("📷 Ambil gambar menggunakan kamera")
    img_file = st.camera_input("Ambil gambar")

    if img_file is not None:
        image = Image.open(img_file)
        image = np.array(image)
        detected_image = detect_fire(image)
        st.image(detected_image, channels="RGB", use_container_width=True)

elif option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        detected_image = detect_fire(image)
        st.image(detected_image, channels="RGB", use_container_width=True)
