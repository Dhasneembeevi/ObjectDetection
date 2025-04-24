# 🚦 Road Object Detection using YOLOv5 and Streamlit

This project is a **web-based object detection app** built using **YOLOv5** (You Only Look Once) and **Streamlit**. It allows users to upload an image and detects objects (like vehicles, pedestrians, etc.) using a custom-trained YOLOv5 model. The app displays a final annotated image with bounding boxes and confidence labels.

---

## 📸 What It Does

- Loads a **custom-trained YOLOv5 model** (`best.pt`)
- Allows image uploads via a **Streamlit UI**
- Preprocesses images (resizing, normalization, etc.)
- Runs inference and applies **Non-Max Suppression**
- Draws **bounding boxes** and **labels with confidence scores**
- Outputs a final image showing all detected objects

---

## 🧱 Tech Stack

- **Python 3.7+**
- **YOLOv5** (from Ultralytics)
- **Streamlit** (for web UI)
- **PIL (Pillow)** (for image handling)
- **PyTorch** (for model inference)
- **NumPy** (for array manipulations)
