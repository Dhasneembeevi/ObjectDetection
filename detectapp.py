import sys
sys.path.append('D:/MeNeM/Object_Detection/yolov5')

from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.plots import scale_coords
import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile
import numpy as np
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

device = torch.device('cpu')
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/MeNeM/Object_Detection/best.pt', force_reload=False)
    model.to('cpu').eval()
    return model

model = load_model()

st.title("🚗 Road Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    img0 = Image.open(img_path).convert("RGB")
    img_np = np.array(img0)
    img = letterbox(img_np, new_shape=640)[0]
    img = img[:, :, ::-1].copy().transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.size).round()

        draw = ImageDraw.Draw(img0)
        try:
            font = ImageFont.truetype("arial.ttf", 45)
        except:
            font = ImageFont.load_default()

        names = model.names if hasattr(model, 'names') else model.module.names

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f'{names[int(cls)]} {conf:.2f}'

            for i in range(4):
                draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline='red')

            padding_x = 10
            padding_y = 8
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            draw.rectangle([x1, y1 - text_height - padding_y * 2, x1 + text_width + padding_x * 2, y1], fill='red')

            draw.text((x1 + padding_x, y1 - text_height - padding_y + 1), label, fill='white', font=font)

        st.image(img0, caption="🧠 Detected Objects", use_container_width=True)
    else:
        st.warning("⚠️ No objects detected.")