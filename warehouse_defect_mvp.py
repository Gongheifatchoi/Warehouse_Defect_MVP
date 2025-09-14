import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import requests

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fOeD5p2bdG-VkgNq7-QNJmlXp5_DaPm1"

@st.cache_data(show_spinner=False)
def download_model_with_progress(url=MODEL_URL, path=MODEL_PATH):
    if os.path.exists(path):
        return path

    st.info("Downloading YOLO model, please wait...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = st.progress(0.0)

    downloaded = 0
    with open(path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                progress_bar.progress(min(downloaded / total_size, 1.0))
    
    st.success("Model downloaded!")
    return path

# Download model
model_file = download_model_with_progress()

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(model_file)
st.write(f"Loaded model task: `{model.task}`")  # detect or classify

# ----------------------------
# 2. Streamlit UI
# ----------------------------
st.title("🏭 Warehouse Concrete Defect Detection/Classification")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO
    results = model(image)
    res = results[0]

    preds = []

    # ----------------------------
    # 3a. Classification
    # ----------------------------
    if model.task == "classify":
        if hasattr(res, 'pred') and res.pred is not None:
            probs = res.pred[0].tolist()
            for idx, conf in enumerate(probs):
                name = res.names[idx]
                preds.append((name, conf))

    # ----------------------------
    # 3b. Detection
    # ----------------------------
    elif model.task == "detect":
        if hasattr(res, 'boxes') and len(res.boxes) > 0:
            for box in res.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                name = res.names[cls_idx]
                preds.append((name, conf))

    # ----------------------------
    # 4. Display predictions
    # ----------------------------
    if preds:
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        top_class, top_conf = preds_sorted[0]

        st.write("✅ Prediction Result")
        st.write(f"{top_class} ({top_conf*100:.2f}% confidence)")

        df = pd.DataFrame(preds_sorted, columns=["Class", "Confidence"])
        st.write("📊 All Class Probabilities / Detections")
        st.dataframe(df)
    else:
        st.write("No predictions available.")
