import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import pandas as pd

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fOeD5p2bdG-VkgNq7-QNJmlXp5_DaPm1"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

model_file = download_model()

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(model_file)
st.write(f"Loaded model task: `{model.task}`")

# ----------------------------
# 2. Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection/Classification")
st.write("Upload an image of concrete surfaces.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)
    res = results[0]

    preds = []

    if model.task == "detect":
        st.write(f"Number of boxes detected: {len(res.boxes)}")
        if len(res.boxes) > 0:
            for box in res.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                name = res.names[cls_idx]
                preds.append((name, conf))
            # Annotated image
            annotated_image = res.plot()
            st.image(annotated_image, caption="Detected Defects", use_column_width=True)
        else:
            st.write("No boxes detected. Try adjusting the image or check training data.")
    
    elif model.task == "classify":
        if hasattr(res, 'pred') and res.pred is not None:
            for idx, conf in enumerate(res.pred[0]):
                name = res.names[idx]
                preds.append((name, float(conf)))
        else:
            st.write("No predictions available.")

    if preds:
        preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
        top_class, top_conf = preds_sorted[0]
        st.write("✅ Top Prediction:")
        st.write(f"{top_class} ({top_conf*100:.2f}% confidence)")

        st.write("📊 All Class Probabilities / Detections")
        st.dataframe(pd.DataFrame(preds_sorted, columns=["Class", "Confidence"]))
