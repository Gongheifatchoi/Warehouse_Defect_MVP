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
    """Download YOLO model if it doesn't exist."""
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        if os.path.exists(path):
            os.remove(path)
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

# Ensure model is downloaded
model_file = download_model()

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load YOLO model once and cache it."""
    return YOLO(model_path)

model = load_yolo_model(model_file)

# ----------------------------
# 2. Streamlit UI
# ----------------------------
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = model(image)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)

    # ----------------------------
    # Extract predictions safely
    # ----------------------------
    res = results[0]
    preds = []

    if hasattr(res, 'boxes') and len(res.boxes) > 0:
        for i in range(len(res.boxes)):
            cls_idx = int(res.boxes.cls[i])       # class index
            conf = float(res.boxes.conf[i])      # confidence
            name = res.names[cls_idx]            # class name
            preds.append((name, conf))

    # ----------------------------
    # Display predictions
    # ----------------------------
    if preds:
        st.write("✅ Prediction Result")
        for name, conf in preds:
            st.write(f"{name} ({conf*100:.2f}% confidence)")

        # Display as a table
        df = pd.DataFrame(preds, columns=["Class", "Confidence"])
        st.write("📊 Class Probabilities")
        st.dataframe(df)
    else:
        st.write("No defects detected.")
