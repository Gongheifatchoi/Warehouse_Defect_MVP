import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown

# --- Model setup ---
MODEL_PATH = "best.pt"

# Automatically download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    url = "https://drive.google.com/file/d/1fOeD5p2bdG-VkgNq7-QNJmlXp5_DaPm1/view?usp=drive_link"  # <-- replace with your Google Drive file ID
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# --- Streamlit UI ---
st.title("Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces and detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Run detection
    results = model(image)
    
    # Render results
    annotated_image = results[0].plot()  # annotated image as numpy array
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)
