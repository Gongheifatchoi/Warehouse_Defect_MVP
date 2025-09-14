import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ===============================
# Load your trained YOLO model
# ===============================
MODEL_PATH = r"C:\Users\WinsonLu\runs\classify\train23\weights\best.pt"
model = YOLO(MODEL_PATH)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Warehouse Defect Classifier", layout="centered")

st.title("🏗️ Warehouse Defect Classification")
st.write("Upload one or more images and the model will classify them.")

# Upload multiple images
uploaded_files = st.file_uploader(
    "📂 Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write("---")  # separator
        st.write(f"### File: {uploaded_file.name}")

        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Run classification
        with st.spinner("🔍 Classifying..."):
            results = model(image)

        # Get prediction
        probs = results[0].probs
        class_names = results[0].names
        top_idx = probs.top1
        confidence = probs.top1conf.item()

        st.subheader("✅ Prediction Result")
        st.write(f"**{class_names[top_idx]}** ({confidence:.2%} confidence)")

        # Show all probabilities
        st.subheader("📊 Class Probabilities")
        prob_dict = {class_names[i]: p for i, p in enumerate(probs.data.tolist())}
        st.bar_chart(prob_dict)
