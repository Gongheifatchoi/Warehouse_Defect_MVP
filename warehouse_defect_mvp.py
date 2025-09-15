import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import requests
import json

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fOeD5p2bdG-VkgNq7-QNJmlXp5_DaPm1"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
        # Remove existing file to allow overwrite
        if os.path.exists(path):
            os.remove(path)
        gdown.download(url, path, quiet=False, fuzzy=True)
        st.success("Model downloaded!")
    return path

# Ensure model is downloaded
model_file = download_model()

# Load YOLO model
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    return YOLO(model_path)

model = load_yolo_model(model_file)

# ----------------------------
# 2. Hugging Face LLM Integration
# ----------------------------
def get_llm_commentary(defects_info, api_key):
    """
    Get AI commentary on detected defects using Hugging Face's free inference API
    """
    if not api_key:
        return "API key is missing. Please enter your Hugging Face API key."
    
    # Prepare the prompt
    prompt = f"""
    As a structural engineering expert, analyze these concrete defects detected in a warehouse:
    {defects_info}
    
    Please provide:
    1. A brief assessment of the severity
    2. Potential causes
    3. Recommended actions
    4. Safety implications
    
    Keep the response concise and professional (under 200 words).
    """
    
    # Hugging Face API endpoint (using a free model)
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    try:
        with st.spinner("Getting expert analysis from AI..."):
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result[0]['generated_text']
    except requests.exceptions.Timeout:
        return "The AI analysis is taking too long. Please try again later."
    except requests.exceptions.HTTPError as err:
        if response.status_code == 401:
            return "Authentication error: Please check your Hugging Face API key."
        elif response.status_code == 503:
            return "The AI model is currently loading. Please try again in a few moments."
        else:
            return f"API Error: {err}"
    except Exception as e:
        return f"Unable to generate AI commentary: {str(e)}"

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis.")

# Get API key from environment variable or user input
api_key = os.environ.get("HUGGINGFACE_API_KEY", "")

# If no API key in environment, show input box (for testing only)
if not api_key:
    api_key = st.text_input("Enter your Hugging Face API key:", type="password")
    if api_key:
        st.success("API key received! You can now use the AI commentary feature.")
    else:
        st.info("To enable AI commentary, please enter your Hugging Face API key above.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run detection
    with st.spinner("Analyzing image for defects..."):
        results = model(image)
    
    # Annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Defects", use_container_width=True)
    
    # Extract defect information
    defects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            defects.append({
                "type": class_name,
                "confidence": confidence,
                "location": [float(coord) for coord in box.xywh[0]]  # x_center, y_center, width, height
            })
    
    # Display defect information
    if defects:
        st.subheader("📊 Detection Results")
        
        # Show defects in a table
        for i, defect in enumerate(defects, 1):
            st.write(f"{i}. **{defect['type']}** ({(defect['confidence']*100):.1f}% confidence)")
        
        # Prepare defect information for LLM
        defects_info = "\n".join([
            f"- {d['type']} (confidence: {d['confidence']:.2f})"
            for d in defects
        ])
        
        # Get and display LLM commentary if API key is available
        if api_key:
            commentary = get_llm_commentary(defects_info, api_key)
            st.subheader("🧠 Expert Analysis")
            st.write(commentary)
        else:
            st.info("AI commentary is disabled. Please provide an API key to enable this feature.")
        
    else:
        st.success("✅ No defects detected! The concrete surface appears to be in good condition.")
