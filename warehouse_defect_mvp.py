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
MODEL_URL = "https://drive.google.com/uc?export=download&id=1FHjz3wjLBWk5c04j7kGBynQcPTyVA19R"


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
# 2. Hugging Face LLM Integration (Updated)
# ----------------------------
def get_llm_commentary(defects_info):
    """
    Get AI commentary on detected defects using a public Hugging Face model
    """
    # Using a public model that doesn't require API key
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
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
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.3,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        with st.spinner("Getting expert analysis from AI..."):
            response = requests.post(API_URL, json=payload, timeout=60)
            
            if response.status_code == 503:
                # Model is loading, wait and try again
                st.info("AI model is loading, please wait a moment...")
                import time
                time.sleep(15)
                response = requests.post(API_URL, json=payload, timeout=60)
            
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No analysis generated')
            elif isinstance(result, dict):
                return result.get('generated_text', 'No analysis generated')
            else:
                return "Received unexpected response format from AI service."
                
    except requests.exceptions.Timeout:
        return "The AI analysis is taking too long. Please try again later."
    except requests.exceptions.HTTPError as err:
        if response.status_code == 503:
            # Try an alternative model if the first one fails
            try:
                alt_api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
                response = requests.post(alt_api_url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'No analysis generated')
            except:
                return "AI service is temporarily unavailable. Please try again later."
        else:
            return f"API Error: {err}"
    except Exception as e:
        return f"Unable to generate AI commentary: {str(e)}"

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis.")

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
        
        # Get and display LLM commentary
        commentary = get_llm_commentary(defects_info)
        
        st.subheader("🧠 Expert Analysis")
        st.write(commentary)
        
        # Add some general advice based on common defects
        if any('crack' in d['type'].lower() for d in defects):
            st.info("💡 **General advice for cracks**: Monitor crack width over time. Cracks wider than 0.3mm may require professional assessment.")
        if any('spall' in d['type'].lower() for d in defects):
            st.info("💡 **General advice for spalling**: Exposed rebar can lead to corrosion. Consider protective coatings or repairs.")
        
    else:
        st.success("✅ No defects detected! The concrete surface appears to be in good condition.")

# Add footer with information
st.markdown("---")
st.markdown("""
**Note**: 
- AI commentary is provided through Hugging Face's public inference API
- Detection accuracy depends on image quality and lighting conditions
""")
