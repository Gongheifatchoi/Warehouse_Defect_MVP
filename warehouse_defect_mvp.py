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
def get_llm_commentary(defects_info):
    """
    Get AI commentary on detected defects using Hugging Face's free inference API
    Using a freely accessible model that doesn't require special permissions
    """
    # Try to get API key from environment or secrets
    api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
    
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
    
    # Use a freely accessible model - Google's Flan-T5
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3,
            "do_sample": True
        }
    }
    
    try:
        with st.spinner("Getting expert analysis from AI..."):
            response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 503:
                # Model is loading, wait and try again
                st.info("AI model is loading, please wait a moment...")
                import time
                time.sleep(10)
                response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            
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
        if response.status_code == 401:
            return "Authentication error: Please check your Hugging Face API key or use without one for limited access."
        elif response.status_code == 404:
            # Try an alternative model if the first one fails
            try:
                alt_api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
                response = requests.post(alt_api_url, headers=headers, json=payload, timeout=45)
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

# Check if we have an API key
api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
if not api_key:
    st.info("ℹ️ Using limited AI access. For better performance, add a Hugging Face API key as an environment variable.")

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
- AI commentary is provided through Hugging Face's free inference API
- For more reliable performance, add a Hugging Face API key as an environment variable
- Detection accuracy depends on image quality and lighting conditions
""")
