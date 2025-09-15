import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import requests
import json
import time

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
# 2. Hugging Face LLM Integration with Llama 3
# ----------------------------
def get_llm_commentary(defects_info):
    """
    Get AI commentary on detected defects using Llama 3 via Hugging Face API
    """
    # Get API key from Streamlit secrets
    try:
        # Try different possible secret names
        if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
            api_key = st.secrets['HUGGINGFACEHUB_API_TOKEN']
        elif 'HUGGINGFACE_API_KEY' in st.secrets:
            api_key = st.secrets['HUGGINGFACE_API_KEY']
        else:
            st.error("Hugging Face API token not found in secrets. Please check your secrets configuration.")
            return "API token configuration error."
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        return "Secrets access error."
    
    # Using Meta's Llama 3 7B model - this is available on Hugging Face Inference API
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Prepare the prompt in chat format for Llama 3
    prompt = f"""
    You are a structural engineering expert analyzing concrete defects in warehouses.
    
    Analyze these detected defects:
    {defects_info}
    
    Please provide:
    1. A brief assessment of the severity
    2. Potential causes
    3. Recommended actions
    4. Safety implications
    
    Keep the response concise and professional (under 200 words).
    """
    
    # Format for chat models
    messages = [
        {"role": "system", "content": "You are a structural engineering expert with 20+ years of experience in analyzing concrete structures and defects."},
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "inputs": messages,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        with st.spinner("Getting expert analysis from Llama 3 AI..."):
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            # Check if model is loading
            if response.status_code == 503:
                try:
                    error_info = response.json()
                    if 'estimated_time' in error_info:
                        wait_time = error_info['estimated_time']
                        st.info(f"Llama 3 model is loading. Estimated wait time: {wait_time:.1f} seconds")
                        time.sleep(wait_time + 5)
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                    else:
                        st.info("Llama 3 model is loading, please wait...")
                        time.sleep(25)
                        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                except:
                    st.info("Model is loading, please wait...")
                    time.sleep(25)
                    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            response.raise_for_status()
            result = response.json()
            
            # Parse the response from Llama 3
            if isinstance(result, list) and len(result) > 0:
                if 'generated_text' in result[0]:
                    return result[0]['generated_text']
                elif isinstance(result[0], dict) and 'generated_text' in result[0]:
                    return result[0]['generated_text']
            
            # Try alternative response format
            if isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
                
            return "Received unexpected response format from AI service."
                
    except requests.exceptions.Timeout:
        return "The AI analysis is taking too long. Please try again later."
    except requests.exceptions.HTTPError as err:
        if response.status_code == 401:
            return "Authentication error: Please check your Hugging Face API key."
        elif response.status_code == 404:
            return "Llama 3 model not found. It may not be available on the inference API."
        elif response.status_code == 503:
            return "Llama 3 model is currently loading. Please try again in a few minutes."
        else:
            return f"API Error: {str(err)}"
    except Exception as e:
        return f"Unable to generate AI commentary: {str(e)}"

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis using Llama 3 AI.")

# Check if we have the API key set up
try:
    has_api_key = 'HUGGINGFACEHUB_API_TOKEN' in st.secrets or 'HUGGINGFACE_API_KEY' in st.secrets
    if not has_api_key:
        st.warning("Hugging Face API key not found in secrets. AI analysis may not work.")
    else:
        st.success("Hugging Face API key found! Using Llama 3 for expert analysis.")
except:
    st.warning("Unable to check secrets configuration. AI analysis may not work.")

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
        
        st.subheader("🧠 Expert Analysis (Powered by Llama 3)")
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
- AI commentary is provided by Meta's Llama 3 model through Hugging Face's inference API
- Detection accuracy depends on image quality and lighting conditions
- The Llama 3 model may take a few moments to load on first use
""")
