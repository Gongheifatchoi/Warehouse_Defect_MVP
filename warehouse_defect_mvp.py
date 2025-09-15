import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import requests
import json
import time
from openai import OpenAI

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
# 2. Hugging Face LLM Integration with OpenAI-compatible API
# ----------------------------
def get_llm_commentary(defects_info):
    """
    Get professional engineering analysis using Hugging Face's OpenAI-compatible API
    """
    # Get API key from Streamlit secrets
    try:
        # Try different possible secret names
        if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
            api_key = st.secrets['HUGGINGFACEHUB_API_TOKEN']
        elif 'HUGGINGFACE_API_KEY' in st.secrets:
            api_key = st.secrets['HUGGINGFACE_API_KEY']
        elif 'HF_TOKEN' in st.secrets:
            api_key = st.secrets['HF_TOKEN']
        else:
            st.error("Hugging Face API token not found in secrets. Please check your secrets configuration.")
            return "API token configuration error."
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        return "Secrets access error."
    
    try:
        # Initialize OpenAI client with Hugging Face endpoint
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        
        # Professional engineering prompt with specific technical requirements
        prompt = f"""
        As a licensed structural engineer with expertise in concrete pathology and warehouse structural assessment, 
        provide a detailed technical analysis of these detected concrete defects:
        
        DEFECTS IDENTIFIED:
        {defects_info}
        
        Please provide a comprehensive engineering assessment including:
        
        1. STRUCTURAL SIGNIFICANCE:
           - Rate severity for each defect type (Minor, Moderate, Severe, Critical)
           - Potential impact on structural integrity and load-bearing capacity
           - Risk of progressive deterioration
        
        2. ROOT CAUSE ANALYSIS:
           - Material deficiencies (concrete mix design, aggregate issues)
           - Construction practices (improper curing, compaction issues)
           - Environmental factors (freeze-thaw cycles, chemical exposure)
           - Loading conditions (overloading, dynamic impacts)
           - Corrosion mechanisms (chloride ingress, carbonation)
        
        3. QUANTITATIVE ASSESSMENT:
           - Estimated remaining service life reduction
           - Crack width classification per ACI 224R or relevant standards
           - Spalling depth and area significance
           - Reinforcement corrosion activity level
        
        4. MITIGATION STRATEGIES:
           - Immediate safety precautions required
           - Short-term stabilization measures
           - Long-term repair methodologies (epoxy injection, cathodic protection, etc.)
           - Monitoring and inspection frequency recommendations
        
        5. COST AND TIMELINE IMPLICATIONS:
           - Urgency of intervention
           - Estimated repair complexity
           - Potential business interruption impacts
        
        Provide specific, actionable recommendations based on engineering best practices and relevant codes (ACI, EN, AS).
        Use technical terminology appropriate for structural engineering professionals.
        """
        
        with st.spinner("Conducting professional structural analysis..."):
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior structural engineering consultant with 25+ years of experience in concrete pathology, structural assessment, and repair design. You provide precise, technical analysis following engineering standards and codes. Your responses are professional, data-driven, and focused on actionable engineering recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=600,
                temperature=0.2,  # Lower temperature for more deterministic, professional output
                top_p=0.8,
                stream=False
            )
            
            return response.choices[0].message.content
                
    except Exception as e:
        return f"Unable to generate engineering analysis: {str(e)}"

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Concrete Structural Assessment")
st.write("Upload an image of concrete surfaces for professional structural defect analysis and engineering recommendations.")

# Check if we have the API key set up
try:
    has_api_key = any(key in st.secrets for key in ['HUGGINGFACEHUB_API_TOKEN', 'HUGGINGFACE_API_KEY', 'HF_TOKEN'])
    if not has_api_key:
        st.warning("Hugging Face API token not found in secrets. Professional engineering analysis may not be available.")
    else:
        st.success("Hugging Face API key authenticated. Ready for professional structural analysis.")
except:
    st.warning("Unable to verify API configuration. Some features may be limited.")

uploaded_file = st.file_uploader("Choose structural inspection image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Structural Inspection Image", use_container_width=True)

    # Run detection
    with st.spinner("Conducting structural defect analysis..."):
        results = model(image)
    
    # Annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Identified Structural Defects", use_container_width=True)
    
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
        st.subheader("📊 Structural Defect Inventory")
        
        # Professional defect table
        st.write("**Defect Classification Summary:**")
        defect_data = []
        for i, defect in enumerate(defects, 1):
            defect_data.append({
                "Defect #": i,
                "Type": defect['type'],
                "Confidence": f"{defect['confidence']*100:.1f}%",
                "Severity": "To be assessed"  # Placeholder for engineering assessment
            })
        
        # Show defects in a professional table format
        for defect in defect_data:
            st.write(f"**{defect['Defect #']}. {defect['Type']}** - Confidence: {defect['Confidence']}")
        
        # Prepare defect information for engineering analysis
        defects_info = "\n".join([
            f"- {d['type']} (detection confidence: {d['confidence']:.2f})"
            for d in defects
        ])
        
        # Get and display professional engineering analysis
        st.subheader("🧠 Professional Engineering Assessment")
        with st.spinner("Generating comprehensive structural analysis..."):
            analysis = get_llm_commentary(defects_info)
        
        st.write(analysis)
        
        # Add technical references
        with st.expander("📚 Technical References & Standards"):
            st.write("""
            **Relevant Engineering Standards:**
            - ACI 201.1R: Guide for Conducting a Visual Inspection of Concrete in Service
            - ACI 224R: Control of Cracking in Concrete Structures
            - ACI 364.1R: Guide for Evaluation of Concrete Structures Prior to Rehabilitation
            - EN 1504: Products and systems for the protection and repair of concrete structures
            - ASTM C856: Standard Practice for Petrographic Examination of Hardened Concrete
            
            **Severity Classification:**
            - **Minor**: Cosmetic issues, no structural impact
            - **Moderate**: Requires monitoring, may need non-structural repairs
            - **Severe**: Structural capacity affected, requires engineering intervention
            - **Critical**: Immediate safety risk, requires urgent structural repairs
            """)
        
    else:
        st.success("✅ No structural defects detected! The concrete elements appear to be in sound condition.")

# Add professional footer
st.markdown("---")
st.markdown("""
**Disclaimer**: 
- This analysis provides preliminary engineering assessment based on visual inspection data
- Field verification and detailed structural analysis by licensed engineers is recommended for final decisions
- All recommendations should be verified against local building codes and specific site conditions
- Detection accuracy is dependent on image quality, lighting, and surface conditions
""")

# Add engineering certification note
st.caption("_Analysis generated using AI-assisted engineering assessment tools. Final engineering decisions should be made by qualified professionals._")
