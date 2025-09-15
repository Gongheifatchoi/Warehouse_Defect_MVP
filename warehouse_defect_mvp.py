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
def get_llm_commentary(defects):
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
        
        # Prepare detailed defect information with explicit types
        defects_info = "DETECTED DEFECT TYPES AND QUANTITIES:\n"
        defect_counts = {}
        for defect in defects:
            defect_type = defect['type']
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        for defect_type, count in defect_counts.items():
            defects_info += f"- {defect_type}: {count} instance(s)\n"
        
        defects_info += "\nINDIVIDUAL DEFECT DETAILS:\n"
        for i, defect in enumerate(defects, 1):
            defects_info += f"{i}. {defect['type']} (confidence: {defect['confidence']:.2f})\n"
        
        # Professional engineering prompt with specific technical requirements
        prompt = f"""
        As a licensed structural engineer with expertise in concrete pathology and warehouse structural assessment, 
        provide a detailed technical analysis of these detected concrete defects:
        
        {defects_info}
        
        For EACH specific defect type identified above, provide a comprehensive engineering assessment including:
        
        1. DEFECT-SPECIFIC ANALYSIS:
           - For each defect type: structural significance, severity rating, and specific implications
           - Detailed description of what each defect type represents in engineering terms
        
        2. ROOT CAUSE ANALYSIS BY DEFECT TYPE:
           - Specific causes for each type of defect (cracks, spalling, corrosion, etc.)
           - Material, construction, environmental, and loading factors for each defect type
        
        3. QUANTITATIVE ASSESSMENT:
           - Severity classification for each defect type (Minor, Moderate, Severe, Critical)
           - Risk assessment for each defect type
           - Potential impact on structural capacity for each defect category
        
        4. DEFECT-SPECIFIC MITIGATION:
           - Recommended repair methods for each specific defect type
           - Urgency of intervention for each defect category
           - Specific repair techniques appropriate for each defect type
        
        Please structure your response by DEFECT TYPE, providing specific analysis for each category of defect detected.
        Use technical engineering terminology and reference appropriate standards.
        """
        
        with st.spinner("Conducting defect-specific structural analysis..."):
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior structural engineering consultant specializing in concrete defect analysis. You provide detailed, defect-specific assessments that explicitly address each type of defect found. Your analysis is organized by defect type and includes specific engineering recommendations for each category."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.2,
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
        
        # Group defects by type for better presentation
        defect_counts = {}
        for defect in defects:
            defect_type = defect['type']
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        st.write("**Defect Type Summary:**")
        for defect_type, count in defect_counts.items():
            st.write(f"• **{defect_type}**: {count} instance(s) detected")
        
        # Show individual defects in a detailed table
        st.write("**Individual Defect Details:**")
        for i, defect in enumerate(defects, 1):
            st.write(f"{i}. **{defect['type']}** - Detection confidence: {defect['confidence']*100:.1f}%")
        
        # Get and display professional engineering analysis
        st.subheader("🧠 Defect-Specific Engineering Assessment")
        with st.spinner("Generating comprehensive defect-type analysis..."):
            analysis = get_llm_commentary(defects)
        
        st.write(analysis)
        
        # Add technical references with defect-specific information
        with st.expander("📚 Technical References & Defect Classification"):
            st.write("""
            **Common Concrete Defect Types:**
            
            **Cracks:**
            - **Hairline cracks**: < 0.1mm width, typically cosmetic
            - **Fine cracks**: 0.1-0.3mm, may indicate early stage issues
            - **Medium cracks**: 0.3-1.0mm, require monitoring and possible repair
            - **Wide cracks**: > 1.0mm, structural concern requiring intervention
            
            **Spalling:**
            - **Surface spalling**: Cosmetic, limited depth
            - **Moderate spalling**: Exposed aggregate, some reinforcement visible
            - **Severe spalling**: Significant concrete loss, reinforcement exposed
            - **Delamination**: Subsurface separation, detectable by sounding
            
            **Corrosion:**
            - **Surface staining**: Early warning signs
            - **Active corrosion**: Rust formation and expansion
            - **Section loss**: Reduction in reinforcement cross-section
            - **Advanced deterioration**: Significant structural compromise
            
            **Relevant Standards:**
            - ACI 201.1R: Guide for Concrete Inspection
            - ACI 224R: Crack Control in Concrete Structures
            - ACI 364.1R: Evaluation Prior to Rehabilitation
            - EN 1504: Concrete Protection and Repair
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
