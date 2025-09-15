import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown
import requests
import json
import time
from openai import OpenAI
from datetime import datetime
import numpy as np

# ----------------------------
# 1. Model setup
# ----------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1FHjz3wjLBWk5c04j7kGBynQcPTyVA19R"

@st.cache_data(show_spinner=False)
def download_model(url=MODEL_URL, path=MODEL_PATH):
    if not os.path.exists(path):
        st.info("Downloading YOLO model, please wait...")
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
def get_comparative_analysis(pre_tenancy_defects, during_tenancy_defects, inspection_context):
    """
    Get professional comparative analysis between pre-tenancy and during-tenancy conditions
    """
    # Get API key from Streamlit secrets
    try:
        if 'HUGGINGFACEHUB_API_TOKEN' in st.secrets:
            api_key = st.secrets['HUGGINGFACEHUB_API_TOKEN']
        elif 'HUGGINGFACE_API_KEY' in st.secrets:
            api_key = st.secrets['HUGGINGFACE_API_KEY']
        elif 'HF_TOKEN' in st.secrets:
            api_key = st.secrets['HF_TOKEN']
        else:
            return "API token configuration error."
    except Exception as e:
        return f"Secrets access error: {e}"
    
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        
        # Prepare defect information for both periods
        pre_info = "PRE-TENANCY DEFECTS:\n"
        pre_counts = {}
        for defect in pre_tenancy_defects:
            defect_type = defect['type']
            pre_counts[defect_type] = pre_counts.get(defect_type, 0) + 1
        
        for defect_type, count in pre_counts.items():
            pre_info += f"- {defect_type}: {count} instance(s)\n"
        
        during_info = "DURING-TENANCY DEFECTS:\n"
        during_counts = {}
        for defect in during_tenancy_defects:
            defect_type = defect['type']
            during_counts[defect_type] = during_counts.get(defect_type, 0) + 1
        
        for defect_type, count in during_counts.items():
            during_info += f"- {defect_type}: {count} instance(s)\n"
        
        # Comparative analysis prompt
        prompt = f"""
        As a licensed structural engineer and building surveyor, provide a comprehensive comparative analysis between pre-tenancy and during-tenancy conditions:
        
        {pre_info}
        
        {during_info}
        
        INSPECTION CONTEXT: {inspection_context}
        
        Please provide a detailed comparative analysis including:
        
        1. DEFECT PROGRESSION ANALYSIS:
           - New defects that appeared during tenancy
           - Existing defects that worsened during tenancy
           - Defects that remained unchanged
           - Defects that improved or were repaired
        
        2. LIABILITY ASSESSMENT:
           - Clear determination of landlord vs tenant responsibility for each defect change
           - Typical timelines for defect development vs tenancy duration
           - Wear and tear vs actual damage assessment
        
        3. QUANTITATIVE COMPARISON:
           - Defect count changes by type
           - Severity progression analysis
           - Rate of deterioration assessment
        
        4. SPECIFIC RECOMMENDATIONS:
           - Immediate safety concerns
           - Repair prioritization based on liability
           - Documentation requirements for dispute resolution
           - Preventive measures for future
        
        5. COST IMPLICATIONS:
           - Estimated repair costs by responsibility party
           - Urgency-based budgeting
           - Insurance claim considerations
        
        Provide a structured comparison with clear before/after analysis for each defect type.
        """
        
        with st.spinner("Generating comprehensive comparative analysis..."):
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert building surveyor specializing in tenancy defect comparisons. You provide clear, structured comparative analysis between pre-tenancy and during-tenancy conditions, with specific liability assessments and practical recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.2,
                top_p=0.8,
                stream=False
            )
            
            return response.choices[0].message.content
                
    except Exception as e:
        return f"Unable to generate comparative analysis: {str(e)}"

# ----------------------------
# 3. Helper Functions
# ----------------------------
def analyze_image(image, label):
    """Analyze a single image and return defects"""
    with st.spinner(f"Analyzing {label} image..."):
        results = model(image)
    
    defects = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            defects.append({
                "type": class_name,
                "confidence": confidence,
                "location": [float(coord) for coord in box.xywh[0]]
            })
    
    annotated_image = results[0].plot()
    return defects, annotated_image

def create_comparison_table(pre_defects, during_defects):
    """Create a comparison table of defects"""
    pre_counts = {}
    for defect in pre_defects:
        pre_counts[defect['type']] = pre_counts.get(defect['type'], 0) + 1
    
    during_counts = {}
    for defect in during_defects:
        during_counts[defect['type']] = during_counts.get(defect['type'], 0) + 1
    
    all_defect_types = set(list(pre_counts.keys()) + list(during_counts.keys()))
    
    comparison_data = []
    for defect_type in sorted(all_defect_types):
        comparison_data.append({
            "Defect Type": defect_type,
            "Pre-Tenancy": pre_counts.get(defect_type, 0),
            "During Tenancy": during_counts.get(defect_type, 0),
            "Change": during_counts.get(defect_type, 0) - pre_counts.get(defect_type, 0)
        })
    
    return comparison_data

# ----------------------------
# 4. Streamlit UI with Side-by-Side Comparison
# ----------------------------
st.title("🏗️ Warehouse Concrete Defect Comparison Analysis")
st.write("Upload both pre-tenancy and during-tenancy images for comprehensive side-by-side analysis.")

# Tenancy context information
st.subheader("📋 Tenancy Information")
col1, col2 = st.columns(2)
with col1:
    tenancy_start = st.date_input("Tenancy Start Date", value=datetime.now().replace(year=datetime.now().year-1))
with col2:
    building_usage = st.selectbox(
        "Building Usage:",
        ["General Storage", "Light Manufacturing", "Heavy Machinery", "Cold Storage", "Distribution Center", "Other"]
    )

# Image upload sections side by side
st.subheader("📸 Upload Comparison Images")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Pre-Tenancy (Before Move-In)")
    pre_tenancy_file = st.file_uploader("Pre-tenancy image...", type=["jpg", "jpeg", "png"], key="pre_tenancy")

with col2:
    st.markdown("### During Tenancy (Current Condition)")
    during_tenancy_file = st.file_uploader("During-tenancy image...", type=["jpg", "jpeg", "png"], key="during_tenancy")

# Check if we have the API key set up
try:
    has_api_key = any(key in st.secrets for key in ['HUGGINGFACEHUB_API_TOKEN', 'HUGGINGFACE_API_KEY', 'HF_TOKEN'])
    if not has_api_key:
        st.warning("Hugging Face API token not found. Comparative analysis may not be available.")
    else:
        st.success("Hugging Face API key authenticated. Ready for comparative analysis.")
except:
    st.warning("Unable to verify API configuration. Some features may be limited.")

if pre_tenancy_file and during_tenancy_file:
    # Display images side by side
    st.subheader("🖼️ Image Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        pre_image = Image.open(pre_tenancy_file)
        st.image(pre_image, caption="Pre-Tenancy Condition", use_container_width=True)
    
    with col2:
        during_image = Image.open(during_tenancy_file)
        st.image(during_image, caption="During-Tenancy Condition", use_container_width=True)
    
    # Analyze both images
    pre_defects, pre_annotated = analyze_image(pre_image, "pre-tenancy")
    during_defects, during_annotated = analyze_image(during_image, "during-tenancy")
    
    # Display annotated images side by side
    st.subheader("🔍 Defect Detection Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(pre_annotated, caption="Pre-Tenancy Defects Detected", use_container_width=True)
    
    with col2:
        st.image(during_annotated, caption="During-Tenancy Defects Detected", use_container_width=True)
    
    # Display defect counts side by side
    st.subheader("📊 Defect Comparison Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pre-Tenancy Defects:**")
        pre_counts = {}
        for defect in pre_defects:
            pre_counts[defect['type']] = pre_counts.get(defect['type'], 0) + 1
        
        for defect_type, count in pre_counts.items():
            st.write(f"• {defect_type}: {count} instance(s)")
        
        st.metric("Total Pre-Tenancy Defects", len(pre_defects))
    
    with col2:
        st.markdown("**During-Tenancy Defects:**")
        during_counts = {}
        for defect in during_defects:
            during_counts[defect['type']] = during_counts.get(defect['type'], 0) + 1
        
        for defect_type, count in during_counts.items():
            st.write(f"• {defect_type}: {count} instance(s)")
        
        st.metric("Total During-Tenancy Defects", len(during_defects))
        change = len(during_defects) - len(pre_defects)
        st.metric("Defect Count Change", change, delta=f"{change} defects")
    
    # Comparative analysis
    if pre_defects or during_defects:
        st.subheader("🧠 Comprehensive Comparative Analysis")
        
        inspection_context = f"""
        Tenancy Duration: {(datetime.now().date() - tenancy_start).days} days
        Building Usage: {building_usage}
        Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        analysis = get_comparative_analysis(pre_defects, during_defects, inspection_context)
        st.write(analysis)
        
        # Detailed comparison table
        st.subheader("📈 Detailed Defect Comparison")
        comparison_data = create_comparison_table(pre_defects, during_defects)
        
        # Display as metrics or table
        for data in comparison_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Pre-Tenancy {data['Defect Type']}", data['Pre-Tenancy'])
            with col2:
                st.metric(f"During-Tenancy {data['Defect Type']}", data['During Tenancy'])
            with col3:
                st.metric("Change", data['Change'], delta=f"{data['Change']}")
        
        # Visual comparison chart
        try:
            import matplotlib.pyplot as plt
            
            st.subheader("📊 Defect Progression Chart")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            defect_types = [data['Defect Type'] for data in comparison_data]
            pre_values = [data['Pre-Tenancy'] for data in comparison_data]
            during_values = [data['During Tenancy'] for data in comparison_data]
            
            x = np.arange(len(defect_types))
            width = 0.35
            
            ax.bar(x - width/2, pre_values, width, label='Pre-Tenancy', alpha=0.8)
            ax.bar(x + width/2, during_values, width, label='During-Tenancy', alpha=0.8)
            
            ax.set_xlabel('Defect Types')
            ax.set_ylabel('Number of Defects')
            ax.set_title('Defect Comparison: Pre-Tenancy vs During-Tenancy')
            ax.set_xticks(x)
            ax.set_xticklabels(defect_types, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except ImportError:
            st.info("Install matplotlib for visual charts: `pip install matplotlib`")
        
        # Liability assessment guide
        with st.expander("📋 Liability Assessment Guidelines"):
            st.write("""
            **Typical Responsibility Classification:**
            
            **Landlord Responsibility:**
            - Pre-existing structural defects
            - Foundation and structural settlement
            - Pre-tenancy corrosion or decay
            - Building code compliance issues
            
            **Tenant Responsibility:**
            - New damage from improper use
            - Accident-related damage
            - Lack of reported maintenance issues
            - Unapproved modifications
            
            **Shared Responsibility:**
            - Normal wear and tear progression
            - Pre-existing conditions exacerbated by use
            - Environmental factors affecting both
            """)
        
    else:
        st.success("✅ No defects detected in either period! Structure appears well-maintained.")
        
        # Even with no defects, provide comparative analysis
        st.info("""
        **Comparative Analysis:** No structural defects detected in either pre-tenancy or during-tenancy inspections. 
        This indicates excellent maintenance and proper usage during the tenancy period.
        """)

# Add professional footer
st.markdown("---")
st.markdown("""
**Disclaimer**: 
- Comparative analysis provided for informational purposes only
- Final liability determinations require professional building survey
- Local tenancy laws and lease agreements take precedence
- Always consult qualified professionals for legal disputes
""")

st.caption("_Comparative analysis generated using AI-assisted assessment. Final determinations require professional inspection._")
