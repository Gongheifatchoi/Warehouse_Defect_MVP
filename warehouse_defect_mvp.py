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
# 2. Public AI Model Integration
# ----------------------------
def get_llm_commentary(defects_info):
    """
    Get AI commentary using publicly accessible models
    """
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
    
    # Try multiple public API endpoints
    endpoints = [
        {
            "name": "Hugging Face Inference API (Free)",
            "url": "https://api-inference.huggingface.co/models/google/flan-t5-base",
            "headers": {},
            "payload": {"inputs": prompt}
        },
        {
            "name": "Hugging Face Inference API (Alternative)",
            "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-small",
            "headers": {},
            "payload": {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
        },
        {
            "name": "Hugging Face Spaces API",
            "url": "https://tomfangio-llama-2-7b-chat.hf.space/api/chat",
            "headers": {"Content-Type": "application/json"},
            "payload": {
                "data": [
                    prompt,
                    "You are a structural engineering expert analyzing concrete defects.",
                    0.7,
                    256,
                    0.9,
                    1.2
                ]
            }
        }
    ]
    
    for endpoint in endpoints:
        try:
            with st.spinner(f"Trying {endpoint['name']}..."):
                response = requests.post(
                    endpoint["url"],
                    headers=endpoint["headers"],
                    json=endpoint["payload"],
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse different response formats
                    if isinstance(result, list):
                        if len(result) > 0:
                            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                                return result[0]['generated_text']
                            elif isinstance(result[0], str):
                                return result[0]
                    elif isinstance(result, dict):
                        if 'generated_text' in result:
                            return result['generated_text']
                        elif 'data' in result and isinstance(result['data'], list) and len(result['data']) > 0:
                            return result['data'][0]
                    
                    # If we can't parse it, just return the raw JSON
                    return json.dumps(result, indent=2)
                    
                elif response.status_code == 503:
                    continue  # Model loading, try next endpoint
                    
        except Exception as e:
            continue  # Try next endpoint if this one fails
    
    # If all APIs fail, use local expert system
    return None

# ----------------------------
# 3. Local Expert System (Primary)
# ----------------------------
def get_local_expert_commentary(defects):
    """
    Generate expert commentary using a comprehensive local rule-based system
    """
    if not defects:
        return "No defects detected. The concrete surface appears to be in good condition."
    
    # Count defects by type
    defect_counts = {}
    confidence_scores = {}
    
    for defect in defects:
        defect_type = defect['type']
        defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        if defect_type not in confidence_scores:
            confidence_scores[defect_type] = []
        confidence_scores[defect_type].append(defect['confidence'])
    
    # Calculate average confidence per defect type
    avg_confidence = {dtype: sum(scores)/len(scores) for dtype, scores in confidence_scores.items()}
    
    # Generate comprehensive commentary
    commentary = "## 🧠 Expert Analysis\n\n"
    commentary += "### 📊 Defect Summary\n"
    
    for defect_type, count in defect_counts.items():
        commentary += f"- **{defect_type.capitalize()}**: {count} detected (avg. confidence: {avg_confidence[defect_type]:.1%})\n"
    
    commentary += "\n### ⚠️ Severity Assessment\n"
    
    # Severity analysis
    total_defects = len(defects)
    if total_defects == 0:
        commentary += "No defects found. Structure appears sound.\n"
    elif total_defects <= 2:
        commentary += "Minor defects detected. Monitor conditions but no immediate action required.\n"
    elif total_defects <= 5:
        commentary += "Moderate defects detected. Recommend inspection and planned maintenance.\n"
    else:
        commentary += "Multiple defects detected. Recommend comprehensive structural assessment.\n"
    
    commentary += "\n### 🔍 Detailed Analysis by Defect Type\n"
    
    # Detailed analysis for each defect type
    for defect_type in defect_counts.keys():
        commentary += f"\n#### {defect_type.capitalize()}\n"
        
        if 'crack' in defect_type.lower():
            commentary += """
            **Characteristics**: Linear fractures in concrete surface
            **Potential Causes**: 
            - Structural overloading or settling
            - Thermal expansion and contraction
            - Shrinkage during curing process
            - Inadequate reinforcement
            
            **Recommended Actions**:
            1. Measure crack width with a crack comparator card
            2. Monitor for progression over 4-6 weeks
            3. For cracks > 0.3mm, consult structural engineer
            4. Consider epoxy injection for active cracks
            
            **Safety Implications**: 
            - Cracks > 1mm may indicate structural issues
            - Can allow water infiltration leading to reinforcement corrosion
            """
            
        elif 'spall' in defect_type.lower():
            commentary += """
            **Characteristics**: Flaking or peeling of concrete surface
            **Potential Causes**: 
            - Corrosion of reinforcing steel (most common)
            - Freeze-thaw cycles in cold climates
            - Impact damage from equipment or vehicles
            - Poor quality concrete or workmanship
            
            **Recommended Actions**:
            1. Remove all loose and deteriorated concrete
            2. Clean and treat exposed reinforcement with anti-corrosion coating
            3. Apply bonding agent to prepared surface
            4. Repair with appropriate concrete patching material
            5. Consider protective coatings for prevention
            
            **Safety Implications**: 
            - Falling concrete fragments can cause injury
            - Exposed rebar may pose puncture hazards
            - Indicates potential loss of structural section capacity
            """
            
        elif 'hole' in defect_type.lower() or 'void' in defect_type.lower():
            commentary += """
            **Characteristics**: Missing sections of concrete
            **Potential Causes**: 
            - Poor consolidation during placement
            - Formwork leaks during pouring
            - Insect or animal activity
            - Deterioration over time
            
            **Recommended Actions**:
            1. Assess depth and extent of void using probe
            2. Clean and prepare the area, removing loose material
            3. Dampen surface before repair (but no standing water)
            4. Fill with non-shrink grout or patching compound
            5. For structural members, consult an engineer
            
            **Safety Implications**: 
            - Small voids typically don't affect structural integrity
            - Large voids may indicate more serious issues with load capacity
            - Can create tripping hazards in walking surfaces
            """
            
        elif 'stain' in defect_type.lower() or 'discolor' in defect_type.lower():
            commentary += """
            **Characteristics**: Discoloration or staining on surface
            **Potential Causes**: 
            - Water infiltration and moisture issues
            - Chemical spills or exposure
            - Biological growth (mold, algae, mildew)
            - Efflorescence (mineral deposits from water migration)
            
            **Recommended Actions**:
            1. Identify and address moisture source if present
            2. Clean with appropriate solutions (avoid acid washes unless necessary)
            3. Apply stain-blocking coatings if needed for aesthetics
            4. Improve drainage and ventilation if water-related
            
            **Safety Implications**: 
            - Usually no direct structural safety concerns
            - May indicate moisture issues that could lead to other problems
            - Slippery surfaces if biological growth is present
            """
            
        else:
            # Generic advice for other defect types
            commentary += f"""
            **Characteristics**: General surface defect
            **Potential Causes**: 
            - Material degradation over time
            - Environmental exposure factors
            - Construction or workmanship issues
            - Loading or impact damage
            
            **Recommended Actions**:
            1. Document location and extent of defect
            2. Monitor for changes over time
            3. Consult with structural engineer for assessment
            4. Develop repair plan based on severity
            
            **Safety Implications**: 
            - Requires professional assessment to determine risk
            - May indicate underlying issues needing attention
            """
    
    commentary += "\n### 🛠️ General Maintenance Recommendations\n"
    commentary += """
    1. **Regular Inspections**: Conduct visual inspections quarterly
    2. **Documentation**: Maintain records of defect progression over time
    3. **Preventive Maintenance**: Address minor issues before they become major problems
    4. **Professional Assessment**: Engage structural engineer for comprehensive evaluation every 2-3 years
    5. **Moisture Control**: Implement proper drainage and waterproofing measures
    """
    
    commentary += "\n### ⚠️ Immediate Action Required if:\n"
    commentary += """
    - Defects are rapidly progressing or changing
    - Structural members show significant deformation
    - Cracks wider than 3mm are observed
    - Spalling exposes more than 10% of reinforcement
    - Any signs of structural movement or instability
    """
    
    return commentary

# ----------------------------
# 4. Streamlit UI
# ----------------------------
st.title("🏗️ Warehouse Concrete Defect Detection")
st.write("Upload an image of concrete surfaces to detect defects and receive expert analysis.")

# Option to try AI commentary
use_ai = st.sidebar.checkbox("Try AI Analysis (Experimental)", value=False, 
                            help="Attempt to use public AI models for additional analysis")

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
                "location": [float(coord) for coord in box.xywh[0]]
            })
    
    # Display defect information
    if defects:
        st.subheader("📊 Detection Results")
        
        # Show defects in a table
        for i, defect in enumerate(defects, 1):
            st.write(f"{i}. **{defect['type']}** ({(defect['confidence']*100):.1f}% confidence)")
        
        # Prepare defect information
        defects_info = "\n".join([
            f"- {d['type']} (confidence: {d['confidence']:.2f})"
            for d in defects
        ])
        
        # Get and display commentary
        st.subheader("🧠 Expert Analysis")
        
        # Always use local expert system as primary
        commentary = get_local_expert_commentary(defects)
        st.write(commentary)
        
        # Optionally try AI analysis
        if use_ai:
            with st.expander("AI Analysis (Experimental)"):
                st.info("Attempting to use public AI models. This may not always work due to availability.")
                ai_commentary = get_llm_commentary(defects_info)
                if ai_commentary:
                    st.write(ai_commentary)
                    st.caption("Generated using public AI models")
                else:
                    st.warning("AI models are currently unavailable. Using local expert analysis only.")
        
    else:
        st.success("✅ No defects detected! The concrete surface appears to be in good condition.")

# Footer
st.markdown("---")
st.markdown("""
**About this app**:
- Defect detection powered by YOLO model
- Comprehensive local expert system for reliable analysis
- Optional AI analysis using public models (experimental)
- Always consult a qualified engineer for critical structural assessments
""")

# Add information about the analysis system
with st.expander("About the Analysis System"):
    st.write("""
    **Local Expert System**:
    - Provides consistent, reliable analysis without external dependencies
    - Based on structural engineering best practices
    - Includes detailed recommendations for each defect type
    
    **AI Analysis (Experimental)**:
    - Uses publicly available models that don't require API keys
    - May be unavailable during high traffic periods
    - Provides supplementary analysis when available
    """)
