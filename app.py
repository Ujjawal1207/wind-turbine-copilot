import streamlit as st
import spacy
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Set up page config
st.set_page_config(layout="wide", page_title="Wind Turbine AI Copilot Demo")

@st.cache_resource
def load_nlp_models():
    # Load spaCy for entity recognition
    nlp = spacy.load("en_core_web_sm")
    
    # Load raw Roberta models directly to handle Transformers v5+ environments
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    return nlp, (model, tokenizer)

nlp, (qa_model, qa_tokenizer) = load_nlp_models()

# --- REPLACE YOUR QA EXTRACTION CODE BLOCK WITH THIS COMPATIBLE HELPER ---
def get_ai_answer(question, context):
    try:
        # Tokenize the input text
        inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = qa_model(**inputs)
            
        # Extract the highest probability answer span
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # Convert tokens back into a human-readable text string
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        
        # Clean up empty responses or special padding tokens
        if not answer.strip() or "<s>" in answer:
            return "I couldn't find a precise answer in the provided manual context."
        return answer
    except Exception as e:
        return f"An processing error occurred: {str(e)}"

# Mock massive, unsearchable 1200-page OEM manual data
MOCK_MANUAL_CONTEXT = """
SECTION 14.2.1 - MAIN TRANSMISSION & GEARBOX ASSEMBLY SPECIFICATIONS
The wind turbine main gearbox contains high-load planetary gears. During routine maintenance or structural replacement of the planetary gear assembly, all primary housing retention bolts must be tightened systematically. The final torque specification for the planetary gear retention bolts is exactly 450 Nm. Technicians must apply this torque in a star pattern sequence to prevent casing warpage. CRITICAL SAFETY PROTOCOL: Before attempting any torque application on the drivetrain, the mechanical rotor lock must be fully engaged, and the hydraulic brake pressure must be vented to 0 bar to prevent accidental rotor rotation. Failure to engage the rotor lock can result in catastrophic structural failure and fatal injury.
"""

st.title("Wind O&M Innovation Demo")
st.subheader("Eliminating the 'Climb-Down Tax' and Unstructured Data Bottlenecks")
st.markdown("---")

# Create the layout: Left Side (The Struggle) vs Right Side (The AI Copilot)
col1, col2 = st.columns([1, 1])

with col1:
    st.header("The Struggle (Traditional Method)")
    st.info("📄 **OEM_Manual_v4.2_FINAL_SECURE_1200_PAGES.pdf**")
    
    # Simulating a dense, hard-to-read wall of text from an unsearchable PDF
    st.markdown(
        """
        <div style="height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 15px; background-color: #f9f9f9; font-family: monospace; font-size: 11px; color: #555;">
        <b>[PAGE 742 OF 1250]</b><br><br>
        ...amendment 4.1.2 tracking code validation parameters. system integrity checks must precede structural interfacing. 
        lubrication viscosity parameters must adhere to ISO VG 320 standards under normal ambient operating envelopes...
        <br><br>
        <b>[PAGE 743 OF 1250]</b><br><br>
        """ + MOCK_MANUAL_CONTEXT + """
        <br><br>
        ...post-torque validation procedures require secondary ultrasonic scan mapping across all radial contact stress paths...
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.caption("**Technician Bottleneck:** Hand-searching this text on a small tablet 150 meters in the air wastes hours of active shift time.")

with col2:
    st.header("The Solution (AI Copilot)")
    st.success("**Intelligent Field Assistant Active**")
    
    # Pre-set conversational, messy technician question
    user_query = st.text_input(
        "Ask the Copilot (Conversational Field Query):",
        value="Hey, I'm working on the planetary gear. What do I torque the bolts to and what safety steps do I need?"
    )
    
    if st.button("Run AI Diagnostics ", type="primary"):
        with st.spinner("Analyzing text schema and extracting specs..."):
            
            # --- STEP 1: SPACHY PROCESSING ---
            doc = nlp(user_query)
            # Find keywords representing the component to mimic a custom NER pipeline
            extracted_components = [token.text for token in doc if token.text.lower() in ["gear", "planetary", "bolts", "rotor"]]
            
            # --- STEP 2: HUGGING FACE TRANSFORMER QA ---
            # The model reads the query, parses the massive manual background text, and grabs the precise answer
            qa_res_torque = qa_pipeline(question="What is the torque specification for the bolts?", context=MOCK_MANUAL_CONTEXT)
            qa_res_safety = qa_pipeline(question="What is the critical safety protocol?", context=MOCK_MANUAL_CONTEXT)
            
            # --- STEP 3: DISPLAY THE HIGH-IMPACT RESULT ---
            st.markdown("### Live Extraction Output")
            
            # Showcase spaCy metadata processing
            st.markdown(f"**Metadata Tags Detected (spaCy):** `{', '.join(set(extracted_components)) if extracted_components else 'General Query'}`")
            
            # The highlighted, bolded answer box perfect for a fast presentation demo
            st.markdown(
                f"""
                <div style="background-color: #e8f4fd; border-left: 5px solid #2196F3; padding: 20px; border-radius: 5px;">
                    <h4 style="color: #0b3c5d; margin-top: 0;">🔧 Engineering Directives Found:</h4>
                    <ul>
                        <li><b>Torque Requirement:</b> <mark style="background-color: #ffeb3b; padding: 2px 5px;"><b>{qa_res_torque['answer']}</b></mark> (Must apply in a star pattern sequence).</li>
                        <li><b>Critical Safety Step:</b> <span style="color: #d32f2f; font-weight: bold;">{qa_res_safety['answer']}</span>.</li>
                    </ul>
                    <p style="font-size: 12px; color: #666; margin-bottom: 0; margin-top: 15px;"><i>Source: Section 14.2.1 - OEM Manual Page 743 (Confidence Score: {qa_res_torque['score']:.2%})</i></p>
                </div>
                """, 
                unsafe_allow_html=True
            )
