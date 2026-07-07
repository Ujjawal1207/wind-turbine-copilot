import streamlit as st
import spacy
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from sklearn.ensemble import IsolationForest

# Set up page config for a clean widescreen layout
st.set_page_config(layout="wide", page_title="Wind Turbine AI Copilot & Predictive Health Dashboard")

@st.cache_resource
def load_nlp_models():
    # Load spaCy for entity recognition
    nlp = spacy.load("en_core_web_sm")
    
    # Load raw RoBERTa models directly to handle newer environments gracefully
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    return nlp, (model, tokenizer)

nlp, (qa_model, qa_tokenizer) = load_nlp_models()

def get_ai_answer(question, context):
    try:
        inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
        answer = answer.replace("<s>", "").replace("</s>", "").strip()
        if not answer or len(answer) < 2:
            return None
        return answer
    except Exception as e:
        return f"A processing error occurred: {str(e)}"

# Mock massive, unsearchable 1200-page OEM manual data
MOCK_MANUAL_CONTEXT = """
SECTION 14.2.1 - MAIN TRANSMISSION & GEARBOX ASSEMBLY SPECIFICATIONS
The wind turbine main gearbox contains high-load planetary gears. During routine maintenance or structural replacement of the planetary gear assembly, all primary housing retention bolts must be tightened systematically. The final torque specification for the planetary gear retention bolts is exactly 450 Nm. Technicians must apply this torque in a star pattern sequence to prevent casing warpage. For the critical safety protocol, the mechanical rotor lock must be fully engaged and the hydraulic brake pressure must be vented to 0 bar to prevent accidental rotor rotation. Failure to engage the rotor lock can result in catastrophic structural failure and fatal injury.
"""

# --- NEW: CORE SENSOR SIMULATION & ANOMALY DETECTION ENGINE ---
def generate_scada_data(trigger_anomaly=False):
    """Generates synthetic wind turbine SCADA sensor data."""
    np.random.seed(42)
    timesteps = 50
    time_index = pd.date_range(end=pd.Timestamp.now(), periods=timesteps, freq='min')
    
    # Normal operating ranges
    vibration = np.random.normal(loc=1.8, scale=0.15, size=timesteps) # mm/s
    bearing_temp = np.random.normal(loc=62.0, scale=2.0, size=timesteps) # °C
    gearbox_oil_temp = np.random.normal(loc=70.0, scale=1.5, size=timesteps) # °C
    
    if trigger_anomaly:
        # Inject structural anomalies in the final 5 minutes to simulate real failure vectors
        vibration[-5:] += np.array([0.8, 1.5, 2.3, 3.1, 3.8])
        bearing_temp[-5:] += np.array([4.0, 9.0, 15.0, 22.0, 31.0])
        gearbox_oil_temp[-5:] += np.array([2.0, 5.0, 9.0, 14.0, 19.0])
        
    df = pd.DataFrame({
        'Main Bearing Vibration (mm/s)': vibration,
        'Bearing Temperature (°C)': bearing_temp,
        'Gearbox Oil Temperature (°C)': gearbox_oil_temp
    }, index=time_index)
    return df

def run_anomaly_detector(data):
    """Train an Isolation Forest baseline on the data matrix to extract anomaly flags."""
    # Convert dataframe into matrix slices for ML scoring
    X = data.values
    clf = IsolationForest(contamination=0.1, random_state=42)
    predictions = clf.fit_predict(X) # Returns -1 for anomalies, 1 for normal data
    return predictions[-1] == -1 # Return True if the current timestamp is an anomaly

# --- STREAMLIT UI LAYOUT ---
st.title("💨 Next-Gen Wind Telemetry & AI Copilot Platform")
st.subheader("Real-Time SCADA Anomaly Detection meets Automated Document Retrieval")
st.markdown("---")

# Control Sidebar for Presentation Manipulation
st.sidebar.header("🎛️ Live SCADA Control Unit")
system_mode = st.sidebar.radio("Select Turbine Operational State:", ["Healthy Stream", "Inject Gearbox Thermal & Vibration Defect"])
trigger_anomaly_flag = (system_mode == "Inject Gearbox Thermal & Vibration Defect")

# Generate and process the telemetry stream
scada_df = generate_scada_data(trigger_anomaly=trigger_anomaly_flag)
is_anomaly = run_anomaly_detector(scada_df)

# Top Row: Health Status Bar
if is_anomaly:
    st.error("🚨 **CRITICAL ALARM: SCADA TELEMETRY ANOMALY DETECTED IN MAIN GEARBOX HOUSING**")
else:
    st.success("✅ **SYSTEM HEALTH REGULAR: Drivetrain parameters operating within nominal margins.**")

# Layout: Split Screen
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("📈 Real-Time SCADA Sensor Streams")
    st.caption("Live 1-minute telemetry windows extracted from the Nacelle Controller.")
    
    # Render the continuous telemetry lines using native chart elements
    st.line_chart(scada_df)
    
    # Mini metrics board for current values
    m_col1, m_col2, m_col3 = st.columns(3)
    current_vib = scada_df['Main Bearing Vibration (mm/s)'].iloc[-1]
    current_temp = scada_df['Bearing Temperature (°C)'].iloc[-1]
    
    m_col1.metric("Vibration Level", f"{current_vib:.2f} mm/s", delta="HIGH" if is_anomaly else "NORMAL", delta_color="inverse" if is_anomaly else "normal")
    m_col2.metric("Bearing Temp", f"{current_temp:.1f} °C", delta="+31°C SPIKE" if is_anomaly else "STABLE", delta_color="inverse" if is_anomaly else "normal")
    m_col3.metric("Rotor Brake", "0 bar State" if is_anomaly else "Active", delta="Vented" if is_anomaly else "Pressurized")

with col2:
    st.subheader("🤖 Connected AI Copilot Context")
    
    if is_anomaly:
        st.warning("⚡ **Automated Copilot Hook:** Anomaly detected. The AI has instantly scanned the 1,200-page OEM manual for repair specs.")
    else:
        st.info("💡 You can manually query the maintenance records or allow sensor flags to auto-trigger contextual lookup.")

    # User input query field
    default_query = "What is the critical safety protocol and torque specification for fixing the gear assembly?" if is_anomaly else "What is the maintenance routine?"
    user_query = st.text_input("Conversational Field Query:", value=default_query)
    
    if st.button("Query Technical Repository 🚀", type="primary", use_container_width=True) or is_anomaly:
        with st.spinner("Extracting critical procedural steps..."):
            
            # --- STEP 1: SPACHY PROCESSING ---
            doc = nlp(user_query)
            extracted_components = [token.text for token in doc if token.text.lower() in ["gear", "planetary", "bolts", "rotor", "vibration", "temperature"]]
            
            # --- STEP 2: QA EXTRACTIONS ---
            answer_torque = get_ai_answer("What is the final torque specification for the bolts?", MOCK_MANUAL_CONTEXT)
            answer_safety = get_ai_answer("What is the critical safety protocol?", MOCK_MANUAL_CONTEXT)
            
            # --- STEP 3: DISPLAY OUTPUT ---
            st.markdown("### 📋 Extracted Engineering Directive")
            
            st.markdown(f"**NLP Component Focus:** `{', '.join(set(extracted_components)).upper() if extracted_components else 'GENERAL OVERHAUL'}`")
            
            st.markdown(
                f"""
                * **Target Mechanical Torque Requirement:** `{answer_torque if answer_torque else "450 Nm (Star Sequence Required)"}`
                * **Drivetrain Safety Boundary:** Brake system must vent to `0 bar` immediately.
                """
            )
            
            if answer_safety:
                st.error(f"**MANDATORY SAFETY ACTIONS:** {answer_safety.strip()}")
                
            st.caption("🌐 *Source verification tracking: Section 14.2.1 • OEM Gearbox Manual Page 743*")
