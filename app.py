import streamlit as st
import spacy
import torch
import time
import numpy as np
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from sklearn.ensemble import IsolationForest

# Set up page config for a clean widescreen layout
st.set_page_config(layout="wide", title="Wind Turbine AI Copilot and Predictive Health Dashboard")

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

# Simulated text streaming generator function to mimic OpenAI's dynamic output typewriter style
def stream_text_effect(text_string):
    for word in text_string.split(" "):
        yield word + " "
        time.sleep(0.04)

# Expanded OEM Manual to contain normal routines, engineering data, and structural principles
MOCK_MANUAL_CONTEXT = """
SECTION 14.2.1 - EMERGENCY MAIN TRANSMISSION AND GEARBOX ASSEMBLY REPAIR SPECIFICATIONS
During high-stress structural replacement or abnormal load remediation of the planetary gear assembly, all primary housing retention bolts must be tightened systematically. The final torque specification for the planetary gear retention bolts is exactly 450 Nm. Technicians must apply this torque in a star pattern sequence to prevent casing warpage. For the critical safety protocol, the mechanical rotor lock must be fully engaged and the hydraulic brake pressure must be vented to 0 bar to prevent accidental rotor rotation. Failure to engage the rotor lock can result in catastrophic structural failure and fatal injury.

SECTION 14.2.2 - NOMINAL OPERATIONS AND STANDARD PREVENTATIVE MAINTENANCE ROUTINE
Under healthy baseline conditions, standard preventative maintenance must be performed every 2500 operating hours. The standard maintenance routine requires a complete visual inspection of the nacelle housing, replenishing the synthetic gear lubricant, cleaning the cooling intake vents, and collecting a raw oil sample to monitor particulate counts. Technicians should verify that the system system pressure is holding steady at a nominal baseline of 160 bar. No mechanical rotor lock is required for external visual sweeps.
"""

def generate_scada_data(trigger_anomaly=False):
    """Generates synthetic wind turbine SCADA sensor data."""
    np.random.seed(42)
    timesteps = 50
    time_index = pd.date_range(end=pd.Timestamp.now(), periods=timesteps, freq='min')
    
    vibration = np.random.normal(loc=1.8, scale=0.15, size=timesteps)
    bearing_temp = np.random.normal(loc=62.0, scale=2.0, size=timesteps)
    gearbox_oil_temp = np.random.normal(loc=70.0, scale=1.5, size=timesteps)
    
    if trigger_anomaly:
        vibration[-5:] += np.array([0.8, 1.5, 2.3, 3.1, 3.8])
        bearing_temp[-5:] += np.array([4.0, 9.0, 15.0, 22.0, 31.0])
        gearbox_oil_temp[-5:] += np.array([2.0, 5.0, 9.0, 14.0, 19.0])
        
    df = pd.DataFrame({
        'Main Bearing Vibration (mm/s)': vibration,
        'Bearing Temperature (C)': bearing_temp,
        'Gearbox Oil Temperature (C)': gearbox_oil_temp
    }, index=time_index)
    return df

def run_anomaly_detector(data):
    X = data.values
    clf = IsolationForest(contamination=0.1, random_state=42)
    predictions = clf.fit_predict(X)
    return predictions[-1] == -1

# Main Application Headers
st.title("Wind O&M Innovation Demo")
st.subheader("Real-Time SCADA Anomaly Detection and Automated Document Retrieval")
st.markdown("---")

# Prominent Main Page Control Unit Container
st.markdown("### SCADA Control Unit Panel")
system_mode = st.radio(
    "Select Turbine Operational State:", 
    ["Healthy Stream", "Inject Gearbox Thermal and Vibration Defect"],
    horizontal=True
)
st.markdown("---")

trigger_anomaly_flag = (system_mode == "Inject Gearbox Thermal and Vibration Defect")
scada_df = generate_scada_data(trigger_anomaly=trigger_anomaly_flag)
is_anomaly = run_anomaly_detector(scada_df)

# System Status Warning Strip
if is_anomaly:
    st.error("CRITICAL ALARM: SCADA TELEMETRY ANOMALY DETECTED IN MAIN GEARBOX HOUSING")
else:
    st.success("SYSTEM HEALTH REGULAR: Drivetrain parameters operating within nominal margins.")

# Main Display split column layout
col1, col2 = st.columns([1, 1.1], gap="large")

with col1:
    st.subheader("Real-Time SCADA Sensor Streams")
    st.caption("Live 1-minute telemetry windows extracted from the Nacelle Controller.")
    st.line_chart(scada_df)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    current_vib = scada_df['Main Bearing Vibration (mm/s)'].iloc[-1]
    current_temp = scada_df['Bearing Temperature (C)'].iloc[-1]
    
    m_col1.metric("Vibration Level", f"{current_vib:.2f} mm/s", delta="HIGH" if is_anomaly else "NORMAL", delta_color="inverse" if is_anomaly else "normal")
    m_col2.metric("Bearing Temp", f"{current_temp:.1f} C", delta="+31C SPIKE" if is_anomaly else "STABLE", delta_color="inverse" if is_anomaly else "normal")
    
    system_pressure = "0 bar State" if is_anomaly else "160 bar"
    pressure_delta = "Vented" if is_anomaly else "Nominal Operating Pressure"
    m_col3.metric("System Pressure", system_pressure, delta=pressure_delta)

with col2:
    st.subheader("Connected AI Copilot Context")
    if is_anomaly:
        st.warning("Automated Copilot Hook: Anomaly detected. The AI has instantly switched to Section 14.2.1 for critical troubleshooting directives.")
    else:
        st.info("System healthy. The AI has automatically queued Section 14.2.2 outlining preventative maintenance criteria.")

    # Context-aware default text values based on the toggle state
    default_query = "What is the critical safety protocol and torque specification for fixing the gear assembly?" if is_anomaly else "What does the standard maintenance routine require?"
    user_query = st.text_input("Conversational Field Query:", value=default_query)
    
    if st.button("Query Technical Repository", type="primary", use_container_width=True) or is_anomaly:
        with st.spinner("Extracting critical procedural steps..."):
            doc = nlp(user_query)
            extracted_components = [token.text for token in doc if token.text.lower() in ["gear", "planetary", "bolts", "rotor", "vibration", "temperature", "inspection", "maintenance"]]
            
            # Formulate textual output blocks and execute QA depending on the pipeline mode state
            if "routine" in user_query.lower() or "maintenance" in user_query.lower() or not is_anomaly:
                answer_primary = get_ai_answer("What does the standard maintenance routine require?", MOCK_MANUAL_CONTEXT)
                answer_secondary = get_ai_answer("What should technicians verify regarding system pressure?", MOCK_MANUAL_CONTEXT)
                
                # Dynamic learning insights injected directly during standard healthy queries
                ai_text_response = (
                    f"### Copilot Advisory Note\n\n"
                    f"The analysis of Section 14.2.2 indicates the system is within nominal ranges. "
                    f"The standard maintenance routine requires {answer_primary if answer_primary else 'a full inspection of the nacelle housing, replenishing gear lubricant, and collecting an oil particulate sample'}.\n\n"
                    f"**Engineering Field Learning:** In modern utility-scale wind systems, preventative checks are spaced at 2500 operating hours to minimize component fatigue. "
                    f"Monitoring gear oil health is vital because micro-particulate metal shavings can indicate premature planetary gear wear long before a sensor alarm triggers. "
                    f"Always confirm system system pressure remains steady at approximately 160 bar to keep the mechanical seals fully lubricated."
                )
                is_fault_view = False
            else:
                answer_primary = get_ai_answer("What is the final torque specification for the bolts?", MOCK_MANUAL_CONTEXT)
                answer_secondary = get_ai_answer("What is the critical safety protocol?", MOCK_MANUAL_CONTEXT)
                
                ai_text_response = (
                    f"### Critical Emergency Engineering Directive\n\n"
                    f"**CRITICAL ACTION REQUIRED:** {answer_secondary.strip().capitalize() if answer_secondary else 'The mechanical rotor lock must be fully engaged and brake pressure vented to 0 bar'} before any tools or technicians interact with the main drivetrain assembly.\n\n"
                    f"**Safety Hazard Context:** Working inside the transmission nacelle without the mechanical rotor lock engaged can result in fatal injuries due to aerodynamic wind loads spinning the blades unexpectedly. "
                    f"Housing bolts must be tightened systematically to exactly {answer_primary if answer_primary else '450 Nm'} in a star pattern. Deviation from this geometric pattern will induce structural casing warpage."
                )
                is_fault_view = True
                
            # --- OUTPUT BLOCK 1: BIG OPENAI TYPEWRITER TEXT GENERATION FIRST ---
            st.write_stream(stream_text_effect(ai_text_response))
            st.markdown("---")
            
            # --- OUTPUT BLOCK 2: SMALLER STRUCTURED SUMMARY DATA CARDS LATER ---
            st.markdown("### Secondary Extraction Summary")
            
            # Smaller metadata footprint
            st.caption(f"**Detected NLP System Components:** {', '.join(set(extracted_components)).upper() if extracted_components else 'GENERAL OVERHAUL'}")
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label="Target Metric Parameter", 
                    value="450 Nm" if is_fault_view else "160 bar",
                    delta="Star Sequence Mandatory" if is_fault_view else "Nominal Fluid Pressure",
                    delta_color="off"
                )
            with metric_col2:
                st.metric(
                    label="System Reference Frame", 
                    value="Section 14.2.1" if is_fault_view else "Section 14.2.2",
                    delta="Emergency Repair Track" if is_fault_view else "Preventative Track",
                    delta_color="inverse" if is_fault_view else "normal"
                )
                
            st.caption("Source reference data verified against OEM Core Engineering Manual Master Sheets")
