import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import datetime
from github import Github

# Initialize layout
st.set_page_config(page_title="Cloud Dashboard & Alerts", layout="wide")

IMG_SIZE = (224, 224)
class_names = ['clean', 'crack', 'dust']

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

st.title("Cloud Dashboard & Automated Alert System ⚡")

# Initialize session state for alert logs
if "alerts" not in st.session_state:
    st.session_state.alerts = []

def log_alert(level, message, duplicate_check=True):
    # Avoid duplicate sequential alerts
    msg_body = f"{level}: {message}"
    if duplicate_check and len(st.session_state.alerts) > 0 and msg_body in st.session_state.alerts[0]:
        return
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.alerts.insert(0, f"[{timestamp}] {msg_body}")

def create_github_issue(title, body):
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "REPLACE_WITH_YOUR_TOKEN") 
    REPO_NAME = "kavyaparnami23-cmd/solar-fault-detection"
    if GITHUB_TOKEN == "REPLACE_WITH_YOUR_TOKEN" or not GITHUB_TOKEN:
        log_alert("WARNING", "GitHub token not configured. Set GITHUB_TOKEN env var to generate actual GitHub Issues.")
        return False
    
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        repo.create_issue(title=title, body=body)
        log_alert("SUCCESS", f"GitHub Issue created: {title}")
        return True
    except Exception as e:
        log_alert("ERROR", f"Failed to create GitHub issue: {str(e)}")
        return False

col1, col2 = st.columns([1, 1.2])

with col1:
    st.info("📸 **Image Upload Guide:** Upload close-up photos of the **solar panel glass**.")
    uploaded_file = st.file_uploader("Upload Image (Simulated Camera Feed)", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        preds = predict(image)

        clean_prob = preds[0] * 100
        crack_prob = preds[1] * 100
        dust_prob = preds[2] * 100

        max_idx = np.argmax(preds)

        if max_idx == 0:
            final_pred = "CLEAN"
            action = "No Action Needed ✅"
            confidence = clean_prob
            volt, curr, eff = 32.4, 8.2, 98
            log_alert("INFO", "Scan clear. System nominal.", False)
        elif max_idx == 1:
            final_pred = "CRACK"
            action = "Critical Maintenance Required ⚠️"
            confidence = crack_prob
            volt, curr, eff = 15.1, 2.3, 21
            log_alert("CRITICAL", "Crack detected! Efficiency catastrophic.")
            create_github_issue("🚨 Urgent: Solar Panel Crack Detected", "A severe crack has been detected via the IoT Monitoring system. Power efficiency dropped to ~21%.\n\nPlease inspect the identified panel immediately.")
        else:
            final_pred = "DUST"
            action = "Cleaning Mechanism Activated 🧹💦"
            confidence = dust_prob
            volt, curr, eff = 28.5, 6.1, 74
            log_alert("WARNING", "Dust detected. Performance dropped. Activating water spray.")

        st.markdown(f"### Classification: **{final_pred}** ({confidence:.1f}%)")
        st.markdown(f"**System Actuator State**: {action}")

with col2:
    st.header("IoT Monitoring & Performance Analysis")
    if uploaded_file:
        power = volt * curr
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Voltage", f"{volt} V", f"{volt - 32.0:.1f} V")
        mcol2.metric("Current", f"{curr} A", f"{curr - 8.0:.1f} A")
        mcol3.metric("Power", f"{power:.1f} W", f"{power - 256.0:.1f} W")
        mcol4.metric("Efficiency", f"{eff}%", f"{eff - 100}%")

        if final_pred == "DUST":
            st.warning("⚙️ **Automated Cleaning Mechanism:** Motor & Pump Active. Water spray initiated.")
        elif final_pred == "CRACK":
            st.error("🚨 **System Alert:** Power extremely low. Maintenance ticket automatically created.")
        else:
            st.success("✅ **System Nominal:** Operating at peak efficiency.")
    else:
        st.write("Awaiting sensor data...")

    st.markdown("---")
    st.header("Alerts & Notification Log")
    
    # CSS to make the alert box scrollable
    st.markdown('''
        <style>
            .alert-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #0e1117;
            }
            .alert-item {
                padding: 8px;
                margin-bottom: 5px;
                border-radius: 4px;
                font-family: monospace;
            }
        </style>
    ''', unsafe_allow_html=True)
    
    if len(st.session_state.alerts) > 0:
        for alert in st.session_state.alerts[:7]: # Show top 7
            if "CRITICAL" in alert or "ERROR" in alert:
                st.error(alert)
            elif "WARNING" in alert:
                st.warning(alert)
            elif "SUCCESS" in alert:
                st.success(alert)
            else:
                st.info(alert)
    else:
        st.info("System initializing. No alerts generated yet.")