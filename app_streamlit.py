import streamlit as st
import numpy as np
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Solar Fault Detection", layout="centered")

st.title("🌞 AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

# =====================================
# CLASS LABELS
# =====================================
class_names = ['CLEAN', 'CRACK', 'DUST']

# =====================================
# FAKE MODEL (SIMULATION)
# =====================================
def predict_image():
    probs = np.random.rand(3)
    probs = probs / np.sum(probs)
    return probs

# =====================================
# UI
# =====================================
uploaded_file = st.file_uploader("📤 Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    probs = predict_image()

    # Probabilities
    clean_prob = probs[0] * 100
    crack_prob = probs[1] * 100
    dust_prob = probs[2] * 100

    # Decision logic (same as your original)
    if dust_prob > 75:
        final_pred = "DUST"
        action = "🧹 Cleaning Required"
        confidence = dust_prob

    elif crack_prob > 60:
        final_pred = "CRACK"
        action = "⚠️ Maintenance Required"
        confidence = crack_prob

    else:
        final_pred = "CLEAN"
        action = "✅ No Action Needed"
        confidence = clean_prob

    # Output
    st.success(f"Prediction: {final_pred}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Action: {action}")

    # Extra details
    with st.expander("🔍 Detailed Probabilities"):
        st.write(f"Clean: {clean_prob:.2f}%")
        st.write(f"Crack: {crack_prob:.2f}%")
        st.write(f"Dust: {dust_prob:.2f}%")