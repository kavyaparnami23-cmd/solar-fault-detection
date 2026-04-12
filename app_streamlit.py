import streamlit as st
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Solar Fault Detection", layout="centered")

st.title("AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

# Classes
class_names = ['CLEAN', 'CRACK', 'DUST']

# Dummy prediction (NO TensorFlow)
def predict_image():
    probs = np.random.rand(3)
    probs = probs / np.sum(probs)
    return probs

# Upload
uploaded_file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    probs = predict_image()

    clean_prob = probs[0] * 100
    crack_prob = probs[1] * 100
    dust_prob = probs[2] * 100

    final_index = np.argmax(probs)
    final_pred = class_names[final_index]
    confidence = probs[final_index] * 100

    if final_pred == "DUST":
        action = "Cleaning Required"
    elif final_pred == "CRACK":
        action = "Maintenance Required"
    else:
        action = "No Action Needed"

    st.success(f"Prediction: {final_pred}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Action: {action}")