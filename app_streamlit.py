import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Solar Fault Detection", layout="centered")

st.title("🌞 AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

# =====================================
# GLOBALS
# =====================================
class_names = ['clean', 'crack', 'dust']
IMG_SIZE = (224, 224)

# =====================================
# LOAD MODEL (CACHED)
# =====================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model.h5")
    return model

model = load_model()

# =====================================
# PREDICTION FUNCTION
# =====================================
def predict_image(image):
    img = image.resize(IMG_SIZE)
    img = np.array(img)

    # Expand dims
    img = np.expand_dims(img, axis=0)

    # Preprocessing (important for your model)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img)[0]
    return preds

# =====================================
# UI
# =====================================
uploaded_file = st.file_uploader("📤 Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    probs = predict_image(image)

    # Probabilities
    dust_prob = probs[2] * 100
    crack_prob = probs[1] * 100
    clean_prob = probs[0] * 100

    # Decision logic (same as your Flask app)
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

    # Optional details
    with st.expander("🔍 Detailed Probabilities"):
        st.write(f"Clean: {clean_prob:.2f}%")
        st.write(f"Crack: {crack_prob:.2f}%")
        st.write(f"Dust: {dust_prob:.2f}%")