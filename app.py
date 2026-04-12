import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Solar Panel Fault Detection", layout="centered")

st.title("AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

# ================================
# LOAD MODEL (cached for speed)
# ================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_model.h5", compile=False)
    return model

model = load_model()

# ================================
# GLOBALS
# ================================
IMG_SIZE = (224, 224)
class_names = ['CLEAN', 'CRACK', 'DUST']

# ================================
# PREDICTION FUNCTION
# ================================
def predict_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)

    # IMPORTANT: match training preprocessing
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    return preds

# ================================
# UI
# ================================
uploaded_file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    preds = predict_image(image)

    # Probabilities
    clean_prob = preds[0] * 100
    crack_prob = preds[1] * 100
    dust_prob = preds[2] * 100

    # Final prediction (argmax)
    final_index = np.argmax(preds)
    prediction = class_names[final_index]
    confidence = preds[final_index] * 100

    # ================================
    # OUTPUT
    # ================================
    st.subheader("Result")
    st.write(f"Prediction: {prediction}")
    st.write(f"Confidence: {confidence:.2f}%")

    # ================================
    # ACTION LOGIC
    # ================================
    if prediction == "DUST":
        action = "Cleaning Required"
    elif prediction == "CRACK":
        action = "Maintenance Required"
    else:
        action = "No Action Needed"

    st.write(f"Action: {action}")

    # ================================
    # DEBUG (optional, can remove later)
    # ================================
    with st.expander("Detailed Probabilities"):
        st.write({
            "Clean": round(clean_prob, 2),
            "Crack": round(crack_prob, 2),
            "Dust": round(dust_prob, 2)
        })