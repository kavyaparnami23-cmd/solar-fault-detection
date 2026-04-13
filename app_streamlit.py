import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

IMG_SIZE = (224, 224)

DUST_THRESHOLD = 75
CRACK_THRESHOLD = 60

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "final_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    img = preprocess_image(image)
    preds = model.predict(img)[0]
    return preds

st.set_page_config(page_title="Solar Fault Detection", page_icon="⚡")

st.title("⚡ Solar Panel Fault Detection")
st.write("Upload an image to detect Clean / Crack / Dust")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    preds = predict(image)

    clean_prob = preds[0] * 100
    crack_prob = preds[1] * 100
    dust_prob = preds[2] * 100

    if dust_prob > DUST_THRESHOLD:
        final_pred = "DUST"
        action = "🧹 Cleaning Required"
        confidence = dust_prob

    elif crack_prob > CRACK_THRESHOLD:
        final_pred = "CRACK"
        action = "⚠️ Maintenance Required"
        confidence = crack_prob

    else:
        final_pred = "CLEAN"
        action = "✅ No Action Needed"
        confidence = clean_prob

    st.success(f"Prediction: {final_pred}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.write(f"Action: {action}")

    st.write("### Probabilities:")
    st.write(f"Clean: {clean_prob:.2f}%")
    st.write(f"Crack: {crack_prob:.2f}%")

    st.write(f"Dust: {dust_prob:.2f}%")

    st.write(f"Dust: {dust_prob:.2f}%")
