import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.h5", compile=False)

model = load_model()

# =========================
# Class Labels
# =========================
CLASSES = ["Clean", "Crack", "Dust"]

# =========================
# Preprocess Image
# =========================
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# UI STARTS HERE
# =========================

st.title("Solar Panel Fault Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    result = CLASSES[class_index]

    st.subheader(f"Prediction: {result}")
