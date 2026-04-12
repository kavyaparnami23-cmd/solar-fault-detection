import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.h5", compile=False)

model = load_model()

# ===============================
# Class Labels (IMPORTANT ORDER)
# ===============================
CLASSES = ["Clean", "Crack", "Dust"]

# ===============================
# Preprocess Image
# ===============================
def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)

    # Match training preprocessing
    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

# ===============================
# Predict Function
# ===============================
def predict(image):
    processed = preprocess(image)
    preds = model.predict(processed, verbose=0)[0]
    return preds

# ===============================
# UI
# ===============================
st.title("Solar Panel Fault Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ===============================
# Prediction Output
# ===============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    preds = predict(image)

    index = int(np.argmax(preds))
    label = CLASSES[index]
    confidence = float(preds[index]) * 100

    st.subheader("Result")
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")