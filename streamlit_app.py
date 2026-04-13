import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf



@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()




CLASSES = ["Clean", "Crack", "Dust"]


def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img).astype(np.float32)

    # Match training preprocessing
    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img


def predict(image):
    processed = preprocess(image)
    preds = model.predict(processed, verbose=0)[0]
    return preds


st.title("Solar Panel Fault Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    result = CLASSES[class_index]

    index = int(np.argmax(preds))
    label = CLASSES[index]
    confidence = float(preds[index]) * 100

    st.subheader("Result")
    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")