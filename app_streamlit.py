import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

IMG_SIZE = (224, 224)
class_names = ['clean', 'crack', 'dust']

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

st.title("Solar Panel Fault Detection ⚡")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    preds = predict(image)

    clean_prob = preds[0] * 100
    crack_prob = preds[1] * 100
    dust_prob = preds[2] * 100

    if dust_prob > 75:
        final_pred = "DUST"
        action = "Cleaning Required 🧹"
        confidence = dust_prob

    elif crack_prob > 60:
        final_pred = "CRACK"
        action = "Maintenance Required ⚠️"
        confidence = crack_prob

    else:
        final_pred = "CLEAN"
        action = "No Action Needed ✅"
        confidence = clean_prob

    st.subheader(final_pred)
    st.write(f"Confidence: {confidence:.2f}%")
    st.write(action)

    st.write("### Probabilities")
    st.write(f"Clean: {clean_prob:.2f}%")
    st.write(f"Crack: {crack_prob:.2f}%")
    st.write(f"Dust: {dust_prob:.2f}%")