import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)
class_names = ['clean', 'crack', 'dust']

def predict(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

st.title("Solar Panel Fault Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    preds = predict(image)

    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.write("Prediction:", predicted_class)
    st.write("Confidence:", round(confidence, 2))