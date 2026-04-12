import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['clean', 'crack', 'dust']
IMG_SIZE = (224, 224)

st.title("AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    index = np.argmax(preds)
    prediction = class_names[index]
    confidence = preds[index] * 100

    if prediction == "dust":
        action = "Cleaning Required"
    elif prediction == "crack":
        action = "Maintenance Required"
    else:
        action = "No Action Needed"

    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Action: {action}")