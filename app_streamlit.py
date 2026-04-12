import streamlit as st
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# ================================
# LOAD MODEL
# ================================
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)
class_names = ['clean', 'crack', 'dust']

# ================================
# PREDICTION FUNCTION
# ================================
def predict(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)

    # normalize (VERY IMPORTANT for correct prediction)
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Solar Fault Detection", layout="centered")

st.title("AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

uploaded_file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preds = predict(image)

    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    st.write("Prediction:", predicted_class)
    st.write("Confidence:", round(confidence, 2))

    # ================================
    # DECISION LOGIC (IMPROVED)
    # ================================
    dust_prob = preds[2] * 100
    crack_prob = preds[1] * 100
    clean_prob = preds[0] * 100

    if dust_prob > 60:
        st.write("Action: Cleaning Required")
    elif crack_prob > 55:
        st.write("Action: Maintenance Required")
    else:
        st.write("Action: No Action Needed")