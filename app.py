import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Solar Fault Detection", layout="centered")

st.title("🌞 AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

# =====================================
# LOAD TFLITE MODEL
# =====================================
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['clean', 'crack', 'dust']
IMG_SIZE = (224, 224)

# =====================================
# PREDICTION FUNCTION
# =====================================
def predict_image(image):
    img = image.resize(IMG_SIZE)
    img = np.array(img).astype(np.float32)

    # Normalize if needed (important!)
    img = img / 255.0  

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

# =====================================
# UI
# =====================================
uploaded_file = st.file_uploader("📤 Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    probs = predict_image(image)

    # SAME LOGIC AS YOUR ORIGINAL CODE
    dust_prob = probs[2] * 100
    crack_prob = probs[1] * 100
    clean_prob = probs[0] * 100

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

    # OUTPUT
    st.success(f"Prediction: {final_pred}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.warning(f"Action: {action}")

    # EXTRA (nice for project)
    with st.expander("🔍 Detailed Probabilities"):
        st.write(f"Clean: {clean_prob:.2f}%")
        st.write(f"Crack: {crack_prob:.2f}%")
        st.write(f"Dust: {dust_prob:.2f}%")