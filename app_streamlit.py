import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
class_names = ['dust', 'crack', 'clean']

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img = np.array(image).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

st.title("Solar Panel Fault Detection ⚡")

st.markdown("### System Parameters")
col1, col2, col3, col4 = st.columns(4)
voltage = col1.number_input("Voltage (V)", value=30.0, step=1.0)
current = col2.number_input("Current (A)", value=8.0, step=0.5)
area = col3.number_input("Area (m²)", value=1.6, step=0.1)
irradiance = col4.number_input("Irradiance (W/m²)", value=1000.0, step=50.0)

power = voltage * current
efficiency = (power / (area * irradiance)) * 100 if (area * irradiance) > 0 else 0.0

st.markdown("### Performance Metrics")
metrics_col1, metrics_col2 = st.columns(2)
metrics_col1.metric("Power Output", f"{power:.2f} W")
metrics_col2.metric("Efficiency", f"{efficiency:.2f} %")

st.divider()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    remove_geotag = st.checkbox("Remove GPS/Geotag Overlay", help="Check this if your image has a timestamp or map overlay at the bottom that might confuse the AI.")
    if remove_geotag:
        width, height = image.size
        # Crop the bottom 30% of the image where geotags usually are
        image = image.crop((0, 0, width, int(height * 0.70)))
        
    st.image(image)

    preds = predict(image)

    dust_prob = preds[0] * 100
    crack_prob = preds[1] * 100
    clean_prob = preds[2] * 100

    max_idx = np.argmax(preds)

    if max_idx == 0:
        final_pred = "DUST"
        action = "Cleaning Required 🧹"
        confidence = dust_prob
    elif max_idx == 1:
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