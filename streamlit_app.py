import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
class_names = ['clean', 'crack', 'dust']

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

    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    img = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

st.title("Solar Panel Fault Detection ⚡")
st.write("Upload an image to detect Clean / Crack / Dust")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preds = predict(image)

    pred_index = np.argmax(preds)
    final_pred = class_names[pred_index].upper()
    confidence = round(preds[pred_index] * 100, 2)

    if final_pred == "DUST":
        action = "Cleaning Required 🧹"
    elif final_pred == "CRACK":
        action = "Maintenance Required ⚠️"
    else:
        action = "No Action Needed ✅"

    st.subheader(f"Prediction: {final_pred}")
    st.write(f"Confidence: {confidence}%")
    st.write(f"Action: {action}")

    st.write("### Probabilities:")
    st.write(f"Clean: {preds[0]*100:.2f}%")
    st.write(f"Crack: {preds[1]*100:.2f}%")
    st.write(f"Dust: {preds[2]*100:.2f}%")
