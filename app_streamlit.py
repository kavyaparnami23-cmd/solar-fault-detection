import streamlit as st
import numpy as np
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Solar Fault Detection", layout="centered")

st.title("AI-Based Solar Panel Fault Detection")
st.write("Upload an image to detect: Clean / Crack / Dust")

# =====================================
# GLOBALS
# =====================================
class_names = ['clean', 'crack', 'dust']
IMG_SIZE = (224, 224)

# =====================================
# LOAD MODEL
# =====================================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("final_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# =====================================
# PREDICTION FUNCTION
# =====================================
def predict_image(image):
    img = image.resize(IMG_SIZE)
    img = np.array(img)

    # Same preprocessing as training
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    return preds

# =====================================
# UI
# =====================================
uploaded_file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    if model is None:
        st.error("Model not loaded. Check deployment.")
    else:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prediction
        probs = predict_image(image)

        # Debug probabilities
        st.subheader("Raw Probabilities")
        st.write({
            "Clean": float(probs[0]),
            "Crack": float(probs[1]),
            "Dust": float(probs[2])
        })

        # Final prediction
        final_index = np.argmax(probs)
        final_pred = class_names[final_index].upper()
        confidence = probs[final_index] * 100

        # Action logic
        if final_pred == "DUST":
            action = "Cleaning Required"
        elif final_pred == "CRACK":
            action = "Maintenance Required"
        else:
            action = "No Action Needed"

        # Output
        st.success(f"Prediction: {final_pred}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.warning(f"Action: {action}")