from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

# ==============================
# APP CONFIG
# ==============================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = 224

# ==============================
# LOAD MODEL (SAFE PATH)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "solar_fault_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['clean', 'crack', 'dust']


# ==============================
# IMAGE PREPROCESSING
# ==============================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)

    # For EfficientNet
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    img = np.expand_dims(img, axis=0)
    return img


# ==============================
# MAIN ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None
    action = None
    error = None

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            error = "⚠️ Please upload an image"
        else:
            filename = secure_filename(file.filename)

            # Validate file type
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                error = "❌ Only JPG/PNG allowed"
            else:
                try:
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)

                    image_path = filepath

                    img = preprocess_image(filepath)

                    preds = model.predict(img)[0]

                    predicted_index = np.argmax(preds)
                    label = CLASS_NAMES[predicted_index]

                    confidence = round(float(preds[predicted_index]) * 100, 2)

                    # 🔥 DECISION LOGIC
                    if label == "crack":
                        prediction = "Crack Detected"
                        action = "⚠️ Repair Required"

                    elif label == "dust":
                        prediction = "Dust Detected"
                        action = "🧹 Cleaning Required"

                    else:
                        prediction = "Clean Panel"
                        action = "✅ No Action Needed"

                    print("Predictions:", preds)

                except Exception as e:
                    error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        action=action,
        image_path=image_path,
        error=error
    )


# ==============================
# RUN SERVER (DEPLOY READY)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)