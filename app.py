# =====================================
# IMPORTS
# =====================================
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

# =====================================
# INIT
# =====================================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================
# GLOBALS
# =====================================
model = None
class_names = ['clean', 'crack', 'dust']
IMG_SIZE = (224, 224)

# =====================================
# LOAD MODEL (LAZY LOADING 🔥)
# =====================================
def load_model_once():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("final_model.h5")
        print("Model loaded successfully")

# =====================================
# PREDICTION FUNCTION
# =====================================
def predict_image(img_path):
    load_model_once()

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img)[0]

    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    return predicted_class, confidence, preds

# =====================================
# ROUTE
# =====================================
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    action = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html")

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        pred, conf, probs = predict_image(path)

        # 🔥 PROBABILITIES
        dust_prob = probs[2] * 100
        crack_prob = probs[1] * 100
        clean_prob = probs[0] * 100

        # 🔥 DECISION LOGIC
        if dust_prob > 75:
            final_pred = "DUST"
            action = "Cleaning Required 🧹"
            confidence = round(dust_prob, 2)

        elif crack_prob > 60:
            final_pred = "CRACK"
            action = "Maintenance Required ⚠️"
            confidence = round(crack_prob, 2)

        else:
            final_pred = "CLEAN"
            action = "No Action Needed ✅"
            confidence = round(clean_prob, 2)

        prediction = final_pred
        image_path = path

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        action=action,
        image_path=image_path
    )

# =====================================
# HEALTH CHECK (IMPORTANT FOR RENDER)
# =====================================
@app.route("/health")
def health():
    return "OK", 200

# =====================================
# RUN (RENDER COMPATIBLE)
# =====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)