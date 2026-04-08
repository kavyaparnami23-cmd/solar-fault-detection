# =====================================
# IMPORTS
# =====================================
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

# =====================================
# INIT
# =====================================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================
# LOAD MODEL
# =====================================
model = tf.keras.models.load_model("final_model .h5")

class_names = ['clean', 'crack', 'dust']
IMG_SIZE = (224,224)

# =====================================
# PREDICTION FUNCTION
# =====================================
def predict_image(img_path):
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
        file = request.files["file"]

        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            pred, conf, probs = predict_image(path)

            # 🔥 OPTIMIZED LOGIC
            dust_prob = probs[2] * 100
            crack_prob = probs[1] * 100
            clean_prob = probs[0] * 100

            # 🔥 Smarter decision
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
# RUN
# =====================================
if __name__ == "__main__":
    app.run()