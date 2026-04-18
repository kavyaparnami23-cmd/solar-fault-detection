from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import io
import os

try:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter

from PIL import Image

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

CLASS_NAMES = ["clean", "crack", "dust"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "model.tflite"

# Load model once at startup
_interpreter = None

def get_interpreter():
    global _interpreter
    if _interpreter is None and os.path.exists(MODEL_PATH):
        _interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    interp = get_interpreter()
    if interp is None:
        return jsonify({"error": "model.tflite not found on server"}), 500

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr   = np.array(image).astype(np.float32)          # EfficientNet: [0,255] float32
    arr   = np.expand_dims(arr, axis=0)

    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()
    interp.set_tensor(in_d[0]["index"], arr)
    interp.invoke()
    preds = interp.get_tensor(out_d[0]["index"])[0]

    idx   = int(np.argmax(preds))
    label = CLASS_NAMES[idx]
    conf  = float(preds[idx])
    probs = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    return jsonify({"label": label, "confidence": conf, "probabilities": probs})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
