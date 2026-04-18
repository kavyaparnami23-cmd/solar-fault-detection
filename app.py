from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/model.tflite")
def model():
    return send_from_directory(".", "model.tflite")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
