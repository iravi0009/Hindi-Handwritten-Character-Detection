from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
from utils.segment import segment_characters
from utils.predict import predict_character

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("model/hindi_cnn_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        processed = preprocess_image(img)
        characters = segment_characters(processed)

        for char in characters:
            extracted_text += predict_character(char, model)

    return render_template("index.html", text=extracted_text)

if __name__ == "__main__":
    app.run(debug=True)

