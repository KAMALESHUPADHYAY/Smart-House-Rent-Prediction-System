from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "rent_model.pkl")
model = pickle.load(open(model_path, "rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    bhk = int(data["bhk"])
    size = int(data["size"])
    bathroom = int(data["bathroom"])
    city = int(data["city"])
    furnish = int(data["furnish"])

    features = np.array([[bhk, size, bathroom, city, furnish]])

    prediction = model.predict(features)

    return jsonify({"rent": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1000, debug=True)