from flask import Flask,request,jsonify,send_file
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("../rent_model.pkl","rb"))

@app.route("/")
def home():
    return send_file("../index.html")

@app.route("/predict",methods=["POST"])
def predict():

    data=request.get_json()

    bhk=int(data["bhk"])
    size=int(data["size"])
    bathroom=int(data["bathroom"])
    city=int(data["city"])
    furnish=int(data["furnish"])

    features=np.array([[bhk,size,bathroom,city,furnish]])

    prediction=model.predict(features)

    return jsonify({"rent":int(prediction[0])})

if __name__=="__main__":
    app.run(debug=True)