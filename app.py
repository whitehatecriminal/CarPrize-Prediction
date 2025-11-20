from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load model + columns
model = joblib.load("random_forest.pkl")
columns = joblib.load("columns.pkl")

@app.route("/")
def home():
    return "Car Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Convert year → Age
    Age = 2024 - int(data["year"])

    # Create a DataFrame from input
    df = pd.DataFrame([{
        "km_driven": data["km_driven"],
        "Age": Age,
        "fuel": data["fuel"],
        "seller_type": data["seller_type"],
        "transmission": data["transmission"],
        "owner": data["owner"]
    }])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Add missing columns
    for col in columns:
        if col not in df:
            df[col] = 0

    # Ensure correct order
    df = df[columns]

    # Predict
    pred = model.predict(df)[0]

    return jsonify({"predicted_price": float(pred)})

if __name__ == "__main__":
    app.run(debug=True)
