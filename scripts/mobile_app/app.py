from flask import Flask, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/heat_wave_model.pkl")

@app.route("/predict", methods=["GET"])
def predict():
    # Dummy data for prediction
    data = {
        "feature1": [value1],
        "feature2": [value2],
        # Add more features as required
    }
    
    df = pd.DataFrame(data)
    prediction = model.predict(df)[0]
    
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
