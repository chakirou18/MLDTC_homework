
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
# Load the model
model_file = 'model_rfr'

with open(model_file, 'rb') as f_in:
    rf_loaded = pickle.load(f_in)

# Create Flask application
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    # Get house data
    house = request.get_json()

    # Convert dictionary to DataFrame
    house_df = pd.DataFrame([house])

    # Extract hour from date
    date = pd.to_datetime(house_df['date'])
    hour = date.dt.hour + date.dt.minute / 60

    # Create hour features
    house_df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    house_df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Select model features
    X_house = house_df[rf_loaded.feature_names_in_]

    # Make prediction
    prediction = rf_loaded.predict(X_house)

    # Prepare result
    result = {
        'Energy': float(prediction[0])
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
