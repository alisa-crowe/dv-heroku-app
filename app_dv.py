from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained model pipeline
model_path = os.path.join(os.path.dirname(__file__), 'domestic_violence_model_pipeline.pkl')
try:
    model_pipeline = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{model_path}' not found. Ensure it exists in the correct location.")

# Define mappings for categorical fields (ensure consistency with training data)
race_map = {
    'HISPANIC': 0,
    'WHITE': 1,
    'OTHER': 2,
    'BLACK': 3,
    'ASIAN': 4,
    'UNKNOWN': 5,
    'PACIFIC ISLANDER': 6,
    'AMERICAN INDIAN': 7
}

day_of_week_map = {
    'MONDAY': 0,
    'TUESDAY': 1,
    'WEDNESDAY': 2,
    'THURSDAY': 3,
    'FRIDAY': 4,
    'SATURDAY': 5,
    'SUNDAY': 6
}

@app.route('/')
def home():
    return "Domestic Violence Predictor is Running!"

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        # Parse input JSON
        data = request.get_json(force=True)

        # Map categorical values to their numeric encodings
        data['Overall Race'] = race_map.get(data['Overall Race'].upper(), -1)
        data['Day of Week'] = day_of_week_map.get(data['Day of Week'].upper(), -1)

        # Validate input
        if -1 in (data['Overall Race'], data['Day of Week']):
            return jsonify({'error': 'Invalid categorical input values'}), 400

        # Create a DataFrame for the input
        input_data = pd.DataFrame([data])

        # Predict using the pipeline
        prob = model_pipeline.predict_proba(input_data)[:, 1][0]
        threshold = 0.4
        prediction = int(prob >= threshold)

        # Return the prediction and probability
        return jsonify({
            'prediction': prediction,
            'probability': round(prob, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Heroku-assigned port or default to 5000
    app.run(host='0.0.0.0', port=port)
