from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://sites.google.com"}})

# Load the trained model and preprocessing pipeline
model_path = os.path.join(os.path.dirname(__file__), 'domestic_violence_model_pipeline.pkl')
try:
    model_pipeline = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{model_path}' not found. Ensure it exists in the correct location.")

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

        # Create a DataFrame for the input
        input_data = pd.DataFrame([data])

        # Predict the probability of domestic violence using the entire pipeline
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
