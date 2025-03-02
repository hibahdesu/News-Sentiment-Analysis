from flask import Flask, request, jsonify
import joblib
import numpy as np
import sys
import os

# Add the 'utils' module to sys.path if it's located in the parent directory (modify the path as needed)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))  # Modify the path as necessary

app = Flask(__name__)

# Load the saved model and vectorizer
try:
    model = joblib.load('saved_models/logistic_model.pkl')  # Ensure the correct path to your model
    vectorizer = joblib.load('saved_models/vectorizer.pkl')  # Ensure the correct path to your vectorizer
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model and vectorizer are loaded
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model or vectorizer not loaded properly.'}), 500

        # Get input from the request
        data = request.get_json()

        # Check if 'news' field is provided
        if 'news' not in data:
            return jsonify({'error': 'No news text provided in the request.'}), 400

        news = data['news']

        # Vectorize the input news using the saved vectorizer
        news_vectorized = vectorizer.transform([news])

        # Predict the sentiment using the saved model
        prediction = model.predict(news_vectorized)

        # Map sentiment prediction to human-readable form
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

        # Return the result as a JSON response
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
