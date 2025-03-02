from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('saved_models/logistic_model.pkl')  # Ensure correct path to your model
vectorizer = joblib.load('saved_models/vectorizer.pkl')  # Ensure correct path to your vectorizer

@app.route('/')
def index():
    # Serve the HTML form to the user
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the request
        news = request.form['news']

        # Vectorize the input news using the saved vectorizer
        news_vectorized = vectorizer.transform([news])

        # Predict the sentiment using the saved model
        prediction = model.predict(news_vectorized)
        
        # Map sentiment prediction to human-readable form
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Return the result as a JSON response
        return render_template('index.html', sentiment=sentiment, news=news)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
