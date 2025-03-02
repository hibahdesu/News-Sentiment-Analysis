import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    "news": "The stock market is performing very well today!"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Sentiment: {result['sentiment']}")
else:
    print("Error:", response.json())
