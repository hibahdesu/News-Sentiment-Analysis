import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "news": [
        {"title": "Apple unveils new iPhone with groundbreaking features", "publishedAt": "2025-03-01T12:00:00Z"},
        {"title": "Tesla's new battery technology aims to revolutionize EVs", "publishedAt": "2025-03-01T14:00:00Z"}
    ]
}

response = requests.post(url, json=data)

print(response.json())  # Check the response
