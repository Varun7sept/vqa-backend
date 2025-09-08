import requests
import base64

# API endpoint
url = "http://127.0.0.1:5000/vqa"

# Convert local image to base64
with open("virat.jpg", "rb") as f:   # 👈 make sure football.jpg is in the same folder
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

# Prepare payload
data = {
    "image": img_base64,
    "question": "What is the boy holding?"
}

# Send POST request
response = requests.post(url, json=data)

print("Response from API:", response.json())
