# app.py
# Flask Backend for Visual Question Answering (VQA)

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViltProcessor, ViltForQuestionAnswering
from pymongo import MongoClient
from PIL import Image
import io, base64, torch, os

# -------------------------
# 1. Setup Flask
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# 2. Setup MongoDB (Atlas + Local Fallback)
# -------------------------
# Try to read from environment variable (for Render)
mongo_uri = os.environ.get(
    "MONGO_URI", 
    "mongodb://localhost:27017/"   # fallback for local dev
)

client = MongoClient(mongo_uri)
db = client["vqa_db"]              # database name
collection = db["history"]         # collection name

# -------------------------
# 3. Load Model
# -------------------------
print("Loading VQA model... this may take 1-2 minutes...")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
print("Model loaded ✅")

# -------------------------
# 4. POST /vqa endpoint
# -------------------------
@app.route("/vqa", methods=["POST"])
def vqa():
    try:
        data = request.json
        image_base64 = data["image"]      # base64 encoded image
        question = data["question"]

        # Decode image
        img_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(img_data))

        # Run model
        inputs = processor(image, question, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        answer = model.config.id2label[outputs.logits.argmax(-1).item()]

        # Save to MongoDB
        entry = {
            "question": question,
            "answer": answer
        }
        collection.insert_one(entry)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# 5. GET /history endpoint
# -------------------------
@app.route("/history", methods=["GET"])
def history():
    try:
        history = list(collection.find({}, {"_id": 0}))
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run Server
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "VQA API is running ✅ Use /vqa for questions and /history to see past results."})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
