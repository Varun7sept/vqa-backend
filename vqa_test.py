# vqa_test.py
# Visual Question Answering with Local Image Support

from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# -------------------------
# 1. Load the Model
# -------------------------
print("Loading VQA model... please wait (first time may take 1-2 mins)...")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
print("Model loaded successfully ✅")

# -------------------------
# 2. Ask user for local image
# -------------------------
image_path = input("\nEnter the path of your image (example: C:/Users/Varun Banda/Visual Question Answering (VQA)/football.jpg): ").strip()

try:
    image = Image.open(image_path)
except Exception as e:
    print(f"❌ Could not open image. Error: {e}")
    exit()

# -------------------------
# 3. Ask user for question
# -------------------------
question = input("Enter your question about the image: ")

# -------------------------
# 4. Run the model
# -------------------------
inputs = processor(image, question, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# -------------------------
# 5. Get the answer
# -------------------------
answer = model.config.id2label[outputs.logits.argmax(-1).item()]

print(f"\n✅ Question: {question}")
print(f"🤖 Answer: {answer}")
