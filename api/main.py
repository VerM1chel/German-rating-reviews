from core.clean import clean_text

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    rating: int
    confidence: float
    probabilities: dict[int, float]

MODEL_NAME = "VerMichel/german-reviews-distilbert"

print("Model loading")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded")

app = FastAPI(title="German Reviews API")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    cleaned_text = clean_text(request.text)
    with torch.no_grad():
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]
        predicted_class_idx = torch.argmax(probs).item()
        rating = predicted_class_idx + 1
        probabilities = {i+1: float(probs[i]) for i in range(6)}
    return {
        "rating": rating,
        "confidence": probabilities[rating],
        "probabilities": probabilities,
    }