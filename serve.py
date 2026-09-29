"""
Minimal FastAPI server exposing the fine-tuned crypto sentiment model.

Run with:
    uvicorn serve:app --reload

Then test with:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Bitcoin just broke $100k, this is huge!"}'
"""

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./sentiment-model"
MAX_LENGTH = 128  # must match train.py

app = FastAPI(title="Crypto Tweet Sentiment Classifier")

# Load the model once at startup, not on every request — loading a
# transformer from disk is slow, so we keep it in memory.
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()  # inference mode: disables dropout etc.

LABELS = {0: "negative", 1: "positive"}


class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(device)

    with torch.no_grad():  # no need to track gradients for inference
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_class = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_class].item())

    return PredictionResponse(
        label=LABELS[predicted_class],
        confidence=round(confidence, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": device}
