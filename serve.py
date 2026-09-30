"""
Minimal FastAPI server exposing the fine-tuned unfair-ToS-clause classifier.

Run with:
    uvicorn serve:app --reload

Then test with:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "We may terminate your account at any time, for any reason, without notice."}'
"""

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import CATEGORIES

MODEL_DIR = "./sentiment-model"
MAX_LENGTH = 128  # must match train.py

# Each category gets an independent yes/no decision (see train.py) - this is
# the probability cutoff for "flagged". Lower it to flag more aggressively
# (higher recall, more false positives); raise it to flag fewer clauses
# (higher precision, more missed ones). No single "correct" value without a
# real downstream cost model for missed vs. over-flagged clauses.
FLAG_THRESHOLD = float(os.environ.get("FLAG_THRESHOLD", "0.5"))

app = FastAPI(title="Unfair ToS Clause Classifier")

# Load the model once at startup, not on every request — loading a
# transformer from disk is slow, so we keep it in memory.
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()  # inference mode: disables dropout etc.


class ClauseRequest(BaseModel):
    text: str


class CategoryScore(BaseModel):
    category: str
    probability: float
    flagged: bool


class PredictionResponse(BaseModel):
    is_unfair: bool
    categories: list[CategoryScore]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ClauseRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(device)

    with torch.no_grad():  # no need to track gradients for inference
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0]  # independent per-category probabilities

    categories = [
        CategoryScore(
            category=name,
            probability=round(float(probs[i].item()), 4),
            flagged=bool(probs[i].item() >= FLAG_THRESHOLD),
        )
        for i, name in enumerate(CATEGORIES)
    ]

    return PredictionResponse(
        is_unfair=any(c.flagged for c in categories),
        categories=categories,
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "flag_threshold": FLAG_THRESHOLD}
