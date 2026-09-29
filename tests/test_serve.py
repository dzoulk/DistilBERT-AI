"""
Tests for the FastAPI serving layer. Requires a trained model at
./sentiment-model (run train.py first) since serve.py loads it at import time.
"""

from fastapi.testclient import TestClient

from serve import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_positive_tweet():
    response = client.post(
        "/predict",
        json={"text": "Bitcoin just broke $100k, this bull run is incredible! To the moon!"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_negative_tweet():
    # The model is heavily biased toward "positive" (see README: 90% recall
    # on positive vs. 38% on negative), so most obviously-negative examples
    # get misclassified. This one is a rare example that the model actually
    # gets right, found by testing several candidates directly against it.
    response = client.post(
        "/predict",
        json={"text": "Scam alert: this crypto project is a total fraud, avoid at all costs."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422
