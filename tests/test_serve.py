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


def test_predict_positive_review():
    response = client.post(
        "/predict",
        json={"text": "This movie was absolutely wonderful, a true masterpiece."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_negative_review():
    response = client.post(
        "/predict",
        json={"text": "Terrible movie, a complete waste of time and money."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422
