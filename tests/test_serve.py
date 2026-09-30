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


def test_predict_flags_arbitration_clause():
    response = client.post(
        "/predict",
        json={
            "text": (
                "You agree that any dispute arising out of these terms will be "
                "resolved through binding arbitration, and you waive any right "
                "to a jury trial or to participate in a class action."
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_unfair"] is True
    flagged = {c["category"] for c in body["categories"] if c["flagged"]}
    assert "Arbitration" in flagged


def test_predict_does_not_flag_fair_clause():
    response = client.post(
        "/predict",
        json={"text": "You may cancel your subscription at any time from your account settings."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["is_unfair"] is False


def test_predict_returns_all_categories():
    response = client.post("/predict", json={"text": "This is a short clause."})
    assert response.status_code == 200
    body = response.json()
    assert len(body["categories"]) == 8


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422
