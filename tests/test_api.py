"""API tests for the Bank GoodCredit credit-risk service."""

import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    """Health endpoint should return HTTP 200."""

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_root_endpoint():
    """Root endpoint should expose service metadata."""

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == (
        "Bank GoodCredit Credit Risk API"
    )

    assert body["version"] == "1.0.0"


def test_prediction_endpoint():
    """Prediction endpoint should return a valid probability."""

    data = pd.read_csv(
        "data/processed/customer_feature_matrix.csv",
        low_memory=False,
        nrows=1,
    )

    row = data.iloc[0].drop(
        labels=[
            "Bad_label",
            "customer_no",
        ],
        errors="ignore",
    )

    row = row.where(
        pd.notna(row),
        None,
    )

    payload = {
        "features": row.to_dict()
    }

    with TestClient(app) as test_client:
        response = test_client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 200

    body = response.json()

    assert "bad_probability" in body
    assert "risk_band" in body
    assert "model_version" in body

    assert 0.0 <= body["bad_probability"] <= 1.0

    assert body["risk_band"] in {
        "Low",
        "Medium",
        "High",
        "Very High",
    }


def test_empty_prediction_payload():
    """Empty features should return HTTP 400."""

    with TestClient(app) as test_client:
        response = test_client.post(
            "/predict",
            json={
                "features": {}
            },
        )

    assert response.status_code == 400