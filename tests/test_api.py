"""API tests for the Bank GoodCredit credit-risk service."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main


class DummyModel:
    """Small deterministic model used only for API tests."""

    def predict_proba(self, features):
        """Return a fixed valid probability distribution."""

        rows = len(features)

        return np.tile(
            np.array([[0.60, 0.40]]),
            (rows, 1),
        )


@pytest.fixture
def test_client(monkeypatch, tmp_path):
    """Create an API client without requiring production artifacts."""

    fake_model_path = tmp_path / "credit_risk_model.joblib"
    fake_model_path.touch()

    monkeypatch.setattr(
        api_main,
        "MODEL_PATH",
        fake_model_path,
    )

    monkeypatch.setattr(
        api_main,
        "load_model",
        lambda: DummyModel(),
    )

    monkeypatch.setattr(
        api_main,
        "prepare_prediction_features",
        lambda model, customer: customer,
    )

    with TestClient(api_main.app) as client:
        yield client


def test_health_endpoint(test_client):
    """Health endpoint should return HTTP 200."""

    response = test_client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_root_endpoint(test_client):
    """Root endpoint should expose service metadata."""

    response = test_client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == (
        "Bank GoodCredit Credit Risk API"
    )

    assert body["version"] == "1.0.0"


def test_prediction_endpoint(test_client):
    """Prediction endpoint should return a valid probability."""

    payload = {
        "features": {
            "feature_1": 1.0,
            "feature_2": 2.0,
            "feature_3": 0.5,
        }
    }

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


def test_empty_prediction_payload(test_client):
    """Empty features should return HTTP 400."""

    response = test_client.post(
        "/predict",
        json={
            "features": {}
        },
    )

    assert response.status_code == 400