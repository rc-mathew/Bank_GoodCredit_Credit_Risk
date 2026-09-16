import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    """Load the trained credit-risk model."""
    model_path = os.path.join(
        model_dir,
        "credit_risk_model.joblib",
    )
    return joblib.load(model_path)


def input_fn(request_body, request_content_type):
    """Convert incoming JSON into a DataFrame."""

    if request_content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {request_content_type}"
        )

    payload = json.loads(request_body)

    features = payload.get("features", payload)

    if not isinstance(features, dict) or not features:
        raise ValueError(
            "Request must contain a non-empty feature dictionary."
        )

    return pd.DataFrame([features])


def predict_fn(input_data, model):
    """Generate bad-credit probability."""

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    if probability < 0.20:
        risk_band = "Low"
    elif probability < 0.40:
        risk_band = "Medium"
    elif probability < 0.60:
        risk_band = "High"
    else:
        risk_band = "Very High"

    return {
        "bad_probability": round(probability, 6),
        "risk_band": risk_band,
        "model_version": "1.0.0",
    }


def output_fn(prediction, accept):
    """Serialize prediction as JSON."""

    if accept not in ("application/json", "*/*"):
        raise ValueError(
            f"Unsupported accept type: {accept}"
        )

    return json.dumps(prediction)
