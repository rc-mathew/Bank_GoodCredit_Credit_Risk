import os
import json
import joblib
import pandas as pd

from flask import Flask, request, jsonify


app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/opt/ml/model/credit_risk_model.joblib"
)

model = None
model_error = None


def load_model():
    global model, model_error

    try:
        model = joblib.load(MODEL_PATH)
        model_error = None
        print(f"Model loaded successfully from {MODEL_PATH}")

    except Exception as exc:
        model = None
        model_error = str(exc)
        print(f"Model loading failed: {exc}")


load_model()


@app.route("/ping", methods=["GET"])
def ping():
    """
    SageMaker health-check endpoint.
    """

    if model is None:
        return jsonify(
            {
                "status": "unhealthy",
                "model_loaded": False,
                "error": model_error,
            }
        ), 500

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
        }
    ), 200


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    SageMaker inference endpoint.
    Expects JSON containing either:

    {"instances": [{...}]}

    or a single feature dictionary:

    {...}
    """

    if model is None:
        return jsonify(
            {
                "error": "Model is not loaded.",
                "details": model_error,
            }
        ), 500

    try:
        payload = request.get_json(force=True)

        if isinstance(payload, dict) and "instances" in payload:
            records = payload["instances"]

        elif isinstance(payload, dict):
            records = [payload]

        elif isinstance(payload, list):
            records = payload

        else:
            return jsonify(
                {"error": "Invalid JSON payload."}
            ), 400

        features = pd.DataFrame(records)

        probabilities = model.predict_proba(features)[:, 1]

        predictions = []

        for probability in probabilities:

            probability = float(probability)

            if probability < 0.10:
                risk_band = "LOW"
            elif probability < 0.25:
                risk_band = "MEDIUM"
            else:
                risk_band = "HIGH"

            predictions.append(
                {
                    "bad_probability": probability,
                    "risk_band": risk_band,
                }
            )

        return jsonify(
            {
                "predictions": predictions
            }
        ), 200

    except Exception as exc:

        return jsonify(
            {
                "error": "Inference failed.",
                "details": str(exc),
            }
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )