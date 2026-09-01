"""FastAPI service for Bank GoodCredit credit-risk inference."""

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import (
    MODEL_PATH,
    assign_risk_band,
    load_model,
    prepare_prediction_features,
)


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the API starts."""

    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    model = load_model()

    print("Credit-risk model loaded successfully.")

    yield

    model = None


app = FastAPI(
    title="Bank GoodCredit Credit Risk API",
    description=(
        "Production-style API for predicting the probability "
        "of bad credit using the Bank GoodCredit XGBoost model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Customer features supplied to the prediction endpoint."""

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Customer-level engineered features used by "
            "the credit-risk model."
        ),
    )


class PredictionResponse(BaseModel):
    """Credit-risk prediction returned by the API."""

    bad_probability: float
    risk_band: str
    model_version: str = "1.0.0"


@app.get("/")
def root() -> dict:
    """API root endpoint."""

    return {
        "service": "Bank GoodCredit Credit Risk API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict:
    """Return API and model health status."""

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_credit_risk(
    request: PredictionRequest,
) -> PredictionResponse:
    """Predict bad-credit probability for one customer."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Credit-risk model is not loaded.",
        )

    if not request.features:
        raise HTTPException(
            status_code=400,
            detail="No customer features were supplied.",
        )

    try:
        customer = pd.DataFrame(
            [request.features]
        )

        features = prepare_prediction_features(
            model,
            customer,
        )

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        risk_band = assign_risk_band(
            probability
        )

        return PredictionResponse(
            bad_probability=round(
                probability,
                6,
            ),
            risk_band=risk_band,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc