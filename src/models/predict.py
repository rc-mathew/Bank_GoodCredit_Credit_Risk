"""Production inference utilities for the Bank GoodCredit model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "credit_risk_model.joblib"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_feature_matrix.csv"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

PREDICTIONS_FILE = (
    PREDICTIONS_DIR
    / "customer_risk_predictions.csv"
)

TARGET_COLUMN = "Bad_label"
ID_COLUMN = "customer_no"


def load_model():
    """Load the persisted credit-risk model pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def assign_risk_band(
    probability: float,
) -> str:
    """
    Convert predicted bad-credit probability into a risk band.

    These thresholds are illustrative portfolio thresholds.
    They are not approved lending-policy cutoffs.
    """

    if probability >= 0.20:
        return "Very High"

    if probability >= 0.10:
        return "High"

    if probability >= 0.05:
        return "Medium"

    return "Low"


def prepare_prediction_features(
    model,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare incoming customer features for model inference.

    Removes target and ID columns and aligns columns with those
    used when the production pipeline was fitted.
    """

    prediction_data = data.copy()

    columns_to_drop = [
        column
        for column in [
            TARGET_COLUMN,
            ID_COLUMN,
        ]
        if column in prediction_data.columns
    ]

    features = prediction_data.drop(
        columns=columns_to_drop
    )

    if hasattr(model, "feature_names_in_"):

        expected_columns = list(
            model.feature_names_in_
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in features.columns
        ]

        for column in missing_columns:
            features[column] = np.nan

        extra_columns = [
            column
            for column in features.columns
            if column not in expected_columns
        ]

        if extra_columns:
            features = features.drop(
                columns=extra_columns
            )

        features = features[
            expected_columns
        ]

    return features


def predict_risk(
    model,
    customer_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate bad-credit probabilities and risk bands.
    """

    features = prepare_prediction_features(
        model,
        customer_data,
    )

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = pd.DataFrame(
        {
            "bad_probability": probabilities,
        },
        index=customer_data.index,
    )

    if ID_COLUMN in customer_data.columns:
        predictions.insert(
            0,
            ID_COLUMN,
            customer_data[ID_COLUMN].values,
        )

    predictions["risk_band"] = (
        predictions["bad_probability"]
        .apply(assign_risk_band)
    )

    return predictions


def main() -> None:
    """Run batch inference on the processed customer dataset."""

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{PROCESSED_DATA_PATH}"
        )

    print("Loading production model...")

    model = load_model()

    print("Loading customer feature matrix...")

    customer_data = pd.read_csv(
        PROCESSED_DATA_PATH,
        low_memory=False,
    )

    print(
        f"Customers loaded: "
        f"{len(customer_data):,}"
    )

    print("Generating risk predictions...")

    predictions = predict_risk(
        model,
        customer_data,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Predictions generated: "
        f"{len(predictions):,}"
    )

    print("\nRisk-band distribution:")

    print(
        predictions["risk_band"]
        .value_counts()
    )

    print(
        f"\nPredictions saved to: "
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()