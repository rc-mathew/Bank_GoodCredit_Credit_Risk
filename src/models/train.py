"""Production model training pipeline for Bank GoodCredit."""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Bad_label"
ID_COLUMN = "customer_no"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def prepare_training_data(
    feature_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare stratified train and test datasets."""

    if TARGET_COLUMN not in feature_matrix.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    modelling_data = feature_matrix.copy()

    modelling_data = modelling_data.dropna(
        subset=[TARGET_COLUMN]
    )

    y = modelling_data[TARGET_COLUMN].astype(int)

    columns_to_drop = [
        column
        for column in [TARGET_COLUMN, ID_COLUMN]
        if column in modelling_data.columns
    ]

    X = modelling_data.drop(
        columns=columns_to_drop
    )

    # Remove completely empty columns.
    X = X.dropna(
        axis=1,
        how="all",
    )

    # Remove constant columns.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    X = X.drop(
        columns=constant_columns
    )

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_preprocessor(
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Build numeric and categorical preprocessing."""

    numeric_columns = (
        X_train
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def calculate_scale_pos_weight(
    y_train: pd.Series,
) -> float:
    """Calculate class weight for the minority bad-credit class."""

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive Bad_label cases."
        )

    return negative_count / positive_count


def build_model(
    scale_pos_weight: float,
) -> XGBClassifier:
    """Create the XGBoost credit-risk classifier."""

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )


def train_model(
    feature_matrix: pd.DataFrame,
):
    """Train the complete preprocessing + XGBoost pipeline."""

    X_train, X_test, y_train, y_test = (
        prepare_training_data(feature_matrix)
    )

    preprocessor = build_preprocessor(
        X_train
    )

    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    classifier = build_model(
        scale_pos_weight
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model_pipeline.fit(
        X_train,
        y_train,
    )

    return (
        model_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )


def save_model(
    model_pipeline: Pipeline,
    filename: str = "credit_risk_model.joblib",
) -> Path:
    """Persist the fitted production model."""

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = ARTIFACT_DIR / filename

    joblib.dump(
        model_pipeline,
        model_path,
    )

    return model_path