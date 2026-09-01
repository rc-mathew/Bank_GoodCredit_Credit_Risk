"""Data validation utilities for the Bank GoodCredit project."""

from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "demographics": [
        "customer_no",
        "Bad_label",
    ],
    "accounts": [
        "customer_no",
    ],
    "enquiries": [
        "customer_no",
    ],
}


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    dataset_name: str,
) -> None:
    """Check whether required columns are present."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def validate_customer_key(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Validate customer_no values."""

    if "customer_no" not in df.columns:
        raise ValueError(
            f"{dataset_name} does not contain customer_no."
        )

    missing_customer_ids = df["customer_no"].isna().sum()

    if missing_customer_ids > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{missing_customer_ids} missing customer_no values."
        )


def validate_target(
    demographics: pd.DataFrame,
) -> None:
    """Validate the Bad_label target."""

    if "Bad_label" not in demographics.columns:
        raise ValueError(
            "Demographics dataset does not contain Bad_label."
        )

    target_values = set(
        demographics["Bad_label"]
        .dropna()
        .unique()
    )

    valid_values = {0, 1}

    if not target_values.issubset(valid_values):
        raise ValueError(
            f"Bad_label contains unexpected values: {target_values}"
        )


def duplicate_summary(
    df: pd.DataFrame,
) -> Dict[str, int]:
    """Return basic duplicate and customer statistics."""

    return {
        "rows": len(df),
        "exact_duplicates": int(df.duplicated().sum()),
        "unique_customers": (
            int(df["customer_no"].nunique())
            if "customer_no" in df.columns
            else 0
        ),
    }


def validate_all(
    demographics: pd.DataFrame,
    accounts: pd.DataFrame,
    enquiries: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Run validation checks for all source datasets."""

    datasets = {
        "demographics": demographics,
        "accounts": accounts,
        "enquiries": enquiries,
    }

    for dataset_name, dataframe in datasets.items():
        validate_required_columns(
            dataframe,
            REQUIRED_COLUMNS[dataset_name],
            dataset_name,
        )

        validate_customer_key(
            dataframe,
            dataset_name,
        )

    validate_target(demographics)

    summary = {
        name: duplicate_summary(dataframe)
        for name, dataframe in datasets.items()
    }

    return summary