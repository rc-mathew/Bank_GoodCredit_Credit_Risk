"""Build the customer-level modelling feature matrix."""

import numpy as np
import pandas as pd

from src.features.account_features import create_account_features
from src.features.enquiry_features import create_enquiry_features


def prepare_demographics(
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the customer-level demographics table."""

    df = demographics.copy()

    # Demographics should contain one row per customer.
    df = df.drop_duplicates(
        subset=["customer_no"],
        keep="first",
    )

    # Convert infinite values to missing values.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


def build_feature_matrix(
    demographics: pd.DataFrame,
    accounts: pd.DataFrame,
    enquiries: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one modelling row per customer.

    Demographics forms the base population.
    Account and enquiry features are joined using customer_no.
    """

    demographic_features = prepare_demographics(
        demographics
    )

    account_features = create_account_features(
        accounts
    )

    enquiry_features = create_enquiry_features(
        enquiries
    )

    feature_matrix = demographic_features.merge(
        account_features,
        on="customer_no",
        how="left",
        validate="one_to_one",
    )

    feature_matrix = feature_matrix.merge(
        enquiry_features,
        on="customer_no",
        how="left",
        validate="one_to_one",
    )

    # Customers with no account/enquiry observations may
    # legitimately have missing engineered features.
    count_columns = [
        column
        for column in feature_matrix.columns
        if (
            column.endswith("_count")
            or column.startswith("count_enquiry_")
        )
    ]

    if count_columns:
        feature_matrix[count_columns] = (
            feature_matrix[count_columns]
            .fillna(0)
        )

    return feature_matrix


def feature_matrix_summary(
    feature_matrix: pd.DataFrame,
) -> dict:
    """Return key modelling-dataset statistics."""

    summary = {
        "rows": int(feature_matrix.shape[0]),
        "columns": int(feature_matrix.shape[1]),
        "unique_customers": int(
            feature_matrix["customer_no"].nunique()
        ),
        "duplicate_customers": int(
            feature_matrix["customer_no"].duplicated().sum()
        ),
        "missing_values": int(
            feature_matrix.isna().sum().sum()
        ),
    }

    if "Bad_label" in feature_matrix.columns:
        summary["bad_customers"] = int(
            (feature_matrix["Bad_label"] == 1).sum()
        )

        summary["good_customers"] = int(
            (feature_matrix["Bad_label"] == 0).sum()
        )

        summary["bad_rate"] = float(
            feature_matrix["Bad_label"].mean()
        )

    return summary