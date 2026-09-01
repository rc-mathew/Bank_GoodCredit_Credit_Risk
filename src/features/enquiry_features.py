"""Credit-enquiry feature engineering for Bank GoodCredit."""

import numpy as np
import pandas as pd


def create_enquiry_features(
    enquiries: pd.DataFrame,
    reference_date=None,
) -> pd.DataFrame:
    """
    Aggregate credit-enquiry records to customer level.

    Creates enquiry-volume, recency, amount and purpose features.
    """

    df = enquiries.copy()

    # -----------------------------
    # Date preparation
    # -----------------------------
    if "dt_enquiry" in df.columns:
        df["dt_enquiry"] = pd.to_datetime(
            df["dt_enquiry"],
            errors="coerce",
        )

    # Use supplied reference date when available.
    # Otherwise use the latest enquiry date in the dataset.
    if reference_date is None:
        if "dt_enquiry" in df.columns:
            reference_date = df["dt_enquiry"].max()
    else:
        reference_date = pd.to_datetime(reference_date)

    # -----------------------------
    # Amount preparation
    # -----------------------------
    if "enq_amt" in df.columns:
        df["enq_amt"] = pd.to_numeric(
            df["enq_amt"],
            errors="coerce",
        )

    # -----------------------------
    # Basic enquiry count
    # -----------------------------
    features = (
        df.groupby("customer_no")
        .size()
        .rename("enquiry_count")
        .reset_index()
    )

    # -----------------------------
    # Enquiry amount features
    # -----------------------------
    if "enq_amt" in df.columns:

        amount_features = (
            df.groupby("customer_no")["enq_amt"]
            .agg(
                [
                    "sum",
                    "mean",
                    "max",
                    "min",
                ]
            )
            .reset_index()
            .rename(
                columns={
                    "sum": "enquiry_amount_sum",
                    "mean": "enquiry_amount_mean",
                    "max": "enquiry_amount_max",
                    "min": "enquiry_amount_min",
                }
            )
        )

        features = features.merge(
            amount_features,
            on="customer_no",
            how="left",
        )

    # -----------------------------
    # Enquiry-purpose features
    # -----------------------------
    if "enq_purpose" in df.columns:

        purpose_count = (
            df.groupby("customer_no")["enq_purpose"]
            .nunique()
            .rename("enquiry_purpose_nunique")
            .reset_index()
        )

        features = features.merge(
            purpose_count,
            on="customer_no",
            how="left",
        )

        purpose_frequency = (
            df.groupby(
                ["customer_no", "enq_purpose"]
            )
            .size()
            .rename("purpose_frequency")
            .reset_index()
        )

        max_frequency = (
            purpose_frequency
            .groupby("customer_no")[
                "purpose_frequency"
            ]
            .max()
            .rename("max_freq_enquiry")
            .reset_index()
        )

        features = features.merge(
            max_frequency,
            on="customer_no",
            how="left",
        )

    # -----------------------------
    # Recency features
    # -----------------------------
    if (
        "dt_enquiry" in df.columns
        and pd.notna(reference_date)
    ):

        df["enquiry_age_days"] = (
            reference_date - df["dt_enquiry"]
        ).dt.days

        for days in [30, 90, 180, 365]:

            recent = (
                df["enquiry_age_days"]
                .between(0, days)
                .astype(int)
            )

            column_name = (
                f"count_enquiry_recency_{days}"
            )

            recent_counts = (
                recent.groupby(df["customer_no"])
                .sum()
                .rename(column_name)
                .reset_index()
            )

            features = features.merge(
                recent_counts,
                on="customer_no",
                how="left",
            )

        # Days since most recent enquiry.
        last_enquiry = (
            df.groupby("customer_no")[
                "enquiry_age_days"
            ]
            .min()
            .rename("days_since_last_enquiry")
            .reset_index()
        )

        features = features.merge(
            last_enquiry,
            on="customer_no",
            how="left",
        )

    # Replace infinite numeric values.
    numeric_columns = features.select_dtypes(
        include=[np.number]
    ).columns

    features[numeric_columns] = (
        features[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    return features