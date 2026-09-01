"""Account-level feature engineering for Bank GoodCredit."""

import numpy as np
import pandas as pd

from src.features.payment_history import calculate_payment_features


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Safely divide two pandas Series."""

    denominator = denominator.replace(0, np.nan)

    return numerator / denominator


def add_payment_history_features(
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Create payment-history features for each account."""

    df = accounts.copy()

    payment_features = df.apply(
        lambda row: calculate_payment_features(
            row.get("paymenthistory1"),
            row.get("paymenthistory2"),
        ),
        axis=1,
    )

    payment_features = pd.DataFrame(
        payment_features.tolist(),
        index=df.index,
    )

    return pd.concat(
        [df, payment_features],
        axis=1,
    )


def create_account_features(
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate account records to one row per customer.

    Returns customer-level credit-risk features.
    """

    df = add_payment_history_features(accounts)

    # Convert important numeric columns safely.
    numeric_columns = [
        "current_balance",
        "credit_limit",
        "amount_past_due",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Account-level credit utilisation.
    if {
        "current_balance",
        "credit_limit",
    }.issubset(df.columns):

        df["account_utilisation"] = safe_divide(
            df["current_balance"],
            df["credit_limit"],
        )

    # Account-level past-due ratio.
    if {
        "amount_past_due",
        "current_balance",
    }.issubset(df.columns):

        df["past_due_balance_ratio"] = safe_divide(
            df["amount_past_due"],
            df["current_balance"].abs(),
        )

    aggregation = {
        "customer_no": "size",
    }

    optional_aggregations = {
        "current_balance": ["sum", "mean", "max"],
        "credit_limit": ["sum", "mean", "max"],
        "amount_past_due": ["sum", "mean", "max"],
        "account_utilisation": ["mean", "max"],
        "past_due_balance_ratio": ["mean", "max"],
        "payment_history_length": ["mean", "max"],
        "payment_history_mean_dpd": ["mean", "max"],
        "payment_history_max_dpd": ["mean", "max"],
        "dpd_0_29_share": ["mean"],
        "dpd_30_59_share": ["mean"],
        "dpd_60_89_share": ["mean"],
        "dpd_90_plus_share": ["mean"],
        "has_30_plus_dpd": ["sum", "mean"],
        "has_90_plus_dpd": ["sum", "mean"],
    }

    available_aggregations = {
        column: functions
        for column, functions
        in optional_aggregations.items()
        if column in df.columns
    }

    customer_features = (
        df.groupby("customer_no")
        .agg(available_aggregations)
    )

    # Flatten MultiIndex column names.
    customer_features.columns = [
        f"{column}_{statistic}"
        for column, statistic
        in customer_features.columns
    ]

    customer_features = (
        customer_features
        .reset_index()
    )

    # Number of accounts per customer.
    account_counts = (
        df.groupby("customer_no")
        .size()
        .rename("account_count")
        .reset_index()
    )

    customer_features = customer_features.merge(
        account_counts,
        on="customer_no",
        how="left",
    )

    # Portfolio-level utilisation.
    required_columns = {
        "current_balance_sum",
        "credit_limit_sum",
    }

    if required_columns.issubset(
        customer_features.columns
    ):
        customer_features[
            "Ratio_currbalance_creditlimit"
        ] = safe_divide(
            customer_features["current_balance_sum"],
            customer_features["credit_limit_sum"],
        )

    return customer_features