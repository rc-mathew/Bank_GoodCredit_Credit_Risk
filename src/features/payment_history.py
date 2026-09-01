"""Payment-history feature engineering for Bank GoodCredit."""

from typing import List

import numpy as np


def parse_payment_history(history) -> List[float]:
    """
    Convert encoded payment history into monthly DPD values.

    Rules
    -----
    STD -> 0 days past due
    Numeric tokens -> corresponding DPD
    Unknown/non-numeric tokens -> NaN
    """

    if history is None:
        return []

    if isinstance(history, float) and np.isnan(history):
        return []

    history = str(history).strip().replace('"', "")

    if not history:
        return []

    tokens = [
        history[i:i + 3]
        for i in range(0, len(history), 3)
        if len(history[i:i + 3]) == 3
    ]

    dpd_values = []

    for token in tokens:
        token = token.strip().upper()

        if token == "STD":
            dpd_values.append(0.0)

        elif token.isdigit():
            dpd_values.append(float(token))

        else:
            dpd_values.append(np.nan)

    return dpd_values


def combine_payment_histories(
    history_1,
    history_2,
) -> List[float]:
    """Combine the two available payment-history fields."""

    return (
        parse_payment_history(history_1)
        + parse_payment_history(history_2)
    )


def calculate_payment_features(
    history_1,
    history_2,
) -> dict:
    """Calculate account-level delinquency features."""

    values = combine_payment_histories(
        history_1,
        history_2,
    )

    valid = np.asarray(
        [value for value in values if not np.isnan(value)],
        dtype=float,
    )

    if valid.size == 0:
        return {
            "payment_history_length": 0,
            "payment_history_mean_dpd": np.nan,
            "payment_history_max_dpd": np.nan,
            "dpd_0_29_share": np.nan,
            "dpd_30_59_share": np.nan,
            "dpd_60_89_share": np.nan,
            "dpd_90_plus_share": np.nan,
            "has_30_plus_dpd": 0,
            "has_90_plus_dpd": 0,
        }

    return {
        "payment_history_length": int(valid.size),

        "payment_history_mean_dpd": float(
            np.mean(valid)
        ),

        "payment_history_max_dpd": float(
            np.max(valid)
        ),

        "dpd_0_29_share": float(
            np.mean((valid >= 0) & (valid <= 29))
        ),

        "dpd_30_59_share": float(
            np.mean((valid >= 30) & (valid <= 59))
        ),

        "dpd_60_89_share": float(
            np.mean((valid >= 60) & (valid <= 89))
        ),

        "dpd_90_plus_share": float(
            np.mean(valid >= 90)
        ),

        "has_30_plus_dpd": int(
            np.any(valid >= 30)
        ),

        "has_90_plus_dpd": int(
            np.any(valid >= 90)
        ),
    }