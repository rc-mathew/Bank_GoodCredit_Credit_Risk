"""Feature drift monitoring using Population Stability Index (PSI)."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-6

STABLE_THRESHOLD = 0.10
MODERATE_THRESHOLD = 0.25


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI) for a numeric feature.

    Parameters
    ----------
    expected:
        Reference/training feature distribution.
    actual:
        Current/production feature distribution.
    bins:
        Number of quantile-based buckets.

    Returns
    -------
    float
        PSI value measuring distribution shift.
    """

    expected = pd.to_numeric(
        expected,
        errors="coerce",
    ).dropna()

    actual = pd.to_numeric(
        actual,
        errors="coerce",
    ).dropna()

    if expected.empty or actual.empty:
        return np.nan

    # Quantile-based bin boundaries derived only from
    # the reference distribution.
    breakpoints = np.unique(
        np.quantile(
            expected,
            np.linspace(0, 1, bins + 1),
        )
    )

    if len(breakpoints) < 2:
        return 0.0

    # Ensure production values outside the training range
    # are still captured.
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints,
    )

    expected_pct = (
        expected_counts / expected_counts.sum()
    )

    actual_pct = (
        actual_counts / actual_counts.sum()
    )

    expected_pct = np.clip(
        expected_pct,
        EPSILON,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        EPSILON,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def classify_drift(psi: float) -> str:
    """Convert a PSI value into an operational drift status."""

    if pd.isna(psi):
        return "UNAVAILABLE"

    if psi < STABLE_THRESHOLD:
        return "STABLE"

    if psi < MODERATE_THRESHOLD:
        return "MODERATE_DRIFT"

    return "SIGNIFICANT_DRIFT"


def monitor_feature_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    features: Iterable[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Calculate PSI across numeric model features.

    The reference dataset should represent the model's
    expected/training distribution. The current dataset
    represents newer production observations.
    """

    if features is None:
        features = reference_data.select_dtypes(
            include=np.number
        ).columns

    results: list[Dict[str, object]] = []

    for feature in features:

        if feature not in reference_data.columns:
            continue

        if feature not in current_data.columns:
            continue

        psi = calculate_psi(
            reference_data[feature],
            current_data[feature],
            bins=bins,
        )

        results.append(
            {
                "feature": feature,
                "psi": round(psi, 6)
                if not pd.isna(psi)
                else np.nan,
                "drift_status": classify_drift(psi),
            }
        )

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            by="psi",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return result_df