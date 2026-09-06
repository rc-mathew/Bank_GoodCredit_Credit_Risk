"""Tests for PSI-based feature drift monitoring."""

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    calculate_psi,
    classify_drift,
    monitor_feature_drift,
)


def test_identical_distributions_have_low_psi():
    """Identical distributions should produce essentially zero drift."""

    reference = pd.Series(np.arange(1, 101))
    current = reference.copy()

    psi = calculate_psi(reference, current)

    assert psi < 0.10
    assert classify_drift(psi) == "STABLE"


def test_shifted_distribution_detects_drift():
    """A strongly shifted population should trigger drift."""

    rng = np.random.default_rng(42)

    reference = pd.Series(
        rng.normal(loc=0.0, scale=1.0, size=5000)
    )

    current = pd.Series(
        rng.normal(loc=2.0, scale=1.0, size=5000)
    )

    psi = calculate_psi(reference, current)

    assert psi >= 0.25
    assert classify_drift(psi) == "SIGNIFICANT_DRIFT"


def test_drift_threshold_classification():
    """PSI thresholds should map to the expected monitoring status."""

    assert classify_drift(0.05) == "STABLE"
    assert classify_drift(0.15) == "MODERATE_DRIFT"
    assert classify_drift(0.30) == "SIGNIFICANT_DRIFT"


def test_monitor_multiple_features():
    """Monitoring should return PSI results for multiple features."""

    rng = np.random.default_rng(42)

    reference = pd.DataFrame(
        {
            "credit_amount": rng.normal(5000, 1000, 2000),
            "account_age": rng.normal(36, 8, 2000),
        }
    )

    current = pd.DataFrame(
        {
            "credit_amount": rng.normal(7500, 1000, 2000),
            "account_age": rng.normal(36, 8, 2000),
        }
    )

    report = monitor_feature_drift(reference, current)

    assert set(report.columns) == {
        "feature",
        "psi",
        "drift_status",
    }

    assert len(report) == 2

    credit_result = report[
        report["feature"] == "credit_amount"
    ].iloc[0]

    assert credit_result["psi"] >= 0.25
    assert (
        credit_result["drift_status"]
        == "SIGNIFICANT_DRIFT"
    )