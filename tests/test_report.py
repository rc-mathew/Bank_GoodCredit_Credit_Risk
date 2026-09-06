"""Tests for production monitoring report generation."""

import json

import pandas as pd

from src.monitoring.report import (
    build_monitoring_report,
    determine_model_health,
    save_monitoring_report,
)


def test_healthy_model_status():
    """Stable features should produce a HEALTHY model status."""

    drift_report = pd.DataFrame(
        {
            "feature": ["credit_amount", "age"],
            "psi": [0.03, 0.06],
            "drift_status": ["STABLE", "STABLE"],
        }
    )

    health = determine_model_health(drift_report)

    assert health == "HEALTHY"


def test_warning_model_status():
    """Moderate drift should produce a WARNING status."""

    drift_report = pd.DataFrame(
        {
            "feature": ["credit_amount", "age"],
            "psi": [0.14, 0.04],
            "drift_status": ["MODERATE_DRIFT", "STABLE"],
        }
    )

    health = determine_model_health(drift_report)

    assert health == "WARNING"


def test_alert_model_status():
    """Significant drift should produce an ALERT status."""

    drift_report = pd.DataFrame(
        {
            "feature": ["credit_amount", "age"],
            "psi": [0.31, 0.05],
            "drift_status": ["SIGNIFICANT_DRIFT", "STABLE"],
        }
    )

    health = determine_model_health(drift_report)

    assert health == "ALERT"


def test_build_monitoring_report():
    """Monitoring report should contain operational summary fields."""

    drift_report = pd.DataFrame(
        {
            "feature": ["credit_amount", "age", "duration"],
            "psi": [0.31, 0.13, 0.04],
            "drift_status": [
                "SIGNIFICANT_DRIFT",
                "MODERATE_DRIFT",
                "STABLE",
            ],
        }
    )

    report = build_monitoring_report(
        drift_report,
        model_version="1.0.0",
    )

    assert report["model_health"] == "ALERT"
    assert report["features_monitored"] == 3
    assert report["stable_features"] == 1
    assert report["moderate_drift_features"] == 1
    assert report["significant_drift_features"] == 1
    assert report["highest_drift_feature"] == "credit_amount"
    assert report["highest_psi"] == 0.31
    assert report["model_version"] == "1.0.0"
    assert len(report["feature_drift"]) == 3


def test_save_monitoring_report(tmp_path):
    """Monitoring report should be persisted as valid JSON."""

    report = {
        "model_version": "1.0.0",
        "model_health": "HEALTHY",
        "features_monitored": 2,
    }

    output_file = tmp_path / "monitoring_report.json"

    saved_path = save_monitoring_report(
        report,
        output_file,
    )

    assert saved_path.exists()

    with saved_path.open("r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report["model_health"] == "HEALTHY"
    assert saved_report["features_monitored"] == 2