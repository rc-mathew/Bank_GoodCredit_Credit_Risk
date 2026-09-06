"""Production monitoring report utilities.

This module converts feature-drift results into a concise model-monitoring
summary suitable for operational review and portfolio evidence.

The PSI calculation itself lives in ``src.monitoring.drift``. This module
focuses only on aggregation, health classification, recommendations, and
report persistence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _normalise_status(status: str) -> str:
    """Normalise drift labels so reporting is robust to naming variations."""
    value = str(status).strip().upper().replace(" ", "_")

    aliases = {
        "NO_DRIFT": "STABLE",
        "LOW_DRIFT": "STABLE",
        "STABLE": "STABLE",
        "MODERATE": "MODERATE_DRIFT",
        "MODERATE_DRIFT": "MODERATE_DRIFT",
        "WARNING": "MODERATE_DRIFT",
        "SIGNIFICANT": "SIGNIFICANT_DRIFT",
        "SIGNIFICANT_DRIFT": "SIGNIFICANT_DRIFT",
        "HIGH_DRIFT": "SIGNIFICANT_DRIFT",
    }

    return aliases.get(value, value)


def determine_model_health(drift_report: pd.DataFrame) -> str:
    """Determine overall model health from feature-level drift results."""

    if drift_report.empty:
        return "UNKNOWN"

    statuses = {
        _normalise_status(status)
        for status in drift_report["drift_status"].dropna()
    }

    if "SIGNIFICANT_DRIFT" in statuses:
        return "ALERT"

    if "MODERATE_DRIFT" in statuses:
        return "WARNING"

    return "HEALTHY"


def monitoring_recommendation(model_health: str) -> str:
    """Return an operational recommendation for the current health state."""

    recommendations = {
        "HEALTHY": (
            "Feature distributions are stable. Continue normal monitoring."
        ),
        "WARNING": (
            "Moderate feature drift detected. Investigate affected features "
            "and increase monitoring frequency."
        ),
        "ALERT": (
            "Significant feature drift detected. Investigate data changes, "
            "evaluate model performance and consider retraining before "
            "continued production use."
        ),
        "UNKNOWN": (
            "Insufficient monitoring information. Verify monitoring inputs "
            "before making a production decision."
        ),
    }

    return recommendations[model_health]


def build_monitoring_report(
    drift_report: pd.DataFrame,
    model_version: str = "1.0.0",
) -> dict[str, Any]:
    """Build a production-style monitoring summary from PSI results."""

    required_columns = {"feature", "psi", "drift_status"}

    missing = required_columns.difference(drift_report.columns)

    if missing:
        raise ValueError(
            "Drift report is missing required columns: "
            + ", ".join(sorted(missing))
        )

    report = drift_report.copy()

    report["drift_status"] = report["drift_status"].map(_normalise_status)

    health = determine_model_health(report)

    stable_count = int(
        (report["drift_status"] == "STABLE").sum()
    )

    moderate_count = int(
        (report["drift_status"] == "MODERATE_DRIFT").sum()
    )

    significant_count = int(
        (report["drift_status"] == "SIGNIFICANT_DRIFT").sum()
    )

    if report.empty:
        highest_psi = None
        highest_drift_feature = None
    else:
        max_index = report["psi"].astype(float).idxmax()
        highest_psi = float(report.loc[max_index, "psi"])
        highest_drift_feature = str(report.loc[max_index, "feature"])

    ranked_features = (
        report.sort_values("psi", ascending=False)
        .loc[:, ["feature", "psi", "drift_status"]]
        .to_dict(orient="records")
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "model_health": health,
        "features_monitored": int(len(report)),
        "stable_features": stable_count,
        "moderate_drift_features": moderate_count,
        "significant_drift_features": significant_count,
        "highest_drift_feature": highest_drift_feature,
        "highest_psi": highest_psi,
        "psi_thresholds": {
            "stable": "PSI < 0.10",
            "moderate_drift": "0.10 <= PSI < 0.25",
            "significant_drift": "PSI >= 0.25",
        },
        "recommendation": monitoring_recommendation(health),
        "feature_drift": ranked_features,
    }

    return summary


def save_monitoring_report(
    report: dict[str, Any],
    output_path: str | Path = "reports/monitoring_report.json",
) -> Path:
    """Persist the monitoring report as JSON."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    return output_path