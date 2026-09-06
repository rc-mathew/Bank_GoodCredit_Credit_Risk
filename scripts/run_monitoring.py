"""Run offline PSI drift validation and generate monitoring artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.monitoring.drift import monitor_feature_drift
from src.monitoring.report import (
    build_monitoring_report,
    save_monitoring_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_feature_matrix.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "monitoring"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET_COLUMN = "Bad_label"
ID_COLUMN = "customer_no"


def main() -> None:
    """Generate offline drift-validation artifacts."""

    print("Loading feature matrix...")

    data = pd.read_csv(
        FEATURE_PATH,
        low_memory=False,
    )

    print(f"Feature matrix shape: {data.shape}")

    # Prefer a temporal split when dt_opened is available.
    if "dt_opened" in data.columns:
        parsed_dates = pd.to_datetime(
            data["dt_opened"],
            errors="coerce",
        )

        valid_ratio = parsed_dates.notna().mean()

        if valid_ratio >= 0.50:
            data = (
                data.assign(_monitor_date=parsed_dates)
                .sort_values("_monitor_date")
                .drop(columns="_monitor_date")
                .reset_index(drop=True)
            )

            split_method = "temporal"
        else:
            split_method = "deterministic_row"
    else:
        split_method = "deterministic_row"

    split_index = int(len(data) * 0.70)

    reference_data = data.iloc[:split_index].copy()
    current_data = data.iloc[split_index:].copy()

    numeric_features = (
        data.select_dtypes(include="number")
        .columns
        .difference(
            [
                ID_COLUMN,
                TARGET_COLUMN,
            ]
        )
        .tolist()
    )

    print(f"Split method: {split_method}")
    print(f"Reference rows: {len(reference_data):,}")
    print(f"Monitoring rows: {len(current_data):,}")
    print(f"Numeric features monitored: {len(numeric_features)}")

    drift_results = monitor_feature_drift(
        reference_data=reference_data,
        current_data=current_data,
        features=numeric_features,
        bins=10,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_csv = (
        REPORT_DIR
        / "psi_feature_drift.csv"
    )

    drift_results.to_csv(
        drift_csv,
        index=False,
    )

    report = build_monitoring_report(
        drift_results,
        model_version="1.0.0",
    )

    report["validation_context"] = {
        "type": "offline_drift_validation",
        "split_method": split_method,
        "reference_share": 0.70,
        "monitoring_share": 0.30,
        "note": (
            "This report validates the monitoring pipeline using "
            "historical project data. It is not live production drift."
        ),
    }

    report_path = save_monitoring_report(
        report,
        REPORT_DIR / "monitoring_report.json",
    )

    top_drift = (
        drift_results
        .dropna(subset=["psi"])
        .head(15)
        .sort_values("psi")
    )

    if not top_drift.empty:
        plt.figure(figsize=(10, 7))

        plt.barh(
            top_drift["feature"],
            top_drift["psi"],
        )

        plt.axvline(
            0.10,
            linestyle="--",
            label="Moderate drift threshold",
        )

        plt.axvline(
            0.25,
            linestyle="--",
            label="Significant drift threshold",
        )

        plt.xlabel("Population Stability Index (PSI)")
        plt.ylabel("Feature")
        plt.title("Top Feature Drift — Offline PSI Validation")
        plt.legend()
        plt.tight_layout()

        figure_path = (
            FIGURE_DIR
            / "psi_feature_drift.png"
        )

        plt.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Drift chart: {figure_path}")

    print("\nMonitoring completed successfully.")
    print(f"PSI results: {drift_csv}")
    print(f"Monitoring report: {report_path}")

    print("\nOverall monitoring status:")
    print(report["model_health"])

    print("\nHighest drift feature:")
    print(report["highest_drift_feature"])

    print("\nHighest PSI:")
    print(report["highest_psi"])

    print("\nTop drift results:")
    print(drift_results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()