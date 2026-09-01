"""Model evaluation and visualization utilities for Bank GoodCredit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

BENCHMARK_GINI = 0.379


def calculate_gini(
    y_true,
    probabilities,
) -> float:
    """Calculate Gini coefficient from ROC-AUC."""

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    return float(2 * auc - 1)


def calculate_ks(
    y_true,
    probabilities,
) -> float:
    """Calculate the Kolmogorov-Smirnov statistic."""

    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "score": np.asarray(probabilities),
        }
    )

    data = data.sort_values(
        "score",
        ascending=False,
    )

    bad_total = (data["target"] == 1).sum()
    good_total = (data["target"] == 0).sum()

    data["cum_bad"] = (
        (data["target"] == 1).cumsum()
        / bad_total
    )

    data["cum_good"] = (
        (data["target"] == 0).cumsum()
        / good_total
    )

    ks = (
        data["cum_bad"]
        - data["cum_good"]
    ).abs().max()

    return float(ks)


def evaluate_model(
    model,
    X_test,
    y_test,
) -> dict:
    """Calculate holdout model-performance metrics."""

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    gini = 2 * auc - 1

    ks = calculate_ks(
        y_test,
        probabilities,
    )

    average_precision = average_precision_score(
        y_test,
        probabilities,
    )

    metrics = {
        "roc_auc": float(auc),
        "gini": float(gini),
        "ks": float(ks),
        "average_precision": float(
            average_precision
        ),
        "benchmark_gini": BENCHMARK_GINI,
        "gini_gap_vs_benchmark": float(
            gini - BENCHMARK_GINI
        ),
    }

    return metrics


def create_decile_table(
    y_true,
    probabilities,
) -> pd.DataFrame:
    """Create risk-decile rank-ordering table."""

    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "probability": np.asarray(probabilities),
        }
    )

    # Highest predicted-risk customers receive decile 10.
    ranked = data["probability"].rank(
        method="first"
    )

    data["risk_decile"] = pd.qcut(
        ranked,
        q=10,
        labels=False,
    ) + 1

    deciles = (
        data.groupby("risk_decile")
        .agg(
            customers=("target", "size"),
            bad_customers=("target", "sum"),
            bad_rate=("target", "mean"),
            mean_score=("probability", "mean"),
        )
        .reset_index()
        .sort_values(
            "risk_decile",
            ascending=False,
        )
    )

    return deciles


def save_roc_curve(
    y_true,
    probabilities,
) -> Path:
    """Save ROC curve for GitHub README."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities,
    )

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    output_path = (
        FIGURES_DIR / "roc_curve.png"
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Model AUC = {auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Credit Risk Model — ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def save_decile_plot(
    deciles: pd.DataFrame,
) -> Path:
    """Save bad-rate-by-risk-decile visualization."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_data = deciles.sort_values(
        "risk_decile"
    )

    output_path = (
        FIGURES_DIR / "bad_rate_deciles.png"
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        plot_data["risk_decile"].astype(str),
        plot_data["bad_rate"],
    )

    plt.xlabel("Risk Decile")
    plt.ylabel("Observed Bad Rate")
    plt.title(
        "Observed Bad Rate by Predicted Risk Decile"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def save_metrics(
    metrics: dict,
) -> Path:
    """Save model metrics as JSON."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        METRICS_DIR / "model_metrics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    return output_path


def run_evaluation(
    model,
    X_test,
    y_test,
) -> dict:
    """Run the complete evaluation workflow."""

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    deciles = create_decile_table(
        y_test,
        probabilities,
    )

    save_metrics(metrics)

    save_roc_curve(
        y_test,
        probabilities,
    )

    save_decile_plot(
        deciles
    )

    return metrics