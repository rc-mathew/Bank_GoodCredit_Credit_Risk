"""Data ingestion utilities for the Bank GoodCredit project."""

from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEMOGRAPHICS_FILE = "Cust_Demographics_FULL.csv"
ACCOUNT_FILE = "Cust_Account_Full.csv"
ENQUIRY_FILE = "Cust_Enquiry_full.csv"


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a semicolon-delimited CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_csv(file_path, sep=";", low_memory=False)


def load_raw_data(
    data_dir: Path = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load demographics, account and enquiry datasets."""

    demographics = load_csv(data_dir / DEMOGRAPHICS_FILE)
    accounts = load_csv(data_dir / ACCOUNT_FILE)
    enquiries = load_csv(data_dir / ENQUIRY_FILE)

    return demographics, accounts, enquiries


def print_dataset_summary(
    demographics: pd.DataFrame,
    accounts: pd.DataFrame,
    enquiries: pd.DataFrame,
) -> None:
    """Print basic information about the loaded datasets."""

    datasets = {
        "Demographics": demographics,
        "Accounts": accounts,
        "Enquiries": enquiries,
    }

    print("\nBank GoodCredit — Raw Data Summary")
    print("-" * 45)

    for name, dataframe in datasets.items():
        print(
            f"{name:<15} "
            f"Rows: {dataframe.shape[0]:>8,} | "
            f"Columns: {dataframe.shape[1]:>3}"
        )


if __name__ == "__main__":
    demo_df, account_df, enquiry_df = load_raw_data()

    print_dataset_summary(
        demo_df,
        account_df,
        enquiry_df,
    )