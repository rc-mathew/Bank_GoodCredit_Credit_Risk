"""Preprocessing utilities for the Bank GoodCredit project."""

from typing import Iterable

import pandas as pd


def remove_exact_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove exact duplicate rows from a dataframe."""

    return df.drop_duplicates().reset_index(drop=True)


def clean_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove leading and trailing whitespace from column names."""

    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    return cleaned


def clean_string_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Strip whitespace from string columns."""

    cleaned = df.copy()

    string_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in string_columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    return cleaned


def parse_date_columns(
    df: pd.DataFrame,
    date_columns: Iterable[str],
) -> pd.DataFrame:
    """Convert selected columns to pandas datetime."""

    parsed = df.copy()

    for column in date_columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(
                parsed[column],
                errors="coerce",
            )

    return parsed


def preprocess_dataframe(
    df: pd.DataFrame,
    date_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Apply common preprocessing operations."""

    processed = clean_column_names(df)

    processed = clean_string_columns(processed)

    processed = remove_exact_duplicates(processed)

    processed = parse_date_columns(
        processed,
        date_columns,
    )

    return processed


def preprocessing_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict:
    """Return a summary of preprocessing changes."""

    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_removed": len(before) - len(after),
        "columns": after.shape[1],
        "missing_values": int(
            after.isna().sum().sum()
        ),
    }