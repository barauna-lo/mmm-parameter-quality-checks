from __future__ import annotations

from typing import Iterable

import pandas as pd

from .constants import (
    CHANNEL_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    VALIDATION_ERRORS_COLUMN,
)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if one or more required columns are missing."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_not_empty(df: pd.DataFrame) -> None:
    """Raise ValueError if the input DataFrame is empty."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def _channel_error(value: object) -> str | None:
    """Return a validation error for Channel, if any."""
    if pd.isna(value):
        return "Channel: value is missing"
    if str(value).strip() == "":
        return "Channel: value is empty"
    return None


def _numeric_error(value: object, column_name: str) -> str | None:
    """Return a validation error for a numeric column, if any."""
    if pd.isna(value):
        return f"{column_name}: value is missing"

    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return f"{column_name}: format is not numerical"

    return None


def collect_row_validation_errors(row: pd.Series) -> str | None:
    """
    Collect all validation errors for a single row.

    Returns
    -------
    str | None
        A semicolon-separated string with all validation errors for the row,
        or None if the row is valid.
    """
    errors: list[str] = []

    channel_error = _channel_error(row[CHANNEL_COLUMN])
    if channel_error:
        errors.append(channel_error)

    for column in NUMERIC_COLUMNS:
        error = _numeric_error(row[column], column)
        if error:
            errors.append(error)

    if not errors:
        return None

    return "; ".join(errors)


def add_validation_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a single validation error column to the DataFrame.

    This function does not raise row-level validation errors. Instead, it records
    them in the `validation_errors` column.
    """
    result = df.copy()
    result[VALIDATION_ERRORS_COLUMN] = result.apply(collect_row_validation_errors, axis=1)
    return result


def validate_dataframe_structure(df: pd.DataFrame) -> None:
    """
    Validate DataFrame-level structure.

    These are hard failures because the pipeline cannot proceed safely without them.
    """
    validate_not_empty(df)
    validate_required_columns(df)