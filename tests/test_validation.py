"""
Tests for the validation layer.

This file covers the functions responsible for validating the input before
business-rule flags are created.

Main goals:
- verify that structural validation works correctly
- verify that row-level validation issues are recorded
- verify that multiple row-level issues are consolidated into a single
  `validation_errors` field

Why this matters:
The project separates structural failures from row-level data quality issues.
Structural failures should stop execution, while row-level problems should be
captured in the output DataFrame.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mmm_quality_checks.validation import (
    add_validation_errors,
    validate_dataframe_structure,
)


def make_valid_df() -> pd.DataFrame:
    """
    Create a minimal valid DataFrame for validation tests.

    This helper keeps tests concise and deterministic.
    """
    return pd.DataFrame(
        {
            "Channel": ["Google PMax", "Paid Social"],
            "Coefficient": [0.8, 0.5],
            "p_value": [0.03, 0.09],
            "Elasticity": [1.2, 2.5],
            "contribution_pct": [25.0, 10.0],
            "adstock_half_life": [14.0, 30.0],
        }
    )


def test_validate_dataframe_structure_passes_for_valid_input() -> None:
    """
    Verify that a structurally valid DataFrame passes validation.

    Expected behavior:
    - no exception is raised
    """
    df = make_valid_df()
    validate_dataframe_structure(df)


def test_validate_dataframe_structure_raises_for_missing_column() -> None:
    """
    Verify that missing required columns raise an error.

    This is a defensive validation test. The pipeline should not continue if the
    required schema is incomplete.
    """
    df = make_valid_df().drop(columns=["p_value"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe_structure(df)


def test_add_validation_errors_collects_multiple_errors_in_same_row() -> None:
    """
    Verify that multiple row-level validation issues are stored in one field.

    The test simulates:
    - one missing numeric value (`p_value`)
    - one invalid non-numeric value (`Elasticity`)

    Expected behavior:
    - no row-level exception is raised
    - both problems are recorded in `validation_errors`
    """
    df = make_valid_df()
    df.loc[0, "p_value"] = None
    df["Elasticity"] = df["Elasticity"].astype(object)
    df.loc[0, "Elasticity"] = "high"

    result = add_validation_errors(df)

    error_text = result.loc[0, "validation_errors"]
    assert error_text is not None
    assert "p_value: value is missing" in error_text
    assert "Elasticity: format is not numerical" in error_text