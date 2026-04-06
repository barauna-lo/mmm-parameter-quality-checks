"""
Tests for the full pipeline.

This file covers the public entry point: `run_quality_checks`.

Main goals:
- verify the end-to-end happy path
- verify stability on threshold edge cases
- verify defensive behavior for invalid structural input

Why this matters:
These are the three scenarios explicitly requested in the assessment:
- 1 happy path test
- 1 edge case test
- 1 failure / invalid input test
"""

from __future__ import annotations

import pandas as pd
import pytest

from mmm_quality_checks.pipeline import run_quality_checks


def test_happy_path_applies_rules_correctly() -> None:
    """
    Happy path test.

    Verifies that the full pipeline:
    - preserves valid rows
    - applies the expected boolean flags
    - returns the correct `has_any_issue` values
    """
    df = pd.DataFrame(
        {
            "Channel": ["Google PMax", "Paid Social", "CTV", "OOH"],
            "Coefficient": [0.8, 0.5, 0.4, 0.2],
            "p_value": [0.03, 0.15, 0.07, 0.02],
            "Elasticity": [1.2, 2.0, 3.4, 0.8],
            "contribution_pct": [25.0, 10.0, 15.0, 0.8],
            "adstock_half_life": [14.0, 30.0, 20.0, 0.5],
        }
    )

    result = run_quality_checks(df)

    # Google PMax: no issues
    assert bool(result.loc[0, "low_significance_flag"]) is False
    assert bool(result.loc[0, "suspicious_elasticity_flag"]) is False
    assert bool(result.loc[0, "over_dominant_contribution_flag"]) is False
    assert bool(result.loc[0, "unrealistic_adstock_flag"]) is False
    assert bool(result.loc[0, "has_any_issue"]) is False
    assert pd.isna(result.loc[0, "validation_errors"])

    # Paid Social: low significance
    assert bool(result.loc[1, "low_significance_flag"]) is True
    assert bool(result.loc[1, "has_any_issue"]) is True

    # CTV: suspicious elasticity
    assert bool(result.loc[2, "suspicious_elasticity_flag"]) is True
    assert bool(result.loc[2, "has_any_issue"]) is True

    # OOH: contribution + adstock issue
    assert bool(result.loc[3, "over_dominant_contribution_flag"]) is True
    assert bool(result.loc[3, "unrealistic_adstock_flag"]) is True
    assert bool(result.loc[3, "has_any_issue"]) is True


def test_edge_case_thresholds_are_not_flagged() -> None:
    """
    Edge case test.

    Verifies that exact threshold values are not flagged.

    The rules use strict comparisons, so the following boundary values should
    remain unflagged:
    - p_value = 0.1
    - Elasticity = 3.0
    - contribution_pct = 1.0 or 60.0
    - adstock_half_life = 1.0 or 150.0
    """
    df = pd.DataFrame(
        {
            "Channel": ["Affiliate", "Influencer", "DOOH"],
            "Coefficient": [0.4, 0.3, 0.2],
            "p_value": [0.1, 0.05, 0.05],
            "Elasticity": [2.0, 3.0, 1.0],
            "contribution_pct": [1.0, 60.0, 10.0],
            "adstock_half_life": [150.0, 10.0, 1.0],
        }
    )

    result = run_quality_checks(df)

    assert result["low_significance_flag"].sum() == 0
    assert result["suspicious_elasticity_flag"].sum() == 0
    assert result["over_dominant_contribution_flag"].sum() == 0
    assert result["unrealistic_adstock_flag"].sum() == 0
    assert result["has_any_issue"].sum() == 0
    assert result["validation_errors"].isna().all()


def test_failure_invalid_input_missing_required_column() -> None:
    """
    Failure / invalid input test.

    Verifies defensive behavior when a required column is missing.

    Expected behavior:
    - the pipeline fails fast
    - a clear ValueError is raised
    """
    df = pd.DataFrame(
        {
            "Channel": ["Google PMax"],
            "Coefficient": [0.8],
            "Elasticity": [1.2],
            "contribution_pct": [25.0],
            "adstock_half_life": [14.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        run_quality_checks(df)