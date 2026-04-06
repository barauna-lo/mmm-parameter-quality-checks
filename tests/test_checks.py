"""
Tests for the business-rule check layer.

This file covers the pure check logic that creates boolean flags from already
prepared input data.

Main goals:
- verify correct boolean flag creation
- verify correct final aggregation into `has_any_issue`

Why this matters:
The assignment requires the output to contain:
- all original columns
- one boolean column per check
- a final `has_any_issue` column
"""

from __future__ import annotations

import pandas as pd

from mmm_quality_checks.checks import apply_quality_checks


def test_apply_quality_checks_happy_path_and_issue_detection() -> None:
    """
    Verify that the check layer applies business rules correctly.

    This test covers:
    - one row with no issue
    - one row with low significance
    - one row with suspicious elasticity
    - one row with contribution and adstock issues
    """
    df = pd.DataFrame(
        {
            "Channel": ["Google PMax", "Paid Social", "CTV", "OOH"],
            "Coefficient": [0.8, 0.5, 0.4, 0.2],
            "p_value": [0.03, 0.15, 0.07, 0.02],
            "Elasticity": [1.2, 2.0, 3.4, 0.8],
            "contribution_pct": [25.0, 10.0, 15.0, 0.8],
            "adstock_half_life": [14.0, 30.0, 20.0, 0.5],
            "validation_errors": [None, None, None, None],
        }
    )

    result = apply_quality_checks(df)

    # Google PMax: no issues
    assert bool(result.loc[0, "low_significance_flag"]) is False
    assert bool(result.loc[0, "suspicious_elasticity_flag"]) is False
    assert bool(result.loc[0, "over_dominant_contribution_flag"]) is False
    assert bool(result.loc[0, "unrealistic_adstock_flag"]) is False
    assert bool(result.loc[0, "has_any_issue"]) is False

    # Paid Social: low significance
    assert bool(result.loc[1, "low_significance_flag"]) is True
    assert bool(result.loc[1, "has_any_issue"]) is True

    # CTV: suspicious elasticity
    assert bool(result.loc[2, "suspicious_elasticity_flag"]) is True
    assert bool(result.loc[2, "has_any_issue"]) is True

    # OOH: contribution and adstock issues
    assert bool(result.loc[3, "over_dominant_contribution_flag"]) is True
    assert bool(result.loc[3, "unrealistic_adstock_flag"]) is True
    assert bool(result.loc[3, "has_any_issue"]) is True