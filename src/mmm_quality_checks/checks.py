from __future__ import annotations

import pandas as pd

from .constants import (
    ADSTOCK_MAX,
    ADSTOCK_MIN,
    CONTRIBUTION_MAX,
    CONTRIBUTION_MIN,
    ELASTICITY_MAX,
    ELASTICITY_MIN,
    HAS_ANY_ISSUE_FLAG,
    LOW_SIGNIFICANCE_FLAG,
    LOW_SIGNIFICANCE_THRESHOLD,
    OVER_DOMINANT_CONTRIBUTION_FLAG,
    SUSPICIOUS_ELASTICITY_FLAG,
    UNREALISTIC_ADSTOCK_FLAG,
    VALIDATION_ERRORS_COLUMN,
)


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce")


def _finalize_flag(mask: pd.Series, source: pd.Series, na_in_missing_data: bool) -> pd.Series:
    """
    Finalize a boolean check column.

    If na_in_missing_data is True, rows where the source is missing/invalid
    become pd.NA. Otherwise they become False.
    """
    result = mask.astype("boolean")

    if na_in_missing_data:
        result = result.mask(source.isna(), pd.NA)
    else:
        result = result.fillna(False)

    return result


def apply_quality_checks(
    df: pd.DataFrame,
    na_in_missing_data: bool = False,
) -> pd.DataFrame:
    """
    Apply MMM quality checks and return an enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with validation metadata already added.
    na_in_missing_data : bool, default False
        If True, checks that cannot be evaluated due to missing/invalid data
        return pd.NA instead of False.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus boolean/nullable-boolean flags and has_any_issue.
    """
    result = df.copy()

    p_value = _safe_to_numeric(result["p_value"])
    elasticity = _safe_to_numeric(result["Elasticity"])
    contribution_pct = _safe_to_numeric(result["contribution_pct"])
    adstock_half_life = _safe_to_numeric(result["adstock_half_life"])

    low_significance_raw = p_value > LOW_SIGNIFICANCE_THRESHOLD
    suspicious_elasticity_raw = (elasticity < ELASTICITY_MIN) | (elasticity > ELASTICITY_MAX)
    over_dominant_contribution_raw = (
        (contribution_pct < CONTRIBUTION_MIN) | (contribution_pct > CONTRIBUTION_MAX)
    )
    unrealistic_adstock_raw = (
        (adstock_half_life < ADSTOCK_MIN) | (adstock_half_life > ADSTOCK_MAX)
    )

    result[LOW_SIGNIFICANCE_FLAG] = _finalize_flag(
        low_significance_raw, p_value, na_in_missing_data
    )
    result[SUSPICIOUS_ELASTICITY_FLAG] = _finalize_flag(
        suspicious_elasticity_raw, elasticity, na_in_missing_data
    )
    result[OVER_DOMINANT_CONTRIBUTION_FLAG] = _finalize_flag(
        over_dominant_contribution_raw, contribution_pct, na_in_missing_data
    )
    result[UNREALISTIC_ADSTOCK_FLAG] = _finalize_flag(
        unrealistic_adstock_raw, adstock_half_life, na_in_missing_data
    )

    flag_columns = [
        LOW_SIGNIFICANCE_FLAG,
        SUSPICIOUS_ELASTICITY_FLAG,
        OVER_DOMINANT_CONTRIBUTION_FLAG,
        UNREALISTIC_ADSTOCK_FLAG,
    ]

    result[HAS_ANY_ISSUE_FLAG] = (
        result[flag_columns].fillna(False).any(axis=1)
        | result[VALIDATION_ERRORS_COLUMN].notna()
    )

    return result