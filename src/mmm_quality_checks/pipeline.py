from __future__ import annotations

import pandas as pd

from .checks import apply_quality_checks
from .validation import add_validation_errors, validate_dataframe_structure


def run_quality_checks(
    df: pd.DataFrame,
    na_in_missing_data: bool = False,
) -> pd.DataFrame:
    """
    Run the full MMM quality check pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    na_in_missing_data : bool, default False
        If True, rule flags that cannot be evaluated because of missing/invalid
        values return pd.NA instead of False.

    Returns
    -------
    pd.DataFrame
        Original data plus validation_errors, rule flags, and has_any_issue.
    """
    validate_dataframe_structure(df)
    validated_df = add_validation_errors(df)
    return apply_quality_checks(
        validated_df,
        na_in_missing_data=na_in_missing_data,
    )