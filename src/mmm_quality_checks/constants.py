from __future__ import annotations

REQUIRED_COLUMNS = (
    "Channel",
    "Coefficient",
    "p_value",
    "Elasticity",
    "contribution_pct",
    "adstock_half_life",
)

NUMERIC_COLUMNS = (
    "Coefficient",
    "p_value",
    "Elasticity",
    "contribution_pct",
    "adstock_half_life",
)

CHANNEL_COLUMN = "Channel"

LOW_SIGNIFICANCE_THRESHOLD = 0.1
ELASTICITY_MIN = 0.0
ELASTICITY_MAX = 3.0
CONTRIBUTION_MIN = 1.0
CONTRIBUTION_MAX = 60.0
ADSTOCK_MIN = 1.0
ADSTOCK_MAX = 150.0

LOW_SIGNIFICANCE_FLAG = "low_significance_flag"
SUSPICIOUS_ELASTICITY_FLAG = "suspicious_elasticity_flag"
OVER_DOMINANT_CONTRIBUTION_FLAG = "over_dominant_contribution_flag"
UNREALISTIC_ADSTOCK_FLAG = "unrealistic_adstock_flag"
HAS_ANY_ISSUE_FLAG = "has_any_issue"
VALIDATION_ERRORS_COLUMN = "validation_errors"