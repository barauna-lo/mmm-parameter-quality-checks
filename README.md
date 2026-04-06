# Marketing Science Product Engineer Assessment

This project implements a reusable Python utility to perform quality checks on MMM outputs.

## Output format

The pipeline returns a DataFrame with:

- all original columns
- one boolean column per product rule
- a final `has_any_issue` column
- one additional `validation_errors` column for row-level validation issues

The boolean check columns remain boolean even when data quality issues exist.
When a rule cannot be evaluated because an input value is missing or invalid, the rule flag remains `False` and the issue is recorded in `validation_errors`.

If multiple validation issues exist in the same row, they are concatenated into a single string separated by semicolons.

## Rules implemented

1. **Low significance**
   - `p_value > 0.1`

2. **Suspicious elasticity**
   - `Elasticity < 0`
   - `Elasticity > 3`

3. **Over-dominant contribution**
   - `contribution_pct < 1`
   - `contribution_pct > 60`

4. **Unrealistic adstock**
   - `adstock_half_life < 1`
   - `adstock_half_life > 150`

## Validation behavior

The pipeline distinguishes between:

- **DataFrame-level structural failures**
  - empty DataFrame
  - missing required columns

These raise `ValueError`.

- **Row-level validation issues**
  - missing values
  - non-numeric values in numeric columns
  - missing or empty channel names

These are recorded in `validation_errors` and do not stop execution.

## Example usage

```python
import pandas as pd
from mmm_quality_checks.pipeline import run_quality_checks

df = pd.read_csv("data/MMM_dummy_outputs.csv")
result = run_quality_checks(df)

print(result)