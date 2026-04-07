# Marketing Science Product Engineer Assessment

Reusable Python utility for validating MMM outputs and flagging parameter quality issues.

## Overview

This project implements a small, reusable Python pipeline for evaluating Marketing Mix Modeling (MMM) channel outputs against product-defined quality rules.

The solution was designed as a product engineering and its purpose is to take an already produced MMM output table and:

- validate the input structure
- identify row-level data quality issues
- apply rule-based parameter checks
- return an enriched DataFrame with boolean flags and a final issue summary

---

## Required input columns

The pipeline expects the following columns:

- `Channel`
- `p_value`
- `Elasticity`
- `contribution_pct`
- `adstock_half_life`

## Output format

The pipeline returns a DataFrame with:

- all original columns
- one boolean column per product rule
- a final `has_any_issue` column
- (extra) one additional `validation_errors` column for row-level validation issues

The boolean check columns remain boolean even when data quality issues exist.
When a rule cannot be evaluated because an input value is missing or invalid, the rule flag remains `False` and the issue is recorded in `validation_errors`.

If multiple validation issues exist in the same row, they are concatenated into a single string separated by semicolons.

## Rules implemented

1. **Low significance**
   - `p_value > 0.1`

2. **Suspicious elasticity**
   - `Elasticity < 0`
   - `Elasticity > 3`

3. **Out-of-range contribution**
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

The project flow is:
```text
input DataFrame
    ↓
validate_dataframe_structure()
    ↓
add_validation_errors()
    ↓
apply_quality_checks()
    ↓
enriched output DataFrame
```


## Project architecture

```text
mmm-parameter-quality-checks/
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
│
├── data/
│   ├── MMM_dummy_outputs.csv
│   └── MMM_dummy_outputs.xlsx
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   └── mmm_quality_checks/
│       ├── __init__.py
│       ├── checks.py
│       ├── constants.py
│       ├── exceptions.py
│       ├── pipeline.py
│       └── validation.py
│
└── tests/
    ├── test_checks.py
    ├── test_pipeline.py
    └── test_validation.py
```

## Code explanation

Below is a short description of what each source file does inside `src/mmm_quality_checks/`.

##### `__init__.py`

Exposes the main public functions of the package so they can be imported more easily by notebooks, scripts, or downstream systems.  
*(In practice, it works as the package entry point.)*

##### `constants.py`

Stores the reusable constants used across the project, such as required column names, numeric column names, thresholds, and output flag names.  
*(This avoids hardcoding the same values in multiple files and makes maintenance easier.)*

##### `exceptions.py`

Defines the custom exception classes used by the project.  
*(This makes pipeline errors clearer and more meaningful than relying only on generic Python exceptions.)*

##### `validation.py`

Contains the input-validation logic. It validates the DataFrame structure, checks required columns, and creates the `validation_errors` column for row-level issues.  
*(This is the data-quality layer of the project.)*

##### `checks.py`

Contains the business-rule logic. It applies the parameter quality rules and creates the boolean issue flags and the final `has_any_issue` column.  
*(This is the analytical rules layer of the project.)*

##### `pipeline.py`

Contains the main orchestration function, `run_quality_checks(df)`, which connects the full workflow: structural validation, row-level validation, and rule application.  
*(This is the main public pipeline entry point.)*


##### `visualizations.py`

Contains the reusable dashboard visualization functions built on top of the quality-check output. It provides chart builders and summary components, such as KPI cards, heatmaps, scatter plots, bar charts, and acceptable-range charts, all designed to make the classifier results easier to inspect and interpret.
*(This is the presentation and monitoring layer of the project.)*
---    

## Installation Guide

### 1. Create and activate a Python > 3.10 environment

#### 1.1 Windows (CMD)

Create the environment with Conda:

```bash
conda create -n mmm-assessment python=3.13 -y
conda activate mmm-assessment
python --version
```

#### 1.2 Linux (Conda)

Create and activate the environment:

```bash
conda create -n mmm-assessment python=3.13 -y
conda activate mmm-assessment
python --version
```

#### 1.3 Linux (base / venv)

Create a virtual environment with Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python --version
```

### 2. In the project root folder

Install the project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```


## Example usage

### Running Tests 

To run the test suite with `pytest`:

* 1 happy path test
* 1 edge case test
* 1 failure or invalid input test

```bash
pytest -q
```
For more detailed output:
```bash
python -m pytest -vv
```

### Testing in notebooks

To test the functions on other datasets:

```python
import pandas as pd
from mmm_quality_checks.pipeline import run_quality_checks

df = pd.read_csv("data/MMM_dummy_outputs.csv")
result = run_quality_checks(df)

print(result)
```