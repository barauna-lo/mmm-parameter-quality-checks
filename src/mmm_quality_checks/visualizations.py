"""
Visualization utilities for the MMM quality checks dashboard.

This module centralizes reusable functions used to build dashboard components
from the classifier output dataframe.

The functions are organized into two groups:

1. Data preparation helpers
   - functions that validate inputs, classify values, or compute summary metrics

2. Visualization builders
   - functions that return Plotly figures ready to be rendered in Hex

Current visualizations included:
- Summary cards: KPI-style indicators with counts of channels, issues, validation errors,
  and rule-level flags
- Parameter range chart: multi-panel chart showing the acceptable range for each metric
  and the observed value of each channel

These utilities are designed to keep the dashboard code modular, readable,
and easy to extend with new charts.
"""


from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mmm_quality_checks.constants import (
    CHANNEL_COLUMN,
    LOW_SIGNIFICANCE_THRESHOLD,
    ELASTICITY_MIN,
    ELASTICITY_MAX,
    CONTRIBUTION_MIN,
    CONTRIBUTION_MAX,
    ADSTOCK_MIN,
    ADSTOCK_MAX,
)


DEFAULT_THRESHOLDS = {
    "p_value": {"min": 0.0, "max": LOW_SIGNIFICANCE_THRESHOLD},
    "Elasticity": {"min": ELASTICITY_MIN, "max": ELASTICITY_MAX},
    "contribution_pct": {"min": CONTRIBUTION_MIN, "max": CONTRIBUTION_MAX},
    "adstock_half_life": {"min": ADSTOCK_MIN, "max": ADSTOCK_MAX},
}

DEFAULT_STATUS_COLORS = {
    "In range": "#2ca02c",
    "Below min": "#ff7f0e",
    "Above max": "#d62728",
    "Missing": "#7f7f7f",
}


def _classify_value(
    value: float | int | None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> str:
    if pd.isna(value):
        return "Missing"
    if min_value is not None and value < min_value:
        return "Below min"
    if max_value is not None and value > max_value:
        return "Above max"
    return "In range"


def _validate_input_dataframe(
    df: pd.DataFrame,
    metrics: Iterable[str],
) -> None:
    required_columns = {CHANNEL_COLUMN, *metrics}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"DataFrame is missing required columns for chart generation: {missing_str}"
        )


def build_parameter_range_chart(
    df: pd.DataFrame,
    metrics: Iterable[str] | None = None,
    thresholds: Mapping[str, Mapping[str, float]] | None = None,
    status_colors: Mapping[str, str] | None = None,
    title: str = "Parâmetros por canal vs faixa aceitável",
    height_per_metric: int = 320,
) -> go.Figure:
    """
    Build a multi-panel Plotly chart where each subplot shows:
    - the acceptable range for a parameter
    - the observed point for each channel
    - point colors indicating whether the value is within range

    Parameters
    ----------
    df : pd.DataFrame
        Classifier output dataframe.
    metrics : iterable[str] | None
        Metrics to display. Defaults to the keys in DEFAULT_THRESHOLDS.
    thresholds : mapping | None
        Dict like:
        {
            "p_value": {"min": 0.0, "max": 0.1},
            ...
        }
    status_colors : mapping | None
        Colors for each classification status.
    title : str
        Chart title.
    height_per_metric : int
        Height allocated to each subplot row.

    Returns
    -------
    go.Figure
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    status_colors = status_colors or DEFAULT_STATUS_COLORS
    metrics = list(metrics or thresholds.keys())

    _validate_input_dataframe(df, metrics)

    df_plot = df.copy()

    for metric in metrics:
        df_plot[metric] = pd.to_numeric(df_plot[metric], errors="coerce")

    subplot_titles = []
    for metric in metrics:
        metric_threshold = thresholds.get(metric)
        if metric_threshold is None:
            raise ValueError(f"No threshold configuration found for metric '{metric}'.")

        min_v = metric_threshold.get("min")
        max_v = metric_threshold.get("max")
        subplot_titles.append(f"{metric} | faixa aceitável: {min_v} → {max_v}")

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    for row_idx, metric in enumerate(metrics, start=1):
        min_v = thresholds[metric]["min"]
        max_v = thresholds[metric]["max"]

        subset = df_plot[[CHANNEL_COLUMN, metric]].copy()
        subset["status"] = subset[metric].apply(
            lambda x: _classify_value(x, min_value=min_v, max_value=max_v)
        )

        range_width = max_v - min_v

        fig.add_trace(
            go.Bar(
                y=subset[CHANNEL_COLUMN],
                x=[range_width] * len(subset),
                base=[min_v] * len(subset),
                orientation="h",
                marker=dict(color="rgba(100, 100, 100, 0.18)"),
                name="Faixa aceitável",
                hovertemplate=(
                    f"{metric}<br>"
                    f"Faixa aceitável: {min_v} → {max_v}<extra></extra>"
                ),
                showlegend=(row_idx == 1),
            ),
            row=row_idx,
            col=1,
        )

        for status, color in status_colors.items():
            tmp = subset[subset["status"] == status]
            if tmp.empty:
                continue

            customdata = np.column_stack(
                [
                    tmp[CHANNEL_COLUMN].astype(str),
                    tmp[metric].astype(str),
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=tmp[metric],
                    y=tmp[CHANNEL_COLUMN],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=color,
                        line=dict(color="black", width=0.6),
                    ),
                    name=status,
                    showlegend=(row_idx == 1),
                    customdata=customdata,
                    hovertemplate=(
                        f"Canal: %{{customdata[0]}}<br>"
                        f"{metric}: %{{customdata[1]}}<br>"
                        f"Status: {status}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

        fig.add_vline(
            x=min_v,
            line_width=1,
            line_dash="dash",
            line_color="black",
            row=row_idx,
            col=1,
        )
        fig.add_vline(
            x=max_v,
            line_width=1,
            line_dash="dash",
            line_color="black",
            row=row_idx,
            col=1,
        )

        fig.update_yaxes(autorange="reversed", row=row_idx, col=1)

    fig.update_layout(
        height=height_per_metric * len(metrics),
        title=title,
        barmode="overlay",
        plot_bgcolor="white",
        legend_title="Status",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig



########################

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mmm_quality_checks.constants import (
    CHANNEL_COLUMN,
    LOW_SIGNIFICANCE_FLAG,
    SUSPICIOUS_ELASTICITY_FLAG,
    OVER_DOMINANT_CONTRIBUTION_FLAG,
    UNREALISTIC_ADSTOCK_FLAG,
    HAS_ANY_ISSUE_FLAG,
    VALIDATION_ERRORS_COLUMN,
)


DEFAULT_FLAG_COLUMNS = [
    LOW_SIGNIFICANCE_FLAG,
    SUSPICIOUS_ELASTICITY_FLAG,
    OVER_DOMINANT_CONTRIBUTION_FLAG,
    UNREALISTIC_ADSTOCK_FLAG,
    HAS_ANY_ISSUE_FLAG,
]


def _safe_true_count(series: pd.Series) -> int:
    """
    Count True values safely, treating NA/null as False.
    """
    return int(series.fillna(False).astype(bool).sum())


def compute_summary_metrics(
    df: pd.DataFrame,
    flag_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute summary metrics for dashboard cards.

    Parameters
    ----------
    df : pd.DataFrame
        Classifier result dataframe.
    flag_columns : list[str] | None
        Flag columns to count. Defaults to the known flag columns.

    Returns
    -------
    dict[str, Any]
        Dictionary with summary metrics.
    """
    flag_columns = flag_columns or DEFAULT_FLAG_COLUMNS

    required_columns = {CHANNEL_COLUMN, HAS_ANY_ISSUE_FLAG, VALIDATION_ERRORS_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"DataFrame is missing required columns for summary cards: {missing_str}"
        )

    total_channels = int(len(df))
    channels_with_issue = _safe_true_count(df[HAS_ANY_ISSUE_FLAG])

    validation_error_count = int(
        df[VALIDATION_ERRORS_COLUMN]
        .notna()
        .sum()
    )

    pct_channels_with_issue = (
        (channels_with_issue / total_channels) * 100 if total_channels > 0 else 0.0
    )

    flag_counts = {}
    for flag_col in flag_columns:
        if flag_col in df.columns:
            flag_counts[flag_col] = _safe_true_count(df[flag_col])

    metrics = {
        "total_channels": total_channels,
        "channels_with_issue": channels_with_issue,
        "validation_error_count": validation_error_count,
        "pct_channels_with_issue": pct_channels_with_issue,
        **flag_counts,
    }

    return metrics


def build_summary_cards(
    df: pd.DataFrame,
    title: str = "General summary",
) -> go.Figure:
    """
    Build Plotly indicator cards for the MMM quality check dashboard.

    Cards included:
    - Total channels
    - Channels with issue
    - Validation errors
    - % channels with issue
    - Count by each rule flag

    Parameters
    ----------
    df : pd.DataFrame
        Classifier result dataframe.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    metrics = compute_summary_metrics(df)

    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[
            [{"type": "indicator"}] * 4,
            [{"type": "indicator"}] * 4,
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )

    cards = [
        ("Total\n of\n Channels", metrics["total_channels"], "number"),
        ("Channels\n with\n issue", metrics["channels_with_issue"], "number"),
        ("Validation\n Errors", metrics["validation_error_count"], "number"),
        ("% issues", metrics["pct_channels_with_issue"], "percent"),
        ("Low\n significance", metrics.get(LOW_SIGNIFICANCE_FLAG, 0), "number"),
        ("Suspicious\n elasticity", metrics.get(SUSPICIOUS_ELASTICITY_FLAG, 0), "number"),
        (
            "Over-dominant contribution",
            metrics.get(OVER_DOMINANT_CONTRIBUTION_FLAG, 0),
            "number",
        ),
        ("Unrealistic adstock", metrics.get(UNREALISTIC_ADSTOCK_FLAG, 0), "number"),
    ]

    for idx, (label, value, value_type) in enumerate(cards):
        row = 1 if idx < 4 else 2
        col = idx + 1 if idx < 4 else idx - 3

        if value_type == "percent":
            display_value = value / 100.0
            number_format = ".1%"
        else:
            display_value = value
            number_format = ",d"

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=display_value,
                number={"valueformat": number_format},
                title={"text": label},
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=30, r=30, t=70, b=30),
        paper_bgcolor="white",
    )

    return fig


#################### HEATMAPS AND OTHER CHARTS CAN BE ADDED BELOW USING THE SAME STRUCTURE OF:

DEFAULT_HEATMAP_FLAG_COLUMNS = [
    LOW_SIGNIFICANCE_FLAG,
    SUSPICIOUS_ELASTICITY_FLAG,
    OVER_DOMINANT_CONTRIBUTION_FLAG,
    UNREALISTIC_ADSTOCK_FLAG,
    HAS_ANY_ISSUE_FLAG,
]


def build_flag_heatmap(
    df: pd.DataFrame,
    flag_columns: list[str] | None = None,
    title: str = "Heatmap de flags por canal",
) -> go.Figure:
    """
    Build a heatmap showing which rule flags were triggered for each channel.

    Visualization type:
        Channel-by-flag heatmap

    What it shows:
    - rows represent channels
    - columns represent rule flags
    - green cells indicate False (no issue)
    - red cells indicate True (issue found)
    - gray cells indicate missing values

    This chart is useful for quickly identifying which channels failed which
    validation rules, making it easier to inspect the overall issue pattern
    across the classifier output.
    """
    flag_columns = flag_columns or DEFAULT_HEATMAP_FLAG_COLUMNS

    required_columns = {CHANNEL_COLUMN, *flag_columns}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"DataFrame is missing required columns for flag heatmap: {missing_str}"
        )

    df_plot = df[[CHANNEL_COLUMN, *flag_columns]].copy()

    pretty_labels = {
        LOW_SIGNIFICANCE_FLAG: "Low significance",
        SUSPICIOUS_ELASTICITY_FLAG: "Suspicious elasticity",
        OVER_DOMINANT_CONTRIBUTION_FLAG: "Over-dominant contribution",
        UNREALISTIC_ADSTOCK_FLAG: "Unrealistic adstock",
        HAS_ANY_ISSUE_FLAG: "Has any issue",
    }

    heatmap_matrix = df_plot[flag_columns].apply(
        lambda col: col.map({False: 0.0, True: 1.0}).astype("Float64")
    ).fillna(0.5)

    text_matrix = df_plot[flag_columns].apply(
        lambda col: col.map({False: "False", True: "True"}).astype("string")
    ).fillna("Missing")

    x_labels = [pretty_labels.get(col, col) for col in flag_columns]
    y_labels = df_plot[CHANNEL_COLUMN].astype(str).tolist()

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=heatmap_matrix.to_numpy(dtype=float),
                x=x_labels,
                y=y_labels,
                text=text_matrix.to_numpy(),
                texttemplate="%{text}",
                textfont={"size": 11},
                colorscale=[
                    [0.00, "#c6efce"],  # False
                    [0.49, "#c6efce"],
                    [0.50, "#d9d9d9"],  # Missing
                    [0.51, "#f4cccc"],
                    [1.00, "#f4cccc"],  # True
                ],
                zmin=0,
                zmax=1,
                showscale=False,
                hovertemplate=(
                    "Canal: %{y}<br>"
                    "Flag: %{x}<br>"
                    "Valor: %{text}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=max(350, 45 * len(y_labels)),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")

    return fig



##################### BAR CHART #########################

def build_contribution_bar_chart(
    df: pd.DataFrame,
    title: str = "Contribution por canal",
    sort_desc: bool = True,
) -> go.Figure:
    """
    Build a horizontal bar chart showing channel contribution percentages.

    Visualization type:
        Horizontal bar chart

    What it shows:
    - each bar represents the contribution percentage of one channel
    - bar color indicates whether the channel has any issue
    - dashed reference lines show the minimum and maximum acceptable
      contribution thresholds

    This chart is useful for identifying which channels concentrate most
    of the model contribution and whether the most important channels are
    also the ones presenting quality issues.
    """
    required_columns = {CHANNEL_COLUMN, "contribution_pct", HAS_ANY_ISSUE_FLAG}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"DataFrame is missing required columns for contribution bar chart: {missing_str}"
        )

    df_plot = df[[CHANNEL_COLUMN, "contribution_pct", HAS_ANY_ISSUE_FLAG]].copy()
    df_plot["contribution_pct"] = pd.to_numeric(
        df_plot["contribution_pct"],
        errors="coerce",
    )

    if VALIDATION_ERRORS_COLUMN in df.columns:
        df_plot[VALIDATION_ERRORS_COLUMN] = df[VALIDATION_ERRORS_COLUMN].astype("string")
    else:
        df_plot[VALIDATION_ERRORS_COLUMN] = "Missing"

    if sort_desc:
        df_plot = df_plot.sort_values(
            by="contribution_pct",
            ascending=False,
            na_position="last",
        )

    def _issue_color(value: object) -> str:
        if pd.isna(value):
            return "#7f7f7f"   # cinza
        if bool(value):
            return "#d62728"   # vermelho
        return "#2ca02c"       # verde

    df_plot["bar_color"] = df_plot[HAS_ANY_ISSUE_FLAG].apply(_issue_color)
    df_plot["issue_label"] = df_plot[HAS_ANY_ISSUE_FLAG].map(
        {True: "Issue", False: "No issue"}
    ).astype("string").fillna("Missing")

    customdata = np.column_stack(
        [
            df_plot[CHANNEL_COLUMN].astype(str),
            df_plot["contribution_pct"].astype(str),
            df_plot["issue_label"].astype(str),
            df_plot[VALIDATION_ERRORS_COLUMN].fillna("None").astype(str),
        ]
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_plot["contribution_pct"],
                y=df_plot[CHANNEL_COLUMN],
                orientation="h",
                marker=dict(
                    color=df_plot["bar_color"],
                    line=dict(color="black", width=0.5),
                ),
                customdata=customdata,
                hovertemplate=(
                    "Canal: %{customdata[0]}<br>"
                    "Contribution: %{customdata[1]}<br>"
                    "Status: %{customdata[2]}<br>"
                    "Validation errors: %{customdata[3]}<extra></extra>"
                ),
            )
        ]
    )

    fig.add_vline(
        x=CONTRIBUTION_MIN,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )
    fig.add_vline(
        x=CONTRIBUTION_MAX,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )

    fig.update_layout(
        title=title,
        height=max(400, 35 * len(df_plot)),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis_title="Contribution (%)",
        yaxis_title="Channel",
        showlegend=False,
    )

    fig.update_yaxes(autorange="reversed")

    return fig


######## SCATER ############
def build_elasticity_vs_contribution_scatter(
    df: pd.DataFrame,
    title: str = "Elasticity vs Contribution by Channel",
) -> go.Figure:
    """
    Build a scatter plot comparing elasticity and contribution percentage
    for each channel.

    Visualization type:
        Scatter plot

    What it shows:
    - each point represents one channel
    - the x-axis shows elasticity
    - the y-axis shows contribution percentage
    - point color indicates whether the channel is within the expected range
    - dashed reference lines show the acceptable elasticity and
      contribution thresholds

    This chart is useful for identifying channels with suspicious
    combinations, such as:
    - negative elasticity
    - elasticity above the expected range
    - very low contribution
    - contribution above the expected maximum
    """

    required_columns = {
        CHANNEL_COLUMN,
        "Elasticity",
        "contribution_pct",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            "DataFrame is missing required columns for elasticity vs contribution "
            f"scatter plot: {missing_str}"
        )

    plot_df = df[[CHANNEL_COLUMN, "Elasticity", "contribution_pct"]].copy()

    plot_df["Elasticity"] = pd.to_numeric(plot_df["Elasticity"], errors="coerce")
    plot_df["contribution_pct"] = pd.to_numeric(
        plot_df["contribution_pct"],
        errors="coerce",
    )

    def _classify_point(elasticity: float, contribution: float) -> str:
        if pd.isna(elasticity) or pd.isna(contribution):
            return "Missing"

        elasticity_issue = (
            elasticity < ELASTICITY_MIN or elasticity > ELASTICITY_MAX
        )
        contribution_issue = (
            contribution < CONTRIBUTION_MIN or contribution > CONTRIBUTION_MAX
        )

        if elasticity_issue or contribution_issue:
            return "Issue"

        return "In range"

    status_colors = {
        "In range": "#2ca02c",   # green
        "Issue": "#d62728",      # red
        "Missing": "#7f7f7f",    # gray
    }

    plot_df["status"] = plot_df.apply(
        lambda row: _classify_point(row["Elasticity"], row["contribution_pct"]),
        axis=1,
    )

    fig = go.Figure()

    for status in ["In range", "Issue", "Missing"]:
        subset = plot_df[plot_df["status"] == status]
        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=subset["Elasticity"],
                y=subset["contribution_pct"],
                mode="markers+text",
                text=subset[CHANNEL_COLUMN],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=status_colors[status],
                    line=dict(color="black", width=0.7),
                ),
                name=status,
                customdata=list(
                    zip(
                        subset[CHANNEL_COLUMN].astype(str),
                        subset["Elasticity"].astype(str),
                        subset["contribution_pct"].astype(str),
                        subset["status"].astype(str),
                    )
                ),
                hovertemplate=(
                    "Channel: %{customdata[0]}<br>"
                    "Elasticity: %{customdata[1]}<br>"
                    "Contribution: %{customdata[2]}<br>"
                    "Status: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=ELASTICITY_MIN,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )
    fig.add_vline(
        x=ELASTICITY_MAX,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )
    fig.add_hline(
        y=CONTRIBUTION_MIN,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )
    fig.add_hline(
        y=CONTRIBUTION_MAX,
        line_width=1,
        line_dash="dash",
        line_color="black",
    )

    fig.update_layout(
        title=title,
        height=550,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis_title="Elasticity",
        yaxis_title="Contribution (%)",
        legend_title="Point status",
    )

    return fig


################################ build_parameter_range_chart


DEFAULT_PARAMETER_RANGE_THRESHOLDS = {
    "p_value": {"min": 0.0, "max": LOW_SIGNIFICANCE_THRESHOLD},
    "Elasticity": {"min": ELASTICITY_MIN, "max": ELASTICITY_MAX},
    "contribution_pct": {"min": CONTRIBUTION_MIN, "max": CONTRIBUTION_MAX},
    "adstock_half_life": {"min": ADSTOCK_MIN, "max": ADSTOCK_MAX},
}

DEFAULT_PARAMETER_RANGE_COLORS = {
    "In range": "#2ca02c",   # green
    "Issue": "#d62728",      # red
    "Missing": "#7f7f7f",    # gray
}


def build_parameter_range_chart(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Acceptable Range by Metric",
    height_per_metric: int = 320,
) -> go.Figure:
    """
    Build a multi-panel chart showing the acceptable range for each metric
    and the observed value of each channel.

    Visualization type:
        Range bar + point overlay chart

    What it shows:
    - one subplot per metric
    - a horizontal bar representing the acceptable interval
    - one point per channel representing the observed metric value
    - point color indicating whether the value is within range, out of range,
      or missing

    This chart is useful for comparing channel-level values against the
    thresholds defined in the quality check constants. Because the thresholds
    are read directly from the constants module, any update to the limits is
    automatically reflected in the visualization.
    """
    thresholds = DEFAULT_PARAMETER_RANGE_THRESHOLDS
    metrics = metrics or list(thresholds.keys())

    required_columns = {CHANNEL_COLUMN, *metrics}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            "DataFrame is missing required columns for parameter range chart: "
            f"{missing_str}"
        )

    plot_df = df[[CHANNEL_COLUMN, *metrics]].copy()

    for metric in metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    def _classify_metric_value(
        value: float,
        min_value: float,
        max_value: float,
    ) -> str:
        if pd.isna(value):
            return "Missing"
        if value < min_value or value > max_value:
            return "Issue"
        return "In range"

    subplot_titles = []
    for metric in metrics:
        min_value = thresholds[metric]["min"]
        max_value = thresholds[metric]["max"]
        subplot_titles.append(
            f"{metric} | acceptable range: {min_value} to {max_value}"
        )

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    for row_idx, metric in enumerate(metrics, start=1):
        min_value = thresholds[metric]["min"]
        max_value = thresholds[metric]["max"]

        metric_df = plot_df[[CHANNEL_COLUMN, metric]].copy()
        metric_df["status"] = metric_df[metric].apply(
            lambda value: _classify_metric_value(value, min_value, max_value)
        )

        range_width = max_value - min_value

        fig.add_trace(
            go.Bar(
                y=metric_df[CHANNEL_COLUMN],
                x=[range_width] * len(metric_df),
                base=[min_value] * len(metric_df),
                orientation="h",
                marker=dict(color="rgba(100, 100, 100, 0.18)"),
                name="Acceptable range",
                hovertemplate=(
                    f"{metric}<br>"
                    f"Acceptable range: {min_value} to {max_value}<extra></extra>"
                ),
                showlegend=(row_idx == 1),
            ),
            row=row_idx,
            col=1,
        )

        for status in ["In range", "Issue", "Missing"]:
            subset = metric_df[metric_df["status"] == status]
            if subset.empty:
                continue

            customdata = np.column_stack(
                [
                    subset[CHANNEL_COLUMN].astype(str),
                    subset[metric].astype(str),
                    subset["status"].astype(str),
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=subset[metric],
                    y=subset[CHANNEL_COLUMN],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color=DEFAULT_PARAMETER_RANGE_COLORS[status],
                        line=dict(color="black", width=0.6),
                    ),
                    name=status,
                    showlegend=(row_idx == 1),
                    customdata=customdata,
                    hovertemplate=(
                        "Channel: %{customdata[0]}<br>"
                        f"{metric}: %{{customdata[1]}}<br>"
                        "Status: %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

        fig.add_vline(
            x=min_value,
            line_width=1,
            line_dash="dash",
            line_color="black",
            row=row_idx,
            col=1,
        )
        fig.add_vline(
            x=max_value,
            line_width=1,
            line_dash="dash",
            line_color="black",
            row=row_idx,
            col=1,
        )

        fig.update_yaxes(autorange="reversed", row=row_idx, col=1)

    fig.update_layout(
        title=title,
        height=height_per_metric * len(metrics),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Metric status",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig
