"""Plotly figures for comparing model-evaluation outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.experiment_viewer.services import EvaluationMetricDefinition


INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
PAPER = "#FFFFFF"
MODEL_COLORS = (
    "#315A9E",
    "#D97706",
    "#2E7D68",
    "#B54747",
    "#7F56D9",
    "#0E7490",
    "#A15C38",
    "#475467",
)


EvaluationSeries = tuple[str, Sequence[Mapping[str, Any]]]
DistributionSeries = tuple[str, Sequence[float]]


def build_evaluation_metric_figure(
    series: Sequence[EvaluationSeries],
    metric: EvaluationMetricDefinition,
) -> go.Figure:
    figure = go.Figure()
    for index, (label, rows) in enumerate(series):
        valid_rows = [row for row in rows if row.get(metric.key) is not None]
        figure.add_trace(
            go.Scatter(
                x=[int(row["horizon_s"]) for row in valid_rows],
                y=[float(row[metric.key]) for row in valid_rows],
                mode="lines+markers",
                name=label,
                line={"color": _color(index), "width": 2.2},
                marker={"color": _color(index), "size": 7},
                hovertemplate=(
                    f"{label}<br>Horizon: %{{x}} s"
                    f"<br>{metric.label}: %{{y:,.2f}} m<extra></extra>"
                ),
            )
        )
    _apply_common_layout(figure, metric.axis_title)
    figure.update_xaxes(title_text="Prediction horizon (s)", dtick=None)
    if not figure.data:
        _add_empty_annotation(figure, "Select at least one compatible model run")
    return figure


def build_evaluation_sample_figure(
    series: Sequence[EvaluationSeries],
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.52, 0.48],
    )
    all_coverages: list[float] = []
    for index, (label, rows) in enumerate(series):
        horizons = [int(row["horizon_s"]) for row in rows]
        coverage = [float(row.get("coverage") or 0.0) for row in rows]
        targets = [int(row.get("evaluated_targets") or 0) for row in rows]
        all_coverages.extend(coverage)
        figure.add_trace(
            go.Scatter(
                x=horizons,
                y=coverage,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line={"color": _color(index), "width": 2},
                marker={"size": 6},
                hovertemplate=(
                    f"{label}<br>Horizon: %{{x}} s"
                    "<br>Coverage: %{y:.1%}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=horizons,
                y=targets,
                name=label,
                legendgroup=label,
                showlegend=False,
                marker={"color": _color(index)},
                opacity=0.72,
                hovertemplate=(
                    f"{label}<br>Horizon: %{{x}} s"
                    "<br>Evaluated targets: %{y:,}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    coverage_floor = max(0.0, min(all_coverages, default=0.8) - 0.04)
    _apply_common_layout(figure, None, margin={"l": 78, "r": 24, "t": 48, "b": 56})
    figure.update_yaxes(
        title_text="Coverage",
        tickformat=".0%",
        range=[coverage_floor, 1.01],
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="Evaluated targets", tickformat=",~s", row=2, col=1)
    figure.update_xaxes(title_text="Prediction horizon (s)", row=2, col=1)
    figure.update_layout(barmode="group")
    if not series:
        _add_empty_annotation(figure, "No evaluation coverage is available")
    return figure


def build_segment_distribution_figure(
    series: Sequence[DistributionSeries],
    metric: EvaluationMetricDefinition,
    horizon_s: int,
) -> go.Figure:
    figure = go.Figure()
    for index, (label, values) in enumerate(series):
        if not values:
            continue
        figure.add_trace(
            go.Box(
                x=[label] * len(values),
                y=list(values),
                name=label,
                marker={"color": _color(index), "size": 3},
                line={"color": _color(index), "width": 1.5},
                fillcolor=_rgba(_color(index), 0.16),
                boxpoints="outliers",
                jitter=0.2,
                pointpos=0,
                hovertemplate=(
                    f"{label}<br>{metric.label}: %{{y:,.2f}} m<extra></extra>"
                ),
            )
        )
    _apply_common_layout(figure, metric.axis_title)
    figure.update_xaxes(title_text=f"Model · {int(horizon_s)} s horizon")
    figure.update_layout(showlegend=False)
    if not figure.data:
        _add_empty_annotation(
            figure, f"No segment-level values are available at {int(horizon_s)} s"
        )
    return figure


def _apply_common_layout(
    figure: go.Figure,
    y_title: str | None,
    margin: dict[str, int] | None = None,
) -> None:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin=margin or {"l": 76, "r": 24, "t": 48, "b": 58},
        hovermode="x unified",
        font={"family": "Segoe UI, Arial, sans-serif", "color": INK, "size": 12},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 10, "color": MUTED},
        },
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
    )
    yaxis_style = {
        "showgrid": True,
        "gridcolor": GRID,
        "zeroline": False,
        "showline": True,
        "linecolor": GRID,
        "ticks": "outside",
        "tickcolor": GRID,
    }
    if y_title is not None:
        yaxis_style["title_text"] = y_title
    figure.update_yaxes(
        **yaxis_style,
    )


def _add_empty_annotation(figure: go.Figure, message: str) -> None:
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": MUTED, "size": 13},
    )


def _color(index: int) -> str:
    return MODEL_COLORS[index % len(MODEL_COLORS)]


def _rgba(hex_color: str, alpha: float) -> str:
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"
