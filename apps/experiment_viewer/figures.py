"""Plotly figure builders for the trajectory viewer."""

from __future__ import annotations

from math import exp, isfinite, pi, sqrt
from statistics import fmean, median, pstdev
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.experiment_viewer.services import (
    ProcessedTrackData,
    SamplingDistributionSummary,
    TrackData,
)


INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
BLUE = "#315A9E"
GREEN = "#2E7D68"
RED = "#B54747"
ORANGE = "#D97706"
PAPER = "#FFFFFF"
SAMPLING_DETAIL_LIMIT_SECONDS = 2.0
SAMPLING_DETAIL_BIN_COUNT = 40


def build_time_series_figure(
    track: TrackData,
    view_revision: str | None = None,
    processed_track: ProcessedTrackData | None = None,
    data_view: str = "raw",
    show_interpolated: bool = True,
    show_suspicious: bool = True,
) -> go.Figure:
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
    )
    specifications = (
        (track.longitude, processed_track.longitude if processed_track else (), "Longitude", "Longitude (°)"),
        (track.latitude, processed_track.latitude if processed_track else (), "Latitude", "Latitude (°)"),
        (
            track.geo_altitude,
            processed_track.geo_altitude if processed_track else (),
            "Geometric altitude",
            "Geometric altitude (m)",
        ),
        (
            track.baro_altitude,
            processed_track.baro_altitude if processed_track else (),
            "Barometric altitude",
            "Barometric altitude (m)",
        ),
    )

    raw_visible = data_view in {"raw", "compare"}
    processed_available = processed_track is not None and processed_track.point_count > 0
    processed_visible = data_view in {"processed", "compare"} and processed_available

    for row, (raw_values, processed_values, name, y_title) in enumerate(
        specifications, start=1
    ):
        if raw_visible:
            if data_view == "raw":
                times, raw_series = _insert_time_gaps(track.time, (raw_values,))
                figure.add_trace(
                    go.Scattergl(
                        x=times,
                        y=raw_series[0],
                        mode="lines+markers",
                        name=name,
                        line={"color": BLUE, "width": 1.5},
                        marker={"color": "rgba(49, 90, 158, 0.72)", "size": 4.5},
                        hovertemplate=f"{name}: %{{y:,.3f}}<extra></extra>",
                        connectgaps=False,
                    ),
                    row=row,
                    col=1,
                )
            else:
                times, raw_series = _insert_time_gaps(track.time, (raw_values,))
                figure.add_trace(
                    go.Scattergl(
                        x=times,
                        y=raw_series[0],
                        mode="lines+markers",
                        name="Raw reports",
                        legendgroup="raw",
                        showlegend=row == 1,
                        line={"color": BLUE, "width": 1.35},
                        marker={"color": "rgba(49, 90, 158, 0.72)", "size": 4.5},
                        hovertemplate=(
                            "Raw report<br>Time: %{x:,.3f} s"
                            f"<br>{name}: %{{y:,.3f}}<extra></extra>"
                        ),
                        connectgaps=False,
                    ),
                    row=row,
                    col=1,
                )

        if processed_visible and processed_track is not None:
            line_time, line_series = _insert_segment_gaps(
                processed_track.time,
                processed_track.segment_id,
                (processed_values,),
            )
            figure.add_trace(
                go.Scattergl(
                    x=line_time,
                    y=line_series[0],
                    mode="lines",
                    name="Standardized trajectory",
                    legendgroup="processed-line",
                    showlegend=row == 1,
                    line={"color": ORANGE, "width": 1.45},
                    hoverinfo="skip",
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )
            _add_processed_time_markers(
                figure,
                processed_track,
                processed_values,
                name,
                row,
                show_interpolated,
                show_suspicious,
            )
        figure.update_yaxes(title_text=y_title, row=row, col=1)

    if data_view == "processed" and not processed_available:
        figure.add_annotation(
            text="No standardized trajectory available for this aircraft",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 13},
        )

    figure.update_xaxes(title_text="Relative time (s)", row=4, col=1)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={
            "l": 72,
            "r": 22,
            "t": 52 if data_view != "raw" else 24,
            "b": 58,
        },
        hovermode="x unified",
        showlegend=data_view != "raw",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.01,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 10, "color": MUTED},
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": INK, "size": 12},
        uirevision=view_revision or f"time-{track.aircraft}",
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
    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
    )
    return figure


def build_spatial_figure(
    track: TrackData,
    view_revision: str | None = None,
    processed_track: ProcessedTrackData | None = None,
    data_view: str = "raw",
    show_interpolated: bool = True,
    show_suspicious: bool = True,
) -> go.Figure:
    valid_points = [
        (time, longitude, latitude)
        for time, longitude, latitude in zip(
            track.time,
            track.longitude,
            track.latitude,
            strict=True,
        )
        if _valid_number(longitude) and _valid_number(latitude)
    ]

    processed_valid = processed_track is not None and processed_track.point_count > 0
    figure = go.Figure()
    if data_view == "raw" and valid_points:
        time = tuple(point[0] for point in valid_points)
        longitude = tuple(point[1] for point in valid_points)
        latitude = tuple(point[2] for point in valid_points)
        _, spatial_series = _insert_time_gaps(time, (longitude, latitude))

        figure.add_trace(
            go.Scattergl(
                x=spatial_series[0],
                y=spatial_series[1],
                mode="lines+markers",
                line={"color": BLUE, "width": 2},
                marker={
                    "color": "rgba(49, 90, 158, 0.72)",
                    "size": 4.5,
                },
                name="Trajectory",
                hovertemplate="Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>",
                connectgaps=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[longitude[0]],
                y=[latitude[0]],
                mode="markers",
                name="Start",
                marker={"color": GREEN, "size": 9, "line": {"color": PAPER, "width": 1}},
                hovertemplate="Start<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scattergl(
                x=[longitude[-1]],
                y=[latitude[-1]],
                mode="markers",
                name="Current position",
                marker={"color": RED, "size": 9},
                customdata=[time[-1]],
                hovertemplate=(
                    "Current position<br>Time: %{customdata:,.3f}"
                    "<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>"
                ),
            )
        )
    elif data_view != "raw":
        if data_view == "compare" and valid_points:
            raw_time = tuple(point[0] for point in valid_points)
            raw_longitude = tuple(point[1] for point in valid_points)
            raw_latitude = tuple(point[2] for point in valid_points)
            raw_time_with_gaps, raw_spatial_series = _insert_time_gaps(
                raw_time,
                (raw_longitude, raw_latitude),
            )
            figure.add_trace(
                go.Scattergl(
                    x=raw_spatial_series[0],
                    y=raw_spatial_series[1],
                    mode="lines+markers",
                    name="Raw reports",
                    line={"color": BLUE, "width": 1.5},
                    marker={"color": "rgba(49, 90, 158, 0.72)", "size": 4.5},
                    customdata=raw_time_with_gaps,
                    hovertemplate=(
                        "Raw report<br>Time: %{customdata:,.3f} s"
                        "<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>"
                    ),
                    connectgaps=False,
                )
            )
        if processed_valid and processed_track is not None:
            line_time, line_series = _insert_segment_gaps(
                processed_track.time,
                processed_track.segment_id,
                (processed_track.longitude, processed_track.latitude),
            )
            figure.add_trace(
                go.Scattergl(
                    x=line_series[0],
                    y=line_series[1],
                    mode="lines",
                    name="Standardized trajectory",
                    line={"color": ORANGE, "width": 1.8},
                    customdata=line_time,
                    hovertemplate=(
                        "Standardized trajectory<br>Time: %{customdata:,.3f} s"
                        "<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>"
                    ),
                    connectgaps=False,
                )
            )
            _add_processed_spatial_markers(
                figure,
                processed_track,
                show_interpolated,
                show_suspicious,
            )
            figure.add_trace(
                go.Scatter(
                    x=[processed_track.longitude[0]],
                    y=[processed_track.latitude[0]],
                    mode="markers",
                    name="Standardized start",
                    showlegend=False,
                    marker={"color": GREEN, "size": 9, "line": {"color": PAPER, "width": 1}},
                    hovertemplate=(
                        "Standardized start<br>Lon: %{x:.5f}°"
                        "<br>Lat: %{y:.5f}°<extra></extra>"
                    ),
                )
            )
            figure.add_trace(
                go.Scattergl(
                    x=[processed_track.longitude[-1]],
                    y=[processed_track.latitude[-1]],
                    mode="markers",
                    name="Current standardized position",
                    showlegend=False,
                    marker={"color": RED, "size": 9},
                    customdata=[processed_track.time[-1]],
                    hovertemplate=(
                        "Current standardized position<br>Time: %{customdata:,.3f} s"
                        "<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>"
                    ),
                )
            )
        elif not valid_points or data_view == "processed":
            figure.add_annotation(
                text="No standardized trajectory available for this aircraft",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"color": MUTED, "size": 13},
            )
    else:
        figure.add_annotation(
            text="No valid longitude/latitude observations",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 13},
        )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={
            "l": 70,
            "r": 24,
            "t": 54 if data_view != "raw" else 24,
            "b": 58,
        },
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.04,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 11, "color": MUTED},
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": INK, "size": 12},
        uirevision=view_revision or f"spatial-{track.aircraft}",
    )
    figure.update_xaxes(
        title_text="Longitude (°)",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
    )
    figure.update_yaxes(
        title_text="Latitude (°)",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
    )
    return figure


def build_sampling_interval_figure(
    track: TrackData,
    view_revision: str | None = None,
) -> go.Figure:
    sample_times = tuple(float(value) for value in track.time[1:])
    sampling_intervals = tuple(
        float(current - previous)
        for previous, current in zip(track.time, track.time[1:])
    )

    figure = go.Figure()
    if sampling_intervals:
        figure.add_trace(
            go.Scattergl(
                x=sample_times,
                y=sampling_intervals,
                mode="lines+markers",
                name="Sampling interval",
                line={"color": BLUE, "width": 1.4},
                marker={"color": "rgba(49, 90, 158, 0.62)", "size": 3.2},
                hovertemplate=(
                    "Relative time: %{x:,.3f} s"
                    "<br>Sampling interval: %{y:,.3f} s<extra></extra>"
                ),
            )
        )
        median_interval = median(sampling_intervals)
        figure.add_hline(
            y=median_interval,
            line={"color": GREEN, "width": 1.2, "dash": "dash"},
            annotation_text=f"Median: {median_interval:,.3f} s",
            annotation_position="top right",
            annotation_font={"color": GREEN, "size": 11},
        )
    else:
        figure.add_annotation(
            text="At least two observations are required",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 13},
        )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={"l": 78, "r": 28, "t": 28, "b": 62},
        hovermode="closest",
        showlegend=False,
        font={"family": "Segoe UI, Arial, sans-serif", "color": INK, "size": 12},
        uirevision=view_revision or f"sampling-{track.aircraft}",
    )
    figure.update_xaxes(
        title_text="Relative time (s)",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
    )
    figure.update_yaxes(
        title_text="Sampling interval Δt (s)",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
    )
    return figure


def build_sampling_distribution_figure(
    track: TrackData,
    view_revision: str | None = None,
) -> go.Figure:
    sampling_intervals = tuple(
        float(current - previous)
        for previous, current in zip(track.time, track.time[1:])
        if isfinite(current - previous) and current >= previous
    )
    return build_sampling_distribution_from_intervals(
        sampling_intervals,
        view_revision=view_revision or f"sampling-distribution-{track.aircraft}",
    )


def build_sampling_distribution_from_intervals(
    intervals: Sequence[float],
    view_revision: str = "sampling-distribution-global",
) -> go.Figure:
    sampling_intervals = tuple(
        float(value) for value in intervals if isfinite(value) and value >= 0
    )
    total_count = len(sampling_intervals)
    regular_intervals = tuple(
        value
        for value in sampling_intervals
        if value <= SAMPLING_DETAIL_LIMIT_SECONDS
    )
    long_gaps = tuple(
        value
        for value in sampling_intervals
        if value > SAMPLING_DETAIL_LIMIT_SECONDS
    )
    _, _, _, detail_counts = _density_histogram(
        regular_intervals,
        0.0,
        SAMPLING_DETAIL_LIMIT_SECONDS,
        SAMPLING_DETAIL_BIN_COUNT,
        max(total_count, 1),
    )
    gap_bin_count = min(24, max(6, round(sqrt(len(long_gaps)))))
    maximum_interval = (
        max(long_gaps) if long_gaps else SAMPLING_DETAIL_LIMIT_SECONDS
    )
    _, _, _, gap_counts = _density_histogram(
        long_gaps,
        SAMPLING_DETAIL_LIMIT_SECONDS,
        maximum_interval,
        gap_bin_count,
        max(total_count, 1),
    )
    summary = SamplingDistributionSummary(
        total_count=total_count,
        detail_count=len(regular_intervals),
        detail_mean=fmean(regular_intervals) if regular_intervals else 0.0,
        detail_standard_deviation=(
            pstdev(regular_intervals) if regular_intervals else 0.0
        ),
        detail_limit=SAMPLING_DETAIL_LIMIT_SECONDS,
        detail_counts=detail_counts,
        maximum_interval=maximum_interval,
        gap_counts=gap_counts,
    )
    return build_sampling_distribution_from_summary(summary, view_revision)


def build_sampling_distribution_from_summary(
    summary: SamplingDistributionSummary,
    view_revision: str = "sampling-distribution-global",
) -> go.Figure:
    detail_limit = summary.detail_limit

    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.78, 0.22],
        horizontal_spacing=0.06,
        subplot_titles=(
            f"0–{detail_limit:g} s detail",
            f">{detail_limit:g} s gaps",
        ),
    )
    if summary.total_count:
        regular_x, regular_density, regular_width = _density_from_counts(
            summary.detail_counts,
            0.0,
            detail_limit,
            summary.total_count,
        )
        figure.add_trace(
            go.Bar(
                x=regular_x,
                y=regular_density,
                width=[regular_width * 0.96] * len(regular_x),
                customdata=summary.detail_counts,
                name=f"Observed intervals (0–{detail_limit:g} s)",
                marker={
                    "color": "rgba(49, 90, 158, 0.62)",
                    "line": {"color": PAPER, "width": 0.5},
                },
                hovertemplate=(
                    "Sampling interval: %{x:,.3f} s"
                    "<br>Count: %{customdata:,}"
                    "<br>Density: %{y:,.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        mean_interval = summary.detail_mean
        standard_deviation = summary.detail_standard_deviation
        if standard_deviation > 0:
            curve_x = tuple(
                detail_limit * index / 240
                for index in range(241)
            )
            regular_fraction = summary.detail_count / summary.total_count
            coefficient = regular_fraction / (standard_deviation * sqrt(2 * pi))
            curve_y = tuple(
                coefficient
                * exp(-0.5 * ((value - mean_interval) / standard_deviation) ** 2)
                for value in curve_x
            )
            figure.add_trace(
                go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode="lines",
                    name=(
                        f"Normal fit (μ={mean_interval:,.3f} s, "
                        f"σ={standard_deviation:,.3f} s)"
                    ),
                    line={"color": RED, "width": 2},
                    hovertemplate=(
                        "Sampling interval: %{x:,.3f} s"
                        "<br>Normal density: %{y:,.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        gap_count = summary.total_count - summary.detail_count
        if gap_count:
            gap_x, gap_density, gap_width = _density_from_counts(
                summary.gap_counts,
                detail_limit,
                summary.maximum_interval,
                summary.total_count,
            )
            figure.add_trace(
                go.Bar(
                    x=gap_x,
                    y=gap_density,
                    width=[gap_width * 0.96] * len(gap_x),
                    customdata=summary.gap_counts,
                    name=f"Long gaps (>{detail_limit:g} s)",
                    marker={
                        "color": "rgba(181, 71, 71, 0.66)",
                        "line": {"color": PAPER, "width": 0.5},
                    },
                    hovertemplate=(
                        "Gap: %{x:,.3f} s"
                        "<br>Count: %{customdata:,}"
                        "<br>Density: %{y:,.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
            gap_percentage = 100 * gap_count / summary.total_count
            figure.add_annotation(
                text=f"Long gaps: {gap_count:,} ({gap_percentage:.2f}%)",
                x=0.5,
                y=0.98,
                xref="x2 domain",
                yref="paper",
                showarrow=False,
                font={"color": RED, "size": 11},
            )
        else:
            figure.add_annotation(
                text=f"No intervals > {detail_limit:g} s",
                x=0.5,
                y=0.5,
                xref="x2 domain",
                yref="paper",
                showarrow=False,
                font={"color": MUTED, "size": 12},
            )
    else:
        figure.add_annotation(
            text="At least two observations are required",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 13},
        )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={"l": 78, "r": 62, "t": 58, "b": 62},
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.04,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 11, "color": MUTED},
        },
        font={"family": "Segoe UI, Arial, sans-serif", "color": INK, "size": 12},
        uirevision=view_revision,
    )
    figure.update_xaxes(
        title_text="Sampling interval Δt (s)",
        range=[0, detail_limit],
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title_text="Long gap Δt (s)",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="Probability density",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Tail density",
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=True,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        row=1,
        col=2,
    )
    return figure


def _density_from_counts(
    counts: Sequence[int],
    lower_bound: float,
    upper_bound: float,
    total_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    span = max(upper_bound - lower_bound, 0.001)
    width = span / max(len(counts), 1)
    centers = tuple(lower_bound + (index + 0.5) * width for index in range(len(counts)))
    density = tuple(count / (total_count * width) for count in counts)
    return centers, density, width


def _density_histogram(
    values: Sequence[float],
    lower_bound: float,
    upper_bound: float,
    bin_count: int,
    total_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...], float, tuple[int, ...]]:
    span = max(upper_bound - lower_bound, 0.001)
    width = span / max(bin_count, 1)
    counts = [0] * max(bin_count, 1)
    for value in values:
        index = min(int((value - lower_bound) / width), len(counts) - 1)
        index = max(index, 0)
        counts[index] += 1
    centers, density, width = _density_from_counts(
        counts,
        lower_bound,
        upper_bound,
        total_count,
    )
    return centers, density, width, tuple(counts)


def _add_processed_time_markers(
    figure: go.Figure,
    track: ProcessedTrackData,
    values: Sequence[float | None],
    value_name: str,
    row: int,
    show_interpolated: bool,
    show_suspicious: bool,
) -> None:
    marker_groups = [
        (
            "Observed 1 Hz points",
            lambda index: not track.is_interpolated[index]
            and not track.is_speed_suspicious[index],
            {"color": ORANGE, "size": 5, "symbol": "circle"},
            "observed",
        )
    ]
    if show_interpolated:
        marker_groups.append(
            (
                "Interpolated points",
                lambda index: track.is_interpolated[index]
                and not track.is_speed_suspicious[index],
                {
                    "color": ORANGE,
                    "size": 7,
                    "symbol": "circle-open",
                    "line": {"color": ORANGE, "width": 1.3},
                },
                "interpolated",
            )
        )
    if show_suspicious:
        marker_groups.append(
            (
                "Suspicious transition endpoint",
                lambda index: track.is_speed_suspicious[index],
                {"color": RED, "size": 8, "symbol": "x", "line": {"width": 1.3}},
                "suspicious",
            )
        )

    for label, predicate, marker, legend_group in marker_groups:
        indices = [
            index
            for index, value in enumerate(values)
            if predicate(index) and _valid_number(value)
        ]
        if not indices:
            continue
        figure.add_trace(
            go.Scattergl(
                x=[track.time[index] for index in indices],
                y=[values[index] for index in indices],
                mode="markers",
                name=label,
                legendgroup=legend_group,
                showlegend=row == 1,
                marker=marker,
                customdata=[track.segment_id[index] for index in indices],
                hovertemplate=(
                    f"{label}<br>Time: %{{x:,.3f}} s"
                    f"<br>{value_name}: %{{y:,.3f}}"
                    "<br>Segment: %{customdata}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def _add_processed_spatial_markers(
    figure: go.Figure,
    track: ProcessedTrackData,
    show_interpolated: bool,
    show_suspicious: bool,
) -> None:
    marker_groups = [
        (
            "Observed 1 Hz points",
            lambda index: not track.is_interpolated[index]
            and not track.is_speed_suspicious[index],
            {"color": ORANGE, "size": 5, "symbol": "circle"},
        )
    ]
    if show_interpolated:
        marker_groups.append(
            (
                "Interpolated points",
                lambda index: track.is_interpolated[index]
                and not track.is_speed_suspicious[index],
                {
                    "color": ORANGE,
                    "size": 7,
                    "symbol": "circle-open",
                    "line": {"color": ORANGE, "width": 1.3},
                },
            )
        )
    if show_suspicious:
        marker_groups.append(
            (
                "Suspicious transition endpoint",
                lambda index: track.is_speed_suspicious[index],
                {"color": RED, "size": 8, "symbol": "x", "line": {"width": 1.3}},
            )
        )

    for label, predicate, marker in marker_groups:
        indices = [index for index in range(track.point_count) if predicate(index)]
        if not indices:
            continue
        figure.add_trace(
            go.Scattergl(
                x=[track.longitude[index] for index in indices],
                y=[track.latitude[index] for index in indices],
                mode="markers",
                name=label,
                marker=marker,
                customdata=[
                    [track.time[index], track.segment_id[index]] for index in indices
                ],
                hovertemplate=(
                    f"{label}<br>Time: %{{customdata[0]:,.3f}} s"
                    "<br>Segment: %{customdata[1]}"
                    "<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>"
                ),
            )
        )


def _insert_segment_gaps(
    time: Sequence[float],
    segment_id: Sequence[int],
    series: Sequence[Sequence[float | None]],
) -> tuple[tuple[float | None, ...], tuple[tuple[float | None, ...], ...]]:
    output_time: list[float | None] = []
    output_series: list[list[float | None]] = [[] for _ in series]
    for index, current_time in enumerate(time):
        if index and segment_id[index] != segment_id[index - 1]:
            output_time.append(None)
            for values in output_series:
                values.append(None)
        output_time.append(current_time)
        for output, values in zip(output_series, series, strict=True):
            output.append(values[index])
    return tuple(output_time), tuple(tuple(values) for values in output_series)


def _insert_time_gaps(
    time: Sequence[float],
    series: Sequence[Sequence[float | None]],
) -> tuple[tuple[float | None, ...], tuple[tuple[float | None, ...], ...]]:
    if len(time) < 3:
        return tuple(time), tuple(tuple(values) for values in series)

    positive_steps = [
        current - previous
        for previous, current in zip(time, time[1:])
        if current > previous
    ]
    if not positive_steps:
        return tuple(time), tuple(tuple(values) for values in series)

    gap_threshold = max(median(positive_steps) * 10, 1.0)
    output_time: list[float | None] = []
    output_series: list[list[float | None]] = [[] for _ in series]

    for index, current_time in enumerate(time):
        if index and current_time - time[index - 1] > gap_threshold:
            output_time.append(None)
            for values in output_series:
                values.append(None)
        output_time.append(current_time)
        for output, values in zip(output_series, series, strict=True):
            output.append(values[index])

    return tuple(output_time), tuple(tuple(values) for values in output_series)


def _valid_number(value: float | None) -> bool:
    return value is not None and isfinite(value)
