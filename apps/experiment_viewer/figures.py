"""Plotly figure builders for the trajectory viewer."""

from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.experiment_viewer.services import TrackData


INK = "#172033"
MUTED = "#667085"
GRID = "#E4E7EC"
BLUE = "#315A9E"
GREEN = "#2E7D68"
RED = "#B54747"
PAPER = "#FFFFFF"


def build_time_series_figure(
    track: TrackData,
    altitude_field: str,
) -> go.Figure:
    altitude, altitude_label = _select_altitude(track, altitude_field)
    times, series = _insert_time_gaps(
        track.time,
        (track.longitude, track.latitude, altitude),
    )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
    )
    specifications = (
        (series[0], "Longitude", "Longitude (°)"),
        (series[1], "Latitude", "Latitude (°)"),
        (series[2], altitude_label, "Altitude (m)"),
    )

    for row, (values, name, y_title) in enumerate(specifications, start=1):
        figure.add_trace(
            go.Scattergl(
                x=times,
                y=values,
                mode="lines",
                name=name,
                line={"color": BLUE, "width": 1.7},
                hovertemplate=f"{name}: %{{y:,.3f}}<extra></extra>",
                connectgaps=False,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=y_title, row=row, col=1)

    figure.update_xaxes(title_text="Relative time", row=3, col=1)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={"l": 72, "r": 22, "t": 24, "b": 58},
        hovermode="x unified",
        showlegend=False,
        font={"family": "Segoe UI, Arial, sans-serif", "color": INK, "size": 12},
        uirevision=f"time-{track.aircraft}-{altitude_field}",
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


def build_spatial_figure(track: TrackData) -> go.Figure:
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

    figure = go.Figure()
    if valid_points:
        time = tuple(point[0] for point in valid_points)
        longitude = tuple(point[1] for point in valid_points)
        latitude = tuple(point[2] for point in valid_points)
        _, spatial_series = _insert_time_gaps(time, (longitude, latitude))

        figure.add_trace(
            go.Scattergl(
                x=spatial_series[0],
                y=spatial_series[1],
                mode="lines",
                line={"color": BLUE, "width": 2},
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
            go.Scatter(
                x=[longitude[-1]],
                y=[latitude[-1]],
                mode="markers",
                name="End",
                marker={"color": RED, "size": 9, "line": {"color": PAPER, "width": 1}},
                hovertemplate="End<br>Lon: %{x:.5f}°<br>Lat: %{y:.5f}°<extra></extra>",
            )
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
        margin={"l": 70, "r": 24, "t": 24, "b": 58},
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
        uirevision=f"spatial-{track.aircraft}",
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


def _select_altitude(
    track: TrackData,
    altitude_field: str,
) -> tuple[Sequence[float | None], str]:
    if altitude_field == "baroAltitude":
        return track.baro_altitude, "Barometric altitude"
    return track.geo_altitude, "Geometric altitude"


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
