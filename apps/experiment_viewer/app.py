"""Local Dash application for inspecting aircraft trajectories."""

from __future__ import annotations

import os
from pathlib import Path

from dash import Dash, Input, Output, dcc, html

from apps.experiment_viewer.figures import (
    build_spatial_figure,
    build_time_series_figure,
)
from apps.experiment_viewer.services import TrajectoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openskyDataset"
    / "subset_1"
    / "set_1.parquet"
)
DATA_PATH = Path(os.environ.get("TRAJECTORY_DATA_PATH", DEFAULT_DATA_PATH))

store = TrajectoryStore(DATA_PATH)
aircraft_summaries = store.aircraft_summaries()
if not aircraft_summaries:
    raise RuntimeError(f"No aircraft trajectories found in {DATA_PATH}")

summary_by_aircraft = {summary.aircraft: summary for summary in aircraft_summaries}
default_aircraft = aircraft_summaries[0].aircraft

app = Dash(__name__, title="Aircraft Trajectory Explorer", update_title=None)
server = app.server

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _figure_heading(title: str, note: str) -> html.Div:
    return html.Div(
        className="figure-heading",
        children=[html.H2(title), html.Span(note)],
    )


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Header(
            className="page-header",
            children=[
                html.Div(
                    children=[
                        html.Div("TRAJECTORY ANALYSIS", className="eyebrow"),
                        html.H1("Aircraft Trajectory Explorer"),
                        html.P(
                            "Single-aircraft inspection of position and altitude over time",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="dataset-badge",
                    children=[
                        html.Span(className="status-dot"),
                        html.Span("OpenSky · Parquet"),
                    ],
                ),
            ],
        ),
        html.Main(
            children=[
                html.Section(
                    className="control-panel",
                    children=[
                        html.Div(
                            className="control-field aircraft-field",
                            children=[
                                html.Label("Aircraft", htmlFor="aircraft-selector"),
                                dcc.Dropdown(
                                    id="aircraft-selector",
                                    options=[
                                        {
                                            "label": (
                                                f"Aircraft {summary.aircraft}"
                                                f"  ·  {summary.point_count:,} observations"
                                            ),
                                            "value": summary.aircraft,
                                        }
                                        for summary in aircraft_summaries
                                    ],
                                    value=default_aircraft,
                                    searchable=True,
                                    clearable=False,
                                    placeholder="Search aircraft ID",
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-field altitude-field",
                            children=[
                                html.Label("Altitude source"),
                                dcc.RadioItems(
                                    id="altitude-selector",
                                    className="radio-group",
                                    inputClassName="radio-input",
                                    labelClassName="radio-label",
                                    options=[
                                        {"label": "Geometric", "value": "geoAltitude"},
                                        {"label": "Barometric", "value": "baroAltitude"},
                                    ],
                                    value="geoAltitude",
                                    inline=True,
                                ),
                            ],
                        ),
                        html.Div(
                            className="source-note",
                            children=[
                                html.Span("Time basis", className="source-note-label"),
                                html.Strong("Relative server time"),
                            ],
                        ),
                    ],
                ),
                html.Section(id="trajectory-meta", className="metric-strip"),
                html.Section(
                    className="figure-grid",
                    children=[
                        html.Article(
                            className="figure-card time-card",
                            children=[
                                _figure_heading(
                                    "Position and altitude over time",
                                    "Linked temporal axes · drag to zoom",
                                ),
                                dcc.Loading(
                                    type="circle",
                                    color="#315A9E",
                                    children=dcc.Graph(
                                        id="time-series-graph",
                                        className="time-series-graph",
                                        config=GRAPH_CONFIG,
                                    ),
                                ),
                            ],
                        ),
                        html.Article(
                            className="figure-card spatial-card",
                            children=[
                                _figure_heading(
                                    "Horizontal trajectory",
                                    "WGS84 longitude–latitude plane",
                                ),
                                dcc.Loading(
                                    type="circle",
                                    color="#315A9E",
                                    children=dcc.Graph(
                                        id="spatial-graph",
                                        className="spatial-graph",
                                        config=GRAPH_CONFIG,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        ),
        html.Footer(
            children=[
                html.Span("Trajectory Prediction Research Workspace"),
                html.Span(f"Source: {DATA_PATH.name}"),
            ]
        ),
    ],
)


@app.callback(
    Output("time-series-graph", "figure"),
    Output("spatial-graph", "figure"),
    Output("trajectory-meta", "children"),
    Input("aircraft-selector", "value"),
    Input("altitude-selector", "value"),
)
def update_trajectory_view(aircraft: int, altitude_field: str):
    track = store.load_track(int(aircraft))
    time_figure = build_time_series_figure(track, altitude_field)
    spatial_figure = build_spatial_figure(track)
    metadata = _build_metadata(track, altitude_field)
    return time_figure, spatial_figure, metadata


def _build_metadata(track, altitude_field: str) -> list[html.Div]:
    summary = summary_by_aircraft[track.aircraft]
    altitude = (
        track.baro_altitude if altitude_field == "baroAltitude" else track.geo_altitude
    )
    altitude_count = sum(value is not None for value in altitude)
    duration = summary.end_time - summary.start_time

    return [
        _metric("Aircraft ID", str(track.aircraft)),
        _metric("Observations", f"{track.point_count:,}"),
        _metric("Spatial coverage", _percentage(summary.spatial_point_count, track.point_count)),
        _metric("Altitude coverage", _percentage(altitude_count, track.point_count)),
        _metric("Track duration", _format_duration(duration)),
    ]


def _metric(label: str, value: str) -> html.Div:
    return html.Div(
        className="metric",
        children=[html.Span(label), html.Strong(value)],
    )


def _percentage(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{numerator / denominator:.1%}"


def _format_duration(value: float) -> str:
    if value < 60:
        return f"{value:.1f} s"
    if value < 3600:
        return f"{value / 60:.1f} min"
    return f"{value / 3600:.2f} h"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
