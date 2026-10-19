"""Local Dash application for inspecting aircraft trajectories."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from apps.experiment_viewer.figures import (
    build_sampling_distribution_figure,
    build_sampling_distribution_from_summary,
    build_sampling_interval_figure,
    build_spatial_figure,
    build_time_series_figure,
)
from apps.experiment_viewer.services import ProcessedTrajectoryStore, TrajectoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "openskyDataset"
    / "subset_1"
    / "set_1.parquet"
)
DATA_PATH = Path(os.environ.get("TRAJECTORY_DATA_PATH", DEFAULT_DATA_PATH)).resolve()
DEFAULT_PROCESSED_DATA_PATH = (
    DEFAULT_DATA_PATH.parent
    / "standardized_1hz"
    / f"{DEFAULT_DATA_PATH.stem}_standardized_1hz.parquet"
)
PROCESSED_DATA_PATH = Path(
    os.environ.get("TRAJECTORY_PROCESSED_DATA_PATH", DEFAULT_PROCESSED_DATA_PATH)
).resolve()

try:
    SOURCE_LABEL = DATA_PATH.relative_to(PROJECT_ROOT).as_posix()
except ValueError:
    SOURCE_LABEL = str(DATA_PATH)
DATASET_LABEL = DATA_PATH.parent.parent.name
SUBSET_LABEL = DATA_PATH.parent.name

store = TrajectoryStore(DATA_PATH)
processed_store = (
    ProcessedTrajectoryStore(PROCESSED_DATA_PATH)
    if PROCESSED_DATA_PATH.is_file()
    else None
)
aircraft_summaries = store.aircraft_summaries()
if not aircraft_summaries:
    raise RuntimeError(f"No aircraft trajectories found in {DATA_PATH}")

summary_by_aircraft = {summary.aircraft: summary for summary in aircraft_summaries}
processed_summaries = (
    processed_store.aircraft_summaries() if processed_store is not None else ()
)
processed_summary_by_aircraft = {
    summary.aircraft: summary for summary in processed_summaries
}
default_aircraft = aircraft_summaries[0].aircraft
default_summary = summary_by_aircraft[default_aircraft]
DEFAULT_DATA_VIEW = "compare" if processed_store is not None else "raw"

app = Dash(
    __name__,
    title="Aircraft Trajectory Explorer",
    update_title=None,
    eager_loading=True,
)
server = app.server

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}
PLAYBACK_INTERVAL_MS = 1000


def _figure_heading(title: str, note: str) -> html.Div:
    return html.Div(
        className="figure-heading",
        children=[html.H2(title), html.Span(note)],
    )


def _timeline_configuration(summary):
    start_time = float(summary.start_time)
    end_time = float(summary.end_time)
    if end_time <= start_time:
        end_time = start_time + 1.0
    step = max((end_time - start_time) / 500, 0.001)
    marks = {
        start_time: {"label": f"{start_time:,.1f}"},
        end_time: {"label": f"{end_time:,.1f}"},
    }
    return start_time, end_time, step, marks


default_start, default_end, default_step, default_marks = _timeline_configuration(
    default_summary
)
default_track = store.load_track(default_aircraft)
default_processed_track = (
    processed_store.load_track(default_aircraft)
    if processed_store is not None
    else None
)
default_view_key = f"{default_aircraft}-{default_start:.6f}-{default_end:.6f}"
default_time_figure = build_time_series_figure(
    default_track,
    view_revision=f"time-{default_view_key}",
    processed_track=default_processed_track,
    data_view=DEFAULT_DATA_VIEW,
)
default_spatial_figure = build_spatial_figure(
    default_track,
    view_revision=f"spatial-{default_view_key}",
    processed_track=default_processed_track,
    data_view=DEFAULT_DATA_VIEW,
)
default_sampling_figure = build_sampling_interval_figure(
    default_track,
    view_revision=f"sampling-{default_aircraft}",
)
default_sampling_distribution_figure = build_sampling_distribution_figure(
    default_track,
    view_revision=f"sampling-distribution-{default_aircraft}",
)
global_sampling_distribution = store.global_sampling_distribution()
global_sampling_distribution_figure = build_sampling_distribution_from_summary(
    global_sampling_distribution,
    view_revision="sampling-distribution-global",
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
                        html.Span(
                            f"{DATASET_LABEL} · {SUBSET_LABEL} · "
                            + ("Raw + standardized" if processed_store else "Raw Parquet")
                        ),
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
                                                f"  ·  {summary.point_count:,} raw"
                                                + (
                                                    f"  ·  {processed_summary_by_aircraft[summary.aircraft].point_count:,} standardized"
                                                    if summary.aircraft in processed_summary_by_aircraft
                                                    else "  ·  no standardized track"
                                                )
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
                            className="control-field view-field",
                            children=[
                                html.Label("Data view"),
                                dcc.RadioItems(
                                    id="data-view-selector",
                                    className="radio-group compact-radio-group",
                                    inputClassName="radio-input",
                                    labelClassName="radio-label",
                                    options=[
                                        {"label": "Raw", "value": "raw"},
                                        {
                                            "label": "Standardized",
                                            "value": "processed",
                                            "disabled": processed_store is None,
                                        },
                                        {
                                            "label": "Compare",
                                            "value": "compare",
                                            "disabled": processed_store is None,
                                        },
                                    ],
                                    value=DEFAULT_DATA_VIEW,
                                    inline=True,
                                ),
                                dcc.Checklist(
                                    id="processed-point-options",
                                    className="point-option-group",
                                    inputClassName="radio-input",
                                    labelClassName="point-option-label",
                                    options=[
                                        {
                                            "label": "Interpolated",
                                            "value": "interpolated",
                                            "disabled": processed_store is None,
                                        },
                                        {
                                            "label": "Suspicious",
                                            "value": "suspicious",
                                            "disabled": processed_store is None,
                                        },
                                    ],
                                    value=(
                                        ["interpolated", "suspicious"]
                                        if processed_store is not None
                                        else []
                                    ),
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
                    className="timeline-panel",
                    children=[
                        html.Div(
                            className="timeline-header",
                            children=[
                                html.Div(
                                    children=[
                                        html.H2("Time window"),
                                        html.P(
                                            "Drag both handles to select a range; Lock window moves both handles during playback."
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="timeline-actions",
                                    children=[
                                        html.Div(
                                            className="speed-field",
                                            children=[
                                                html.Label(
                                                    "Speed",
                                                    htmlFor="playback-speed",
                                                ),
                                                dcc.Dropdown(
                                                    id="playback-speed",
                                                    options=[
                                                        {"label": "0.5×", "value": 0.5},
                                                        {
                                                            "label": "1× real time",
                                                            "value": 1.0,
                                                        },
                                                        {"label": "2×", "value": 2.0},
                                                        {"label": "4×", "value": 4.0},
                                                    ],
                                                    value=1.0,
                                                    clearable=False,
                                                    searchable=False,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="window-lock-field",
                                            children=dcc.Checklist(
                                                id="playback-window-lock",
                                                options=[
                                                    {
                                                        "label": "Lock window",
                                                        "value": "locked",
                                                    }
                                                ],
                                                value=[],
                                                inputStyle={"marginRight": "6px"},
                                            ),
                                            title=(
                                                "Keep the selected time-window width "
                                                "fixed during playback"
                                            ),
                                        ),
                                        html.Button(
                                            "▶ Play",
                                            id="playback-button",
                                            className="playback-button",
                                            n_clicks=0,
                                            type="button",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="timeline-slider-wrap",
                            children=dcc.RangeSlider(
                                id="timeline-slider",
                                min=default_start,
                                max=default_end,
                                step=default_step,
                                value=[default_start, default_end],
                                marks=default_marks,
                                allowCross=False,
                                updatemode="mouseup",
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ),
                        dcc.Interval(
                            id="playback-timer",
                            interval=PLAYBACK_INTERVAL_MS,
                            n_intervals=0,
                            disabled=True,
                        ),
                        dcc.Store(
                            id="playback-state",
                            data={
                                "status": "ready",
                                "start_time": default_start,
                                "end_time": default_end,
                            },
                        ),
                        dcc.Store(id="active-aircraft", data=default_aircraft),
                    ],
                ),
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
                                dcc.Graph(
                                    id="time-series-graph",
                                    className="time-series-graph",
                                    config=GRAPH_CONFIG,
                                    figure=default_time_figure,
                                    style={"height": "820px", "minHeight": "820px"},
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
                                dcc.Graph(
                                    id="spatial-graph",
                                    className="spatial-graph",
                                    config=GRAPH_CONFIG,
                                    figure=default_spatial_figure,
                                    style={"height": "820px", "minHeight": "820px"},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="sampling-grid",
                    children=[
                        html.Article(
                            className="figure-card sampling-card",
                            children=[
                                _figure_heading(
                                    "Sampling interval",
                                    "Raw reports · full trajectory · independent of playback",
                                ),
                                dcc.Graph(
                                    id="sampling-interval-graph",
                                    className="sampling-interval-graph",
                                    config=GRAPH_CONFIG,
                                    figure=default_sampling_figure,
                                    style={"height": "360px", "minHeight": "360px"},
                                ),
                            ],
                        ),
                        html.Article(
                            className="figure-card sampling-card",
                            children=[
                                _figure_heading(
                                    "Sampling-interval distribution",
                                    "Raw reports · 0–2 s detail · long gaps separated",
                                ),
                                dcc.Graph(
                                    id="sampling-distribution-graph",
                                    className="sampling-interval-graph",
                                    config=GRAPH_CONFIG,
                                    figure=default_sampling_distribution_figure,
                                    style={"height": "360px", "minHeight": "360px"},
                                ),
                            ],
                        ),
                        html.Article(
                            className="figure-card sampling-card",
                            children=[
                                _figure_heading(
                                    "Global sampling-interval distribution",
                                    "Raw reports · adjacent intervals computed within each aircraft",
                                ),
                                dcc.Graph(
                                    id="global-sampling-distribution-graph",
                                    className="sampling-interval-graph",
                                    config=GRAPH_CONFIG,
                                    figure=global_sampling_distribution_figure,
                                    style={"height": "360px", "minHeight": "360px"},
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
                html.Span(
                    f"Raw: {SOURCE_LABEL}"
                    + (
                        f" · Standardized: {PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT).as_posix()}"
                        if processed_store is not None
                        and PROCESSED_DATA_PATH.is_relative_to(PROJECT_ROOT)
                        else ""
                    )
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("timeline-slider", "min"),
    Output("timeline-slider", "max"),
    Output("timeline-slider", "step"),
    Output("timeline-slider", "marks"),
    Output("timeline-slider", "value"),
    Output("active-aircraft", "data"),
    Input("aircraft-selector", "value"),
)
def configure_timeline(aircraft: int):
    start_time, end_time, step, marks = _timeline_configuration(
        summary_by_aircraft[int(aircraft)]
    )
    return start_time, end_time, step, marks, [start_time, end_time], int(aircraft)


@app.callback(
    Output("sampling-interval-graph", "figure"),
    Output("sampling-distribution-graph", "figure"),
    Input("active-aircraft", "data"),
)
def update_sampling_figures(aircraft: int):
    track = store.load_track(int(aircraft))
    return (
        build_sampling_interval_figure(
            track,
            view_revision=f"sampling-{aircraft}",
        ),
        build_sampling_distribution_figure(
            track,
            view_revision=f"sampling-distribution-{aircraft}",
        ),
    )


@app.callback(
    Output("time-series-graph", "figure"),
    Output("spatial-graph", "figure"),
    Output("trajectory-meta", "children"),
    Output("playback-timer", "disabled"),
    Output("playback-button", "children"),
    Output("playback-state", "data"),
    Input("active-aircraft", "data"),
    Input("data-view-selector", "value"),
    Input("processed-point-options", "value"),
    Input("timeline-slider", "value"),
    State("playback-state", "data"),
)
def update_trajectory_view(
    aircraft: int,
    data_view: str,
    processed_point_options: list[str] | None,
    time_range: list[float],
    playback_state: dict | None,
):
    full_track = store.load_track(int(aircraft))
    full_processed_track = (
        processed_store.load_track(int(aircraft))
        if processed_store is not None
        else None
    )
    if processed_store is None:
        data_view = "raw"
    start_time, end_time = _normalize_time_range(
        time_range,
        summary_by_aircraft[int(aircraft)],
    )
    visible_track = full_track.between(start_time, end_time)
    visible_processed_track = (
        full_processed_track.between(start_time, end_time)
        if full_processed_track is not None
        else None
    )
    point_options = set(processed_point_options or [])
    view_key = f"{aircraft}-{start_time:.6f}-{end_time:.6f}-{data_view}"
    time_figure = build_time_series_figure(
        visible_track,
        view_revision=f"time-{view_key}",
        processed_track=visible_processed_track,
        data_view=data_view,
        show_interpolated="interpolated" in point_options,
        show_suspicious="suspicious" in point_options,
    )
    spatial_figure = build_spatial_figure(
        visible_track,
        view_revision=f"spatial-{view_key}",
        processed_track=visible_processed_track,
        data_view=data_view,
        show_interpolated="interpolated" in point_options,
        show_suspicious="suspicious" in point_options,
    )
    metadata = _build_metadata(
        visible_track,
        full_track,
        visible_processed_track,
        full_processed_track,
    )
    automated_slider_change = (
        ctx.triggered_id == "timeline-slider"
        and playback_state
        and playback_state.get("status") in {"playing", "complete"}
    )
    ready_state = _playback_state("ready", start_time, end_time)
    return (
        time_figure,
        spatial_figure,
        metadata,
        no_update if automated_slider_change else True,
        no_update if automated_slider_change else "▶ Play",
        no_update if automated_slider_change else ready_state,
    )


@app.callback(
    Output("playback-timer", "disabled", allow_duplicate=True),
    Output("playback-button", "children", allow_duplicate=True),
    Output("playback-state", "data", allow_duplicate=True),
    Output("timeline-slider", "value", allow_duplicate=True),
    Input("playback-button", "n_clicks"),
    Input("playback-timer", "n_intervals"),
    State("timeline-slider", "value"),
    State("timeline-slider", "min"),
    State("timeline-slider", "max"),
    State("playback-speed", "value"),
    State("playback-window-lock", "value"),
    State("playback-timer", "disabled"),
    State("playback-state", "data"),
    prevent_initial_call=True,
)
def control_playback(
    _play_clicks: int,
    _timer_ticks: int,
    time_range: list[float],
    minimum_time: float,
    maximum_time: float,
    playback_speed: float,
    window_lock: list[str] | None,
    timer_disabled: bool,
    playback_state: dict | None,
):
    if not time_range or len(time_range) != 2:
        return True, "▶ Play", _playback_state("empty", 0.0, 0.0), no_update

    start_time, end_time = sorted(float(value) for value in time_range)
    minimum_time = float(minimum_time)
    maximum_time = float(maximum_time)
    speed = max(float(playback_speed or 1.0), 0.1)
    window_locked = "locked" in (window_lock or [])
    state = playback_state or _playback_state("ready", start_time, end_time)
    trigger = ctx.triggered_id

    if trigger == "playback-button":
        if not timer_disabled:
            return (
                True,
                "▶ Continue",
                _playback_state("paused", start_time, end_time),
                no_update,
            )

        if state.get("status") == "paused":
            return (
                False,
                "❚❚ Pause",
                _playback_state("playing", start_time, end_time, monotonic()),
                no_update,
            )

        next_start = start_time
        next_end = end_time
        if end_time >= maximum_time:
            if window_locked:
                window_width = end_time - start_time
                next_start = minimum_time
                next_end = min(minimum_time + window_width, maximum_time)
            else:
                next_end = start_time
        return (
            False,
            "❚❚ Pause",
            _playback_state("playing", next_start, next_end, monotonic()),
            (
                [next_start, next_end]
                if next_start != start_time or next_end != end_time
                else no_update
            ),
        )

    if trigger == "playback-timer":
        if timer_disabled or state.get("status") != "playing":
            return no_update, no_update, no_update, no_update

        now = monotonic()
        previous_tick = float(state.get("last_tick", now))
        elapsed_real_seconds = max(now - previous_tick, 0.0)
        state_start = float(state.get("start_time", start_time))
        state_end = float(state.get("end_time", end_time))
        next_end = min(state_end + elapsed_real_seconds * speed, maximum_time)
        next_start = (
            state_start + (next_end - state_end)
            if window_locked
            else start_time
        )
        if next_end >= maximum_time:
            return (
                True,
                "↻ Replay",
                _playback_state("complete", next_start, maximum_time),
                [next_start, maximum_time],
            )
        return (
            False,
            "❚❚ Pause",
            _playback_state("playing", next_start, next_end, now),
            [next_start, next_end],
        )

    return no_update, no_update, no_update, no_update


def _playback_state(
    status: str,
    start_time: float,
    end_time: float,
    last_tick: float | None = None,
) -> dict:
    state = {
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
    }
    if last_tick is not None:
        state["last_tick"] = last_tick
    return state


def _build_metadata(
    track,
    full_track,
    processed_track,
    full_processed_track,
) -> list[html.Div]:
    spatial_count = sum(
        longitude is not None and latitude is not None
        for longitude, latitude in zip(track.longitude, track.latitude, strict=True)
    )
    duration = track.time[-1] - track.time[0] if len(track.time) > 1 else 0.0
    raw_observation_value = _visible_total(track.point_count, full_track.point_count)

    if full_processed_track is None or full_processed_track.point_count == 0:
        processed_point_value = "—"
        observed_point_value = "—"
        interpolated_value = "—"
        segment_value = "—"
        suspicious_value = "—"
        trusted_value = "—"
        split_value = "—"
    else:
        visible_processed = processed_track.point_count if processed_track else 0
        visible_observed = processed_track.observed_point_count if processed_track else 0
        visible_interpolated = (
            processed_track.interpolated_point_count if processed_track else 0
        )
        visible_segments = processed_track.segment_count if processed_track else 0
        visible_suspicious = processed_track.suspicious_point_count if processed_track else 0
        processed_point_value = _visible_total(
            visible_processed, full_processed_track.point_count
        )
        observed_point_value = _visible_total(
            visible_observed, full_processed_track.observed_point_count
        )
        interpolated_value = _visible_total(
            visible_interpolated, full_processed_track.interpolated_point_count
        )
        segment_value = _visible_total(
            visible_segments, full_processed_track.segment_count
        )
        suspicious_value = _visible_total(
            visible_suspicious, full_processed_track.suspicious_point_count
        )
        trusted_value = str(full_processed_track.trusted or "—").title()
        split_value = str(full_processed_track.split or "—").title()

    return [
        _metric("Aircraft ID", str(track.aircraft)),
        _metric("Raw observations", raw_observation_value),
        _metric("Standardized points", processed_point_value),
        _metric("Observed 1 Hz", observed_point_value),
        _metric("Interpolated", interpolated_value),
        _metric("Segments", segment_value),
        _metric("Suspicious endpoints", suspicious_value),
        _metric("Quality / split", f"{trusted_value} · {split_value}"),
        _metric("Raw spatial coverage", _percentage(spatial_count, track.point_count)),
        _metric("Visible duration", _format_duration(duration)),
    ]


def _normalize_time_range(time_range, summary) -> tuple[float, float]:
    summary_start = float(summary.start_time)
    summary_end = float(summary.end_time)
    if not time_range or len(time_range) != 2:
        return summary_start, summary_end
    start_time, end_time = sorted(float(value) for value in time_range)
    start_time = max(start_time, summary_start)
    end_time = min(end_time, summary_end)
    if start_time > end_time:
        return summary_start, summary_end
    return start_time, end_time


def _metric(label: str, value: str) -> html.Div:
    return html.Div(
        className="metric",
        children=[html.Span(label), html.Strong(value)],
    )


def _visible_total(visible: int, total: int) -> str:
    if visible == total:
        return f"{total:,}"
    return f"{visible:,} / {total:,}"


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
