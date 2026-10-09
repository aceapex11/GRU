
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="GRU Drift Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("GRU Drift Intelligence")
st.caption(
    "Predict → Detect → Diagnose → Adapt → Evaluate | GRU benchmark"
)

DATA_CANDIDATES = [
    ROOT / "industrial_sensor_drift_dataset.csv",
    ROOT / "data" / "industrial_sensor_drift_dataset.csv",
]


def resolve_dataset():
    for path in DATA_CANDIDATES:
        if path.is_file() and path.stat().st_size > 0:
            return path

    for folder in [ROOT, ROOT / "data"]:
        if folder.is_dir():
            matches = list(
                folder.glob("*industrial_sensor_drift_dataset*.csv")
            )
            for path in matches:
                if path.stat().st_size > 0:
                    return path

    return None


@st.cache_data(show_spinner=False)
def load_data(path, modified_time, file_size):
    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Dataset contains no rows.")

    data = data.loc[:, ~data.columns.duplicated()].copy()

    if "timestamp" not in data.columns:
        return data

    raw = data["timestamp"]
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().mean() > 0.95:
        valid = numeric.dropna()
        magnitude = float(valid.abs().median()) if len(valid) else 0

        if magnitude >= 1e17:
            parsed = pd.to_datetime(numeric, unit="ns", errors="coerce")
        elif magnitude >= 1e14:
            parsed = pd.to_datetime(numeric, unit="us", errors="coerce")
        elif magnitude >= 1e11:
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
        elif magnitude >= 1e8:
            parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
        else:
            parsed = pd.Series(pd.NaT, index=data.index)
            data["timestamp_is_index"] = True

        data["timestamp"] = parsed
    else:
        data["timestamp"] = pd.to_datetime(raw, errors="coerce")

    return data


def read_result(filename):
    path = RESULTS / filename

    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        result = pd.read_csv(path)
        return result if not result.empty else None
    except (EmptyDataError, ParserError, OSError):
        return None


def to_flags(series):
    if series.dtype == object:
        return series.astype(str).str.strip().str.lower().isin(
            ["1", "true", "yes", "alarm", "drift"]
        )

    return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)


def get_event_positions(event_frame, dataset, x_column):
    if event_frame is None or event_frame.empty:
        return []

    if dataset is None or dataset.empty:
        return []

    if x_column not in dataset.columns:
        return []

    candidates = [
        x_column,
        "timestamp",
        "window_start",
        "timestamp_end",
        "adapted_at",
        "adaptation_timestamp",
        "adaptation_point",
        "observation_index",
        "observation",
        "row_index",
        "sample_index",
        "row",
        "index",
        "window",
    ]

    event_col = next(
        (col for col in candidates if col in event_frame.columns),
        None,
    )

    if event_col is None:
        return []

    values = event_frame[event_col].dropna()

    if values.empty:
        return []

    if x_column == "timestamp":
        parsed = pd.to_datetime(values, errors="coerce")

        if parsed.notna().any():
            return parsed.dropna().tolist()

    numeric = pd.to_numeric(values, errors="coerce")

    if numeric.notna().mean() > 0.9:
        numbers = numeric.dropna().to_numpy(dtype=float)

        if (
            len(numbers)
            and np.isfinite(numbers).all()
            and (numbers >= 0).all()
            and (numbers < len(dataset)).all()
            and np.equal(numbers, np.floor(numbers)).all()
        ):
            positions = numbers.astype(int)
            return (
                dataset.iloc[positions][x_column]
                .dropna()
                .tolist()
            )

        return numeric.dropna().tolist()

    return []


def add_event_markers(
    fig,
    x_column,
    dataset,
    detector_frame=None,
    adaptation_frame=None,
    max_markers=24,
):
    def draw(values, color, dash, label):
        cleaned = []
        seen = set()

        for value in values:
            try:
                if pd.isna(value):
                    continue

                key = str(value)
                if key not in seen:
                    seen.add(key)
                    cleaned.append(value)
            except (TypeError, ValueError):
                continue

        if not cleaned:
            return

        total = len(cleaned)

        if total > max_markers:
            indexes = np.linspace(
                0, total - 1, max_markers, dtype=int
            )
            cleaned = [cleaned[i] for i in indexes]

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{label} ({total})",
                line=dict(color=color, dash=dash, width=2),
                hoverinfo="skip",
            )
        )

        for value in cleaned:
            try:
                fig.add_vline(
                    x=value,
                    line_color=color,
                    line_dash=dash,
                    line_width=1.2,
                    opacity=0.75,
                )
            except (TypeError, ValueError, OverflowError):
                continue

    if detector_frame is not None and not detector_frame.empty:
        alarm_columns = [
            col
            for col in detector_frame.columns
            if any(
                token in str(col).lower()
                for token in [
                    "alarm",
                    "drift_detected",
                    "change_detected",
                ]
            )
        ]

        if alarm_columns:
            flags = detector_frame[alarm_columns].apply(to_flags)
            alarm_rows = detector_frame.loc[flags.any(axis=1)]

            draw(
                get_event_positions(alarm_rows, dataset, x_column),
                "#D97706",
                "dash",
                "Detector alarm",
            )

    if adaptation_frame is not None and not adaptation_frame.empty:
        draw(
            get_event_positions(
                adaptation_frame, dataset, x_column
            ),
            "#0F766E",
            "dot",
            "Adaptation",
        )

    return fig


def add_drift_regions(fig, dataset):
    if "timestamp" not in dataset.columns:
        return fig

    if not pd.api.types.is_datetime64_any_dtype(dataset["timestamp"]):
        return fig

    flags = np.zeros(len(dataset), dtype=int)

    for col, weight in [
        ("data_drift", 1),
        ("concept_drift", 2),
    ]:
        if col in dataset.columns:
            flags += to_flags(dataset[col]).to_numpy(dtype=int) * weight

    if not np.any(flags):
        return fig

    timestamps = dataset["timestamp"]
    valid = timestamps.notna().to_numpy()
    flags = np.where(valid, flags, 0)

    starts = np.where(
        np.r_[True, flags[1:] != flags[:-1]]
    )[0]
    ends = np.r_[starts[1:], len(flags)]

    colors = {
        1: "rgba(245,166,35,0.10)",
        2: "rgba(120,90,200,0.10)",
        3: "rgba(220,80,80,0.13)",
    }

    for start, end in zip(starts, ends):
        state = int(flags[start])

        if state == 0 or end <= start:
            continue

        x0 = timestamps.iloc[start]
        x1 = timestamps.iloc[end - 1]

        if pd.isna(x0) or pd.isna(x1):
            continue

        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=colors[state],
            line_width=0,
            layer="below",
        )

    return fig


def align_predictions(predictions, dataset):
    pred = predictions.copy()

    row_candidates = [
        "row",
        "row_index",
        "observation",
        "index",
        "sample_index",
    ]

    row_col = next(
        (col for col in row_candidates if col in pred.columns),
        None,
    )

    real_timestamps = (
        "timestamp" in dataset.columns
        and pd.to_datetime(
            dataset["timestamp"], errors="coerce"
        ).notna().any()
    )

    if real_timestamps:
        dates = pd.to_datetime(
            dataset["timestamp"], errors="coerce"
        )

        if "timestamp" not in pred.columns:
            if row_col:
                ids = pd.to_numeric(
                    pred[row_col], errors="coerce"
                )
                valid = ids.notna() & ids.between(
                    0, len(dataset) - 1
                )
                pred["timestamp"] = pd.NaT

                positions = ids[valid].astype(int).to_numpy()
                pred.loc[valid, "timestamp"] = (
                    dates.iloc[positions].to_numpy()
                )
            elif len(pred) == len(dataset):
                pred["timestamp"] = dates.to_numpy()
            else:
                pred["timestamp"] = pd.NaT
        else:
            parsed = pd.to_datetime(
                pred["timestamp"], errors="coerce"
            )
            numeric = pd.to_numeric(
                pred["timestamp"], errors="coerce"
            )

            looks_like_index = (
                numeric.notna().sum() > 0
                and numeric.dropna().between(
                    0, len(dataset) - 1
                ).mean() > 0.9
                and parsed.isna().mean() > 0.5
            )

            if looks_like_index:
                ids = (
                    pd.to_numeric(pred[row_col], errors="coerce")
                    if row_col
                    else numeric
                )
                valid = ids.notna() & ids.between(
                    0, len(dataset) - 1
                )
                pred["timestamp"] = pd.NaT

                positions = ids[valid].astype(int).to_numpy()
                pred.loc[valid, "timestamp"] = (
                    dates.iloc[positions].to_numpy()
                )
            else:
                pred["timestamp"] = parsed

        pred["timestamp"] = pd.to_datetime(
            pred["timestamp"], errors="coerce"
        )

        return (
            pred.dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )

    if row_col:
        pred["observation_index"] = pd.to_numeric(
            pred[row_col], errors="coerce"
        )
    elif len(pred) == len(dataset):
        pred["observation_index"] = np.arange(len(pred))
    elif "timestamp" in pred.columns:
        pred["observation_index"] = pd.to_numeric(
            pred["timestamp"], errors="coerce"
        )
    else:
        pred["observation_index"] = np.nan

    return (
        pred.dropna(subset=["observation_index"])
        .sort_values("observation_index")
    )


# ============================================================
# SIDEBAR
# ============================================================

DATA = resolve_dataset()

with st.sidebar:
    st.header("Experiment controls")

    if DATA:
        st.success(f"Dataset: {DATA.name}")
    else:
        st.error("Dataset not found")

    full_training = st.checkbox(
        "Full training (slower)",
        value=False,
    )

    run_experiment = st.button(
        "Run GRU experiment",
        type="primary",
        width="stretch",
    )

    if run_experiment:
        script = ROOT / "experiment.py"

        if DATA is None:
            st.error("Put the dataset CSV beside app.py.")
        elif not script.is_file():
            st.error("experiment.py was not found.")
        else:
            RESULTS.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                str(script),
                "--data",
                str(DATA),
                "--out",
                str(RESULTS),
            ]

            if full_training:
                command.append("--full")

            try:
                with st.spinner("Running GRU experiment..."):
                    result = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=7200,
                    )

                if result.returncode == 0:
                    st.success("Experiment completed.")
                    if result.stdout:
                        st.code(result.stdout[-5000:])
                else:
                    st.error("Experiment failed.")
                    st.code(
                        (
                            (result.stderr or "")
                            + "\n"
                            + (result.stdout or "")
                        )[-8000:]
                    )

            except subprocess.TimeoutExpired:
                st.error("Experiment timed out.")

            st.cache_data.clear()
            st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

if DATA is None:
    st.error(
        "Dataset not found. Place "
        "industrial_sensor_drift_dataset.csv beside app.py."
    )
    st.stop()

try:
    df = load_data(
        str(DATA),
        DATA.stat().st_mtime,
        DATA.stat().st_size,
    )
except (
    EmptyDataError,
    ParserError,
    UnicodeDecodeError,
    ValueError,
    OSError,
) as exc:
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

if "timestamp" not in df.columns:
    st.error("Dataset requires a timestamp column.")
    st.stop()

if df["timestamp"].notna().sum() == 0:
    df["observation_index"] = np.arange(len(df))
    X_COL = "observation_index"
    X_LABEL = "Observation index"
else:
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    X_COL = "timestamp"
    X_LABEL = "Time"

metrics = read_result("strategy_metrics.csv")
events = read_result("adaptation_events.csv")
predictions = read_result("predictions.csv")
detector_metrics = read_result("detector_window_metrics.csv")

sensor_features = [
    col
    for col in [
        "temperature_C",
        "pressure_bar",
        "vibration_mm_s_rms",
        "load_pct",
        "humidity_pct",
        "motor_speed_rpm",
        "operating_hours",
    ]
    if col in df.columns
]

targets = [
    col
    for col in ["health_deterioration"]
    if col in df.columns
]

# ============================================================
# SUMMARY METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(df):,}")
c2.metric("Sensor inputs", len(sensor_features))

c3.metric(
    "Regimes",
    df["regime"].nunique() if "regime" in df.columns else "—",
)

c4.metric(
    "Adaptation events",
    len(events) if events is not None else "—",
)

tabs = st.tabs([
    "Overview",
    "Sensor Explorer",
    "Drift Timeline",
    "Detector Analysis",
    "Strategy Comparison",
    "Predictions",
    "Adaptation Events",
    "Data & Notes",
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tabs[0]:
    st.subheader("Sensor stream")

    choices = sensor_features + targets

    if choices:
        signal = st.selectbox(
            "Signal",
            choices,
            key="overview_signal",
        )

        stride = max(1, len(df) // 3000)
        sample = df.iloc[::stride]

        fig = px.line(
            sample,
            x=X_COL,
            y=signal,
            title=f"{signal} over time",
        )

        fig = add_event_markers(
            fig,
            X_COL,
            df,
            detector_metrics,
            events,
        )

        if X_COL == "timestamp":
            fig = add_drift_regions(fig, df)

        fig.update_layout(
            height=500,
            xaxis_title=X_LABEL,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No recognized sensor columns.")

    if metrics is not None:
        view = metrics.copy()

        if "Phase" in view.columns:
            overall = view[
                view["Phase"].astype(str).str.lower() == "overall"
            ]
            if not overall.empty:
                view = overall

        if "MAE" in view.columns:
            view = view.sort_values("MAE")

        st.subheader("Strategy metrics")
        st.dataframe(view, width="stretch", hide_index=True)
    else:
        st.info("Run the experiment to generate strategy metrics.")


# ============================================================
# TAB 2: SENSOR EXPLORER
# ============================================================

with tabs[1]:
    st.subheader("Explore individual signals")

    choices = sensor_features + targets

    if choices:
        signal = st.selectbox(
            "Sensor or target",
            choices,
            key="sensor_signal",
        )

        max_points = st.slider(
            "Maximum chart points",
            500,
            10000,
            3000,
            step=500,
        )

        stride = max(1, len(df) // max_points)
        sample = df.iloc[::stride]

        fig = px.line(
            sample,
            x=X_COL,
            y=signal,
            title=f"{signal}: alarms and adaptations",
        )

        fig = add_event_markers(
            fig,
            X_COL,
            df,
            detector_metrics,
            events,
        )

        if X_COL == "timestamp":
            fig = add_drift_regions(fig, df)

        fig.update_layout(
            height=500,
            xaxis_title=X_LABEL,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

        st.subheader("Descriptive statistics")
        st.dataframe(
            df[choices].describe().T,
            width="stretch",
        )
    else:
        st.warning("No sensor columns found.")


# ============================================================
# TAB 3: DRIFT TIMELINE
# ============================================================

with tabs[2]:
    st.subheader("Synthetic drift regimes")

    if "regime" in df.columns:
        counts = (
            df.groupby("regime", dropna=False)
            .size()
            .reset_index(name="records")
        )

        fig = px.bar(
            counts,
            x="regime",
            y="records",
            title="Records by regime",
        )

        st.plotly_chart(fig, width="stretch")

        choices = [
            col
            for col in [
                "temperature_C",
                "pressure_bar",
                "vibration_mm_s_rms",
                "health_deterioration",
            ]
            if col in df.columns
        ]

        if choices:
            signal = st.selectbox(
                "Timeline signal",
                choices,
                key="drift_timeline_signal",
            )

            sample = df.iloc[::max(1, len(df) // 3000)]

            fig = px.line(
                sample,
                x=X_COL,
                y=signal,
                color="regime",
                title=f"{signal} across regimes",
            )

            fig = add_event_markers(
                fig,
                X_COL,
                df,
                detector_metrics,
                events,
            )

            if X_COL == "timestamp":
                fig = add_drift_regions(fig, df)

            fig.update_layout(height=550, hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Ground-truth labels are for offline evaluation. "
            "They should not be inputs to the drift detectors."
        )
    else:
        st.warning("No regime column found.")


# ============================================================
# TAB 4: DETECTOR ANALYSIS
# ============================================================

with tabs[3]:
    st.subheader("KS, PSI and Page-Hinkley")

    st.markdown(
        """
        - **KS:** compares reference and recent distributions.
        - **PSI:** measures population distribution changes.
        - **Page-Hinkley:** monitors changes in a signal, such as
          prediction errors.
        """
    )

    if detector_metrics is None:
        st.info("Run the experiment to generate detector metrics.")
    else:
        st.dataframe(
            detector_metrics.tail(500),
            width="stretch",
            hide_index=True,
        )

        alarm_columns = [
            col
            for col in [
                "KS_alarm",
                "PSI_alarm",
                "Page_Hinkley_alarm",
            ]
            if col in detector_metrics.columns
        ]

        if alarm_columns:
            alarm_counts = (
                detector_metrics[alarm_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum()
                .rename_axis("Detector")
                .reset_index(name="Alarm windows")
            )

            fig = px.bar(
                alarm_counts,
                x="Detector",
                y="Alarm windows",
                title="Alarm windows by detector",
            )

            st.plotly_chart(fig, width="stretch")

        for column, title in [
            ("max_PSI", "Maximum PSI"),
            ("max_KS_stat", "Maximum KS statistic"),
        ]:
            if column not in detector_metrics.columns:
                continue

            xcol = (
                "window_start"
                if "window_start" in detector_metrics.columns
                else None
            )

            if xcol is None:
                st.caption(
                    f"{title}: no window_start column available."
                )
                continue

            fig = px.line(
                detector_metrics,
                x=xcol,
                y=column,
                title=title,
            )

            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 5: STRATEGY COMPARISON
# ============================================================

with tabs[4]:
    st.subheader("GRU adaptation strategy comparison")

    if metrics is None:
        st.info("No strategy metrics found. Run the experiment.")
    else:
        view = metrics.copy()

        if "Phase" in view.columns:
            phases = sorted(
                view["Phase"].dropna().astype(str).unique()
            )

            if phases:
                phase = st.selectbox(
                    "Evaluation phase",
                    phases,
                    key="strategy_phase",
                )

                view = view[
                    view["Phase"].astype(str) == phase
                ]

        if "MAE" in view.columns:
            view = view.sort_values("MAE")

        st.dataframe(view, width="stretch", hide_index=True)

        available = [
            col for col in ["MAE", "RMSE", "R2"]
            if col in view.columns
        ]

        if "Strategy" in view.columns and available:
            selected_metric = st.selectbox(
                "Comparison metric",
                available,
                key="comparison_metric",
            )

            fig = px.bar(
                view,
                x="Strategy",
                y=selected_metric,
                title=f"{selected_metric} by strategy",
                text_auto=".3f",
            )

            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Compare strategies on the same evaluation period. "
            "Lower MAE/RMSE is better; higher R² is generally better."
        )


# ============================================================
# TAB 6: ACTUAL VS PREDICTED
# ============================================================

with tabs[5]:
    st.subheader("Actual vs predicted")

    if predictions is None:
        st.info("Run the experiment to generate predictions.")
    else:
        aligned = align_predictions(predictions, df)

        if aligned.empty:
            st.error(
                "Could not align predictions. Check predictions.csv "
                "for timestamps or observation indices."
            )
        else:
            actual_candidates = [
                "actual",
                "actual_target",
                "y_true",
                "target_actual",
            ]

            actual_col = next(
                (
                    col for col in actual_candidates
                    if col in aligned.columns
                ),
                None,
            )

            metadata = {
                "timestamp",
                "row",
                "row_index",
                "observation",
                "index",
                "sample_index",
                "observation_index",
                "actual",
                "actual_target",
                "y_true",
                "target_actual",
            }

            model_columns = [
                col
                for col in aligned.columns
                if col not in metadata
                and pd.api.types.is_numeric_dtype(aligned[col])
            ]

            if actual_col is None:
                st.warning(
                    "Actual target column not found. Showing available "
                    "numeric prediction columns."
                )

            if actual_col is None and not model_columns:
                st.info("No numeric prediction columns found.")
            else:
                max_points = st.slider(
                    "Maximum prediction chart points",
                    500,
                    12000,
                    5000,
                    step=500,
                    key="prediction_max_points",
                )

                stride = max(1, len(aligned) // max_points)
                sample = aligned.iloc[::stride].copy()

                if (
                    "timestamp" in sample.columns
                    and pd.to_datetime(
                        sample["timestamp"], errors="coerce"
                    ).notna().any()
                ):
                    plot_x = "timestamp"
                else:
                    if "observation_index" not in sample.columns:
                        sample["observation_index"] = np.arange(
                            len(sample)
                        )
                    plot_x = "observation_index"

                fig = go.Figure()

                if actual_col:
                    fig.add_trace(
                        go.Scatter(
                            x=sample[plot_x],
                            y=sample[actual_col],
                            mode="lines",
                            name="Actual target",
                            line=dict(width=2.5),
                        )
                    )

                for col in model_columns:
                    fig.add_trace(
                        go.Scatter(
                            x=sample[plot_x],
                            y=sample[col],
                            mode="lines",
                            name=str(col),
                            line=dict(width=1.5),
                        )
                    )

                marker_dataset = df.copy()

                if (
                    plot_x == "observation_index"
                    and "observation_index" not in marker_dataset.columns
                ):
                    marker_dataset["observation_index"] = np.arange(
                        len(marker_dataset)
                    )

                if plot_x == "timestamp":
                    fig = add_drift_regions(fig, df)

                fig = add_event_markers(
                    fig,
                    plot_x,
                    marker_dataset,
                    detector_metrics,
                    events,
                )

                fig.update_layout(
                    title="Health deterioration: actual vs predictions",
                    xaxis_title=(
                        "Time"
                        if plot_x == "timestamp"
                        else "Observation index"
                    ),
                    yaxis_title="Health deterioration",
                    height=620,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                    ),
                    margin=dict(l=20, r=20, t=90, b=20),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={
                        "displaylogo": False,
                        "responsive": True,
                    },
                )

                st.caption(
                    "Drift shading uses ground-truth labels. Amber "
                    "dashed lines are detector alarms; teal dotted "
                    "lines are recorded adaptations. Event markers "
                    "are capped to reduce clutter."
                )

        st.subheader("Prediction data")
        st.dataframe(
            predictions.tail(100),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 7: ADAPTATION EVENTS
# ============================================================

with tabs[6]:
    st.subheader("Adaptation event log")

    if events is None:
        st.info("No adaptation event file found.")
    else:
        st.dataframe(
            events,
            width="stretch",
            hide_index=True,
        )

        if events.empty:
            st.info("No adaptations were recorded.")
        else:
            event_x = next(
                (
                    col for col in [
                        "adapted_at",
                        "adaptation_timestamp",
                        "timestamp",
                        "observation_index",
                        "adaptation_point",
                    ]
                    if col in events.columns
                ),
                None,
            )

            sample_col = next(
                (
                    col for col in [
                        "n_samples",
                        "samples_used",
                        "replay_samples",
                        "buffer_size",
                    ]
                    if col in events.columns
                ),
                None,
            )

            if event_x and sample_col:
                hover_cols = [
                    col for col in [
                        "Strategy",
                        "strategy",
                        "reason",
                        "drift_type",
                    ]
                    if col in events.columns
                ]

                fig = px.scatter(
                    events,
                    x=event_x,
                    y=sample_col,
                    title="Adaptation events and samples used",
                    hover_data=hover_cols,
                )

                st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 8: DATA AND NOTES
# ============================================================

with tabs[7]:
    st.subheader("Dataset preview")

    st.caption(
        f"Dataset: {DATA.name} | "
        f"File size: {DATA.stat().st_size:,} bytes"
    )

    st.dataframe(
        df.head(30),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Missing values")

    missing = df.isna().sum().rename(
        "missing_count"
    ).to_frame()

    missing["missing_pct"] = (
        missing["missing_count"] / max(len(df), 1) * 100
    ).round(3)

    st.dataframe(missing, width="stretch")

    st.subheader("Methodology notes")

    st.markdown(
        """
        - The dataset is synthetic.
        - `health_deterioration` is a generated target score, not a
          measured probability of equipment failure.
        - Ground-truth drift labels are used for offline evaluation.
        - KS and PSI monitor changes in feature distributions.
        - Page-Hinkley can monitor a signal such as prediction errors.
        - Detector alarms and true drift labels are different.
        - Static prediction, transfer learning, full retraining, and
          continual learning should use consistent evaluation windows.
        """
    )

    st.subheader("Expected experiment outputs")

    expected_files = [
        "strategy_metrics.csv",
        "predictions.csv",
        "detector_window_metrics.csv",
        "adaptation_events.csv",
    ]

    status = pd.DataFrame([
        {
            "File": name,
            "Status": (
                "Available"
                if (RESULTS / name).is_file()
                else "Not generated"
            ),
        }
        for name in expected_files
    ])

    st.dataframe(status, width="stretch", hide_index=True)
