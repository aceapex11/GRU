
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="GRU Drift Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("GRU Drift Intelligence")
st.caption(
    "PyTorch GRU | Data drift | Concept drift | "
    "KS | PSI | Page-Hinkley | Continual learning"
)


# ============================================================
# DATA AND RESULT HELPERS
# ============================================================

def find_dataset():
    candidates = [
        ROOT / "industrial_sensor_drift_dataset.csv",
        ROOT / "data" / "industrial_sensor_drift_dataset.csv",
    ]

    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path

    return None


@st.cache_data(show_spinner=False)
def load_csv(path, modified, size):
    data = pd.read_csv(path)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"], errors="coerce"
        )

    data["row_index"] = np.arange(len(data))
    return data


def result_csv(filename):
    path = RESULTS / filename

    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def flag_values(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series, errors="coerce"
        ).fillna(0).ne(0)

    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin(["1", "true", "yes", "alarm", "drift"])
    )


def event_row(event, candidates):
    """Return the first usable row-index field from an event."""
    for col in candidates:
        if col in event.index:
            value = pd.to_numeric(
                pd.Series([event[col]]), errors="coerce"
            ).iloc[0]

            if pd.notna(value) and float(value).is_integer():
                return int(value)

    return None


def row_to_x(row, data, has_dates):
    if row is None or not 0 <= row < len(data):
        return None

    if has_dates:
        value = data.iloc[row]["timestamp"]
        return value if pd.notna(value) else None

    return row


def add_drift_regions(fig, data, has_dates):
    """Subtle shaded regions for offline ground-truth labels."""
    if not has_dates:
        return fig

    timestamp = pd.to_datetime(data["timestamp"], errors="coerce")

    if timestamp.notna().sum() == 0:
        return fig

    state = np.zeros(len(data), dtype=int)

    if "data_drift" in data.columns:
        state += flag_values(data["data_drift"]).to_numpy(dtype=int)

    if "concept_drift" in data.columns:
        state += (
            2 * flag_values(data["concept_drift"]).to_numpy(dtype=int)
        )

    state[timestamp.isna().to_numpy()] = 0

    starts = np.where(np.r_[True, state[1:] != state[:-1]])[0]
    ends = np.r_[starts[1:], len(state)]

    colors = {
        1: "rgba(245,166,35,0.10)",
        2: "rgba(120,90,200,0.10)",
        3: "rgba(220,80,80,0.12)",
    }

    for start, end in zip(starts, ends):
        label = int(state[start])

        if label == 0:
            continue

        x0 = timestamp.iloc[start]
        x1 = timestamp.iloc[end - 1]

        if pd.notna(x0) and pd.notna(x1):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=colors[label],
                line_width=0,
                layer="below",
            )

    return fig


def add_adaptation_markers(
    fig,
    data,
    events,
    has_dates,
    prediction_column=None,
    prediction_data=None,
):
    """Show adaptation points as diamonds, not vertical lines."""
    if events is None or events.empty:
        return fig

    x_values = []
    y_values = []
    hover_text = []

    for _, event in events.iterrows():
        row = event_row(
            event,
            ["adapted_at", "adaptation_point", "row_index"],
        )

        x_value = row_to_x(row, data, has_dates)

        if x_value is None:
            continue

        y_value = None

        if (
            prediction_column is not None
            and prediction_data is not None
            and row is not None
        ):
            match = prediction_data[
                prediction_data["_row_id"] == row
            ]

            if not match.empty:
                value = pd.to_numeric(
                    pd.Series([match.iloc[0][prediction_column]]),
                    errors="coerce",
                ).iloc[0]

                if pd.notna(value):
                    y_value = float(value)

        if y_value is None:
            # A horizontal reference marker if predictions are unavailable.
            y_value = np.nan

        if pd.notna(y_value):
            x_values.append(x_value)
            y_values.append(y_value)

            detected = event.get("detected_at", "N/A")
            adapted = event.get("adapted_at", "N/A")
            samples = event.get("n_samples", "N/A")

            hover_text.append(
                f"Detected at row: {detected}"
                f"<br>Adapted at row: {adapted}"
                f"<br>Training samples: {samples}"
            )

    if x_values:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name="Adaptation",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color="#00897B",
                    line=dict(color="white", width=1),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    return fig


def calculate_metrics(actual, predicted):
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")

    valid = (
        actual.notna()
        & predicted.notna()
        & np.isfinite(actual)
        & np.isfinite(predicted)
    )

    if not valid.any():
        return None

    y_true = actual.loc[valid].to_numpy()
    y_pred = predicted.loc[valid].to_numpy()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Valid predictions": int(valid.sum()),
    }


def make_prediction_frame(predictions, data, has_dates):
    """Align prediction rows with dataset positions and timestamps."""
    pred = predictions.copy()

    if "row_index" in pred.columns:
        row_values = pd.to_numeric(
            pred["row_index"], errors="coerce"
        )
    elif "row" in pred.columns:
        row_values = pd.to_numeric(pred["row"], errors="coerce")
    else:
        row_values = pd.Series(
            np.arange(len(pred)), index=pred.index
        )

    valid = (
        row_values.notna()
        & row_values.between(0, len(data) - 1)
    )

    pred = pred.loc[valid].copy()
    row_values = row_values.loc[valid].astype(int)

    pred["_row_id"] = row_values.to_numpy()

    pred["_plot_x"] = [
        row_to_x(int(row), data, has_dates)
        for row in pred["_row_id"]
    ]

    return pred


# ============================================================
# SIDEBAR
# ============================================================

DATA = find_dataset()

with st.sidebar:
    st.header("Experiment")

    if DATA is None:
        st.error("Dataset not found.")
        st.caption(
            "Place industrial_sensor_drift_dataset.csv "
            "beside app.py or inside the data folder."
        )
    else:
        st.success(f"Dataset: {DATA.name}")

    full_training = st.checkbox(
        "Full training",
        value=False,
        help="Uses more epochs and takes longer.",
    )

    if st.button(
        "Run GRU experiment",
        type="primary",
        width="stretch",
    ):
        script = ROOT / "experiment.py"

        if DATA is None:
            st.error("Dataset file was not found.")
        elif not script.is_file():
            st.error("experiment.py was not found.")
        else:
            command = [
                sys.executable,
                str(script),
                "--data", str(DATA),
                "--out", str(RESULTS),
            ]

            if full_training:
                command.append("--full")

            try:
                with st.spinner(
                    "Training and evaluating the GRU experiment..."
                ):
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
                        )[-9000:]
                    )

            except subprocess.TimeoutExpired:
                st.error("Experiment exceeded the 2-hour timeout.")

            st.cache_data.clear()
            st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

if DATA is None:
    st.stop()

try:
    df = load_csv(
        str(DATA),
        DATA.stat().st_mtime,
        DATA.stat().st_size,
    )
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

HAS_DATES = (
    "timestamp" in df.columns
    and df["timestamp"].notna().any()
)

X_COL = "timestamp" if HAS_DATES else "row_index"
X_LABEL = "Time" if HAS_DATES else "Observation index"

metrics = result_csv("strategy_metrics.csv")
predictions = result_csv("predictions.csv")
detectors = result_csv("detector_window_metrics.csv")
events = result_csv("adaptation_events.csv")

sensor_cols = [
    col for col in [
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

target_cols = [
    col for col in ["health_deterioration"]
    if col in df.columns
]


# ============================================================
# TOP SUMMARY
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(df):,}")
c2.metric("Sensor features", str(len(sensor_cols)))

if "regime" in df.columns:
    c3.metric("Regimes", str(df["regime"].nunique()))
else:
    c3.metric("Regimes", "Not available")

c4.metric(
    "Adaptation events",
    str(len(events)) if events is not None else "Not run",
)


# ============================================================
# TABS
# ============================================================

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
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:
    st.subheader("Dataset overview")

    signals = sensor_cols + target_cols

    if signals:
        signal = st.selectbox(
            "Select signal",
            signals,
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

        if HAS_DATES:
            fig = add_drift_regions(fig, df, HAS_DATES)

        fig.update_layout(
            height=480,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

    if metrics is not None and not metrics.empty:
        st.subheader("Overall model comparison")

        overall = metrics.copy()

        if "Phase" in overall.columns:
            overall = overall[
                overall["Phase"].astype(str).str.lower() == "overall"
            ]

        st.dataframe(
            overall,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Run the experiment to see model results.")


# ============================================================
# TAB 2 — SENSOR EXPLORER
# ============================================================

with tabs[1]:
    st.subheader("Explore sensor behaviour")

    signals = sensor_cols + target_cols

    if not signals:
        st.warning("No expected sensor columns were found.")
    else:
        signal = st.selectbox(
            "Sensor or target",
            signals,
            key="sensor_signal",
        )

        max_points = st.slider(
            "Maximum sensor chart points",
            min_value=500,
            max_value=10000,
            value=3000,
            step=500,
        )

        stride = max(1, len(df) // max_points)
        sample = df.iloc[::stride]

        fig = px.line(
            sample,
            x=X_COL,
            y=signal,
            title=f"{signal} readings",
        )

        if HAS_DATES:
            fig = add_drift_regions(fig, df, HAS_DATES)

        fig.update_layout(
            height=520,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

        st.subheader("Descriptive statistics")
        st.dataframe(
            df[signals].describe().T,
            width="stretch",
        )


# ============================================================
# TAB 3 — DRIFT TIMELINE
# ============================================================

with tabs[2]:
    st.subheader("Drift timeline")

    if "regime" in df.columns:
        counts = (
            df.groupby("regime", dropna=False)
            .size()
            .reset_index(name="Records")
        )

        st.plotly_chart(
            px.bar(
                counts,
                x="regime",
                y="Records",
                title="Records by regime",
            ),
            width="stretch",
        )
    else:
        st.info("No regime column found.")

    drift_signals = [
        col for col in [
            "temperature_C",
            "pressure_bar",
            "vibration_mm_s_rms",
            "health_deterioration",
        ]
        if col in df.columns
    ]

    if drift_signals:
        signal = st.selectbox(
            "Timeline signal",
            drift_signals,
            key="timeline_signal",
        )

        sample = df.iloc[::max(1, len(df) // 3000)]

        fig = px.line(
            sample,
            x=X_COL,
            y=signal,
            color="regime" if "regime" in sample.columns else None,
            title=f"{signal} through the stream",
        )

        if HAS_DATES:
            fig = add_drift_regions(fig, df, HAS_DATES)

        fig.update_layout(
            height=550,
            hovermode="x unified",
        )

        st.plotly_chart(fig, width="stretch")

    st.caption(
        "Shaded areas use the dataset's ground-truth labels for "
        "offline evaluation. They are not detector predictions."
    )


# ============================================================
# TAB 4 — DETECTOR ANALYSIS
# ============================================================

with tabs[3]:
    st.subheader("Detector analysis")

    st.markdown(
        """
        - **KS:** tests whether sensor distributions differ from a reference.
        - **PSI:** measures the magnitude of a distribution shift.
        - **Page-Hinkley:** monitors changes in the prediction-residual signal.
        """
    )

    if detectors is None or detectors.empty:
        st.info("Run the experiment to generate detector metrics.")
    else:
        alarm_cols = [
            col for col in [
                "KS_alarm",
                "PSI_alarm",
                "Page_Hinkley_alarm",
            ]
            if col in detectors.columns
        ]

        if alarm_cols:
            alarm_summary = pd.DataFrame({
                "Detector": alarm_cols,
                "Alarm windows": [
                    int(flag_values(detectors[col]).sum())
                    for col in alarm_cols
                ],
            })

            st.plotly_chart(
                px.bar(
                    alarm_summary,
                    x="Detector",
                    y="Alarm windows",
                    title="Alarm windows by detector",
                ),
                width="stretch",
            )

        value_col = st.selectbox(
            "Detector metric",
            [
                col for col in [
                    "max_PSI",
                    "max_KS_stat",
                    "min_KS_p",
                ]
                if col in detectors.columns
            ] or ["No detector metric available"],
            key="detector_metric",
        )

        if value_col != "No detector metric available":
            x_col = (
                "timestamp"
                if "timestamp" in detectors.columns
                else "window_start"
            )

            fig = px.line(
                detectors,
                x=x_col,
                y=value_col,
                title=value_col,
            )

            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")

        with st.expander("View detector window records"):
            st.dataframe(
                detectors,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# TAB 5 — STRATEGY COMPARISON
# ============================================================

with tabs[4]:
    st.subheader("Compare adaptation strategies")

    if metrics is None or metrics.empty:
        st.info("Run the experiment to generate strategy metrics.")
    else:
        view = metrics.copy()

        if "Phase" in view.columns:
            phases = sorted(
                view["Phase"].dropna().astype(str).unique()
            )

            if phases:
                selected_phase = st.selectbox(
                    "Evaluation phase",
                    phases,
                    key="comparison_phase",
                )

                view = view[
                    view["Phase"].astype(str) == selected_phase
                ]

        st.dataframe(
            view,
            width="stretch",
            hide_index=True,
        )

        metric_options = [
            col for col in ["MAE", "RMSE", "R2"]
            if col in view.columns
        ]

        if metric_options and "Strategy" in view.columns:
            selected_metric = st.selectbox(
                "Performance metric",
                metric_options,
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


# ============================================================
# TAB 6 — PREDICTIONS
# ============================================================

with tabs[5]:
    st.subheader("Actual vs predicted")

    if predictions is None or predictions.empty:
        st.info("Run the experiment to generate predictions.")
    else:
        pred = make_prediction_frame(
            predictions, df, HAS_DATES
        )

        actual_col = next(
            (
                col for col in [
                    "actual",
                    "actual_target",
                    "y_true",
                ]
                if col in pred.columns
            ),
            None,
        )

        excluded = {
            "timestamp",
            "_plot_x",
            "_row_id",
            "row",
            "row_index",
            "actual",
            "actual_target",
            "y_true",
        }

        model_cols = [
            col for col in pred.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(pred[col])
        ]

        if actual_col is None:
            st.error(
                "The actual target column was not found in predictions.csv."
            )
        elif not model_cols:
            st.error(
                "No numeric model prediction columns were found."
            )
        else:
            selected_model = st.selectbox(
                "Choose a model",
                model_cols,
                key="prediction_model",
            )

            max_points = st.slider(
                "Maximum prediction chart points",
                min_value=1000,
                max_value=20000,
                value=8000,
                step=1000,
                key="prediction_max_points",
            )

            stride = max(1, int(np.ceil(len(pred) / max_points)))
            sample = pred.iloc[::stride].copy()

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=sample["_plot_x"],
                    y=sample[actual_col],
                    mode="lines",
                    name="Actual target",
                    line=dict(
                        color="#1565C0",
                        width=2.3,
                    ),
                    connectgaps=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=sample["_plot_x"],
                    y=sample[selected_model],
                    mode="lines",
                    name=selected_model,
                    line=dict(
                        color="#E53935",
                        width=1.8,
                    ),
                    connectgaps=False,
                )
            )

            if HAS_DATES:
                fig = add_drift_regions(fig, df, HAS_DATES)

            fig = add_adaptation_markers(
                fig,
                df,
                events,
                HAS_DATES,
                prediction_column=selected_model,
                prediction_data=pred,
            )

            fig.update_layout(
                title=f"Actual target vs {selected_model}",
                xaxis_title=X_LABEL,
                yaxis_title="Health deterioration",
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                margin=dict(
                    l=35,
                    r=25,
                    t=100,
                    b=35,
                ),
            )

            st.plotly_chart(fig, width="stretch")

            scores = calculate_metrics(
                pred[actual_col],
                pred[selected_model],
            )

            if scores is not None:
                a, b, c = st.columns(3)
                a.metric("MAE", f"{scores['MAE']:.4f}")
                b.metric("RMSE", f"{scores['RMSE']:.4f}")
                c.metric(
                    "Valid predictions",
                    f"{scores['Valid predictions']:,}",
                )

            st.caption(
                "MAE and RMSE above are calculated over all valid "
                "predictions for the selected model."
            )

            with st.expander("Inspect prediction rows"):
                st.dataframe(
                    pred.tail(100),
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# TAB 7 — ADAPTATION EVENTS
# ============================================================

with tabs[6]:
    st.subheader("Adaptation events")

    if events is None or events.empty:
        st.info(
            "No adaptation events are available. "
            "Run the experiment and check its detector output."
        )
    else:
        st.metric("Total adaptation events", len(events))

        st.dataframe(
            events,
            width="stretch",
            hide_index=True,
        )

        x_col = next(
            (
                col for col in [
                    "adapted_at",
                    "adaptation_point",
                    "row_index",
                ]
                if col in events.columns
            ),
            None,
        )

        y_col = next(
            (
                col for col in [
                    "n_samples",
                    "training_seconds",
                ]
                if col in events.columns
            ),
            None,
        )

        if x_col and y_col:
            fig = px.scatter(
                events,
                x=x_col,
                y=y_col,
                hover_data=[
                    col for col in [
                        "detected_at",
                        "detectors",
                        "n_samples",
                        "training_seconds",
                    ]
                    if col in events.columns
                ],
                title=f"{y_col} by adaptation point",
            )

            st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 8 — DATA AND NOTES
# ============================================================

with tabs[7]:
    st.subheader("Dataset preview")
    st.write(f"Dataset file: `{DATA.name}`")

    st.dataframe(
        df.head(30),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Missing values")

    missing_summary = pd.DataFrame({
        "Missing count": df.isna().sum(),
        "Missing percentage": (
            df.isna().mean() * 100
        ).round(3),
    })

    st.dataframe(
        missing_summary,
        width="stretch",
    )

    st.subheader("Experiment output files")

    output_names = [
        "strategy_metrics.csv",
        "predictions.csv",
        "detector_window_metrics.csv",
        "adaptation_events.csv",
        "ground_truth_offline_only.csv",
    ]

    output_status = pd.DataFrame([
        {
            "File": filename,
            "Status": (
                "Available"
                if (RESULTS / filename).is_file()
                else "Not generated"
            ),
        }
        for filename in output_names
    ])

    st.dataframe(
        output_status,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Ground-truth drift labels are for offline evaluation. "
        "They must not be used as live detector inputs."
    )
