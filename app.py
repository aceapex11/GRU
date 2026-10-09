
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="GRU Drift Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("GRU Drift Intelligence")
st.caption("PyTorch GRU | KS | PSI | Page-Hinkley | Continual learning")


def find_dataset():
    candidates = [
        ROOT / "industrial_sensor_drift_dataset.csv",
        ROOT / "data" / "industrial_sensor_drift_dataset.csv",
    ]

    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path

    for folder in [ROOT, ROOT / "data"]:
        if folder.is_dir():
            matches = list(
                folder.glob("*industrial_sensor_drift_dataset*.csv")
            )
            for path in matches:
                if path.is_file() and path.stat().st_size:
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


def result_csv(name):
    path = RESULTS / name

    if not path.is_file() or not path.stat().st_size:
        return None

    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def flag_values(series):
    if series.dtype == object:
        return series.astype(str).str.lower().str.strip().isin(
            ["1", "true", "yes", "alarm", "drift"]
        )

    return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)


def row_positions(frame, dataset):
    """Map event row IDs to the dataset's real x-axis."""
    if frame is None or frame.empty:
        return []

    index_col = next(
        (
            col for col in [
                "row_index",
                "row",
                "observation_index",
                "sample_index",
                "detected_at",
                "adapted_at",
                "adaptation_point",
                "window_start",
            ]
            if col in frame.columns
        ),
        None,
    )

    if index_col is None:
        return []

    ids = pd.to_numeric(frame[index_col], errors="coerce").dropna()
    result = []

    for value in ids:
        if not float(value).is_integer():
            continue

        idx = int(value)

        if 0 <= idx < len(dataset):
            if (
                "timestamp" in dataset.columns
                and dataset["timestamp"].notna().any()
            ):
                x = dataset.iloc[idx]["timestamp"]
            else:
                x = idx

            if pd.notna(x):
                result.append(x)

    return result


def add_events(fig, dataset, x_column, detector_data, adaptation_data):
    def add_lines(values, color, dash, label, maximum=12):
        values = list(dict.fromkeys(str(v) for v in values if pd.notna(v)))

        if not values:
            return

        count = len(values)

        if count > maximum:
            indices = np.linspace(0, count - 1, maximum, dtype=int)
            values = [values[i] for i in indices]

        if x_column == "timestamp":
            values = pd.to_datetime(values, errors="coerce")
            values = [v for v in values if pd.notna(v)]
        else:
            values = pd.to_numeric(
                pd.Series(values), errors="coerce"
            ).dropna().tolist()

        if not values:
            return

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{label} ({count} total)",
                line=dict(color=color, dash=dash, width=2),
                hoverinfo="skip",
            )
        )

        for value in values:
            try:
                fig.add_vline(
                    x=value,
                    line_color=color,
                    line_dash=dash,
                    line_width=1.7,
                    opacity=0.85,
                )
            except (ValueError, TypeError, OverflowError):
                pass

    if detector_data is not None and not detector_data.empty:
        alarm_cols = [
            col for col in [
                "KS_alarm",
                "PSI_alarm",
                "Page_Hinkley_alarm",
            ]
            if col in detector_data.columns
        ]

        if alarm_cols:
            mask = detector_data[alarm_cols].apply(
                flag_values
            ).any(axis=1)

            alarm_rows = detector_data.loc[mask]

            add_lines(
                row_positions(alarm_rows, dataset),
                "#E87500",
                "dash",
                "Detector alarms",
            )

    if adaptation_data is not None and not adaptation_data.empty:
        add_lines(
            row_positions(adaptation_data, dataset),
            "#008B8B",
            "dot",
            "Adaptations",
        )

    return fig


def add_truth_regions(fig, data):
    if "timestamp" not in data.columns:
        return fig

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")

    if timestamps.notna().sum() == 0:
        return fig

    flags = np.zeros(len(data), dtype=int)

    for col, weight in [
        ("data_drift", 1),
        ("concept_drift", 2),
    ]:
        if col in data.columns:
            flags += flag_values(data[col]).to_numpy(dtype=int) * weight

    flags = np.where(timestamps.notna().to_numpy(), flags, 0)
    starts = np.where(np.r_[True, flags[1:] != flags[:-1]])[0]
    ends = np.r_[starts[1:], len(flags)]

    colors = {
        1: "rgba(245,166,35,0.12)",
        2: "rgba(120,90,200,0.12)",
        3: "rgba(220,80,80,0.14)",
    }

    for start, end in zip(starts, ends):
        state = int(flags[start])

        if state == 0:
            continue

        x0 = timestamps.iloc[start]
        x1 = timestamps.iloc[end - 1]

        if pd.notna(x0) and pd.notna(x1):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=colors[state],
                line_width=0,
                layer="below",
            )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

DATA = find_dataset()

with st.sidebar:
    st.header("Experiment")

    if DATA:
        st.success(DATA.name)
    else:
        st.error("Dataset not found")

    full = st.checkbox("Full training", value=False)

    if st.button("Run GRU experiment", type="primary", width="stretch"):
        script = ROOT / "experiment.py"

        if DATA is None:
            st.error("Place the dataset CSV beside app.py.")
        elif not script.exists():
            st.error("experiment.py not found.")
        else:
            command = [
                sys.executable,
                str(script),
                "--data", str(DATA),
                "--out", str(RESULTS),
            ]

            if full:
                command.append("--full")

            try:
                with st.spinner("Running experiment..."):
                    result = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=7200,
                    )

                if result.returncode == 0:
                    st.success("Experiment completed.")
                    st.code((result.stdout or "")[-4000:])
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


if DATA is None:
    st.error(
        "Dataset not found. Expected "
        "industrial_sensor_drift_dataset.csv."
    )
    st.stop()

try:
    df = load_csv(
        str(DATA),
        DATA.stat().st_mtime,
        DATA.stat().st_size,
    )
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

if df.empty:
    st.error("Dataset is empty.")
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

targets = [
    col for col in ["health_deterioration"]
    if col in df.columns
]

# ============================================================
# SUMMARY
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(df):,}")
c2.metric("Sensor features", len(sensor_cols))
c3.metric(
    "Regimes",
    df["regime"].nunique() if "regime" in df.columns else "—",
)
c4.metric(
    "Adaptations",
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
    choices = sensor_cols + targets

    if choices:
        signal = st.selectbox(
            "Signal",
            choices,
            key="overview_signal",
        )

        sample = df.iloc[::max(1, len(df) // 3000)]

        fig = px.line(
            sample,
            x=X_COL,
            y=signal,
            title=f"{signal} over time",
        )

        fig = add_events(fig, df, X_COL, detectors, events)

        if HAS_DATES:
            fig = add_truth_regions(fig, df)

        fig.update_layout(
            height=520,
            hovermode="x unified",
            xaxis_title=X_LABEL,
        )

        st.plotly_chart(fig, width="stretch")

    if metrics is not None:
        st.subheader("Strategy metrics")
        view = metrics.copy()

        if "Phase" in view.columns:
            overall = view[
                view["Phase"].astype(str).str.lower() == "overall"
            ]
            if not overall.empty:
                view = overall

        st.dataframe(view, width="stretch", hide_index=True)
    else:
        st.info("Run the experiment to generate results.")


# ============================================================
# TAB 2: SENSOR EXPLORER
# ============================================================

with tabs[1]:
    choices = sensor_cols + targets

    if choices:
        signal = st.selectbox(
            "Sensor or target",
            choices,
            key="sensor_signal",
        )

        max_points = st.slider(
            "Maximum plotted points",
            500, 10000, 3000, step=500,
        )

        sample = df.iloc[::max(1, len(df) // max_points)]

        fig = px.line(
            sample,
            x=X_COL,
            y=signal,
            title=signal,
        )

        fig = add_events(fig, df, X_COL, detectors, events)

        if HAS_DATES:
            fig = add_truth_regions(fig, df)

        fig.update_layout(height=520, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Descriptive statistics")
        st.dataframe(
            df[choices].describe().T,
            width="stretch",
        )


# ============================================================
# TAB 3: DRIFT TIMELINE
# ============================================================

with tabs[2]:
    if "regime" in df.columns:
        counts = (
            df.groupby("regime", dropna=False)
            .size()
            .reset_index(name="records")
        )

        st.plotly_chart(
            px.bar(counts, x="regime", y="records"),
            width="stretch",
        )

        choices = [
            col for col in [
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
                key="timeline_signal",
            )

            sample = df.iloc[::max(1, len(df) // 3000)]

            fig = px.line(
                sample,
                x=X_COL,
                y=signal,
                color="regime",
                title=f"{signal} by regime",
            )

            fig = add_events(fig, df, X_COL, detectors, events)

            if HAS_DATES:
                fig = add_truth_regions(fig, df)

            fig.update_layout(height=550, hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No regime column found.")


# ============================================================
# TAB 4: DETECTOR ANALYSIS
# ============================================================

with tabs[3]:
    st.markdown(
        """
        **KS:** sensor distribution changes.

        **PSI:** magnitude of distribution changes.

        **Page-Hinkley:** changes in the monitored prediction residual.
        """
    )

    if detectors is None:
        st.info("Run the experiment to generate detector results.")
    else:
        st.dataframe(
            detectors.tail(500),
            width="stretch",
            hide_index=True,
        )

        alarm_cols = [
            col for col in [
                "KS_alarm",
                "PSI_alarm",
                "Page_Hinkley_alarm",
            ]
            if col in detectors.columns
        ]

        if alarm_cols:
            counts = (
                detectors[alarm_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum()
                .rename_axis("Detector")
                .reset_index(name="Alarm windows")
            )

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Detector",
                    y="Alarm windows",
                ),
                width="stretch",
            )

        for col, title in [
            ("max_PSI", "Maximum PSI"),
            ("max_KS_stat", "Maximum KS statistic"),
        ]:
            if col in detectors.columns:
                xcol = (
                    "timestamp"
                    if "timestamp" in detectors.columns
                    else "window_start"
                )

                if xcol not in detectors.columns:
                    continue

                fig = px.line(
                    detectors,
                    x=xcol,
                    y=col,
                    title=title,
                )

                st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 5: STRATEGY COMPARISON
# ============================================================

with tabs[4]:
    if metrics is None:
        st.info("No strategy metrics available.")
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
                view = view[view["Phase"].astype(str) == phase]

        st.dataframe(view, width="stretch", hide_index=True)

        choices = [
            col for col in ["MAE", "RMSE", "R2"]
            if col in view.columns
        ]

        if "Strategy" in view.columns and choices:
            metric = st.selectbox(
                "Metric",
                choices,
                key="strategy_metric",
            )

            fig = px.bar(
                view,
                x="Strategy",
                y=metric,
                title=f"{metric} by strategy",
                text_auto=".3f",
            )

            st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 6: PREDICTIONS
# ============================================================

with tabs[5]:
    if predictions is None:
        st.info("Run the experiment to generate predictions.")
    else:
        pred = predictions.copy()

        if "row" in pred.columns:
            ids = pd.to_numeric(pred["row"], errors="coerce")
        elif "row_index" in pred.columns:
            ids = pd.to_numeric(pred["row_index"], errors="coerce")
        else:
            ids = pd.Series(np.arange(len(pred)), index=pred.index)

        valid = ids.notna() & ids.between(0, len(df) - 1)
        pred = pred.loc[valid].copy()
        ids = ids.loc[valid].astype(int)

        if HAS_DATES:
            pred["plot_x"] = df.iloc[ids.to_numpy()][
                "timestamp"
            ].to_numpy()
        else:
            pred["plot_x"] = ids.to_numpy()

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
            "plot_x",
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

        sample = pred.iloc[::max(1, len(pred) // 5000)]
        fig = go.Figure()

        if actual_col:
            fig.add_trace(
                go.Scatter(
                    x=sample["plot_x"],
                    y=sample[actual_col],
                    mode="lines",
                    name="Actual target",
                    line=dict(width=2.5),
                )
            )

        for col in model_cols:
            fig.add_trace(
                go.Scatter(
                    x=sample["plot_x"],
                    y=sample[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1.5),
                )
            )

        if HAS_DATES:
            fig = add_truth_regions(fig, df)

        fig = add_events(fig, df, X_COL, detectors, events)

        fig.update_layout(
            title="Actual vs predicted",
            xaxis_title=X_LABEL,
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
        )

        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            predictions.tail(100),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 7: ADAPTATION EVENTS
# ============================================================

with tabs[6]:
    if events is None:
        st.info("No adaptation events available.")
    else:
        st.dataframe(events, width="stretch", hide_index=True)

        if not events.empty:
            ycol = next(
                (
                    col for col in [
                        "n_samples",
                        "training_seconds",
                    ]
                    if col in events.columns
                ),
                None,
            )

            xcol = next(
                (
                    col for col in [
                        "adapted_at",
                        "detected_at",
                        "row_index",
                    ]
                    if col in events.columns
                ),
                None,
            )

            if xcol and ycol:
                st.plotly_chart(
                    px.scatter(
                        events,
                        x=xcol,
                        y=ycol,
                        title="Adaptation events",
                    ),
                    width="stretch",
                )


# ============================================================
# TAB 8: DATA & NOTES
# ============================================================

with tabs[7]:
    st.subheader("Dataset preview")
    st.write(f"Dataset: `{DATA.name}`")

    st.dataframe(
        df.head(30),
        width="stretch",
        hide_index=True,
    )

    missing = df.isna().sum().rename(
        "missing_count"
    ).to_frame()

    missing["missing_pct"] = (
        missing["missing_count"] / max(len(df), 1) * 100
    ).round(3)

    st.subheader("Missing values")
    st.dataframe(missing, width="stretch")

    st.subheader("Output files")

    expected = [
        "strategy_metrics.csv",
        "predictions.csv",
        "detector_window_metrics.csv",
        "adaptation_events.csv",
        "ground_truth_offline_only.csv",
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
        for name in expected
    ])

    st.dataframe(status, width="stretch", hide_index=True)

    st.caption(
        "Ground-truth drift labels are for offline evaluation only "
        "and must not be used as detector inputs."
    )
