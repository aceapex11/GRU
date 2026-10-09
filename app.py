from pathlib import Path
import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA_CANDIDATES = [
    ROOT / "industrial_sensor_drift_dataset.csv",
    ROOT / "industrial_sensor_drift_dataset (1).csv",
    ROOT / "industrial_sensor_drift_dataset (7).csv",
    ROOT / "data" / "industrial_sensor_drift_dataset.csv",
]

st.set_page_config(page_title="GRU Drift Intelligence", page_icon="📈", layout="wide")
st.title("GRU Drift Intelligence")
st.caption("Predict → Detect → Diagnose → Adapt → Evaluate | GRU deep-learning benchmark")
st.info("The dataset is synthetic. Ground-truth drift labels are used only for offline evaluation.")


def resolve_dataset():
    for candidate in DATA_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    for folder in (ROOT, ROOT / "data"):
        if folder.is_dir():
            for candidate in folder.glob("*.csv"):
                if "industrial_sensor_drift_dataset" in candidate.name.lower() and candidate.stat().st_size > 0:
                    return candidate
    return None


@st.cache_data(show_spinner=False)
def load_data(path, modified_time, file_size):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("CSV contains no data rows.")
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def read_result(name):
    path = RESULTS / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
        return frame if not frame.empty else None
    except (EmptyDataError, ParserError, OSError):
        return None


def add_event_markers(fig, x_column, detector_frame=None, adaptation_frame=None):
    """Overlay alarm and adaptation points where event timestamps/indices are available."""
    if detector_frame is not None and not detector_frame.empty:
        xcol = next(
            (c for c in [x_column, "timestamp", "window_start", "row", "index"]
             if c in detector_frame.columns),
            None,
        )
        alarm_cols = [
            c for c in ["KS_alarm", "PSI_alarm", "Page_Hinkley_alarm", "alarm", "drift_alarm"]
            if c in detector_frame.columns
        ]
        if xcol and alarm_cols:
            alarm_mask = (
                detector_frame[alarm_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(bool)
                .any(axis=1)
            )
            for value in detector_frame.loc[alarm_mask, xcol].dropna().tolist():
                try:
                    fig.add_vline(
                        x=value, line_dash="dash", line_width=1,
                        line_color="#E45756", opacity=0.75,
                    )
                except (TypeError, ValueError):
                    pass

    if adaptation_frame is not None and not adaptation_frame.empty:
        adapt_col = next(
            (c for c in ["adapted_at", "adaptation_timestamp", "adaptation_point", x_column]
             if c in adaptation_frame.columns),
            None,
        )
        if adapt_col:
            for value in adaptation_frame[adapt_col].dropna().tolist():
                try:
                    fig.add_vline(
                        x=value, line_dash="dot", line_width=2,
                        line_color="#2A9D8F", opacity=0.9,
                    )
                except (TypeError, ValueError):
                    pass

    return fig


DATA = resolve_dataset()
metrics = None
events = None
predictions = None
detector_metrics = None

with st.sidebar:
    st.header("Experiment controls")
    st.write(f"Dataset: {DATA.name}" if DATA else "Dataset: not found")
    full = st.checkbox("Full training (slower)", value=False)
    run = st.button("Run GRU experiment", type="primary", use_container_width=True)
    if run:
        if DATA is None:
            st.error("Upload a non-empty industrial_sensor_drift_dataset.csv beside app.py.")
        elif not (ROOT / "experiment.py").is_file():
            st.error(f"Experiment script not found: {ROOT / 'experiment.py'}")
        else:
            RESULTS.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(ROOT / "experiment.py"),
                "--data", str(DATA), "--out", str(RESULTS),
            ]
            if full:
                command.append("--full")
            try:
                with st.spinner("Running GRU experiment..."):
                    proc = subprocess.run(
                        command, cwd=str(ROOT), capture_output=True,
                        text=True, timeout=7200,
                    )
                if proc.returncode == 0:
                    st.success("Experiment finished.")
                    if proc.stdout:
                        st.code(proc.stdout[-5000:])
                else:
                    st.error("Experiment failed:")
                    st.code(((proc.stderr or "") + "\n" + (proc.stdout or ""))[-8000:])
            except subprocess.TimeoutExpired:
                st.error("Experiment timed out. Try with Full training unchecked.")
            st.cache_data.clear()
            st.rerun()

if DATA is None:
    st.error("Dataset not found or empty. Put the CSV beside app.py.")
    st.write("Files in repository root:")
    try:
        st.code("\n".join(sorted(p.name for p in ROOT.iterdir() if p.is_file())))
    except OSError:
        pass
    st.stop()

try:
    df = load_data(str(DATA), DATA.stat().st_mtime, DATA.stat().st_size)
except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
    st.error(f"Could not load dataset `{DATA.name}`: {exc}")
    st.stop()

if "timestamp" not in df.columns:
    st.error("Dataset must contain a `timestamp` column.")
    st.stop()

metrics = read_result("strategy_metrics.csv")
events = read_result("adaptation_events.csv")
predictions = read_result("predictions.csv")
detector_metrics = read_result("detector_window_metrics.csv")

sensor_features = [
    c for c in [
        "temperature_C", "pressure_bar", "vibration_mm_s_rms",
        "load_pct", "humidity_pct", "motor_speed_rpm", "operating_hours",
    ] if c in df.columns
]
targets = [c for c in ["health_deterioration"] if c in df.columns]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Sensor inputs", str(len(sensor_features)))
c3.metric("Drift regimes", str(df["regime"].nunique()) if "regime" in df else "—")
c4.metric("Adaptation events", f"{len(events):,}" if events is not None else "Run experiment")

tabs = st.tabs([
    "Overview", "Sensor Explorer", "Drift Timeline",
    "Detector Analysis (KS / PSI / Page-Hinkley)",
    "GRU Strategy Comparison", "Predictions", "Adaptation Events", "Data & Notes",
])

with tabs[0]:
    st.subheader("Sensor stream")
    choices = sensor_features + targets
    if choices:
        signal = st.selectbox("Signal", choices, key="overview_signal")
        plot_df = df.iloc[::max(1, len(df) // 3000)]
        fig = px.line(plot_df, x="timestamp", y=signal, title=f"{signal} over time")
        fig = add_event_markers(fig, "timestamp", detector_metrics, events)
        st.plotly_chart(fig, use_container_width=True)
    if metrics is not None:
        view = metrics.copy()
        if "Phase" in view.columns:
            overall = view[view["Phase"].astype(str).str.lower() == "overall"]
            if not overall.empty:
                view = overall
        if "MAE" in view.columns:
            view = view.sort_values("MAE")
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.warning("No GRU results yet. Run the experiment from the sidebar.")

with tabs[1]:
    choices = sensor_features + targets
    if choices:
        sensor = st.selectbox("Sensor / target", choices)
        max_points = st.slider("Maximum plotted points", 500, 10000, 3000, step=500)
        plot_df = df.iloc[::max(1, len(df) // max_points)]
        fig = px.line(plot_df, x="timestamp", y=sensor, title=f"{sensor} — alarms and adaptation")
        fig = add_event_markers(fig, "timestamp", detector_metrics, events)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df[choices].describe().T, use_container_width=True)
    else:
        st.warning("No recognized sensor columns found.")

with tabs[2]:
    st.subheader("Designed drift regimes")
    if "regime" in df.columns:
        counts = df.groupby("regime", as_index=False).size()
        st.plotly_chart(px.bar(counts, x="regime", y="size", title="Rows by regime"), use_container_width=True)
        choices = [c for c in ["temperature_C", "pressure_bar", "vibration_mm_s_rms", "health_deterioration"] if c in df.columns]
        if choices:
            signal = st.selectbox("Timeline signal", choices, key="timeline_signal")
            plot_df = df.iloc[::max(1, len(df) // 3000)]
            fig = px.line(plot_df, x="timestamp", y=signal, color="regime", title=f"{signal}: drift, alarms and adaptations")
            fig = add_event_markers(fig, "timestamp", detector_metrics, events)
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Ground-truth regimes are for offline evaluation only; detectors must not use them.")
    else:
        st.warning("No `regime` column found.")

with tabs[3]:
    st.subheader("KS, PSI and Page-Hinkley")
    st.markdown(
        "- **KS test:** compares recent and reference sensor distributions.\n"
        "- **PSI:** measures the magnitude of distribution changes.\n"
        "- **Page-Hinkley:** monitors baseline-GRU prediction errors.\n\n"
        "Dashed red lines indicate detector alarms; dotted teal lines indicate adaptation events "
        "when event timestamps are available."
    )
    if detector_metrics is None:
        st.info("Run the experiment to populate detector-window metrics.")
    else:
        st.dataframe(detector_metrics.tail(500), use_container_width=True, hide_index=True)
        alarm_cols = [c for c in ["KS_alarm", "PSI_alarm", "Page_Hinkley_alarm"] if c in detector_metrics.columns]
        if alarm_cols:
            counts = detector_metrics[alarm_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().rename_axis("Detector").reset_index(name="Alarm windows")
            st.plotly_chart(px.bar(counts, x="Detector", y="Alarm windows", title="Alarm windows by detector"), use_container_width=True)
        for column, title in [("max_PSI", "Maximum PSI"), ("max_KS_stat", "Maximum KS statistic")]:
            xcol = "window_start" if "window_start" in detector_metrics.columns else None
            if column in detector_metrics.columns and xcol:
                fig = px.line(detector_metrics, x=xcol, y=column, title=f"{title}: alarms and adaptations")
                fig = add_event_markers(fig, xcol, detector_metrics, events)
                st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    if metrics is None:
        st.warning("Run the experiment to create strategy_metrics.csv.")
    else:
        view = metrics.copy()
        if "Phase" in view.columns:
            phases = sorted(view["Phase"].dropna().astype(str).unique().tolist())
            if phases:
                phase = st.selectbox("Evaluation phase", phases, key="metric_phase")
                view = view[view["Phase"].astype(str) == phase]
        if "MAE" in view.columns:
            view = view.sort_values("MAE")
        st.dataframe(view, use_container_width=True, hide_index=True)
        available = [c for c in ["MAE", "RMSE", "R2"] if c in view.columns]
        if "Strategy" in view.columns and available:
            metric = st.selectbox("Plot metric", available)
            st.plotly_chart(px.bar(view, x="Strategy", y=metric, title=f"{metric} by strategy", text_auto=".3f"), use_container_width=True)

with tabs[5]:
    if predictions is None:
        st.warning("Predictions appear after running the experiment.")
    else:
        strategies = [c for c in predictions.columns if c not in ["row", "timestamp", "actual"]]
        if strategies:
            strategy = st.selectbox("Strategy prediction", strategies)
            if "timestamp" in predictions.columns:
                n = min(len(predictions), 6000)
                sample = predictions.iloc[::max(1, len(predictions) // max(1, n))]
                columns = [c for c in ["timestamp", "actual", strategy] if c in sample.columns]
                if len(columns) >= 2:
                    chart = sample[columns].melt(id_vars="timestamp", var_name="series", value_name="health_deterioration")
                    fig = px.line(chart, x="timestamp", y="health_deterioration", color="series", title="Actual vs predicted — alarms and adaptations")
                    fig = add_event_markers(fig, "timestamp", detector_metrics, events)
                    st.plotly_chart(fig, use_container_width=True)
        st.dataframe(predictions.tail(100), use_container_width=True, hide_index=True)

with tabs[6]:
    if events is None:
        st.info("No adaptation events recorded yet; no alarms may have fired.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)
        if "adapted_at" in events.columns and "n_samples" in events.columns:
            st.plotly_chart(px.scatter(events, x="adapted_at", y="n_samples", title="Adaptation time and samples used"), use_container_width=True)

with tabs[7]:
    st.subheader("Dataset preview")
    st.caption(f"Loaded {DATA.name} ({DATA.stat().st_size:,} bytes).")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(3)
    st.subheader("Missing values")
    st.dataframe(missing, use_container_width=True)
    st.subheader("Scientific cautions")
    st.markdown(
        "- Dataset is synthetic, not field-collected telemetry.\n"
        "- `health_deterioration` is a generated score, not a measured failure probability.\n"
        "- Ground-truth drift labels are for offline evaluation only.\n"
        "- KS/PSI inspect feature distributions; Page-Hinkley monitors prediction errors.\n"
        "- This project compares GRU adaptation strategies only."
    )
