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
    if frame.columns.duplicated().any():
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    if "timestamp" in frame.columns:
        raw = frame["timestamp"]
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_ratio = numeric.notna().mean() if len(raw) else 0.0

        # Numeric timestamps may be Unix epochs, or merely row numbers.
        # Do not let row numbers silently become dates in January 1970.
        if numeric_ratio > 0.95:
            values = numeric.dropna()
            median_abs = float(values.abs().median()) if len(values) else 0.0
            if median_abs >= 1e17:
                parsed = pd.to_datetime(numeric, unit="ns", errors="coerce")
            elif median_abs >= 1e14:
                parsed = pd.to_datetime(numeric, unit="us", errors="coerce")
            elif median_abs >= 1e11:
                parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
            elif median_abs >= 1e8:
                parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
            else:
                # These are observation IDs, not trustworthy calendar dates.
                # Keep them numeric and use an observation-index axis below.
                parsed = pd.Series(pd.NaT, index=frame.index)
                frame["timestamp_is_index"] = True
            frame["timestamp"] = parsed
        else:
            frame["timestamp"] = pd.to_datetime(raw, errors="coerce")
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


def _event_x_values(event_frame, x_column, dataset_frame):
    """Safely align event timestamps/indices with the plotted dataset axis."""
    if (
        event_frame is None or event_frame.empty
        or dataset_frame is None or dataset_frame.empty
        or x_column not in dataset_frame.columns
    ):
        return []

    candidates = [
        x_column, "timestamp", "window_start", "timestamp_end",
        "adapted_at", "adaptation_timestamp", "adaptation_point",
        "observation_index", "observation", "row_index",
        "sample_index", "row", "index", "window",
    ]
    event_col = next((c for c in candidates if c in event_frame.columns), None)
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
        nums = numeric.to_numpy(dtype=float)
        if (
            len(nums) and np.isfinite(nums).all()
            and (nums >= 0).all()
            and (nums < len(dataset_frame)).all()
            and np.equal(nums, np.floor(nums)).all()
        ):
            return dataset_frame.iloc[nums.astype(int)][x_column].dropna().tolist()
        return numeric.dropna().tolist()

    if x_column == "timestamp":
        return pd.to_datetime(values, errors="coerce").dropna().tolist()
    return []


def add_event_markers(
    fig, x_column, dataset_frame, detector_frame=None,
    adaptation_frame=None, max_markers=24
):
    """Overlay readable, deduplicated detector alarms and adaptation events."""
    if dataset_frame is None or x_column not in dataset_frame.columns:
        return fig

    def draw(values, color, dash, label):
        cleaned, seen = [], set()
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
            picks = np.linspace(0, total - 1, max_markers, dtype=int)
            cleaned = [cleaned[i] for i in picks]

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"{label} ({total})",
            line=dict(color=color, dash=dash, width=2),
            hoverinfo="skip",
        ))
        for value in cleaned:
            try:
                fig.add_vline(
                    x=value, line_color=color, line_dash=dash,
                    line_width=1.3, opacity=0.78
                )
            except (TypeError, ValueError, OverflowError):
                continue

    if detector_frame is not None and not detector_frame.empty:
        alarm_cols = [
            c for c in detector_frame.columns
            if any(token in str(c).lower() for token in
                   ("alarm", "drift_detected", "change_detected"))
        ]
        if alarm_cols:
            flags = detector_frame[alarm_cols].apply(
                lambda col: (
                    col.astype(str).str.strip().str.lower().isin(
                        ["1", "true", "yes", "alarm", "drift"]
                    )
                    | pd.to_numeric(col, errors="coerce").fillna(0).ne(0)
                )
            )
            alarm_rows = detector_frame.loc[flags.any(axis=1)]
            draw(
                _event_x_values(alarm_rows, x_column, dataset_frame),
                "#D97706", "dash", "New drift alarm"
            )

    if adaptation_frame is not None and not adaptation_frame.empty:
        draw(
            _event_x_values(adaptation_frame, x_column, dataset_frame),
            "#0F766E", "dot", "Adaptation"
        )

    return fig


def align_prediction_timestamps(prediction_frame, dataset_frame):
    """Align predictions to real timestamps or to row indices when no real dates exist."""
    pred = prediction_frame.copy()
    row_candidates = ["row", "row_index", "observation", "index", "sample_index"]
    row_col = next((c for c in row_candidates if c in pred.columns), None)

    if dataset_frame["timestamp"].notna().any():
        dataset_x = pd.to_datetime(dataset_frame["timestamp"], errors="coerce")
        if "timestamp" not in pred.columns:
            if row_col:
                row_ids = pd.to_numeric(pred[row_col], errors="coerce")
                valid = row_ids.notna() & row_ids.between(0, len(dataset_frame) - 1)
                pred["timestamp"] = pd.NaT
                pred.loc[valid, "timestamp"] = dataset_x.iloc[row_ids[valid].astype(int).to_numpy()].to_numpy()
            elif len(pred) == len(dataset_frame):
                pred["timestamp"] = dataset_x.to_numpy()
            else:
                pred["timestamp"] = pd.NaT
        else:
            numeric = pd.to_numeric(pred["timestamp"], errors="coerce")
            parsed = pd.to_datetime(pred["timestamp"], errors="coerce")
            looks_like_index = numeric.notna().mean() > 0.9 and numeric.dropna().between(0, len(dataset_frame)-1).mean() > 0.9
            if parsed.isna().mean() > 0.5 or looks_like_index:
                ids = pd.to_numeric(pred[row_col], errors="coerce") if row_col else numeric
                valid = ids.notna() & ids.between(0, len(dataset_frame)-1)
                pred["timestamp"] = pd.NaT
                pred.loc[valid, "timestamp"] = dataset_x.iloc[ids[valid].astype(int).to_numpy()].to_numpy()
            else:
                pred["timestamp"] = parsed
        pred["timestamp"] = pd.to_datetime(pred["timestamp"], errors="coerce")
        pred = pred.dropna(subset=["timestamp"]).sort_values("timestamp")
        return pred

    # Dataset timestamp column was only a row ID. Use observation_index, not 1970.
    if "observation_index" not in dataset_frame.columns:
        dataset_frame = dataset_frame.copy()
        dataset_frame["observation_index"] = np.arange(len(dataset_frame))

    if row_col:
        ids = pd.to_numeric(pred[row_col], errors="coerce")
        pred["observation_index"] = ids
    elif len(pred) == len(dataset_frame):
        pred["observation_index"] = np.arange(len(pred))
    elif "timestamp" in pred.columns:
        pred["observation_index"] = pd.to_numeric(pred["timestamp"], errors="coerce")
    else:
        pred["observation_index"] = np.nan

    return pred.dropna(subset=["observation_index"]).sort_values("observation_index")


def add_drift_regions(fig, dataset_frame):
    """Shade continuous true drift periods as background diagnostic context."""
    if "timestamp" not in dataset_frame.columns:
        return fig
    frame = dataset_frame[["timestamp"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    flags = []
    for col in ["data_drift", "concept_drift"]:
        if col in dataset_frame.columns:
            values = dataset_frame[col]
            if values.dtype == object:
                active = values.astype(str).str.lower().isin(["true", "1", "yes", "drift"])
            else:
                active = pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
            flags.append((col, active.to_numpy()))
    if not flags:
        return fig

    combined = np.zeros(len(frame), dtype=int)
    for name, active in flags:
        combined += active.astype(int) * (1 if name == "data_drift" else 2)
    # Draw contiguous segments; 1=data drift, 2=concept drift, 3=both.
    valid = frame["timestamp"].notna().to_numpy()
    values = np.where(valid, combined, 0)
    starts = np.where(np.r_[True, values[1:] != values[:-1]])[0]
    ends = np.r_[starts[1:], len(values)]
    colors = {1: "rgba(245, 166, 35, 0.10)", 2: "rgba(120, 90, 200, 0.10)", 3: "rgba(220, 80, 80, 0.13)"}
    for start, end in zip(starts, ends):
        state = int(values[start])
        if state == 0 or end <= start:
            continue
        x0 = frame["timestamp"].iloc[start]
        x1 = frame["timestamp"].iloc[end - 1]
        if pd.isna(x0) or pd.isna(x1):
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor=colors.get(state), line_width=0, layer="below")
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
    run = st.button("Run GRU experiment", type="primary", width="stretch")
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

# If timestamp values were only row IDs, use observation index on charts rather
# than displaying a misleading 1970 calendar axis.
if df["timestamp"].notna().sum() == 0:
    df["observation_index"] = np.arange(len(df))
    X_COL = "observation_index"
    X_LABEL = "Observation index (timestamp column contains row IDs)"
else:
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    X_COL = "timestamp"
    X_LABEL = "Time"

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
        fig = px.line(plot_df, x=X_COL, y=signal, title=f"{signal} over time")
        fig = add_event_markers(fig, X_COL, df, detector_metrics, events)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})
    if metrics is not None:
        view = metrics.copy()
        if "Phase" in view.columns:
            overall = view[view["Phase"].astype(str).str.lower() == "overall"]
            if not overall.empty:
                view = overall
        if "MAE" in view.columns:
            view = view.sort_values("MAE")
        st.dataframe(view, width="stretch", hide_index=True)
    else:
        st.warning("No GRU results yet. Run the experiment from the sidebar.")

with tabs[1]:
    choices = sensor_features + targets
    if choices:
        sensor = st.selectbox("Sensor / target", choices)
        max_points = st.slider("Maximum plotted points", 500, 10000, 3000, step=500)
        plot_df = df.iloc[::max(1, len(df) // max_points)]
        fig = px.line(plot_df, x=X_COL, y=sensor, title=f"{sensor} — alarms and adaptation")
        fig = add_event_markers(fig, X_COL, df, detector_metrics, events)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})
        st.dataframe(df[choices].describe().T, width="stretch")
    else:
        st.warning("No recognized sensor columns found.")

with tabs[2]:
    st.subheader("Designed drift regimes")
    if "regime" in df.columns:
        counts = df.groupby("regime", as_index=False).size()
        st.plotly_chart(px.bar(counts, x="regime", y="size", title="Rows by regime"), width="stretch")
        choices = [c for c in ["temperature_C", "pressure_bar", "vibration_mm_s_rms", "health_deterioration"] if c in df.columns]
        if choices:
            signal = st.selectbox("Timeline signal", choices, key="timeline_signal")
            plot_df = df.iloc[::max(1, len(df) // 3000)]
            fig = px.line(plot_df, x=X_COL, y=signal, color="regime", title=f"{signal}: drift, alarms and adaptations")
            fig = add_event_markers(fig, X_COL, df, detector_metrics, events)
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})
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
        st.dataframe(detector_metrics.tail(500), width="stretch", hide_index=True)
        alarm_cols = [c for c in ["KS_alarm", "PSI_alarm", "Page_Hinkley_alarm"] if c in detector_metrics.columns]
        if alarm_cols:
            counts = detector_metrics[alarm_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().rename_axis("Detector").reset_index(name="Alarm windows")
            st.plotly_chart(px.bar(counts, x="Detector", y="Alarm windows", title="Alarm windows by detector"), width="stretch")
        for column, title in [("max_PSI", "Maximum PSI"), ("max_KS_stat", "Maximum KS statistic")]:
            xcol = "window_start" if "window_start" in detector_metrics.columns else None
            if column in detector_metrics.columns and xcol:
                fig = px.line(detector_metrics, x=xcol, y=column, title=f"{title}: alarms and adaptations")
                marker_frame = df.copy()
                if xcol not in marker_frame.columns:
                    marker_frame[xcol] = np.arange(len(marker_frame))
                fig = add_event_markers(fig, xcol, marker_frame, detector_metrics, events)
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})

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
        st.dataframe(view, width="stretch", hide_index=True)
        available = [c for c in ["MAE", "RMSE", "R2"] if c in view.columns]
        if "Strategy" in view.columns and available:
            metric = st.selectbox("Plot metric", available)
            st.plotly_chart(px.bar(view, x="Strategy", y=metric, title=f"{metric} by strategy", text_auto=".3f"), width="stretch")

with tabs[5]:
    st.subheader("Actual vs predicted — all GRU strategies")
    if predictions is None:
        st.warning("Predictions appear after running the experiment.")
    else:
        aligned = align_prediction_timestamps(predictions, df)
        if aligned.empty:
            st.error("Could not align prediction rows to valid dataset timestamps. Check predictions.csv row/index columns.")
        else:
            actual_col = next((c for c in ["actual", "actual_target", "y_true", "target_actual"] if c in aligned.columns), None)
            metadata = {"timestamp", "row", "row_index", "observation", "index", "actual", "actual_target", "y_true", "target_actual"}
            model_cols = [c for c in aligned.columns if c not in metadata and pd.api.types.is_numeric_dtype(aligned[c])]
            if actual_col is None:
                st.warning("No actual-target column was found in predictions.csv. Showing available model prediction columns only.")
            if not model_cols and actual_col is None:
                st.info("No numeric prediction columns are available.")
            else:
                max_points = st.slider("Maximum points on chart", 500, 12000, 5000, step=500, key="prediction_points")
                stride = max(1, len(aligned) // max_points)
                sample = aligned.iloc[::stride].copy()
                plot_x_col = "timestamp" if "timestamp" in sample.columns and sample["timestamp"].notna().any() else "observation_index"
                fig = go.Figure()
                if actual_col:
                    fig.add_trace(go.Scatter(
                        x=sample[plot_x_col], y=sample[actual_col], mode="lines",
                        name="Actual target", line=dict(color="#596273", width=2.6),
                    ))
                palette = ["#8DA0CB", "#66C2A5", "#4C78A8", "#B279A2", "#F58518", "#54A24B", "#E45756"]
                for i, col in enumerate(model_cols):
                    fig.add_trace(go.Scatter(
                        x=sample[plot_x_col], y=sample[col], mode="lines", name=str(col),
                        line=dict(color=palette[i % len(palette)], width=1.5), opacity=0.9,
                    ))
                # Use a plotting frame whose x-axis matches prediction rows.
                # This prevents event row IDs from being interpreted as 1970 dates.
                marker_frame = df.copy()
                if plot_x_col == "observation_index" and "observation_index" not in marker_frame.columns:
                    marker_frame["observation_index"] = np.arange(len(marker_frame))

                if plot_x_col == "timestamp":
                    fig = add_drift_regions(fig, df)
                fig = add_event_markers(
                    fig, plot_x_col, marker_frame, detector_metrics, events
                )
                fig.update_layout(
                    title="Health deterioration — actual vs all models",
                    xaxis_title=("Time" if plot_x_col == "timestamp" else "Observation index"),
                    yaxis_title="Health deterioration",
                    height=620, autosize=True, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(l=20, r=20, t=90, b=20),
                )
                min_time = sample[plot_x_col].min()
                max_time = sample[plot_x_col].max()
                if pd.notna(min_time) and pd.notna(max_time):
                    axis_options = {"range": [min_time, max_time], "rangeslider": {"visible": True}}
                    if plot_x_col == "timestamp":
                        axis_options["rangeselector"] = {"buttons": [
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(step="all", label="All"),
                        ]}
                    fig.update_xaxes(**axis_options)
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})
                alarm_columns = (
                    [c for c in detector_metrics.columns
                     if any(k in str(c).lower() for k in ["alarm", "drift_detected"])]
                    if detector_metrics is not None else []
                )
                if not alarm_columns:
                    st.warning(
                        "No detector alarm columns were found in "
                        "results/detector_window_metrics.csv. Red alarm lines "
                        "cannot be verified until experiment.py writes actual alarm flags."
                    )
                elif not events is None and events.empty:
                    st.info("The adaptation event file is empty; no adaptation lines can be shown.")
                if events is None:
                    st.warning(
                        "results/adaptation_events.csv is missing or empty. "
                        "Teal lines will appear only after the adaptation pipeline records events."
                    )
                st.caption(
                    "Red dashed lines are derived from detector alarm flags in the result CSV; "
                    "teal dotted lines come from recorded adaptation events. Shaded backgrounds "
                    "show synthetic ground-truth drift labels for offline evaluation only."
                )
        st.dataframe(predictions.tail(100), width="stretch", hide_index=True)

with tabs[6]:
    if events is None:
        st.info("No adaptation events recorded yet; no alarms may have fired.")
    else:
        st.dataframe(events, width="stretch", hide_index=True)
        if "adapted_at" in events.columns and "n_samples" in events.columns:
            st.plotly_chart(px.scatter(events, x="adapted_at", y="n_samples", title="Adaptation time and samples used"), width="stretch")

with tabs[7]:
    st.subheader("Dataset preview")
    st.caption(f"Loaded {DATA.name} ({DATA.stat().st_size:,} bytes).")
    st.dataframe(df.head(30), width="stretch", hide_index=True)
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(3)
    st.subheader("Missing values")
    st.dataframe(missing, width="stretch")
    st.subheader("Scientific cautions")
    st.markdown(
        "- Dataset is synthetic, not field-collected telemetry.\n"
        "- `health_deterioration` is a generated score, not a measured failure probability.\n"
        "- Ground-truth drift labels are for offline evaluation only.\n"
        "- KS/PSI inspect feature distributions; Page-Hinkley monitors prediction errors.\n"
        "- This project compares GRU adaptation strategies only."
    )
