from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

DATA_CANDIDATES = [
    ROOT / "data" / "drift_regression_dataset_20000.csv",
    ROOT / "drift_regression_dataset_20000.csv",
    ROOT / "industrial_sensor_drift_dataset.csv",
]

FEATURES = [
    "temperature", "vibration", "pressure", "load",
    "operating_hours", "humidity", "rpm", "oil_viscosity",
]
TARGET = "health_deterioration"

st.set_page_config(
    page_title="GRU Drift Intelligence",
    page_icon="📈",
    layout="wide",
)
st.title("GRU Drift Intelligence")
st.caption(
    "Predict → Detect → Diagnose → Adapt → Evaluate | PyTorch GRU"
)
st.info(
    "Synthetic research dataset. Ground-truth drift labels are used "
    "for offline evaluation, not as detector inputs."
)


def resolve_dataset():
    for path in DATA_CANDIDATES:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def read_result(filename):
    path = RESULTS / filename
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
        return frame if not frame.empty else None
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


DATA = resolve_dataset()

with st.sidebar:
    st.header("Experiment controls")
    st.write(f"Dataset: {DATA.name}" if DATA else "Dataset not found")
    full_training = st.checkbox("Full training (slower)", value=False)
    run = st.button(
        "Run GRU experiment",
        type="primary",
        use_container_width=True,
    )

    if run:
        if DATA is None:
            st.error(
                "Place drift_regression_dataset_20000.csv in the "
                "project's data folder."
            )
        elif not (ROOT / "experiment.py").exists():
            st.error("experiment.py was not found beside app.py.")
        else:
            RESULTS.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ROOT / "experiment.py"),
                "--data", str(DATA),
                "--out", str(RESULTS),
            ]
            if full_training:
                command.append("--full")

            with st.spinner("Running GRU experiment..."):
                try:
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
                            (result.stderr + "\n" + result.stdout)[-8000:]
                        )
                except subprocess.TimeoutExpired:
                    st.error("Experiment timed out.")

            st.rerun()

if DATA is None:
    st.error(
        "Dataset not found. Place "
        "`drift_regression_dataset_20000.csv` inside the data folder."
    )
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path, modified, size):
    frame = pd.read_csv(path)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="coerce"
        )
    return frame


try:
    df = load_data(
        str(DATA), DATA.stat().st_mtime, DATA.stat().st_size
    )
except Exception as exc:
    st.error(f"Unable to read dataset: {exc}")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Dataset is missing these columns: {missing}")
    st.stop()

if df.empty:
    st.error("Dataset contains no rows.")
    st.stop()

if "timestamp" in df.columns and df["timestamp"].notna().any():
    df = df.sort_values("timestamp").reset_index(drop=True)
    X_COL = "timestamp"
else:
    df["observation_index"] = np.arange(len(df))
    X_COL = "observation_index"

metrics = read_result("strategy_metrics.csv")
events = read_result("adaptation_events.csv")
predictions = read_result("predictions.csv")
detectors = read_result("detector_window_metrics.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Sensor inputs", str(len(FEATURES)))
c3.metric(
    "Drift regimes",
    str(df["regime"].nunique()) if "regime" in df else "—",
)
c4.metric(
    "Adaptation events",
    str(len(events)) if events is not None else "Not run",
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

with tabs[0]:
    st.subheader("Dataset overview")
    st.write(f"Dataset: `{DATA.name}`")
    st.write(f"Initial training rows configured in experiment: `5000`")

    signal = st.selectbox(
        "Select signal",
        FEATURES + [TARGET],
        key="overview_signal",
    )
    fig = px.line(
        df.iloc[::max(1, len(df) // 3000)],
        x=X_COL,
        y=signal,
        title=f"{signal} over time",
    )
    st.plotly_chart(fig, use_container_width=True)

    if metrics is not None:
        overall = metrics
        if "Phase" in overall:
            overall = overall[
                overall["Phase"].astype(str).str.lower() == "overall"
            ]
        st.subheader("Overall model comparison")
        st.dataframe(
            overall.sort_values("MAE")
            if "MAE" in overall else overall,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run the experiment to view model metrics.")


with tabs[1]:
    st.subheader("Sensor Explorer")
    sensor = st.selectbox(
        "Sensor or target",
        FEATURES + [TARGET],
        key="sensor_explorer",
    )
    st.plotly_chart(
        px.line(
            df.iloc[::max(1, len(df) // 5000)],
            x=X_COL,
            y=sensor,
            title=sensor,
        ),
        use_container_width=True,
    )
    st.dataframe(
        df[FEATURES + [TARGET]].describe().T,
        use_container_width=True,
    )


with tabs[2]:
    st.subheader("Ground-truth drift timeline")

    if "regime" in df:
        regime_counts = df["regime"].value_counts().rename_axis(
            "Regime"
        ).reset_index(name="Rows")

        st.plotly_chart(
            px.bar(
                regime_counts,
                x="Regime",
                y="Rows",
                title="Rows by regime",
            ),
            use_container_width=True,
        )

        signal = st.selectbox(
            "Timeline signal",
            FEATURES + [TARGET],
            key="timeline_signal",
        )
        fig = px.line(
            df.iloc[::max(1, len(df) // 4000)],
            x=X_COL,
            y=signal,
            color="regime",
            title=f"{signal} by labelled regime",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Ground-truth labels are for offline evaluation only. "
            "They must not be used by the live detector."
        )
    else:
        st.warning("No regime column was found.")


with tabs[3]:
    st.subheader("KS, PSI and Page-Hinkley")

    if detectors is None:
        st.info("Run the experiment to generate detector metrics.")
    else:
        st.dataframe(
            detectors,
            use_container_width=True,
            hide_index=True,
        )

        detector_columns = [
            c for c in [
                "KS_alarm", "PSI_alarm", "Page_Hinkley_alarm", "alarm"
            ] if c in detectors
        ]

        if detector_columns:
            counts = (
                detectors[detector_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum()
                .rename_axis("Detector")
                .reset_index(name="Positive windows")
            )

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Detector",
                    y="Positive windows",
                    title="Positive detector windows",
                ),
                use_container_width=True,
            )

        for column in ["max_KS_stat", "max_PSI", "mean_residual"]:
            if column in detectors:
                plot_x = (
                    "window_start"
                    if "window_start" in detectors
                    else detectors.columns[0]
                )
                st.plotly_chart(
                    px.line(
                        detectors,
                        x=plot_x,
                        y=column,
                        title=column,
                    ),
                    use_container_width=True,
                )


with tabs[4]:
    st.subheader("GRU strategy comparison")

    if metrics is None:
        st.info("Run the experiment to generate strategy_metrics.csv.")
    else:
        view = metrics.copy()

        if "Phase" in view:
            phases = view["Phase"].dropna().unique().tolist()
            selected_phase = st.selectbox(
                "Evaluation phase",
                phases,
                key="comparison_phase",
            )
            view = view[view["Phase"] == selected_phase]

        if "MAE" in view:
            view = view.sort_values("MAE")

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
        )

        available_metrics = [
            c for c in ["MAE", "RMSE", "R2"] if c in view
        ]

        if "Strategy" in view and available_metrics:
            metric = st.selectbox(
                "Metric to plot",
                available_metrics,
                key="comparison_metric",
            )
            st.plotly_chart(
                px.bar(
                    view,
                    x="Strategy",
                    y=metric,
                    title=f"{metric} by strategy",
                ),
                use_container_width=True,
            )


with tabs[5]:
    st.subheader("Actual vs predicted")

    if predictions is None:
        st.info("Run the experiment to generate predictions.csv.")
    else:
        pred = predictions.copy()

        if "timestamp" in pred:
            pred["timestamp"] = pd.to_datetime(
                pred["timestamp"], errors="coerce"
            )

        actual_col = next(
            (
                c for c in ["actual", "y_true", "actual_target"]
                if c in pred.columns
            ),
            None,
        )

        metadata = {
            "timestamp", "row", "row_index",
            "actual", "y_true", "actual_target",
        }

        model_cols = [
            c for c in pred.columns
            if c not in metadata
            and pd.api.types.is_numeric_dtype(pred[c])
        ]

        if actual_col is None:
            st.warning("Actual target column was not found.")
        else:
            xcol = (
                "timestamp"
                if "timestamp" in pred and pred["timestamp"].notna().any()
                else "row_index"
                if "row_index" in pred
                else "row"
            )

            if xcol not in pred:
                pred["row"] = np.arange(len(pred))
                xcol = "row"

            sample = pred.iloc[
                ::max(1, len(pred) // 6000)
            ].copy()

            fig = px.line(
                sample,
                x=xcol,
                y=actual_col,
                title="Actual and predicted deterioration",
                labels={xcol: "Time / observation", actual_col: "Actual"},
            )

            for column in model_cols:
                fig.add_scatter(
                    x=sample[xcol],
                    y=sample[column],
                    mode="lines",
                    name=column,
                )

            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            pred.tail(100),
            use_container_width=True,
            hide_index=True,
        )


with tabs[6]:
    st.subheader("Adaptation events")

    if events is None:
        st.info("No adaptation events recorded yet.")
    else:
        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True,
        )


with tabs[7]:
    st.subheader("Dataset preview")
    st.dataframe(
        df.head(50),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Missing values")
    missing_values = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(3),
    })
    st.dataframe(
        missing_values,
        use_container_width=True,
    )

    st.subheader("Interpretation notes")
    st.markdown("""
- The dataset is synthetic, not field-collected telemetry.
- `health_deterioration` is a generated score, not a measured failure probability.
- `data_drift` and `concept_drift` are ground-truth labels for offline evaluation.
- KS and PSI monitor sensor distribution changes.
- Page-Hinkley monitors prediction residuals.
- One positive detector window is not necessarily a separate drift episode.
""")
