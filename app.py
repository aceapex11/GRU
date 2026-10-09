from pathlib import Path
import subprocess
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
from pandas.errors import EmptyDataError, ParserError

# Resolve files relative to this app.py, not the current working directory.
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# Support the intended filename and common accidental duplicate names.
DATA_CANDIDATES = [
    ROOT / "industrial_sensor_drift_dataset.csv",
    ROOT / "industrial_sensor_drift_dataset (1).csv",
    ROOT / "industrial_sensor_drift_dataset (7).csv",
    ROOT / "data" / "industrial_sensor_drift_dataset.csv",
]

st.set_page_config(page_title="GRU Drift Intelligence", page_icon="📈", layout="wide")
st.title("GRU Drift Intelligence")
st.caption("Predict → Detect → Diagnose → Adapt → Evaluate | GRU deep-learning benchmark")
st.info(
    "The included sensor dataset is synthetic. Drift truth labels are used only "
    "for offline evaluation, never as live detector inputs."
)

def resolve_dataset():
    """Find a non-empty dataset file in the repository's expected locations."""
    for candidate in DATA_CANDIDATES:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    # Fallback: case-insensitive filename search at root and in data/.
    for folder in (ROOT, ROOT / "data"):
        if folder.is_dir():
            for candidate in folder.glob("*.csv"):
                if "industrial_sensor_drift_dataset" in candidate.name.lower():
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        return candidate
    return None

@st.cache_data(show_spinner=False)
def load_data(path, modified_time, file_size):
    # modified_time and file_size invalidate Streamlit's cache after replacement.
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("CSV has a header but contains zero data rows.")
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame

def read_result(name):
    path = RESULTS / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        result = pd.read_csv(path)
        return result if not result.empty else None
    except (EmptyDataError, ParserError, OSError):
        return None

DATA = resolve_dataset()

with st.sidebar:
    st.header("Experiment controls")
    if DATA:
        st.write(f"Dataset: {DATA.name}")
    else:
        st.write("Dataset: not found")
    full = st.checkbox(
        "Full training (slower)",
        value=False,
        help="Unchecked runs the faster configuration. Full training uses more CPU."
    )
    st.caption("The experiment runs on the Streamlit server. CPU throttling may affect runtime.")
    run = st.button("Run GRU experiment", type="primary", use_container_width=True)

    if run:
        if DATA is None:
            st.error("Cannot start: upload a non-empty industrial_sensor_drift_dataset.csv beside app.py.")
        else:
            experiment_script = ROOT / "experiment.py"
            if not experiment_script.is_file():
                st.error(f"Experiment script not found: {experiment_script}")
            else:
                RESULTS.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(experiment_script),
                    "--data", str(DATA),
                    "--out", str(RESULTS),
                ]
                if full:
                    cmd.append("--full")
                try:
                    with st.spinner("Running the GRU experiment…"):
                        proc = subprocess.run(
                            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200
                        )
                    if proc.returncode == 0:
                        st.success("Experiment finished. Results refreshed.")
                        if proc.stdout:
                            st.code(proc.stdout[-5000:])
                    else:
                        st.error("Experiment failed. Diagnostic output:")
                        st.code(((proc.stderr or "") + "\n" + (proc.stdout or ""))[-8000:])
                except subprocess.TimeoutExpired:
                    st.error("Experiment exceeded the 2-hour timeout. Try with Full training unchecked.")
                except Exception as exc:
                    st.error(f"Could not launch experiment: {type(exc).__name__}: {exc}")
                st.cache_data.clear()
                st.rerun()

if DATA is None:
    st.error(
        "Dataset not found or all matching files are empty. Expected a non-empty CSV named "
        "`industrial_sensor_drift_dataset.csv` in the same folder as `app.py` (repository root)."
    )
    st.write("Files visible in the app folder:")
    try:
        visible = sorted(p.name for p in ROOT.iterdir() if p.is_file())
        st.code("\n".join(visible) if visible else "(No files found)")
    except OSError:
        st.write("Unable to list repository files.")
    st.stop()

try:
    df = load_data(str(DATA), DATA.stat().st_mtime, DATA.stat().st_size)
except EmptyDataError:
    st.error(f"`{DATA.name}` is empty. Upload the actual 20,000-row CSV, not an empty renamed file.")
    st.stop()
except (ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(f"Could not read `{DATA.name}` as a valid CSV: {exc}")
    st.stop()
except OSError as exc:
    st.error(f"Could not open dataset `{DATA.name}`: {exc}")
    st.stop()

required_for_app = {"timestamp"}
if not required_for_app.issubset(df.columns):
    st.error(
        "The CSV was found, but it does not contain the required `timestamp` column. "
        f"Columns found: {', '.join(map(str, df.columns))}"
    )
    st.stop()

metrics = read_result("strategy_metrics.csv")
events = read_result("adaptation_events.csv")
preds = read_result("predictions.csv")
truth = read_result("ground_truth_offline_only.csv")
detector_metrics = read_result("detector_window_metrics.csv")

sensor_features = [
    c for c in [
        "temperature_C", "pressure_bar", "vibration_mm_s_rms", "load_pct",
        "humidity_pct", "motor_speed_rpm", "operating_hours"
    ] if c in df.columns
]
targets = [c for c in ["health_deterioration"] if c in df.columns]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Sensor inputs", str(len(sensor_features)))
k3.metric("Drift regimes", str(df["regime"].nunique()) if "regime" in df.columns else "—")
k4.metric("Adaptation events", f"{len(events):,}" if events is not None else "Run experiment")

tabs = st.tabs([
    "Overview", "Sensor Explorer", "Drift Timeline",
    "Detector Analysis (KS / PSI / Page-Hinkley)",
    "GRU Strategy Comparison", "Predictions", "Adaptation Events", "Data & Notes"
])

with tabs[0]:
    st.subheader("Sensor stream")
    choices = sensor_features + targets
    if choices:
        chosen = st.selectbox("Signal", choices, key="overview_signal")
        plot_df = df.iloc[::max(1, len(df) // 3000)]
        st.plotly_chart(px.line(plot_df, x="timestamp", y=chosen, title=f"{chosen} over time"), use_container_width=True)
    if metrics is not None:
        st.subheader("Strategy performance")
        view = metrics[metrics["Phase"].astype(str).str.lower().eq("overall")] if "Phase" in metrics.columns else metrics
        if "MAE" in view.columns:
            view = view.sort_values("MAE")
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.warning("No GRU results found yet. Click Run GRU experiment in the sidebar.")

with tabs[1]:
    choices = sensor_features + targets
    if not choices:
        st.warning("No recognized sensor columns were found in the CSV.")
    else:
        sensor = st.selectbox("Sensor / target", choices)
        max_points = st.slider("Maximum plotted points", 500, 10000, 3000, step=500)
        plot_df = df.iloc[::max(1, len(df) // max_points)]
        st.plotly_chart(px.line(plot_df, x="timestamp", y=sensor, title=sensor), use_container_width=True)
        st.dataframe(df[choices].describe().T, use_container_width=True)

with tabs[2]:
    st.subheader("Designed drift regimes (offline ground truth)")
    if "regime" in df.columns:
        counts = df.groupby("regime", as_index=False).size()
        st.plotly_chart(px.bar(counts, x="regime", y="size", title="Rows by designed regime"), use_container_width=True)
        st.caption("Regime labels are for offline evaluation/visualization only; the live detectors must not read them.")
        choices = [c for c in ["temperature_C", "pressure_bar", "vibration_mm_s_rms", "health_deterioration"] if c in df.columns]
        if choices:
            sig = st.selectbox("Timeline signal", choices, key="drift_signal")
            stride = max(1, len(df) // 3000)
            st.plotly_chart(
                px.line(df.iloc[::stride], x="timestamp", y=sig, color="regime", title=f"{sig} with offline regime annotations"),
                use_container_width=True
            )
    else:
        st.warning("No `regime` column is available for the offline drift timeline.")

with tabs[3]:
    st.subheader("Three complementary drift detectors")
    st.markdown(
        "- **KS test:** compares recent sensor distributions with a reference window.\n"
        "- **PSI:** measures the magnitude of feature-distribution changes.\n"
        "- **Page-Hinkley:** monitors normalized baseline-GRU prediction errors."
    )
    if detector_metrics is None:
        st.info("Run the GRU experiment to populate detector-window metrics.")
    else:
        st.dataframe(detector_metrics.tail(500), use_container_width=True, hide_index=True)
        cols = [c for c in ["KS_alarm", "PSI_alarm", "Page_Hinkley_alarm"] if c in detector_metrics.columns]
        if cols:
            rates = detector_metrics[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().rename_axis("Detector").reset_index(name="Alarm windows")
            st.plotly_chart(px.bar(rates, x="Detector", y="Alarm windows", title="Alarm windows by detector"), use_container_width=True)
        for col, title in [
            ("max_PSI", "Maximum PSI by stream window"),
            ("max_KS_stat", "Maximum KS statistic by stream window"),
        ]:
            if col in detector_metrics.columns and "window_start" in detector_metrics.columns:
                st.plotly_chart(px.line(detector_metrics, x="window_start", y=col, title=title), use_container_width=True)
    st.caption("KS and PSI target feature-distribution drift. Page-Hinkley targets changes in prediction-error behaviour. Thresholds need calibration.")

with tabs[4]:
    if metrics is None:
        st.warning("Run the experiment to create results/strategy_metrics.csv.")
    else:
        st.subheader("Metrics by strategy and phase")
        if "Phase" in metrics.columns:
            phases = sorted(metrics["Phase"].dropna().astype(str).unique().tolist())
            phase = st.selectbox("Evaluation phase", phases, key="metric_phase")
            view = metrics[metrics["Phase"].astype(str) == phase]
        else:
            view = metrics
        if "MAE" in view.columns:
            view = view.sort_values("MAE")
        st.dataframe(view, use_container_width=True, hide_index=True)
        available_metrics = [c for c in ["MAE", "RMSE", "R2"] if c in view.columns]
        if "Strategy" in view.columns and available_metrics:
            metric_name = st.selectbox("Plot metric", available_metrics)
            st.plotly_chart(px.bar(view, x="Strategy", y=metric_name, title=f"{metric_name} by strategy", text_auto=".3f"), use_container_width=True)

with tabs[5]:
    if preds is None:
        st.warning("Predictions are unavailable until the experiment has run.")
    else:
        available = [c for c in preds.columns if c not in ["row", "timestamp", "actual"]]
        if available:
            strategy = st.selectbox("Strategy prediction", available)
            if "timestamp" in preds.columns:
                n = min(len(preds), 6000)
                sample = preds.iloc[::max(1, len(preds) // n)]
                chart_cols = [c for c in ["timestamp", "actual", strategy] if c in sample.columns]
                chart = sample[chart_cols].melt(id_vars="timestamp", var_name="series", value_name="health_deterioration")
                st.plotly_chart(px.line(chart, x="timestamp", y="health_deterioration", color="series", title="Actual vs predicted deterioration"), use_container_width=True)
            st.dataframe(preds.tail(100), use_container_width=True, hide_index=True)
        else:
            st.dataframe(preds.tail(100), use_container_width=True, hide_index=True)

with tabs[6]:
    if events is None:
        st.info("No adaptation events recorded yet. An empty event table may also mean no alarms fired.")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)
        if "adapted_at" in events.columns and "n_samples" in events.columns:
            st.plotly_chart(px.scatter(events, x="adapted_at", y="n_samples", title="Adaptation time and new samples used"), use_container_width=True)

with tabs[7]:
    st.subheader("Dataset preview")
    st.caption(f"Loaded `{DATA.name}` ({DATA.stat().st_size:,} bytes).")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.subheader("Missing values")
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(3)
    st.dataframe(missing, use_container_width=True)
    st.subheader("Scientific cautions")
    st.markdown(
        "- Dataset is synthetic, not field-collected telemetry.\n"
        "- The target is a generated 0–100 deterioration score, not a measured failure probability.\n"
        "- Drift truth labels are for offline evaluation only.\n"
        "- KS and PSI inspect feature distributions; Page-Hinkley monitors baseline-GRU prediction errors.\n"
        "- This repository compares GRU adaptation strategies only."
    )
