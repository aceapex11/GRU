from pathlib import Path
import subprocess, sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "industrial_sensor_drift_dataset.csv"
RESULTS = ROOT / "results"

st.set_page_config(page_title="GRU Drift Intelligence", page_icon="📈", layout="wide")
st.title("GRU Drift Intelligence")
st.caption("Predict → Detect → Diagnose → Adapt → Evaluate | GRU deep-learning benchmark")
st.info("The included sensor dataset is synthetic. Drift truth labels are used only for offline evaluation, never as live detector inputs.")

@st.cache_data(show_spinner=False)
def load_data(path, mtime):
    return pd.read_csv(path, parse_dates=["timestamp"])

def read_result(name):
    p = RESULTS / name
    if p.exists():
        try: return pd.read_csv(p)
        except Exception: return None
    return None

with st.sidebar:
    st.header("Experiment controls")
    st.write("Dataset: 60,000 hourly records")
    full = st.checkbox("Full training (slower)", value=False, help="Unchecked runs the fast smoke-test configuration.")
    st.caption("Run from the repository root. Training time depends on CPU/GPU.")
    run = st.button("Run GRU experiment", type="primary", use_container_width=True)
    if run:
        with st.spinner("Running the GRU experiment…"):
            cmd = [sys.executable, str(ROOT / "experiment.py"), "--data", str(DATA), "--out", str(RESULTS)] + (["--full"] if full else [])
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=7200)
        if proc.returncode == 0:
            st.success("Experiment finished. Results refreshed.")
            if proc.stdout: st.code(proc.stdout[-5000:])
        else:
            st.error("Experiment failed. See diagnostic output below.")
            st.code((proc.stderr or proc.stdout)[-8000:])
        st.cache_data.clear()

if not DATA.exists():
    st.error(f"Dataset not found: {DATA}. Upload `industrial_sensor_drift_dataset.csv` beside `app.py`.")
    st.stop()
df = load_data(str(DATA), DATA.stat().st_mtime)
metrics = read_result("strategy_metrics.csv")
events = read_result("adaptation_events.csv")
preds = read_result("predictions.csv")
truth = read_result("ground_truth_offline_only.csv")

k1,k2,k3,k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Sensor inputs", "7")
k3.metric("Drift regimes", str(df['regime'].nunique()) if 'regime' in df else "—")
k4.metric("Adaptation events", f"{len(events):,}" if events is not None else "Run experiment")

detector_metrics = read_result("detector_window_metrics.csv")
tabs = st.tabs(["Overview", "Sensor Explorer", "Drift Timeline", "Detector Analysis (KS / PSI / Page-Hinkley)", "GRU Strategy Comparison", "Predictions", "Adaptation Events", "Data & Notes"])
with tabs[0]:
    st.subheader("Sensor stream")
    choices=[c for c in ["temperature_C","pressure_bar","vibration_mm_s_rms","load_pct","health_deterioration"] if c in df]
    chosen=st.selectbox("Signal", choices, key="overview_signal")
    st.plotly_chart(px.line(df.iloc[::max(1,len(df)//3000)], x="timestamp", y=chosen, title=f"{chosen} over time"), use_container_width=True)
    st.subheader("Strategy performance")
    if metrics is not None and not metrics.empty:
        view=metrics[metrics["Phase"].astype(str).eq("overall")].sort_values("MAE")
        st.dataframe(view, use_container_width=True, hide_index=True)
    else: st.warning("No GRU results found yet. Click **Run GRU experiment** in the sidebar.")
with tabs[1]:
    sensor_cols=["temperature_C","pressure_bar","vibration_mm_s_rms","load_pct","humidity_pct","motor_speed_rpm","operating_hours","health_deterioration"]
    sensor=st.selectbox("Sensor / target", sensor_cols)
    max_points=st.slider("Maximum plotted points", 500, 10000, 3000, step=500)
    plot_df=df.iloc[::max(1,len(df)//max_points)]
    st.plotly_chart(px.line(plot_df, x="timestamp", y=sensor, title=sensor), use_container_width=True)
    st.dataframe(df[sensor_cols].describe().T, use_container_width=True)
with tabs[2]:
    st.subheader("Designed drift regimes (offline ground truth)")
    if "regime" in df:
        counts=df.groupby("regime",as_index=False).size()
        st.plotly_chart(px.bar(counts,x="regime",y="size",title="Rows by designed regime"),use_container_width=True)
        st.caption("This plot uses known synthetic regime annotations for evaluation/visualization only. The live detector does not read these labels.")
    sig=st.selectbox("Timeline signal",["temperature_C","pressure_bar","vibration_mm_s_rms","health_deterioration"],key="drift_signal")
    st.plotly_chart(px.line(df.iloc[::20],x="timestamp",y=sig,color="regime",title=f"{sig} with offline regime annotations"),use_container_width=True)
with tabs[3]:
    st.subheader("Three complementary drift detectors")
    st.markdown("""- **KS test:** compares recent sensor distributions with the baseline reference.
- **PSI:** measures the magnitude of feature-distribution shift.
- **Page-Hinkley:** monitors normalized GRU baseline prediction errors for relational/concept changes.""")
    if detector_metrics is None or detector_metrics.empty:
        st.info("Run the GRU experiment to populate detector-window metrics.")
    else:
        st.dataframe(detector_metrics.tail(500), use_container_width=True, hide_index=True)
        cols=[c for c in ["KS_alarm","PSI_alarm","Page_Hinkley_alarm"] if c in detector_metrics]
        if cols:
            rates=detector_metrics[cols].sum().rename_axis("Detector").reset_index(name="Alarm windows")
            st.plotly_chart(px.bar(rates,x="Detector",y="Alarm windows",title="Alarm windows by detector"),use_container_width=True)
        if "max_PSI" in detector_metrics:
            st.plotly_chart(px.line(detector_metrics,x="window_start",y="max_PSI",title="Maximum PSI by stream window"),use_container_width=True)
        if "max_KS_stat" in detector_metrics:
            st.plotly_chart(px.line(detector_metrics,x="window_start",y="max_KS_stat",title="Maximum KS statistic by stream window"),use_container_width=True)
        if "Page_Hinkley_alarm" in detector_metrics:
            st.plotly_chart(px.scatter(detector_metrics,x="window_start",y="Page_Hinkley_alarm",title="Page-Hinkley alarms"),use_container_width=True)
    st.caption("KS and PSI target feature-distribution drift. Page-Hinkley targets changes in prediction-error behavior. Thresholds need calibration on validation data.")

with tabs[4]:
    if metrics is None or metrics.empty:
        st.warning("Run the experiment to create results/strategy_metrics.csv.")
    else:
        st.subheader("Metrics by strategy and phase")
        phase=st.selectbox("Evaluation phase", sorted(metrics["Phase"].dropna().unique().tolist()), key="metric_phase")
        m=metrics[metrics["Phase"]==phase].sort_values("MAE")
        st.dataframe(m,use_container_width=True,hide_index=True)
        metric_name=st.selectbox("Plot metric",["MAE","RMSE","R2"])
        st.plotly_chart(px.bar(m,x="Strategy",y=metric_name,title=f"{metric_name} — {phase}",text_auto=".3f"),use_container_width=True)
        st.caption("Lower MAE/RMSE is better; higher R² is better. Compare multiple seeds before drawing research conclusions.")
with tabs[5]:
    if preds is None or preds.empty:
        st.warning("Predictions are unavailable until the experiment has run.")
    else:
        available=[c for c in preds.columns if c not in ["row","timestamp","actual"]]
        strategy=st.selectbox("Strategy prediction",available)
        n=min(len(preds),6000)
        sample=preds.iloc[::max(1,len(preds)//n)]
        chart=sample[[c for c in ["timestamp","actual",strategy] if c in sample]].melt(id_vars="timestamp",var_name="series",value_name="health_deterioration")
        st.plotly_chart(px.line(chart,x="timestamp",y="health_deterioration",color="series",title="Actual vs predicted deterioration"),use_container_width=True)
        st.dataframe(preds.tail(100),use_container_width=True,hide_index=True)
with tabs[6]:
    if events is None or events.empty: st.info("No adaptation events recorded yet. Run the experiment; an empty event table can also mean no alarms fired.")
    else:
        st.dataframe(events,use_container_width=True,hide_index=True)
        if "adapted_at" in events:
            st.plotly_chart(px.scatter(events,x="adapted_at",y="n_samples",title="Adaptation time and new samples used"),use_container_width=True)
with tabs[7]:
    st.subheader("Dataset preview")
    st.dataframe(df.head(30),use_container_width=True,hide_index=True)
    st.subheader("Missing values")
    miss=df.isna().sum().rename("missing_count").to_frame()
    miss["missing_pct"]=(miss["missing_count"]/len(df)*100).round(3)
    st.dataframe(miss,use_container_width=True)
    st.subheader("Scientific cautions")
    st.markdown("""- Dataset is synthetic, not field-collected telemetry.
- The target is a generated 0–100 deterioration score, not a measured failure probability.
- `regime`, `data_drift`, `concept_drift`, `drift_type`, and `true_drift_point` are offline evaluation labels only.
- KS and PSI inspect recent feature windows; Page-Hinkley inspects normalized baseline-GRU prediction errors. Any detector alarm can trigger a shared adaptation event.
- This repository is GRU-only; no classic-ML models or legacy ML result files are included.""")
