import copy, time, random, argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from detectors import PageHinkley, ks_detector, psi_detector

# Avoid severe CPU oversubscription in Colab and small VMs.
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

FEATURES = ["temperature_C", "pressure_bar", "vibration_mm_s_rms", "load_pct",
            "humidity_pct", "motor_speed_rpm", "operating_hours"]
TARGET = "health_deterioration"
SEED = 42
SEQ_LEN = 12
TRAIN_END = 14_400
VAL_END = 16_000
STREAM_START = 16_000
CHUNK = 24
MIN_ADAPT = 150
MAX_ADAPT = 500
COOLDOWN = 48
BASE_EPOCHS = 35
ADAPT_EPOCHS = 10
BATCH_SIZE = 64
PH_DELTA = 0.15
PH_THRESHOLD = 25.0
PH_WARMUP = 40

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class GRURegressor(nn.Module):
    def __init__(self, n_features=len(FEATURES), hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        seq, _ = self.gru(x)
        return self.head(seq[:, -1]).squeeze(-1)

def make_windows(x, y, seq_len):
    # sample t uses only X[t-seq_len+1:t+1] and predicts y[t]
    Xw = np.stack([x[i-seq_len+1:i+1] for i in range(seq_len-1, len(x))]).astype("float32")
    yw = y[seq_len-1:].astype("float32")
    return Xw, yw

def fit_model(model, X, y, epochs, lr, device, train_params=None, replay=None):
    params = list(model.parameters()) if train_params is None else list(train_params)
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    Xt=torch.from_numpy(np.asarray(X, dtype=np.float32)); yt=torch.from_numpy(np.asarray(y, dtype=np.float32))
    if replay is not None:
        Xr=torch.from_numpy(np.asarray(replay[0], dtype=np.float32)); yr=torch.from_numpy(np.asarray(replay[1], dtype=np.float32))
    model.to(device); t0=time.perf_counter()
    for _ in range(epochs):
        model.train()
        order=torch.randperm(len(Xt))
        for start in range(0, len(order), BATCH_SIZE):
            ids=order[start:start+BATCH_SIZE]
            xb,yb=Xt[ids].to(device),yt[ids].to(device)
            if replay is not None and len(Xr):
                count=max(1, len(ids))
                rid=torch.randint(0,len(Xr),(count,))
                xb=torch.cat([xb,Xr[rid].to(device)],dim=0)
                yb=torch.cat([yb,yr[rid].to(device)],dim=0)
            opt.zero_grad(); loss_fn(model(xb),yb).backward(); opt.step()
    return model, time.perf_counter()-t0

def predict_scaled(model, X, device):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),2048):
            out.append(model(torch.from_numpy(X[i:i+2048]).to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def score(y, p):
    return {"MAE":float(mean_absolute_error(y,p)), "RMSE":float(mean_squared_error(y,p)**.5),
            "R2":float(r2_score(y,p)) if len(y)>1 else float("nan")}

def run_experiment(data_path="industrial_sensor_drift_dataset.csv", out_dir="results",
                   fast=True, seed=SEED):
    set_seed(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    df=pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    missing=set(FEATURES+[TARGET])-set(df.columns)
    if missing: raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(df) <= VAL_END+MIN_ADAPT: raise ValueError("Dataset is too short for the configured splits.")
    xraw=df[FEATURES].to_numpy(dtype=np.float32); yraw=df[TARGET].to_numpy(dtype=np.float32)
    xs=StandardScaler().fit(xraw[:TRAIN_END])
    x=xs.transform(xraw).astype(np.float32)
    y_mu=float(yraw[:TRAIN_END].mean()); y_sd=float(yraw[:TRAIN_END].std() or 1.0)
    ys=((yraw-y_mu)/y_sd).astype(np.float32)
    W, yw=make_windows(x, ys, SEQ_LEN)
    # W[i] maps to raw row i+SEQ_LEN-1
    offset=SEQ_LEN-1
    row_indices=np.arange(offset,len(df))
    base_end=TRAIN_END-offset; val_end=VAL_END-offset; stream_start=STREAM_START-offset
    epochs=3 if fast else BASE_EPOCHS
    adapt_epochs=2 if fast else ADAPT_EPOCHS
    base=GRURegressor()
    base,_=fit_model(base,W[:base_end],yw[:base_end],epochs,3e-3,device)
    val_pred=(predict_scaled(base,W[base_end:val_end],device)*y_sd)+y_mu
    ref_mae=max(float(np.mean(np.abs(val_pred-yraw[TRAIN_END:VAL_END])),),0.1)
    method_names=["Static / No Adaptation","Transfer Learning — Frozen GRU",
                  "Full Retraining","Continual + Replay 100","Continual + Replay 500"]
    models={name:copy.deepcopy(base).to(device) for name in method_names}
    preds={name:np.full(len(df),np.nan) for name in method_names}
    events=[]; detector=PageHinkley(); pending=None; cooldown_until=-1; detector_log=[]
    reference_raw=xraw[max(0, TRAIN_END-1200):TRAIN_END]
    replay_idx=np.arange(max(0,base_end-600),base_end)
    replay_x=W[replay_idx].copy(); replay_y=yw[replay_idx].copy()
    # Live stream is test-then-train: predictions first; current labels only enter detector/adaptation after prediction.
    for start in range(stream_start, len(W), CHUNK):
        end=min(start+CHUNK,len(W))
        raw_lo=row_indices[start]; raw_hi=row_indices[end-1]+1
        for name,model in models.items():
            preds[name][raw_lo:raw_hi]=(predict_scaled(model,W[start:end],device)*y_sd)+y_mu
        # Three detectors: KS and PSI monitor sensor-distribution drift; Page-Hinkley
        # monitors prediction-residual / relational drift. Truth annotations are never inputs.
        current_raw=xraw[max(STREAM_START, raw_lo-120):raw_hi]
        ks_alarm, ks_detail=ks_detector(reference_raw, current_raw)
        psi_alarm, psi_detail=psi_detector(reference_raw, current_raw)
        static_p=preds["Static / No Adaptation"][raw_lo:raw_hi]
        errs=np.abs(static_p-yraw[raw_lo:raw_hi])/ref_mae
        ph_alarm=False
        for j,t in enumerate(range(raw_lo,raw_hi)):
            if detector.update(errs[j],t): ph_alarm=True
        alarms=[]
        if ks_alarm: alarms.append("KS")
        if psi_alarm: alarms.append("PSI")
        if ph_alarm: alarms.append("Page-Hinkley")
        detector_log.append({"window_start":raw_lo,"window_end":raw_hi-1,"KS_alarm":int(ks_alarm),
                             "PSI_alarm":int(psi_alarm),"Page_Hinkley_alarm":int(ph_alarm),
                             "alarms":"|".join(alarms),
                             "max_KS_stat":max([d["statistic"] for d in ks_detail],default=0.0),
                             "min_KS_p":min([d["p_value"] for d in ks_detail],default=1.0),
                             "max_PSI":max([d["psi"] for d in psi_detail],default=0.0)})
        if pending is None and raw_lo>=cooldown_until and alarms:
            pending={"detected_at":raw_lo,"change_est":raw_lo,"detectors":"|".join(alarms)}
        if pending is not None and raw_hi-pending["change_est"]>=MIN_ADAPT:
            lo=max(int(pending["change_est"]),raw_hi-MAX_ADAPT)
            # Keep only windows whose target is already observed at this point.
            wi0=max(0,lo-offset); wi1=max(wi0+1,raw_hi-offset)
            Xnew,ynew=W[wi0:wi1],yw[wi0:wi1]
            n_new=len(Xnew); train_secs={}
            for name in method_names:
                model=models[name]
                if name=="Static / No Adaptation": continue
                if name=="Transfer Learning — Frozen GRU":
                    model=copy.deepcopy(model)
                    for p in model.gru.parameters(): p.requires_grad=False
                    model,secs=fit_model(model,Xnew,ynew,adapt_epochs,1e-3,device,train_params=model.head.parameters())
                    for p in model.gru.parameters(): p.requires_grad=True
                elif name=="Full Retraining":
                    # All labels observed so far, not future rows.
                    hist_end=raw_hi-offset
                    model=GRURegressor()
                    model,secs=fit_model(model,W[:hist_end],yw[:hist_end],adapt_epochs,2e-3,device)
                else:
                    size=100 if name.endswith("100") else 500
                    if len(replay_x)>size:
                        rng=np.random.default_rng(seed+raw_hi+size)
                        ids=rng.choice(len(replay_x),size=size,replace=False)
                        rx,ry=replay_x[ids],replay_y[ids]
                    else: rx,ry=replay_x,replay_y
                    model=copy.deepcopy(model)
                    model,secs=fit_model(model,Xnew,ynew,adapt_epochs,1e-3,device,replay=(rx,ry))
                models[name]=model
                train_secs[name]=secs
            # Update bounded replay pool only with observed data.
            replay_x=np.concatenate([replay_x,Xnew])[-2000:]
            replay_y=np.concatenate([replay_y,ynew])[-2000:]
            events.append({**pending,"adapted_at":raw_hi,"n_samples":n_new,
                           "training_seconds":sum(train_secs.values()),"per_method_seconds":str(train_secs)})
            pending=None; detector.reset(); cooldown_until=raw_hi+COOLDOWN
    # Scores by phase; ground-truth columns are used here only, offline, for evaluation.
    rows=[]
    phases={"normal_stream":(16_000,24_000),"data_drift":(24_000,30_000),
            "concept_drift":(30_000,42_000),"combined_drift":(42_000,len(df))}
    for name in method_names:
        p=preds[name]
        valid=np.isfinite(p)
        overall=score(yraw[valid],p[valid])
        rows.append({"Strategy":name,"Phase":"overall",**overall})
        for phase,(a,b) in phases.items():
            mask=(np.arange(len(df))>=a)&(np.arange(len(df))<b)&valid
            if mask.sum()>1: rows.append({"Strategy":name,"Phase":phase,**score(yraw[mask],p[mask])})
        # Old-regime retention: final model's performance on baseline stream rows is scored using predictions made then;
        # do not retrospectively apply the final model to historical points.
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"strategy_metrics.csv",index=False)
    pd.DataFrame(events).drop(columns=["per_method_seconds"],errors="ignore").to_csv(out/"adaptation_events.csv",index=False)
    pd.DataFrame(detector_log).to_csv(out/"detector_window_metrics.csv",index=False)
    pred_df=pd.DataFrame({"row":np.arange(len(df)),"timestamp":df.timestamp,"actual":yraw})
    for name,p in preds.items(): pred_df[name]=p
    pred_df.to_csv(out/"predictions.csv",index=False)
    df[["timestamp","regime","data_drift","concept_drift","drift_type","true_drift_point"]].to_csv(out/"ground_truth_offline_only.csv",index=False)
    print(f"Device: {device}; rows: {len(df):,}; features: {len(FEATURES)}")
    print(f"Adaptation events: {len(events)}; detector windows: {len(detector_log)}")
    print(pd.DataFrame(rows).query("Phase == 'overall'").to_string(index=False))
    print(f"Results written to: {out.resolve()}")
    return pd.DataFrame(rows), pd.DataFrame(events)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="industrial_sensor_drift_dataset.csv")
    ap.add_argument("--out",default="results")
    ap.add_argument("--full",action="store_true",help="Use longer training settings; default is fast smoke-test mode.")
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args()
    run_experiment(args.data,args.out,fast=not args.full,seed=args.seed)
