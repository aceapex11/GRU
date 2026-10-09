
import copy
import time
import random
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from detectors import PageHinkley, ks_detector, psi_detector

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

FEATURES = [
    "temperature_C",
    "pressure_bar",
    "vibration_mm_s_rms",
    "load_pct",
    "humidity_pct",
    "motor_speed_rpm",
    "operating_hours",
]
TARGET = "health_deterioration"

SEED = 42
SEQ_LEN = 12
TRAIN_END = 14400
VAL_END = 16000
STREAM_START = 16000
CHUNK = 24
MIN_ADAPT = 150
MAX_ADAPT = 500
COOLDOWN = 48
BASE_EPOCHS = 35
ADAPT_EPOCHS = 10
BATCH_SIZE = 64


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GRURegressor(nn.Module):
    def __init__(self, n_features=len(FEATURES), hidden=32):
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def make_windows(x, y, seq_len):
    if len(x) < seq_len:
        raise ValueError("Not enough rows to create GRU windows.")

    Xw = np.stack([
        x[i - seq_len + 1:i + 1]
        for i in range(seq_len - 1, len(x))
    ]).astype("float32")

    yw = y[seq_len - 1:].astype("float32")
    return Xw, yw


def fit_model(
    model,
    X,
    y,
    epochs,
    lr,
    device,
    train_params=None,
    replay=None,
):
    if len(X) == 0:
        return model, 0.0

    params = (
        list(model.parameters())
        if train_params is None
        else list(train_params)
    )

    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))

    if replay is not None:
        Xr = torch.from_numpy(
            np.asarray(replay[0], dtype=np.float32)
        )
        yr = torch.from_numpy(
            np.asarray(replay[1], dtype=np.float32)
        )
    else:
        Xr = yr = None

    model.to(device)
    start_time = time.perf_counter()

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xt))

        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            xb = Xt[ids].to(device)
            yb = yt[ids].to(device)

            if replay is not None and len(Xr):
                count = max(1, len(ids))
                rid = torch.randint(0, len(Xr), (count,))
                xb = torch.cat([xb, Xr[rid].to(device)], dim=0)
                yb = torch.cat([yb, yr[rid].to(device)], dim=0)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model, time.perf_counter() - start_time


def predict_scaled(model, X, device):
    model.eval()
    output = []

    with torch.no_grad():
        for start in range(0, len(X), 2048):
            batch = torch.from_numpy(
                np.asarray(X[start:start + 2048], dtype=np.float32)
            ).to(device)
            output.append(model(batch).cpu().numpy())

    return np.concatenate(output) if output else np.array([])


def score(y, prediction):
    return {
        "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(mean_squared_error(y, prediction) ** 0.5),
        "R2": (
            float(r2_score(y, prediction))
            if len(y) > 1
            else float("nan")
        ),
    }


def run_experiment(
    data_path="industrial_sensor_drift_dataset.csv",
    out_dir="results",
    fast=True,
    seed=SEED,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(data_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if len(df) <= VAL_END + MIN_ADAPT:
        raise ValueError(
            f"Dataset has {len(df)} rows but the configured splits "
            f"require more than {VAL_END + MIN_ADAPT}."
        )

    # Clean numeric inputs without using ground-truth drift labels.
    x_frame = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    x_frame = x_frame.interpolate(limit_direction="both")
    x_frame = x_frame.fillna(x_frame.median()).fillna(0.0)

    y_series = pd.to_numeric(
        df[TARGET], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    valid = y_series.notna()
    df = df.loc[valid].reset_index(drop=True)
    x_raw = x_frame.loc[valid].to_numpy(dtype=np.float32)
    y_raw = y_series.loc[valid].to_numpy(dtype=np.float32)

    if len(df) <= VAL_END + MIN_ADAPT:
        raise ValueError("Too few valid target rows after cleaning.")

    # Fit preprocessing only on the initial training period.
    scaler = StandardScaler()
    scaler.fit(x_raw[:TRAIN_END])
    x = scaler.transform(x_raw).astype(np.float32)

    y_mean = float(y_raw[:TRAIN_END].mean())
    y_std = float(y_raw[:TRAIN_END].std())
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    y_scaled = ((y_raw - y_mean) / y_std).astype(np.float32)

    W, yw = make_windows(x, y_scaled, SEQ_LEN)
    offset = SEQ_LEN - 1
    row_indices = np.arange(offset, len(df))

    base_end = TRAIN_END - offset
    val_end = VAL_END - offset
    stream_start = STREAM_START - offset

    epochs = 3 if fast else BASE_EPOCHS
    adapt_epochs = 2 if fast else ADAPT_EPOCHS

    print(f"Training baseline on {TRAIN_END:,} rows...")
    base = GRURegressor(n_features=len(FEATURES))
    base, _ = fit_model(
        base, W[:base_end], yw[:base_end],
        epochs, 3e-3, device,
    )

    val_prediction = (
        predict_scaled(base, W[base_end:val_end], device) * y_std
        + y_mean
    )
    val_actual = y_raw[TRAIN_END:VAL_END]
    residual_scale = float(
        np.mean(np.abs(val_prediction - val_actual))
    )
    residual_scale = max(residual_scale, 0.1)

    method_names = [
        "Static / No Adaptation",
        "Transfer Learning — Frozen GRU",
        "Full Retraining",
        "Continual + Replay 100",
        "Continual + Replay 500",
    ]

    models = {
        name: copy.deepcopy(base).to(device)
        for name in method_names
    }

    # Freeze the GRU encoder for the transfer-learning strategy.
    transfer = models["Transfer Learning — Frozen GRU"]
    for parameter in transfer.gru.parameters():
        parameter.requires_grad = False
    transfer.compile if False else None

    predictions = {
        name: np.full(len(df), np.nan)
        for name in method_names
    }

    events = []
    detector_log = []

    ph = PageHinkley(
        delta=0.15,
        threshold=25.0,
        warmup=40,
        cooldown=20,
    )

    reference_raw = x_raw[max(0, TRAIN_END - 1200):TRAIN_END]
    replay_indices = np.arange(max(0, base_end - 600), base_end)
    replay_x = W[replay_indices].copy()
    replay_y = yw[replay_indices].copy()

    pending = None
    cooldown_until = -1
    last_alarm_row = -10**9

    print("Monitoring the chronological stream...")

    for start in range(stream_start, len(W), CHUNK):
        end = min(start + CHUNK, len(W))
        raw_lo = int(row_indices[start])
        raw_hi = int(row_indices[end - 1]) + 1

        # Predict before using the current target for adaptation.
        for name, model in models.items():
            pred_scaled = predict_scaled(model, W[start:end], device)
            predictions[name][raw_lo:raw_hi] = (
                pred_scaled * y_std + y_mean
            )

        # Recent sensor window versus a fixed training reference.
        current_start = max(STREAM_START, raw_lo - 120)
        current_raw = x_raw[current_start:raw_hi]

        ks_alarm, ks_details = ks_detector(
            reference_raw, current_raw,
            p_threshold=0.001,
        )
        psi_alarm, psi_details = psi_detector(
            reference_raw, current_raw,
            threshold=0.25,
        )

        static_prediction = predictions[
            "Static / No Adaptation"
        ][raw_lo:raw_hi]

        residuals = (
            np.abs(static_prediction - y_raw[raw_lo:raw_hi])
            / residual_scale
        )

        ph_alarm = False
        for residual, row in zip(residuals, range(raw_lo, raw_hi)):
            if ph.update(residual, row):
                ph_alarm = True

        alarms = []
        if ks_alarm:
            alarms.append("KS")
        if psi_alarm:
            alarms.append("PSI")
        if ph_alarm:
            alarms.append("Page-Hinkley")

        timestamp = (
            df["timestamp"].iloc[raw_lo]
            if "timestamp" in df.columns
            else pd.NaT
        )

        detector_log.append({
            "row_index": raw_lo,
            "timestamp": timestamp,
            "window_start": raw_lo,
            "window_end": raw_hi - 1,
            "KS_alarm": int(ks_alarm),
            "PSI_alarm": int(psi_alarm),
            "Page_Hinkley_alarm": int(ph_alarm),
            "alarms": "|".join(alarms),
            "max_KS_stat": max(
                [d["statistic"] for d in ks_details],
                default=0.0,
            ),
            "min_KS_p": min(
                [d["p_value"] for d in ks_details],
                default=1.0,
            ),
            "max_PSI": max(
                [d["psi"] for d in psi_details],
                default=0.0,
            ),
        })

        if alarms:
            last_alarm_row = raw_lo
            if pending is None and raw_lo >= cooldown_until:
                pending = {
                    "detected_at": raw_lo,
                    "change_est": raw_lo,
                    "timestamp": timestamp,
                    "detectors": "|".join(alarms),
                }

        # Adapt only after enough new observations have arrived.
        if (
            pending is not None
            and raw_hi - pending["detected_at"] >= MIN_ADAPT
        ):
            lo = max(
                int(pending["detected_at"]),
                raw_hi - MAX_ADAPT,
            )

            # W index k corresponds to raw row k + offset.
            wi0 = max(0, lo - offset)
            wi1 = min(len(W), raw_hi - offset)

            if wi1 <= wi0:
                pending = None
                continue

            X_new = W[wi0:wi1]
            y_new = yw[wi0:wi1]
            train_seconds = {}

            for name in method_names:
                if name == "Static / No Adaptation":
                    continue

                model = copy.deepcopy(models[name])

                if name == "Transfer Learning — Frozen GRU":
                    for parameter in model.gru.parameters():
                        parameter.requires_grad = False

                    model, seconds = fit_model(
                        model,
                        X_new,
                        y_new,
                        adapt_epochs,
                        1e-3,
                        device,
                        train_params=model.head.parameters(),
                    )

                    for parameter in model.gru.parameters():
                        parameter.requires_grad = False

                elif name == "Full Retraining":
                    hist_end = max(1, min(len(W), raw_hi - offset))
                    model = GRURegressor(
                        n_features=len(FEATURES)
                    )
                    model, seconds = fit_model(
                        model,
                        W[:hist_end],
                        yw[:hist_end],
                        adapt_epochs,
                        2e-3,
                        device,
                    )

                else:
                    size = 100 if name.endswith("100") else 500
                    take = min(size, len(replay_x))
                    rx = replay_x[-take:]
                    ry = replay_y[-take:]

                    model, seconds = fit_model(
                        model,
                        X_new,
                        y_new,
                        adapt_epochs,
                        1e-3,
                        device,
                        replay=(rx, ry),
                    )

                models[name] = model
                train_seconds[name] = seconds

            replay_x = np.concatenate(
                [replay_x, X_new], axis=0
            )[-2000:]
            replay_y = np.concatenate(
                [replay_y, y_new], axis=0
            )[-2000:]

            events.append({
                **pending,
                "row_index": int(pending["detected_at"]),
                "adapted_at": raw_hi,
                "adaptation_point": raw_hi,
                "n_samples": len(X_new),
                "training_seconds": sum(train_seconds.values()),
            })

            pending = None
            ph.reset()
            cooldown_until = raw_hi + COOLDOWN

    # Evaluate predictions offline. Truth labels are not detector inputs.
    rows = []
    phases = {
        "normal_stream": (16000, 24000),
        "data_drift": (24000, 30000),
        "concept_drift": (30000, 42000),
        "combined_drift": (42000, len(df)),
    }

    for name in method_names:
        p = predictions[name]
        valid_pred = np.isfinite(p)

        if valid_pred.sum() > 1:
            rows.append({
                "Strategy": name,
                "Phase": "overall",
                **score(y_raw[valid_pred], p[valid_pred]),
            })

        for phase, (a, b) in phases.items():
            mask = (
                (np.arange(len(df)) >= a)
                & (np.arange(len(df)) < b)
                & valid_pred
            )

            if mask.sum() > 1:
                rows.append({
                    "Strategy": name,
                    "Phase": phase,
                    **score(y_raw[mask], p[mask]),
                })

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        out / "strategy_metrics.csv", index=False
    )
    pd.DataFrame(events).to_csv(
        out / "adaptation_events.csv", index=False
    )
    pd.DataFrame(detector_log).to_csv(
        out / "detector_window_metrics.csv", index=False
    )

    prediction_frame = pd.DataFrame({
        "row": np.arange(len(df)),
        "row_index": np.arange(len(df)),
        "timestamp": (
            df["timestamp"]
            if "timestamp" in df.columns
            else pd.NaT
        ),
        "actual": y_raw,
    })

    for name, values in predictions.items():
        prediction_frame[name] = values

    prediction_frame.to_csv(
        out / "predictions.csv", index=False
    )

    truth_columns = [
        col for col in [
            "timestamp",
            "regime",
            "data_drift",
            "concept_drift",
            "drift_type",
            "true_drift_point",
        ]
        if col in df.columns
    ]

    if truth_columns:
        df[truth_columns].to_csv(
            out / "ground_truth_offline_only.csv",
            index=False,
        )

    print(f"Device: {device}; rows: {len(df):,}")
    print(f"Detector windows: {len(detector_log)}")
    print(f"Adaptation events: {len(events)}")
    print(f"Results written to: {out.resolve()}")
    print(pd.DataFrame(rows).to_string(index=False))

    return pd.DataFrame(rows), pd.DataFrame(events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="industrial_sensor_drift_dataset.csv",
    )
    parser.add_argument("--out", default="results")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    run_experiment(
        args.data,
        args.out,
        fast=not args.full,
        seed=args.seed,
    )
