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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from detectors import PageHinkley, ks_detector, psi_detector

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

FEATURES = [
    "temperature", "vibration", "pressure", "load",
    "operating_hours", "humidity", "rpm", "oil_viscosity",
]
TARGET = "health_deterioration"

SEED = 42
SEQ_LEN = 12
TRAIN_END = 5000
VAL_END = 5500
STREAM_START = 5500

CHUNK = 24
MIN_ADAPT = 72
MAX_ADAPT = 240
COOLDOWN = 120

BASE_EPOCHS = 35
ADAPT_EPOCHS = 10
BATCH_SIZE = 64

PH_DELTA = 0.20
PH_THRESHOLD = 35.0
PH_WARMUP = 60
KS_STAT_THRESHOLD = 0.15
PSI_THRESHOLD = 0.35
MIN_MOVED_FEATURES = 3
DATA_CONFIRM_WINDOWS = 2
RECOVERY_WINDOWS = 3


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
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
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
    Xw = np.stack([
        x[i - seq_len + 1:i + 1]
        for i in range(seq_len - 1, len(x))
    ]).astype(np.float32)

    return Xw, y[seq_len - 1:].astype(np.float32)


def fit_model(
    model, X, y, epochs, lr, device,
    train_params=None, replay=None,
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

    if replay is not None and len(replay[0]) > 0:
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

            if Xr is not None:
                replay_ids = torch.randint(
                    0, len(Xr), (len(ids),)
                )
                xb = torch.cat(
                    [xb, Xr[replay_ids].to(device)], dim=0
                )
                yb = torch.cat(
                    [yb, yr[replay_ids].to(device)], dim=0
                )

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model, time.perf_counter() - start_time


def predict_scaled(model, X, device):
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(X), 2048):
            xb = torch.from_numpy(
                np.asarray(X[start:start + 2048], dtype=np.float32)
            ).to(device)
            outputs.append(model(xb).cpu().numpy())

    return np.concatenate(outputs) if outputs else np.array([])


def score(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }


def run_experiment(
    data_path="data/drift_regression_dataset_20000.csv",
    out_dir="results",
    fast=True,
    seed=SEED,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(data_path)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a timestamp column.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(df) <= VAL_END + MIN_ADAPT:
        raise ValueError("Dataset is too short for these split settings.")

    Xframe = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    Xframe = Xframe.interpolate(limit_direction="both")
    Xframe = Xframe.fillna(Xframe.median()).fillna(0)

    Xraw = Xframe.to_numpy(dtype=np.float32)
    yraw = pd.to_numeric(
        df[TARGET], errors="coerce"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(yraw).all():
        raise ValueError("Target contains missing or infinite values.")

    # Fit preprocessing on the initial normal training segment only.
    scaler = StandardScaler().fit(Xraw[:TRAIN_END])
    X = scaler.transform(Xraw).astype(np.float32)

    y_mean = float(yraw[:TRAIN_END].mean())
    y_std = max(float(yraw[:TRAIN_END].std()), 1e-6)
    y_scaled = ((yraw - y_mean) / y_std).astype(np.float32)

    W, yw = make_windows(X, y_scaled, SEQ_LEN)
    offset = SEQ_LEN - 1
    row_indices = np.arange(offset, len(df))

    base_end = TRAIN_END - offset
    val_end = VAL_END - offset
    stream_start = STREAM_START - offset

    epochs = 3 if fast else BASE_EPOCHS
    adapt_epochs = 2 if fast else ADAPT_EPOCHS

    print(f"Training initial GRU on {TRAIN_END} normal rows.")
    base = GRURegressor()
    base, _ = fit_model(
        base, W[:base_end], yw[:base_end],
        epochs, 3e-3, device,
    )

    val_pred = (
        predict_scaled(base, W[base_end:val_end], device)
        * y_std + y_mean
    )
    val_actual = yraw[TRAIN_END:VAL_END]
    common = min(len(val_pred), len(val_actual))
    reference_mae = max(
        float(np.mean(np.abs(
            val_pred[:common] - val_actual[:common]
        ))),
        0.1,
    )

    names = [
        "Static / No Adaptation",
        "Transfer Learning — Frozen GRU",
        "Full Retraining",
        "Continual + Replay 100",
        "Continual + Replay 500",
    ]

    models = {
        name: copy.deepcopy(base).to(device)
        for name in names
    }

    for param in models["Transfer Learning — Frozen GRU"].gru.parameters():
        param.requires_grad = False

    predictions = {
        name: np.full(len(df), np.nan)
        for name in names
    }

    reference_raw = Xraw[max(0, TRAIN_END - 1000):TRAIN_END]
    monitored_indices = [
        i for i, feature in enumerate(FEATURES)
        if feature != "operating_hours"
    ]

    replay_idx = np.arange(max(0, base_end - 600), base_end)
    replay_X = W[replay_idx].copy()
    replay_y = yw[replay_idx].copy()

    ph = PageHinkley(
        delta=PH_DELTA,
        threshold=PH_THRESHOLD,
        warmup=PH_WARMUP,
    )

    detector_log = []
    adaptation_events = []

    data_candidate_streak = 0
    data_active = False
    data_quiet = 0
    relational_active = False
    relational_quiet = 0
    episode_latched = False
    pending = None
    cooldown_until = -1

    print("Starting streaming predictions and drift detection.")

    for start in range(stream_start, len(W), CHUNK):
        end = min(start + CHUNK, len(W))
        raw_lo = int
