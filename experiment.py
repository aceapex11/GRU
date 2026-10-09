
import copy
import time
import random
import argparse
import os
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from detectors import PageHinkley, ks_detector, psi_detector


# ============================================================
# CONFIGURATION
# ============================================================

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

# Alarm management based on the working Drift project.
DATA_DRIFT_MIN_FEATURES = 3
DATA_DRIFT_CONFIRM_WINDOWS = 2
ALARM_HISTORY_WINDOWS = 3
ALARM_CONFIRM_COUNT = 2
RECOVERY_WINDOWS = 3
EVENT_COOLDOWN = 300

# Data drift thresholds.
KS_STAT_THRESHOLD = 0.15
PSI_THRESHOLD = 0.35

# Residual-based relational drift.
PH_DELTA = 0.20
PH_THRESHOLD = 35.0
PH_WARMUP = 60
RESID_CLIP = 4.0
RESID_RECOVERY_THRESHOLD = 1.25

BASE_EPOCHS = 35
ADAPT_EPOCHS = 10
BATCH_SIZE = 64


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GRU MODEL
# ============================================================

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
    """Predict target y[t] using sensor rows ending at t."""
    if len(x) < seq_len:
        raise ValueError("Not enough rows to create GRU windows.")

    Xw = np.stack([
        x[i - seq_len + 1:i + 1]
        for i in range(seq_len - 1, len(x))
    ]).astype(np.float32)

    yw = y[seq_len - 1:].astype(np.float32)

    return Xw, yw


# ============================================================
# MODEL TRAINING AND PREDICTION
# ============================================================

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

    parameters = (
        list(model.parameters())
        if train_params is None
        else list(train_params)
    )

    optimizer = torch.optim.Adam(parameters, lr=lr)
    loss_function = nn.MSELoss()

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
        Xr = None
        yr = None

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
                count = len(ids)
                replay_ids = torch.randint(
                    0, len(Xr), (count,)
                )

                xb = torch.cat(
                    [xb, Xr[replay_ids].to(device)],
                    dim=0,
                )
                yb = torch.cat(
                    [yb, yr[replay_ids].to(device)],
                    dim=0,
                )

            optimizer.zero_grad()
            loss = loss_function(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model, time.perf_counter() - start_time


def predict_scaled(model, X, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 2048):
            xb = torch.from_numpy(
                np.asarray(X[start:start + 2048], dtype=np.float32)
            ).to(device)

            predictions.append(
                model(xb).cpu().numpy()
            )

    if not predictions:
        return np.array([], dtype=float)

    return np.concatenate(predictions)


def score(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) > 1
            else float("nan")
        ),
    }


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    data_path="industrial_sensor_drift_dataset.csv",
    out_dir="results",
    fast=True,
    seed=SEED,
):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Load and validate
    # --------------------------------------------------------

    df = pd.read_csv(data_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) <= VAL_END + MIN_ADAPT:
        raise ValueError(
            f"Dataset has {len(df)} rows. "
            f"More than {VAL_END + MIN_ADAPT} rows are required."
        )

    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    X_frame = (
        df[FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    X_frame = X_frame.interpolate(limit_direction="both")
    X_frame = X_frame.fillna(X_frame.median()).fillna(0.0)

    y_series = (
        pd.to_numeric(df[TARGET], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    valid_target = y_series.notna()

    df = df.loc[valid_target].reset_index(drop=True)
    X_raw = X_frame.loc[valid_target].to_numpy(dtype=np.float32)
    y_raw = y_series.loc[valid_target].to_numpy(dtype=np.float32)

    if len(df) <= VAL_END + MIN_ADAPT:
        raise ValueError(
            "Too few valid target rows after data cleaning."
        )

    # --------------------------------------------------------
    # Fit scalers on initial training data only
    # --------------------------------------------------------

    scaler = StandardScaler()
    scaler.fit(X_raw[:TRAIN_END])

    X = scaler.transform(X_raw).astype(np.float32)

    y_mean = float(np.mean(y_raw[:TRAIN_END]))
    y_std = float(np.std(y_raw[:TRAIN_END]))

    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    y_scaled = ((y_raw - y_mean) / y_std).astype(np.float32)

    W, yw = make_windows(X, y_scaled, SEQ_LEN)

    # Window W[i] predicts the original row i + offset.
    offset = SEQ_LEN - 1
    row_indices = np.arange(offset, len(df))

    base_end = TRAIN_END - offset
    val_end = VAL_END - offset
    stream_start = STREAM_START - offset

    epochs = 3 if fast else BASE_EPOCHS
    adaptation_epochs = 2 if fast else ADAPT_EPOCHS

    # --------------------------------------------------------
    # Train baseline GRU
    # --------------------------------------------------------

    print(f"Device: {device}")
    print(f"Training baseline using {TRAIN_END:,} rows...")

    base_model = GRURegressor()
    base_model, _ = fit_model(
        base_model,
        W[:base_end],
        yw[:base_end],
        epochs,
        3e-3,
        device,
    )

    validation_prediction = (
        predict_scaled(
            base_model,
            W[base_end:val_end],
            device,
        ) * y_std + y_mean
    )

    validation_actual = y_raw[TRAIN_END:VAL_END]

    reference_mae = float(
        np.mean(
            np.abs(validation_prediction - validation_actual)
        )
    )
    reference_mae = max(reference_mae, 0.1)

    # --------------------------------------------------------
    # Five model strategies
    # --------------------------------------------------------

    strategy_names = [
        "Static / No Adaptation",
        "Transfer Learning — Frozen GRU",
        "Full Retraining",
        "Continual + Replay 100",
        "Continual + Replay 500",
    ]

    models = {
        name: copy.deepcopy(base_model).to(device)
        for name in strategy_names
    }

    for parameter in models[
        "Transfer Learning — Frozen GRU"
    ].gru.parameters():
        parameter.requires_grad = False

    predictions = {
        name: np.full(len(df), np.nan, dtype=float)
        for name in strategy_names
    }

    # --------------------------------------------------------
    # Reference distribution and replay pool
    # --------------------------------------------------------

    reference_raw = X_raw[
        max(0, TRAIN_END - 1200):TRAIN_END
    ]

    replay_indices = np.arange(
        max(0, base_end - 600),
        base_end,
    )

    replay_X = W[replay_indices].copy()
    replay_y = yw[replay_indices].copy()

    # --------------------------------------------------------
    # Detector state
    # --------------------------------------------------------

    ph = PageHinkley(
        delta=PH_DELTA,
        threshold=PH_THRESHOLD,
        warmup=PH_WARMUP,
    )

    alarm_history = deque(maxlen=ALARM_HISTORY_WINDOWS)

    data_candidate_streak = 0
    data_active = False
    data_quiet_windows = 0

    relational_active = False
    relational_quiet_windows = 0

    episode_latched = False
    pending = None
    cooldown_until = -1

    detector_log = []
    adaptation_events = []

    # --------------------------------------------------------
    # Streaming loop: predict, detect, adapt
    # --------------------------------------------------------

    print("Monitoring stream with persistent drift detection...")

    for start in range(stream_start, len(W), CHUNK):
        end = min(start + CHUNK, len(W))

        raw_start = int(row_indices[start])
        raw_end = int(row_indices[end - 1]) + 1

        # A. Predict first, before updating any model.
        for name, model in models.items():
            pred_scaled = predict_scaled(
                model,
                W[start:end],
                device,
            )

            predictions[name][raw_start:raw_end] = (
                pred_scaled * y_std + y_mean
            )

        # B. Data drift: compare several features to a fixed reference.
        current_start = max(STREAM_START, raw_start - 120)
        current_raw = X_raw[current_start:raw_end]

        _, ks_details = ks_detector(
            reference_raw,
            current_raw,
        )

        _, psi_details = psi_detector(
            reference_raw,
            current_raw,
            threshold=PSI_THRESHOLD,
        )

        # Count features with evidence from KS statistic or PSI.
        ks_by_feature = {
            item["feature_index"]: item["statistic"]
            for item in ks_details
        }

        psi_by_feature = {
            item["feature_index"]: item["psi"]
            for item in psi_details
        }

        moved_features = []

        for j in range(len(FEATURES)):
            ks_value = ks_by_feature.get(j, 0.0)
            psi_value = psi_by_feature.get(j, 0.0)

            if (
                ks_value >= KS_STAT_THRESHOLD
                or psi_value >= PSI_THRESHOLD
            ):
                moved_features.append(FEATURES[j])

        data_candidate = (
            len(moved_features) >= DATA_DRIFT_MIN_FEATURES
        )

        if data_candidate:
            data_candidate_streak += 1
            data_quiet_windows = 0
        else:
            data_candidate_streak = 0
            data_quiet_windows += 1

        data_alarm = (
            data_candidate_streak >= DATA_DRIFT_CONFIRM_WINDOWS
            and not data_active
        )

        if data_alarm:
            data_active = True

        # Clear data drift only after several consecutive quiet windows.
        if data_active and data_quiet_windows >= RECOVERY_WINDOWS:
            data_active = False
            data_candidate_streak = 0

        # C. Relational drift: frozen baseline GRU residuals.
        # Current targets are only used after their predictions are recorded.
        static_prediction = predictions[
            "Static / No Adaptation"
        ][raw_start:raw_end]

        residuals = (
            np.abs(static_prediction - y_raw[raw_start:raw_end])
            / reference_mae
        )

        residuals = np.clip(
            residuals,
            0.0,
            RESID_CLIP,
        )

        ph_fired = False

        for residual, row in zip(
            residuals,
            range(raw_start, raw_end),
        ):
            if ph.update(residual, row):
                ph_fired = True
                ph.reset()

        relational_alarm = (
            ph_fired and not relational_active
        )

        if ph_fired:
            relational_active = True
            relational_quiet_windows = 0
        elif relational_active:
            if float(np.mean(residuals)) < RESID_RECOVERY_THRESHOLD:
                relational_quiet_windows += 1
            else:
                relational_quiet_windows = 0

            if relational_quiet_windows >= RECOVERY_WINDOWS:
                relational_active = False
                relational_quiet_windows = 0
                ph.reset()

        # D. Confirm persistent evidence across windows.
        raw_flags = {
            "data": bool(data_alarm),
            "relational": bool(relational_alarm),
        }

        alarm_history.append(raw_flags)

        confirmed_flags = {
            name: (
                sum(window[name] for window in alarm_history)
                >= ALARM_CONFIRM_COUNT
            )
            for name in raw_flags
        }

        # The first window is allowed to open an episode when the
        # detector itself already requires persistence.
        candidate_alarm = (
            data_alarm or relational_alarm
        )

        # Episode-level alarm: never start another adaptation cycle
        # while an earlier drift episode remains active.
        diagnosis_active = data_active or relational_active

        new_alarm = (
            candidate_alarm
            and not episode_latched
            and raw_start >= cooldown_until
        )

        if new_alarm:
            episode_latched = True
            pending = {
                "detected_at": raw_start,
                "change_est": raw_start,
                "timestamp": (
                    df["timestamp"].iloc[raw_start]
                    if "timestamp" in df.columns
                    else pd.NaT
                ),
                "detectors": "|".join(
                    name
                    for name, fired in raw_flags.items()
                    if fired
                ),
            }

        # Rearm after all detectors report recovery.
        if not diagnosis_active and episode_latched:
            episode_latched = False
            cooldown_until = raw_end + EVENT_COOLDOWN

        diagnosis = (
            "both"
            if data_active and relational_active
            else "data"
            if data_active
            else "relational"
            if relational_active
            else "none"
        )

        # Log both raw and confirmed evidence.
        timestamp = (
            df["timestamp"].iloc[raw_start]
            if "timestamp" in df.columns
            else pd.NaT
        )

        detector_log.append({
            "row_index": raw_start,
            "timestamp": timestamp,
            "window_start": raw_start,
            "window_end": raw_end - 1,
            "KS_alarm": int(
                any(
                    value >= KS_STAT_THRESHOLD
                    for value in ks_by_feature.values()
                )
            ),
            "PSI_alarm": int(
                any(
                    value >= PSI_THRESHOLD
                    for value in psi_by_feature.values()
                )
            ),
            "Page_Hinkley_alarm": int(ph_fired),
            "data_candidate": int(data_candidate),
            "data_alarm": int(data_alarm),
            "relational_alarm": int(relational_alarm),
            "alarm": int(new_alarm),
            "confirmed_data": int(data_active),
            "confirmed_relational": int(relational_active),
            "episode_active": int(episode_latched),
            "diagnosis": diagnosis,
            "raw_alarm_count": int(sum(raw_flags.values())),
            "confirmed_alarm_count": int(
                sum(confirmed_flags.values())
            ),
            "features_moved": "|".join(moved_features),
            "feature_count_moved": len(moved_features),
            "max_KS_stat": max(ks_by_feature.values(), default=0.0),
            "min_KS_p": min(
                [item["p_value"] for item in ks_details],
                default=1.0,
            ),
            "max_PSI": max(psi_by_feature.values(), default=0.0),
            "mean_residual": float(np.mean(residuals)),
        })

        # E. Adapt once per confirmed episode, after MIN_ADAPT
        # observations have arrived following the first alarm.
        if (
            pending is not None
            and raw_end - pending["detected_at"] >= MIN_ADAPT
        ):
            lo = max(
                int(pending["detected_at"]),
                raw_end - MAX_ADAPT,
            )

            wi0 = max(0, lo - offset)
            wi1 = min(len(W), raw_end - offset)

            if wi1 > wi0:
                X_new = W[wi0:wi1]
                y_new = yw[wi0:wi1]

                training_seconds = {}

                for name in strategy_names:
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
                            adaptation_epochs,
                            1e-3,
                            device,
                            train_params=model.head.parameters(),
                        )

                    elif name == "Full Retraining":
                        hist_end = max(
                            1,
                            min(len(W), raw_end - offset),
                        )

                        model = GRURegressor()
                        model, seconds = fit_model(
                            model,
                            W[:hist_end],
                            yw[:hist_end],
                            adaptation_epochs,
                            2e-3,
                            device,
                        )

                    else:
                        replay_size = (
                            100 if name.endswith("100") else 500
                        )

                        take = min(replay_size, len(replay_X))
                        replay_batch_X = replay_X[-take:]
                        replay_batch_y = replay_y[-take:]

                        model, seconds = fit_model(
                            model,
                            X_new,
                            y_new,
                            adaptation_epochs,
                            1e-3,
                            device,
                            replay=(
                                replay_batch_X,
                                replay_batch_y,
                            ),
                        )

                    models[name] = model
                    training_seconds[name] = seconds

                # Update bounded replay pool with observed windows only.
                replay_X = np.concatenate(
                    [replay_X, X_new],
                    axis=0,
                )[-2000:]

                replay_y = np.concatenate(
                    [replay_y, y_new],
                    axis=0,
                )[-2000:]

                adaptation_events.append({
                    **pending,
                    "row_index": int(pending["detected_at"]),
                    "adapted_at": raw_end,
                    "adaptation_point": raw_end,
                    "n_samples": len(X_new),
                    "training_seconds": sum(
                        training_seconds.values()
                    ),
                })

            pending = None

    # ========================================================
    # OFFLINE EVALUATION
    # ========================================================

    metric_rows = []

    phases = {
        "normal_stream": (16000, 24000),
        "data_drift": (24000, 30000),
        "concept_drift": (30000, 42000),
        "combined_drift": (42000, len(df)),
    }

    for name in strategy_names:
        prediction = predictions[name]
        valid = np.isfinite(prediction)

        if valid.sum() > 1:
            metric_rows.append({
                "Strategy": name,
                "Phase": "overall",
                **score(y_raw[valid], prediction[valid]),
            })

        for phase, (lo, hi) in phases.items():
            mask = (
                (np.arange(len(df)) >= lo)
                & (np.arange(len(df)) < hi)
                & valid
            )

            if mask.sum() > 1:
                metric_rows.append({
                    "Strategy": name,
                    "Phase": phase,
                    **score(y_raw[mask], prediction[mask]),
                })

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metric_rows).to_csv(
        output / "strategy_metrics.csv",
        index=False,
    )

    pd.DataFrame(adaptation_events).to_csv(
        output / "adaptation_events.csv",
        index=False,
    )

    pd.DataFrame(detector_log).to_csv(
        output / "detector_window_metrics.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame({
        "row": np.arange(len(df)),
        "row_index": np.arange(len(df)),
        "timestamp": (
            df["timestamp"]
            if "timestamp" in df.columns
            else pd.Series([pd.NaT] * len(df))
        ),
        "actual": y_raw,
    })

    for name, prediction in predictions.items():
        prediction_frame[name] = prediction

    prediction_frame.to_csv(
        output / "predictions.csv",
        index=False,
    )

    truth_columns = [
        column for column in [
            "timestamp",
            "regime",
            "data_drift",
            "concept_drift",
            "drift_type",
            "true_drift_point",
        ]
        if column in df.columns
    ]

    if truth_columns:
        df[truth_columns].to_csv(
            output / "ground_truth_offline_only.csv",
            index=False,
        )

    print(f"Device: {device}")
    print(f"Rows: {len(df):,}")
    print(f"Detector windows: {len(detector_log)}")
    print(f"Adaptation events: {len(adaptation_events)}")

    alarm_count = sum(
        int(row["alarm"])
        for row in detector_log
    )

    print(f"Confirmed new alarm episodes: {alarm_count}")
    print(f"Results saved to: {output.resolve()}")

    if metric_rows:
        print(
            pd.DataFrame(metric_rows)
            .query("Phase == 'overall'")
            .to_string(index=False)
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(adaptation_events)


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
        data_path=args.data,
        out_dir=args.out,
        fast=not args.full,
        seed=args.seed,
    )
