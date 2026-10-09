"""
Synthetic industrial-machine stream for drift research.
20,000 hourly rows, eight sensor features, labelled drift scenarios.
Target semantics: health_deterioration is a 0-100 score; 100 = worse deterioration.
Run: python data_generator.py
Output: data/drift_regression_dataset_20000.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter

FEATURES = ["temperature", "vibration", "pressure", "load",
            "operating_hours", "humidity", "rpm", "oil_viscosity"]
MONITORED = [f for f in FEATURES if f != "operating_hours"]
TARGET = "health_deterioration"
N_ROWS = 20_000
TRAIN_END = 5_000  # only 5,000 normal rows before the first drift
WINDOW = 24

NOM_MEAN = dict(temperature=70., vibration=2., pressure=30., load=60.,
                operating_hours=2000., humidity=45., rpm=1500., oil_viscosity=40.)
NOM_SD = dict(temperature=6., vibration=.5, pressure=3., load=12.,
              operating_hours=1155., humidity=10., rpm=120., oil_viscosity=4.)

# The relation weights alter sensor-to-target relationships during concept drift.
W_BASE = np.array([.25, .30, .12, .18, .05, .08, .02])
W_R1   = np.array([.12, .52, .10, .05, .04, .17, .02])
W_R2   = np.array([.06, .22, .28, .08, .04, .32, .02])

SHIFT_A = {"temperature": (1.8, 1.0), "vibration": (1.1, 1.25), "rpm": (.9, 1.0)}
SHIFT_B = {"load": (1.6, 1.0), "temperature": (0.0, 1.45),
           "humidity": (1.2, 1.0), "pressure": (-.8, 1.0)}
SHIFT_C = {"temperature": (1.2, 1.0), "load": (1.0, 1.0),
           "oil_viscosity": (-1.2, 1.15), "vibration": (.7, 1.0)}

# The stream has 5,000 normal training rows, then six labelled events.
# Each event is separated by a stable interval for recovery evaluation.
EVENTS = [
    (5_500,  6_500,  "data_drift",       "sudden",  SHIFT_A, None),
    (7_000,  8_000,  "relational_drift", "sudden",  None,    W_R1),
    (8_500, 10_000,  "both",             "gradual", SHIFT_C, W_R2),
    (10_500, 11_500, "relational_drift", "sudden",  None,    W_R1),
    (12_000, 13_500, "data_drift",       "sudden",  SHIFT_B, None),
    (14_000, 16_000, "both",             "sudden",  SHIFT_A, W_R2),
]
GRADUAL_RAMP = 240
GLITCH = (15_100, 15_108)  # sensor-only glitch; true target ignores it


def _ar1(n, phi, rng):
    return lfilter([1.0], [1.0, -phi], rng.normal(size=n)) * np.sqrt(1.0 - phi**2)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate(n=N_ROWS, seed=7):
    if n <= TRAIN_END:
        raise ValueError(f"n must exceed TRAIN_END={TRAIN_END}; got {n}.")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = np.sin(2 * np.pi * t / 24.0)
    yearly = np.sin(2 * np.pi * t / 8760.0)

    data_strength = np.zeros(n)
    weights = np.tile(W_BASE, (n, 1))
    regime = np.full(n, "stable", dtype=object)
    data_flag = np.zeros(n, dtype=int)
    concept_flag = np.zeros(n, dtype=int)
    drift_point = np.zeros(n, dtype=int)
    drift_mode = np.full(n, "none", dtype=object)

    for s, e, name, mode, shift, relation in EVENTS:
        s, e = min(s, n), min(e, n)
        if e <= s:
            continue
        local_t = np.arange(s, e)
        ramp = _smoothstep((local_t - s) / max(GRADUAL_RAMP, 1)) if mode == "gradual" else np.ones(e - s)
        regime[s:e] = name
        drift_mode[s:e] = mode
        drift_point[s] = 1
        if shift is not None:
            data_strength[s:e] = ramp
            data_flag[s:e] = 1
        if relation is not None:
            weights[s:e] = (1.0 - ramp)[:, None] * W_BASE + ramp[:, None] * relation
            concept_flag[s:e] = 1

    z = {
        "temperature": .80 * _ar1(n, .92, rng) + .35 * daily + .20 * yearly,
        "vibration": _ar1(n, .90, rng) + .12 * daily,
        "pressure": _ar1(n, .93, rng) - .10 * yearly,
        "load": .80 * _ar1(n, .88, rng) + .45 * daily,
        "humidity": .90 * _ar1(n, .94, rng) - .20 * yearly,
        "rpm": .90 * _ar1(n, .85, rng) + .20 * rng.normal(size=n),
        "oil_viscosity": .90 * _ar1(n, .95, rng),
    }
    z["vibration"] += .30 * (z["temperature"] + z["load"]) / 2.0

    raw = {}
    for feature, signal in z.items():
        shift_amount = np.zeros(n)
        scale_amount = np.ones(n)
        for s, e, _, _, shift, _ in EVENTS:
            if shift is None or feature not in shift:
                continue
            s2, e2 = min(s, n), min(e, n)
            if e2 <= s2:
                continue
            strength = data_strength[s2:e2]
            mean_shift, scale_multiplier = shift[feature]
            shift_amount[s2:e2] = strength * mean_shift
            scale_amount[s2:e2] = 1.0 + strength * (scale_multiplier - 1.0)
        raw[feature] = NOM_MEAN[feature] + NOM_SD[feature] * (
            signal * scale_amount + shift_amount
        )

    raw["operating_hours"] = (t % 4000).astype(float)

    # Target-generating hidden state is captured before injecting sensor glitch.
    target_signals = {
        f: ((raw[f] - NOM_MEAN[f]) / NOM_SD[f]).copy()
        for f in raw
    }
    g0, g1 = min(GLITCH[0], n), min(GLITCH[1], n)
    raw["temperature"][g0:g1] += 6.0 * NOM_SD["temperature"]
    raw["vibration"][g0:g1] += 5.0 * NOM_SD["vibration"]

    # Keep condition signals bounded to avoid unrealistic target spikes.
    target_order = ["temperature", "vibration", "pressure", "load",
                    "humidity", "rpm", "oil_viscosity"]
    sensor_vector = np.column_stack([
        np.clip(target_signals[f], -3.0, 3.0) for f in target_order
    ])
    condition = np.sum(weights * sensor_vector, axis=1)

    # A gradual deterioration baseline plus small bounded sensor effects.
    lifetime_fraction = t / max(n - 1, 1)
    wear = 12.0 + 52.0 * lifetime_fraction
    maintenance_cycle = t // 4000
    wear -= 3.0 * (maintenance_cycle > 0)
    noise = rng.normal(0.0, 0.8, n)
    target_unclipped = wear + 5.0 * condition + noise
    target = np.clip(target_unclipped, 0.0, 100.0)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2018-01-01", periods=n, freq="h")
    })
    for f in FEATURES:
        df[f] = raw[f]
    df[TARGET] = target
    df["regime"] = regime
    df["data_drift"] = data_flag
    df["concept_drift"] = concept_flag
    df["true_drift_point"] = drift_point
    df["drift_mode"] = drift_mode
    df["sensor_glitch"] = 0
    if g1 > g0:
        df.loc[g0:g1 - 1, "sensor_glitch"] = 1
    return df


def validate_dataset(df):
    if len(df) != N_ROWS:
        raise AssertionError(f"Expected {N_ROWS} rows, got {len(df)}.")
    numeric = FEATURES + [TARGET]
    if df[numeric].isna().any().any():
        raise AssertionError("Missing numeric values found.")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise AssertionError("Non-finite numeric values found.")
    if not df[TARGET].between(0, 100).all():
        raise AssertionError("Target outside [0, 100].")
    if df["data_drift"].sum() == 0 or df["concept_drift"].sum() == 0:
        raise AssertionError("Expected both data and concept drift labels.")
    max_step = float(df[TARGET].diff().abs().max())
    if max_step > 25.0:
        raise AssertionError(f"Unrealistic one-hour target jump: {max_step:.2f}.")
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
        "target_max_abs_hourly_change": max_step,
        "data_drift_rows": int(df["data_drift"].sum()),
        "concept_drift_rows": int(df["concept_drift"].sum()),
        "glitch_rows": int(df["sensor_glitch"].sum()),
    }


if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate(n=N_ROWS, seed=7)
    summary = validate_dataset(dataset)
    output_path = output_dir / "drift_regression_dataset_20000.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Saved: {output_path.resolve()}")
    print(f"Shape: {dataset.shape}")
    print(pd.Series(summary).to_string())
    print("\nRegime summary:")
    print(dataset.groupby("regime", sort=False)[TARGET]
          .agg(["size", "mean", "std", "min", "max"]).round(2).to_string())
    print("\nDrift label summary:")
    print(dataset.groupby("regime", sort=False)[
        ["data_drift", "concept_drift", "true_drift_point"]
    ].sum().to_string())
