import numpy as np
import pandas as pd

def generate_dataset(n=60_000, seed=42, start="2025-01-01"):
    """Generate controlled, autocorrelated industrial sensor data with known drift labels.
    This is a synthetic benchmark, not a claim of real plant measurements.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    load = np.zeros(n)
    temp = np.zeros(n); pressure = np.zeros(n); vibration = np.zeros(n)
    humidity = np.zeros(n); rpm = np.zeros(n)
    load[0] = 62.0
    for i in range(1, n):
        target_load = 62 + 8*np.sin(2*np.pi*i/(24*7)) + 3*np.sin(2*np.pi*i/24) + rng.normal(0, 2)
        load[i] = np.clip(0.92*load[i-1] + 0.08*target_load + rng.normal(0, 1.4), 30, 90)
    for i in range(n):
        if i < 24_000: data_level = 0.0
        elif i < 30_000: data_level = (i-24_000)/6_000
        elif i < 42_000: data_level = 0.0
        else: data_level = 0.75
        temp_mu = 58 + .20*(load[i]-60) + 4*np.sin(2*np.pi*i/(24*3)) + 5*data_level
        press_mu = 5.6 - .006*(load[i]-60) + .12*np.sin(2*np.pi*i/(24*2)) - .35*data_level
        vib_mu = 2.0 + .018*max(load[i]-45, 0) + .25*np.sin(2*np.pi*i/(24*2)) + .55*data_level
        hum_mu = 45 + 7*np.sin(2*np.pi*i/(24*30))
        rpm_mu = 1480 + 2*(load[i]-60)
        if i == 0:
            temp[i], pressure[i], vibration[i] = temp_mu, press_mu, vib_mu
            humidity[i], rpm[i] = hum_mu, rpm_mu
        else:
            temp[i] = .85*temp[i-1] + .15*temp_mu + rng.normal(0, .65)
            pressure[i] = .78*pressure[i-1] + .22*press_mu + rng.normal(0, .055)
            vibration[i] = .72*vibration[i-1] + .28*vib_mu + rng.normal(0, .11)
            humidity[i] = .92*humidity[i-1] + .08*hum_mu + rng.normal(0, .7)
            rpm[i] = .8*rpm[i-1] + .2*rpm_mu + rng.normal(0, 2.0)
    temp = np.clip(temp, 45, 90); pressure = np.clip(pressure, 4.2, 6.8)
    vibration = np.clip(vibration, .5, 6.5); humidity = np.clip(humidity, 20, 75)
    rpm = np.clip(rpm, 1380, 1580)
    temp_sev = np.clip((temp-55)/25, 0, 1)
    vib_sev = np.clip((vibration-1)/5, 0, 1)
    press_sev = np.clip(np.abs(pressure-5.6)/1.4, 0, 1)
    load_sev = np.clip((load-30)/60, 0, 1)
    target = np.zeros(n)
    for i in range(n):
        w = np.array([.30, .35, .15, .20])
        if 36_000 <= i < 48_000:
            alpha = (i-36_000)/12_000
            w = (1-alpha)*w + alpha*np.array([.18, .48, .10, .24])
        elif i >= 48_000:
            w = np.array([.18, .48, .10, .24])
        target[i] = np.clip(100*np.dot(w, [temp_sev[i], vib_sev[i], press_sev[i], load_sev[i]])
                            + min(12, .0012*i) + rng.normal(0, 1.8), 0, 100)
    regime = np.where(t < 24_000, "baseline", np.where(t < 30_000, "data_drift",
                     np.where(t < 42_000, "concept_drift", "combined_drift")))
    data_drift = ((t >= 24_000) & (t < 30_000)) | (t >= 42_000)
    concept_drift = t >= 36_000
    drift_type = np.where(t < 24_000, "none", np.where(t < 30_000, "data",
                      np.where(t < 42_000, "concept", "both")))
    true_point = np.where(np.isin(t, [24_000, 30_000, 42_000]), drift_type, "")
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="h"),
        "temperature_C": temp.round(3), "pressure_bar": pressure.round(4),
        "vibration_mm_s_rms": vibration.round(4), "load_pct": load.round(3),
        "humidity_pct": humidity.round(3), "motor_speed_rpm": rpm.round(2),
        "operating_hours": t.astype(float),
        "health_deterioration": target.round(3), "regime": regime,
        "data_drift": data_drift.astype(int), "concept_drift": concept_drift.astype(int),
        "drift_type": drift_type, "true_drift_point": true_point
    })

if __name__ == "__main__":
    from pathlib import Path
    out = Path("data/industrial_sensor_drift_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset(n=60_000)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame):,} rows to {out}")
