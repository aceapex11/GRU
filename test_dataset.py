from src.data_generator import generate_dataset

def test_dataset_shape_and_ranges():
    df = generate_dataset(n=60_000, seed=42)
    assert len(df) == 60_000
    assert df["temperature_C"].between(45, 90).all()
    assert df["pressure_bar"].between(4.2, 6.8).all()
    assert df["vibration_mm_s_rms"].between(.5, 6.5).all()
    assert df["load_pct"].between(30, 90).all()
    assert df["humidity_pct"].between(20, 75).all()
    assert df["motor_speed_rpm"].between(1380, 1580).all()
    assert df["health_deterioration"].between(0, 100).all()

def test_drift_boundaries():
    df = generate_dataset(n=60_000, seed=42)
    assert df.loc[23_999, "drift_type"] == "none"
    assert df.loc[24_000, "drift_type"] == "data"
    assert df.loc[30_000, "drift_type"] == "concept"
    assert df.loc[42_000, "drift_type"] == "both"
