# Deep-learning drift detection and adaptation (GRU)

A reproducible research repository that translates the uploaded drift-adaptation notebook's streaming experiment into a **GRU-only deep-learning benchmark**. It compares five strategies under a shared drift-detection schedule.

> **Important:** `industrial_sensor_drift_dataset.csv` is a controlled synthetic benchmark with plausible bounded sensor ranges and autocorrelated readings. It is not a real plant dataset and must not be presented as field-collected evidence.

## What is included

- `app.py` — integrated Streamlit dashboard for sensors, drift timeline, strategy metrics, predictions, and adaptation events.
- `notebooks/GRU_Drift_Adaptation_Research.ipynb` — walkthrough notebook.
- `data/industrial_sensor_drift_dataset.csv` — 60,000 hourly rows, six sensor features, a bounded deterioration score, and offline-only drift annotations.
- `src/data_generator.py` — deterministic dataset generator.
- `src/experiment.py` — GRU training, shared streaming adaptation, five GRU strategies, and evaluation.
- `src/detectors.py` — Kolmogorov–Smirnov (KS), Population Stability Index (PSI), and Page-Hinkley detectors.
- `results/` — experiment output folder (populated when run).
- `tests/test_dataset.py` — range and drift schedule sanity checks.

## Strategies

1. **Static / No Adaptation** — frozen baseline model.
2. **Transfer Learning — Frozen GRU** — freeze recurrent feature extractor and train the regression head.
3. **Full Retraining** — initialize a new GRU and train on all labelled history available at adaptation time.
4. **Continual Learning + Replay (100)** — update the current GRU with new labelled samples plus up to 100 replay samples.
5. **Continual Learning + Replay (500)** — same, with up to 500 replay samples.

All strategies begin with the same baseline training set and use a shared adaptation schedule. KS and PSI monitor sensor feature distributions; Page-Hinkley monitors normalized prediction errors from the static baseline GRU. Any detector alarm can trigger adaptation, so the strategies receive the same adaptation opportunities.

## Drift schedule and target

| Rows (0-based, end-exclusive) | Regime | Designed change |
|---|---|---|
| 0–23,999 | Baseline | Stable relationship |
| 24,000–29,999 | Data drift | Sensor distributions shift gradually; target relationship held constant |
| 30,000–41,999 | Concept drift | Sensor-to-target weights change gradually; sensor distributions return toward baseline |
| 42,000–59,999 | Combined drift | Sensor distributions shift and target relationship differs |

`regime`, `data_drift`, `concept_drift`, `drift_type`, and `true_drift_point` are **offline ground truth only**. The live detector/adaptation code does not read them. They are used only to score results and draw evaluation plots.

## Sensor ranges

The generator uses bounded values in plausible industrial ranges for a generic motor/pump scenario: temperature (°C), pressure (bar), vibration velocity RMS (mm/s), load (%), relative humidity (%), motor speed (rpm), and cumulative operating hours. These are illustrative design ranges, not universal engineering limits; actual limits depend on machine, sensor placement, rating, and operating standard. The deterioration target is a synthetic 0–100 score, not a measured probability of failure.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m src.data_generator
python -m src.experiment

# Launch the integrated dashboard in another terminal
streamlit run app.py
```

The default run is a fast smoke-test setting. For a longer run:

```bash
python -m src.experiment --full
```

## Run in Google Colab

1. Upload this repository ZIP to Colab and extract it, or clone your own GitHub repository.
2. Set the notebook working directory to the extracted repository root.
3. Run `notebooks/GRU_Drift_Adaptation_Research.ipynb` from top to bottom.
4. Do not force-reinstall NumPy or downgrade Colab's preinstalled packages unless you have a pinned environment and have restarted the runtime.

## Outputs

- `results/strategy_metrics.csv`: MAE, RMSE and R² by strategy and phase.
- `results/adaptation_events.csv`: detection/adaptation timestamps, detector names, and sample counts.
- `results/detector_window_metrics.csv`: KS/PSI/Page-Hinkley alarm flags and window statistics.
- `results/predictions.csv`: actual and strategy predictions by timestamp.
- `results/ground_truth_offline_only.csv`: separate truth file for offline evaluation only.

## Scientific cautions

- This is a controlled synthetic experiment. It tests whether the code behaves as intended; it cannot establish real-world performance by itself.
- Sensor ranges are scenario assumptions and must be validated against the intended equipment's manual, sensor specification, and operating logs.
- The target is generated from the sensor variables. It is therefore not independent evidence of real failure risk.
- Use chronological splits; fit scalers only on baseline training rows.
- Current labels are consumed only after the corresponding prediction has been made (test-then-train).
- Compare multiple seeds and include real public datasets before drawing strong research conclusions.
- The replay methods deliberately trade off new-regime adaptation and retention of old patterns. Report both instead of choosing only the best metric.

## Research references and dataset provenance

The repository structure and experiment design were adapted from the user's uploaded notebook and the requested GitHub repository link. I could not reliably retrieve the target repository's files through the current web fetch, so this implementation does not claim a line-by-line copy of its source.

A BibTeX file is included at `references.bib`. Useful public benchmark sources:
- NASA Prognostics Center of Excellence data repository (C-MAPSS, bearings and other datasets): https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA IMS Bearings dataset: https://data.nasa.gov/dataset/ims-bearings
- Paderborn University Bearing DataCenter: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
- IEEE IES Industrial AI Lab benchmark dataset overview: https://ieee-ies-industrial-ai-lab.github.io/industrial-ai-hub/datasets/
- MetroPT predictive-maintenance benchmark paper: https://arxiv.org/abs/2207.05466
- SCANIA Component X dataset paper: https://arxiv.org/abs/2401.15199
- Syed et al. (2025), *A systematic review of time series algorithms and analytics in predictive maintenance*, Decision Analytics Journal, DOI: https://doi.org/10.1016/j.dajour.2025.100573
- A real-world IIoT dataset for predictive maintenance of metalworking fluids (2025), Data in Brief, DOI: https://doi.org/10.1016/j.dib.2025.112020

These references describe real datasets or relevant research context; they do not validate the synthetic sensor values or establish that this benchmark represents every industrial machine.


## Integrated dashboard

The root `app.py` is the GRU dashboard. It reads the current dataset and `results/` files, and its sidebar can launch the GRU experiment. By default it runs the fast smoke-test configuration; check **Full training (slower)** for the longer run. Historical files from the previous classic-ML project are retained under `legacy_ml/` for traceability, not merged into current GRU metrics.

The old app depended on its own `DualRunner` backend, so the old frontend is archived rather than falsely presented as compatible with the GRU backend. The new dashboard implements corresponding high-level views against the GRU experiment outputs.
