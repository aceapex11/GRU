
"""Drift detectors for the PyTorch GRU streaming experiment."""

import numpy as np
from scipy.stats import ks_2samp


# Conservative defaults; validate these against offline drift labels.
KS_P_THRESHOLD = 0.001
PSI_THRESHOLD = 0.25


def _prepare(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    if current.ndim == 1:
        current = current.reshape(-1, 1)

    if reference.ndim != 2 or current.ndim != 2:
        return None, None

    if reference.shape[1] != current.shape[1]:
        return None, None

    # Drop invalid rows consistently.
    reference = reference[np.isfinite(reference).all(axis=1)]
    current = current[np.isfinite(current).all(axis=1)]

    return reference, current


def ks_detector(
    reference,
    current,
    p_threshold=KS_P_THRESHOLD,
    min_reference=50,
    min_current=20,
):
    """Return per-feature KS statistics and a distribution-change alarm."""
    reference, current = _prepare(reference, current)

    if reference is None or current is None:
        return False, []

    if (
        len(reference) < min_reference
        or len(current) < min_current
    ):
        return False, []

    details = []
    feature_count = reference.shape[1]

    # Bonferroni correction controls false positives across features.
    adjusted_threshold = p_threshold / max(feature_count, 1)

    for j in range(feature_count):
        statistic, p_value = ks_2samp(
            reference[:, j],
            current[:, j],
            alternative="two-sided",
            method="auto",
        )

        details.append({
            "feature_index": j,
            "statistic": float(statistic),
            "p_value": float(p_value),
        })

    alarm = any(
        item["p_value"] < adjusted_threshold
        for item in details
    )

    return bool(alarm), details


def _psi_one(reference, current, bins=10, eps=1e-6):
    """Calculate PSI using bins determined by the reference data."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    # Constant or nearly constant reference feature.
    if len(edges) < 3:
        scale = max(float(np.max(np.abs(reference))), 1.0)
        changed_fraction = np.mean(
            np.abs(current - reference[0]) > 1e-8 * scale
        )
        return float(changed_fraction)

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(reference, bins=edges)[0]
    current_counts = np.histogram(current, bins=edges)[0]

    reference_pct = (
        reference_counts.astype(float) / len(reference) + eps
    )
    current_pct = (
        current_counts.astype(float) / len(current) + eps
    )

    reference_pct /= reference_pct.sum()
    current_pct /= current_pct.sum()

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def psi_detector(
    reference,
    current,
    threshold=PSI_THRESHOLD,
    bins=10,
    min_reference=50,
    min_current=20,
):
    """Return per-feature PSI scores and a distribution-change alarm."""
    reference, current = _prepare(reference, current)

    if reference is None or current is None:
        return False, []

    if (
        len(reference) < min_reference
        or len(current) < min_current
    ):
        return False, []

    details = []

    for j in range(reference.shape[1]):
        value = _psi_one(
            reference[:, j],
            current[:, j],
            bins=bins,
        )

        details.append({
            "feature_index": j,
            "psi": float(value),
        })

    return (
        bool(any(item["psi"] >= threshold for item in details)),
        details,
    )


class PageHinkley:
    """Page-Hinkley detector for increases in a monitored signal."""

    def __init__(
        self,
        delta=0.15,
        threshold=25.0,
        warmup=40,
        alpha=1.0,
    ):
        if delta < 0:
            raise ValueError("delta must be non-negative.")
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        if warmup < 1:
            raise ValueError("warmup must be at least 1.")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")

        self.delta = float(delta)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0
        self.change_est = None

    def update(self, value, t=None):
        """Consume one observation and return True when an alarm fires."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        if not np.isfinite(value):
            return False

        self.n += 1
        previous_mean = self.mean
        self.mean += (value - self.mean) / self.n

        if self.n <= self.warmup:
            return False

        self.cum = (
            self.alpha * self.cum
            + value - previous_mean - self.delta
        )

        if self.cum < self.cum_min:
            self.cum_min = self.cum

        if self.cum - self.cum_min > self.threshold:
            self.change_est = t
            return True

        return False
