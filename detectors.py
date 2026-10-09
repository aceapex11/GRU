
"""Unsupervised drift detectors for the PyTorch GRU experiment."""

import numpy as np
from scipy.stats import ks_2samp


def _prepare_windows(reference, current):
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

    reference = reference[np.isfinite(reference).all(axis=1)]
    current = current[np.isfinite(current).all(axis=1)]
    return reference, current


def ks_detector(
    reference,
    current,
    p_threshold=0.001,
    min_reference=50,
    min_current=20,
    correction="bonferroni",
):
    """Compare feature distributions using the two-sample KS test."""
    reference, current = _prepare_windows(reference, current)

    if reference is None:
        return False, []
    if (
        len(reference) < min_reference
        or len(current) < min_current
        or reference.shape[1] == 0
    ):
        return False, []

    details = []
    n_features = reference.shape[1]

    for j in range(n_features):
        ref = reference[:, j]
        cur = current[:, j]

        statistic, p_value = ks_2samp(
            ref, cur, alternative="two-sided", method="auto"
        )
        details.append({
            "feature_index": j,
            "statistic": float(statistic),
            "p_value": float(p_value),
        })

    threshold = (
        p_threshold / n_features
        if correction == "bonferroni"
        else p_threshold
    )
    alarm = any(item["p_value"] < threshold for item in details)
    return bool(alarm), details


def _psi_one(reference, current, bins=10, eps=1e-6):
    """Calculate PSI with bins defined by the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        scale = max(abs(float(reference[0])), 1.0)
        return float(
            np.mean(
                np.abs(current - reference[0]) > 1e-8 * scale
            )
        )

    edges[0], edges[-1] = -np.inf, np.inf

    r = np.histogram(reference, bins=edges)[0].astype(float)
    c = np.histogram(current, bins=edges)[0].astype(float)

    rp = np.clip(r / max(r.sum(), 1.0), eps, None)
    cp = np.clip(c / max(c.sum(), 1.0), eps, None)
    rp /= rp.sum()
    cp /= cp.sum()

    return float(np.sum((cp - rp) * np.log(cp / rp)))


def psi_detector(
    reference,
    current,
    threshold=0.25,
    bins=10,
    min_reference=50,
    min_current=20,
):
    """Compare feature distributions using PSI."""
    reference, current = _prepare_windows(reference, current)

    if reference is None:
        return False, []
    if (
        len(reference) < min_reference
        or len(current) < min_current
        or reference.shape[1] == 0
    ):
        return False, []

    details = []
    for j in range(reference.shape[1]):
        value = _psi_one(
            reference[:, j], current[:, j], bins=bins
        )
        details.append({
            "feature_index": j,
            "psi": float(value),
        })

    return bool(any(item["psi"] >= threshold for item in details)), details


class PageHinkley:
    """Page-Hinkley detector for an increasing monitored scalar signal."""

    def __init__(
        self,
        delta=0.15,
        threshold=25.0,
        warmup=40,
        cooldown=20,
    ):
        if delta < 0 or threshold <= 0:
            raise ValueError("delta must be non-negative and threshold positive.")
        if warmup < 2 or cooldown < 0:
            raise ValueError("Invalid warmup or cooldown.")

        self.delta = float(delta)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.cooldown = int(cooldown)
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0
        self.change_est = None
        self.last_alarm_t = None
        self.cooldown_remaining = 0

    def update(self, value, t):
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

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False

        self.cum += value - previous_mean - self.delta
        self.cum_min = min(self.cum_min, self.cum)

        if self.cum - self.cum_min > self.threshold:
            self.change_est = t
            self.last_alarm_t = t
            self.cum = 0.0
            self.cum_min = 0.0
            self.cooldown_remaining = self.cooldown
            return True

        return False
