
"""Unsupervised drift detectors used by the GRU streaming experiment."""

import numpy as np
from scipy.stats import ks_2samp


# ============================================================
# INPUT VALIDATION
# ============================================================

def _prepare_windows(reference, current):
    """Convert inputs to finite 2D arrays and validate their dimensions."""
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

    # Remove rows with non-finite sensor values.
    reference = reference[np.isfinite(reference).all(axis=1)]
    current = current[np.isfinite(current).all(axis=1)]

    return reference, current


# ============================================================
# KOLMOGOROV-SMIRNOV DETECTOR
# ============================================================

def ks_detector(
    reference,
    current,
    p_threshold=0.001,
    min_reference=50,
    min_current=20,
    correction="bonferroni",
):
    """
    Compare reference and current feature distributions.

    Returns:
        alarm: bool
        details: list of per-feature statistics and p-values

    Bonferroni correction controls the family-wise error rate
    across the tested features.
    """
    reference, current = _prepare_windows(reference, current)

    if reference is None:
        return False, []

    if (
        len(reference) < min_reference
        or len(current) < min_current
        or reference.shape[1] == 0
    ):
        return False, []

    n_features = reference.shape[1]
    details = []

    for j in range(n_features):
        ref = reference[:, j]
        cur = current[:, j]

        if np.all(ref == ref[0]) and np.all(cur == cur[0]):
            statistic = 0.0 if ref[0] == cur[0] else 1.0
            p_value = 1.0 if ref[0] == cur[0] else 0.0
        else:
            statistic, p_value = ks_2samp(
                ref,
                cur,
                alternative="two-sided",
                method="auto",
            )

        details.append({
            "feature_index": j,
            "statistic": float(statistic),
            "p_value": float(p_value),
        })

    if correction == "bonferroni":
        adjusted_threshold = p_threshold / max(n_features, 1)
    else:
        adjusted_threshold = p_threshold

    alarm = any(
        item["p_value"] < adjusted_threshold
        for item in details
    )

    return bool(alarm), details


# ============================================================
# POPULATION STABILITY INDEX
# ============================================================

def _psi_one(reference, current, bins=10, eps=1e-6):
    """Calculate PSI using bins derived from the reference window."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Quantile-based reference bins.
    edges = np.unique(
        np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
    )

    if len(edges) < 3:
        # A constant reference cannot define meaningful quantile bins.
        # Compare the current values against the constant reference.
        scale = max(abs(float(reference[0])), 1.0)
        changed_fraction = np.mean(
            np.abs(current - reference[0]) > 1e-8 * scale
        )
        return float(changed_fraction)

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1.0)
    cur_pct = cur_counts / max(cur_counts.sum(), 1.0)

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    # Renormalize after clipping.
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def psi_detector(
    reference,
    current,
    threshold=0.25,
    bins=10,
    min_reference=50,
    min_current=20,
):
    """
    Compare feature distributions using PSI.

    Returns:
        alarm: bool
        details: per-feature PSI values
    """
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
            reference[:, j],
            current[:, j],
            bins=bins,
        )

        details.append({
            "feature_index": j,
            "psi": float(value),
        })

    alarm = any(
        item["psi"] >= threshold
        for item in details
    )

    return bool(alarm), details


# ============================================================
# PAGE-HINKLEY DETECTOR
# ============================================================

class PageHinkley:
    """
    Detect sustained increases in a monitored scalar signal.

    Intended input:
        A prediction error or residual, preferably on a stable,
        documented scale.

    The detector has:
        - warm-up period
        - running mean
        - cumulative deviation
        - minimum cumulative statistic
        - reset after alarm
        - cooldown to avoid immediate repeated alarms

    Note:
        delta and threshold depend on the scale of the input.
    """

    def __init__(
        self,
        delta=0.005,
        threshold=5.0,
        warmup=40,
        cooldown=20,
    ):
        if delta < 0:
            raise ValueError("delta must be non-negative.")

        if threshold <= 0:
            raise ValueError("threshold must be positive.")

        if warmup < 2:
            raise ValueError("warmup must be at least 2.")

        if cooldown < 0:
            raise ValueError("cooldown must be non-negative.")

        self.delta = float(delta)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.cooldown = int(cooldown)

        self.reset()

    def reset(self):
        """Reset detector statistics and internal state."""
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0
        self.change_est = None
        self.last_alarm_t = None
        self._cooldown_remaining = 0

    def update(self, value, t):
        """
        Update the detector with one observation.

        Returns:
            True if an alarm is raised; otherwise False.
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False

        if not np.isfinite(value):
            return False

        self.n += 1

        # Update mean using the current observation.
        previous_mean = self.mean
        self.mean += (value - self.mean) / self.n

        # Warm-up: establish a baseline before testing for drift.
        if self.n <= self.warmup:
            return False

        # Cooldown prevents consecutive alarms from the same episode.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        # Use the previous mean as the baseline for this observation.
        deviation = value - previous_mean - self.delta
        self.cum += deviation

        if self.cum < self.cum_min:
            self.cum_min = self.cum

        statistic = self.cum - self.cum_min

        if statistic > self.threshold:
            self.change_est = t
            self.last_alarm_t = t

            # Reset cumulative statistics but retain the running mean.
            # This lets the detector look for a later new change.
            self.cum = 0.0
            self.cum_min = 0.0
            self._cooldown_remaining = self.cooldown

            return True

        return False
