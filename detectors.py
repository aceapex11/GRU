"""Unsupervised drift detectors used by the GRU streaming experiment."""
import numpy as np
from scipy.stats import ks_2samp


def ks_detector(reference, current, p_threshold=0.001):
    """Feature-wise Kolmogorov-Smirnov test. Returns alarm and diagnostics."""
    details = []
    if len(current) < 20 or len(reference) < 50:
        return False, details
    for j in range(reference.shape[1]):
        ref = reference[:, j]
        cur = current[:, j]
        stat, p = ks_2samp(ref, cur, alternative="two-sided", method="auto")
        details.append({"feature_index": j, "statistic": float(stat), "p_value": float(p)})
    # Conservative multiple-feature rule: at least one feature has a very small p-value.
    return any(d["p_value"] < p_threshold for d in details), details


def _psi_one(reference, current, bins=10, eps=1e-6):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=edges)[0].astype(float)
    c = np.histogram(current, bins=edges)[0].astype(float)
    rp = np.clip(r / max(r.sum(), 1), eps, None)
    cp = np.clip(c / max(c.sum(), 1), eps, None)
    return float(np.sum((cp-rp) * np.log(cp/rp)))


def psi_detector(reference, current, threshold=0.25):
    """Feature-wise Population Stability Index; alarm if any feature exceeds threshold."""
    if len(current) < 20 or len(reference) < 50:
        return False, []
    vals = [_psi_one(reference[:, j], current[:, j]) for j in range(reference.shape[1])]
    detail = [{"feature_index": j, "psi": value} for j, value in enumerate(vals)]
    return any(v >= threshold for v in vals), detail


class PageHinkley:
    """Page-Hinkley detector for increasing normalized prediction residuals."""
    def __init__(self, delta=0.15, threshold=25.0, warmup=40):
        self.delta, self.threshold, self.warmup = delta, threshold, warmup
        self.reset()
    def reset(self):
        self.n=0; self.mean=0.; self.cum=0.; self.cum_min=0.; self.change_est=None
    def update(self, value, t):
        value=float(value); self.n += 1
        self.mean += (value-self.mean)/self.n
        if self.n < self.warmup: return False
        self.cum += value-self.mean-self.delta
        if self.cum < self.cum_min:
            self.cum_min=self.cum; self.change_est=t
        return (self.cum-self.cum_min) > self.threshold
