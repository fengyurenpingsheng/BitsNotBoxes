"""Statistics used in the paper: bootstrap intervals, paired Wilcoxon tests, Holm correction and pooled comparisons
with a sign-flip test."""
from __future__ import annotations

import numpy as np
from scipy import stats


def boot(x, n: int = 2000, seed: int = 0):
    """Mean and 95% bootstrap interval over questions (NaNs dropped): (mean, lo, hi)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(a, b) -> dict:
    """Paired comparison a - b: mean difference, bootstrap interval and two-sided Wilcoxon signed-rank p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    mu, lo, hi = boot(d)
    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return {"diff": mu, "lo": lo, "hi": hi, "p": float(p), "n": int(len(d))}


def holm(ps) -> np.ndarray:
    """Holm-adjusted p-values."""
    ps = np.asarray(ps, float)
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def pooled(cluster_values, n_boot: int = 10000, n_flip: int = 20000, seed: int = 0) -> dict:
    """Pooled mean of per-cluster values (one cluster = one question, averaged over models and/or windows), with a
    bootstrap interval over clusters and a two-sided sign-flip test."""
    rng = np.random.default_rng(seed)
    c = np.asarray(cluster_values, float)
    boots = rng.choice(c, size=(n_boot, len(c)), replace=True).mean(1)
    obs = c.mean()
    flips = (rng.choice([-1.0, 1.0], size=(n_flip, len(c))) * c).mean(1)
    p = (1 + np.sum(np.abs(flips) >= abs(obs) - 1e-12)) / (n_flip + 1)
    return {"diff": float(obs), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p": float(p), "n_clusters": int(len(c))}
