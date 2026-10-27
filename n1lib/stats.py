"""Paired statistical tests: Wilcoxon signed-rank over (fold, seed) pairs and exact McNemar."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, wilcoxon


def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, float(p)


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan, 1.0
    try:
        r = wilcoxon(a, b, zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return np.nan, 1.0


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj
