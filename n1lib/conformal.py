"""Split conformal prediction: LAC/THR, APS, RAPS; marginal and class-conditional (Mondrian).

Scores s(x, y) are 'non-conformity' (higher = less conforming). With n calibration scores the
threshold is the ceil((n+1)(1-alpha))/n empirical quantile, giving P(Y in C(X)) >= 1-alpha
under exchangeability (Vovk et al. 2005; Angelopoulos & Bates 2023)."""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.inf
    lev = np.ceil((n + 1) * (1 - alpha)) / n
    if lev > 1:
        return np.inf
    return float(np.quantile(scores, lev, method="higher"))


def _aps_all(probs, u, lam=0.0, k_reg=0):
    """Score for every label: cumulative mass of labels ranked above + u * p(label) (+ RAPS penalty)."""
    n, C = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    ps = np.take_along_axis(probs, order, 1)
    cum = np.cumsum(ps, 1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(C)[None].repeat(n, 0), 1)
    cum_label = np.take_along_axis(cum, rank, 1)  # inclusive cum mass at label position
    s = cum_label - (1 - u[:, None]) * probs       # randomised: subtract (1-u) p_y  -> cum_before + u p_y
    if lam > 0:
        s = s + lam * np.maximum(rank + 1 - k_reg, 0)
    return s


def scores_all(probs, method, u=None, lam=0.01, k_reg=2):
    if method == "lac":
        return 1.0 - probs
    if u is None:
        u = np.ones(len(probs))
    if method == "aps":
        return _aps_all(probs, u)
    if method == "raps":
        return _aps_all(probs, u, lam, k_reg)
    raise KeyError(method)


def conformal_sets(cal_probs, cal_y, test_probs, method="lac", alpha=0.1, mondrian=False,
                   seed=0, lam=0.01, k_reg=2):
    # Calibration rows with non-finite probabilities (e.g. an fp16/AMP overflow giving NaN logits)
    # would turn the empirical quantile into NaN and empty every prediction set -> drop them.
    # evaluate.py counts and logs these rows per run; this guard also protects cross/gradcam.
    cal_probs = np.asarray(cal_probs)
    cal_y = np.asarray(cal_y)
    ok = np.isfinite(cal_probs).all(1)
    if not ok.all():
        cal_probs, cal_y = cal_probs[ok], cal_y[ok]
    rng = np.random.default_rng(seed)
    u_cal = rng.uniform(size=len(cal_probs))
    u_te = rng.uniform(size=len(test_probs))
    s_cal_all = scores_all(cal_probs, method, u_cal, lam, k_reg)
    s_cal = s_cal_all[np.arange(len(cal_y)), cal_y]
    s_te = scores_all(test_probs, method, u_te, lam, k_reg)
    C = cal_probs.shape[1]
    if mondrian:
        q = np.array([conformal_quantile(s_cal[cal_y == k], alpha) for k in range(C)])
        sets = s_te <= q[None, :]
    else:
        q = conformal_quantile(s_cal, alpha)
        sets = s_te <= q
    return sets, q


def set_metrics(sets, y, probs=None) -> dict:
    C = sets.shape[1]
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    cls_cov = np.array([covered[y == k].mean() if (y == k).any() else np.nan for k in range(C)])
    d = dict(coverage=float(covered.mean()), avg_size=float(size.mean()),
             median_size=float(np.median(size)), singleton_rate=float((size == 1).mean()),
             empty_rate=float((size == 0).mean()), worst_class_cov=float(np.nanmin(cls_cov)),
             class_cov_gap=float(np.nanmax(cls_cov) - np.nanmin(cls_cov)))
    if probs is not None:
        # selective prediction by conformal set size: accept only singletons
        acc_mask = size == 1
        pred = probs.argmax(1)
        d["sel_singleton_coverage"] = float(acc_mask.mean())
        d["sel_singleton_acc"] = float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else np.nan
    d["class_cov"] = cls_cov.tolist()
    return d
