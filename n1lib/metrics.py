"""Classification, calibration and selective-prediction metrics (numpy, probabilities in)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def ece_bins(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    cnt = np.bincount(ids, minlength=n_bins)
    s_conf = np.bincount(ids, weights=conf, minlength=n_bins)
    s_acc = np.bincount(ids, weights=correct.astype(float), minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_conf = s_conf / cnt
        avg_acc = s_acc / cnt
    return cnt, avg_conf, avg_acc


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    cnt, c, a = ece_bins(conf, correct, n_bins)
    m = cnt > 0
    return float(np.sum(cnt[m] / len(y) * np.abs(c[m] - a[m])))


def mce(probs, y, n_bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    cnt, c, a = ece_bins(conf, correct, n_bins)
    m = cnt > 0
    return float(np.max(np.abs(c[m] - a[m]))) if m.any() else 0.0


def classwise_ece(probs, y, n_bins=15):
    """Mean over classes of the ECE of p_k vs 1[y=k] (Kull et al. 2019)."""
    vals = []
    for k in range(probs.shape[1]):
        pk = probs[:, k]
        ok = (y == k).astype(float)
        cnt, c, _ = ece_bins(pk, ok, n_bins)
        edges = np.linspace(0, 1, n_bins + 1)
        ids = np.clip(np.digitize(pk, edges[1:-1], right=True), 0, n_bins - 1)
        freq = np.bincount(ids, weights=ok, minlength=n_bins)
        m = cnt > 0
        vals.append(np.sum(np.abs(c[m] * cnt[m] - freq[m])) / len(y))
    return float(np.mean(vals))


def brier(probs, y):
    oh = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - oh) ** 2, 1)))


def nll(probs, y, eps=1e-12):
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1))))


def risk_coverage(conf, correct):
    """Sort by confidence desc; returns coverage, selective risk arrays and AURC."""
    order = np.argsort(-conf, kind="stable")
    err = 1 - correct[order].astype(float)
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk, float(np.mean(risk))


def acc_at_coverage(conf, correct, c):
    order = np.argsort(-conf, kind="stable")
    k = max(1, int(np.floor(c * len(conf))))
    return float(correct[order[:k]].mean())


def eaurc(conf, correct):
    _, _, a = risk_coverage(conf, correct)
    r = 1 - correct.mean()
    opt = r + (1 - r) * np.log(1 - r + 1e-12) if r > 0 else 0.0
    return float(a - opt)


def all_metrics(probs, y, class_names, n_bins=15) -> dict:
    pred = probs.argmax(1)
    correct = pred == y
    conf = probs.max(1)
    f1s = f1_score(y, pred, labels=range(len(class_names)), average=None, zero_division=0)
    _, _, aurc = risk_coverage(conf, correct)
    d = dict(acc=float(correct.mean()),
             macro_f1=float(f1_score(y, pred, average="macro", labels=range(len(class_names)), zero_division=0)),
             ece=ece(probs, y, n_bins), cw_ece=classwise_ece(probs, y, n_bins), mce=mce(probs, y, n_bins),
             brier=brier(probs, y), nll=nll(probs, y), aurc=aurc, eaurc=eaurc(conf, correct),
             acc_cov80=acc_at_coverage(conf, correct, 0.8), acc_cov90=acc_at_coverage(conf, correct, 0.9),
             mean_conf=float(conf.mean()))
    for c, v in zip(class_names, f1s):
        d[f"f1::{c}"] = float(v)
    return d
