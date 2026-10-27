"""Unit tests on synthetic data: calibration metrics, temperature scaling, conformal coverage,
duplicate grouping and leakage-free splits.  Run:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n1lib.calib import apply_temperature, fit_temperature, softmax_np  # noqa: E402
from n1lib.conformal import conformal_quantile, conformal_sets, set_metrics  # noqa: E402
from n1lib.dedup import group_duplicates  # noqa: E402
from n1lib.metrics import acc_at_coverage, brier, classwise_ece, ece, mce, nll, risk_coverage  # noqa: E402
from n1lib.splits import grouped_cv_splits, leakage_report, naive_split  # noqa: E402
from n1lib.stats import holm, mcnemar_exact  # noqa: E402


def synth(n=20000, C=5, T_true=2.0, seed=0):
    """Labels drawn from softmax(logits/T_true): calibrated at T_true, overconfident at T=1."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 3, (n, C))
    p = softmax_np(z / T_true)
    y = np.array([rng.choice(C, p=pi) for pi in p])
    return z, y, p


def test_ece_perfectly_calibrated_is_small():
    z, y, p = synth()
    assert ece(p, y) < 0.02
    assert classwise_ece(p, y) < 0.02


def test_ece_overconfident_is_large_and_ts_recovers_temperature():
    z, y, p = synth()
    raw = softmax_np(z)
    assert ece(raw, y) > 0.08
    T = fit_temperature(z[:10000], y[:10000])
    assert abs(T - 2.0) < 0.15
    assert ece(apply_temperature(z[10000:], T), y[10000:]) < 0.02
    assert nll(apply_temperature(z[10000:], T), y[10000:]) < nll(raw[10000:], y[10000:])


def test_ece_hand_example():
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    y = np.array([0, 1, 0, 0])
    # bins: conf .9 acc .5 (gap .4, weight .5); conf .6 acc 1 (gap .4, weight .5)
    assert ece(probs, y, 15) == pytest.approx(0.4)
    assert mce(probs, y, 15) == pytest.approx(0.4)
    assert brier(np.eye(2)[[0, 1]], np.array([0, 1])) == 0


@pytest.mark.parametrize("method", ["lac", "aps", "raps"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_marginal_coverage(method, alpha):
    covs = []
    for s in range(20):
        z, y, p = synth(3000, 6, 2.0, seed=s)
        P = softmax_np(z)  # deliberately miscalibrated: conformal must still cover
        sets, _ = conformal_sets(P[:1000], y[:1000], P[1000:], method, alpha, seed=s)
        covs.append(set_metrics(sets, y[1000:])["coverage"])
    m = np.mean(covs)
    assert m >= 1 - alpha - 0.01, (method, alpha, m)
    assert m <= 1 - alpha + 0.03, (method, alpha, m)


def test_mondrian_class_conditional_coverage():
    rng = np.random.default_rng(1)
    z, y, p = synth(12000, 4, 2.0, seed=3)
    z[y == 3] *= 0.3  # one hard class
    P = softmax_np(z)
    alpha = 0.1
    sm_m = set_metrics(conformal_sets(P[:6000], y[:6000], P[6000:], "lac", alpha)[0], y[6000:])
    sm_c = set_metrics(conformal_sets(P[:6000], y[:6000], P[6000:], "lac", alpha, mondrian=True)[0], y[6000:])
    assert sm_c["worst_class_cov"] >= 1 - alpha - 0.03
    assert sm_c["worst_class_cov"] >= sm_m["worst_class_cov"] - 0.01


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 10, dtype=float)  # n=9, alpha=.1 -> ceil(10*.9)/9 = 1 -> max
    assert conformal_quantile(s, 0.1) == 9
    assert conformal_quantile(s[:5], 0.05) == np.inf


def test_risk_coverage_and_aurc():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0], bool)
    cov, risk, aurc = risk_coverage(conf, correct)
    assert risk[-1] == 0.5 and risk[0] == 0
    assert aurc == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert acc_at_coverage(conf, correct, 0.5) == 1.0


def test_dedup_groups_and_grouped_splits_are_leak_free():
    rng = np.random.default_rng(0)
    n = 300
    ph = rng.integers(0, 2 ** 63, n, dtype=np.int64).astype(np.uint64)
    dh = rng.integers(0, 2 ** 63, n, dtype=np.int64).astype(np.uint64)
    md5 = [f"m{i}" for i in range(n)]
    for k in range(0, 60, 2):  # 30 near-duplicate pairs (1 bit flipped)
        ph[k + 1] = ph[k] ^ np.uint64(1)
        dh[k + 1] = dh[k] ^ np.uint64(2)
    md5[299] = md5[298]  # exact duplicate
    y = np.repeat(np.arange(3), 100)
    h = pd.DataFrame(dict(path=[f"p{i}" for i in range(n)], label=y, md5=md5, phash=ph, dhash=dh))
    groups, pairs, summ = group_duplicates(h, 4, 4)
    assert summ["n_dup_groups"] == 31
    assert groups[0] == groups[1] and groups[298] == groups[299]
    folds = grouped_cv_splits(y, groups, 5, 0.2, 1)
    allt = np.concatenate([f["test"] for f in folds])
    assert len(np.unique(allt)) == n  # every image tested exactly once
    for f in folds:
        assert leakage_report(f, groups)["pct_test_dupgroup_in_train"] == 0
    nv = naive_split(y, 2, 0)
    rep = leakage_report(nv, groups)
    assert rep["pct_test_source_image_in_train"] > 50  # augmented copies leak


def test_stats():
    a = np.array([1, 1, 1, 0, 1, 1, 0, 1] * 20, bool)
    b = np.array([1, 0, 1, 0, 0, 1, 0, 1] * 20, bool)
    nb, nc, p = mcnemar_exact(a, b)
    assert nb == 40 and nc == 0 and p < 1e-6
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
