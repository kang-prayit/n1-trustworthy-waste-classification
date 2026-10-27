"""Unit tests for the optimism-gap statistics (paired Wilcoxon + Holm), analyze.optimism_gap_tests."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n1lib.analyze import optimism_gap_tests  # noqa: E402


def _rows(ds, model, protocol, folds, seeds, base, rng, noise=0.003):
    out = []
    for f in folds:
        for s in seeds:
            acc = base + rng.normal(0, noise)
            out.append(dict(dataset=ds, protocol=protocol, preprocess="none", model=model, loss="ce", fold=f, seed=s,
                            method="raw", acc=acc, macro_f1=acc - 0.02))
    return out


def _frame(naive_folds, naive_seeds, gap=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("dsA", "dsB"):
        for m in ("resnet101", "hybrid_r101_incv3"):
            rows += _rows(ds, m, "grouped", range(5), (0, 1, 2), 0.85, rng)
            rows += _rows(ds, m, "naive_aug", naive_folds, naive_seeds, 0.85 + gap, rng)
    # rows that must be ignored: ensembles (seed -1), other method / loss
    rows += [dict(dataset="dsA", protocol="naive_aug", preprocess="none", model="resnet101", loss="ce", fold=0, seed=-1,
                  method="raw", acc=0.0, macro_f1=0.0),
             dict(dataset="dsA", protocol="naive_aug", preprocess="none", model="resnet101", loss="ce", fold=0, seed=0,
                  method="ts", acc=0.0, macro_f1=0.0)]
    return pd.DataFrame(rows)


def test_optimism_gap_many_pairs_significant_and_holm():
    M = _frame(range(5), (0, 1, 2))  # 15 (fold, seed) pairs per comparison
    T = optimism_gap_tests(M, naive_protocols=("naive_aug",))
    assert len(T) == 2 * 2 * 2  # 2 metrics x 2 datasets x 2 models
    assert (T["Pairs"] == 15).all()
    assert (T["Mean diff (naive - grouped)"] > 0.03).all()
    assert (T["p"] < 0.01).all()
    assert (T["p (Holm)"] >= T["p"] - 1e-12).all() and (T["p (Holm)"] <= 1).all()
    assert np.allclose(T["Min attainable p"], 2 / 2 ** 15)
    # Holm family = metric x naive protocol, 4 comparisons (2 datasets x 2 models)
    assert T.groupby("Family").size().eq(4).all()


def test_optimism_gap_single_split_three_seeds_cannot_reach_significance():
    M = _frame((0,), (0, 1, 2), gap=0.20)  # full.yaml design: naive = fold 0 only, seeds 0-2
    T = optimism_gap_tests(M, naive_protocols=("naive_aug",))
    assert (T["Pairs"] == 3).all()
    assert np.allclose(T["Min attainable p"], 0.25)
    assert (T["p"] >= 0.25 - 1e-12).all()


def test_optimism_gap_empty_without_naive_runs():
    M = _frame(range(5), (0, 1, 2))
    T = optimism_gap_tests(M[M.protocol == "grouped"])
    assert len(T) == 0 and "p (Holm)" in T.columns
