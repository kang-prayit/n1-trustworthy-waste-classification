"""Non-finite handling (see docs/NONFINITE_HANDLING.md): rows with NaN/inf probabilities are dropped
from the conformal calibration set and from the test set, counted, and written to nonfinite_rows.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n1lib.calib import softmax_np  # noqa: E402
from n1lib.conformal import conformal_sets  # noqa: E402
from n1lib.evaluate import evaluate_all  # noqa: E402


def test_conformal_drops_nonfinite_calibration_rows():
    rng = np.random.default_rng(0)
    p_cal = softmax_np(rng.normal(0, 2, (300, 4)))
    y_cal = rng.integers(0, 4, 300)
    p_te = softmax_np(rng.normal(0, 2, (50, 4)))
    ref, q_ref = conformal_sets(p_cal, y_cal, p_te, "lac", 0.1)
    bad = p_cal.copy()
    bad = np.vstack([bad, np.full((1, 4), np.nan)])
    y_bad = np.append(y_cal, 0)
    got, q = conformal_sets(bad, y_bad, p_te, "lac", 0.1)
    assert np.isfinite(q) and q == q_ref
    assert np.array_equal(got, ref)


def test_evaluate_drops_and_reports_nonfinite_test_row(tmp_path):
    rng = np.random.default_rng(1)
    C, classes = 3, ["a", "b", "c"]
    z = {}
    for part, n in (("calfit", 120), ("calconf", 120), ("test", 60)):
        y = rng.integers(0, C, n)
        logits = rng.normal(0, 1, (n, C)) + 3 * np.eye(C)[y]
        z[f"logits_{part}"], z[f"y_{part}"] = logits.astype(np.float32), y
    z["logits_test"][7] = np.nan  # e.g. fp16 overflow
    run = tmp_path / "out" / "runs" / "toy-grouped-m-ce-f0-s0"
    run.mkdir(parents=True)
    np.savez(run / "preds.npz", **z)
    info = dict(run_id=run.name, dataset="toy", protocol="grouped", preprocess="none", model="m", loss="ce",
                fold=0, seed=0, block="A", classes=classes)
    (run / "done.json").write_text(json.dumps(info))
    cfg = dict(out_dir=str(tmp_path / "out"), eval=dict(alphas=[0.1], conformal_scores=["lac"],
                                                         conformal_methods=["raw", "ts"]))
    evaluate_all(cfg, log=lambda *a: None)
    res = tmp_path / "out" / "results"
    M = pd.read_csv(res / "metrics_long.csv")
    assert np.isfinite(M[["ece", "nll", "brier"]].to_numpy()).all()
    NF = pd.read_csv(res / "nonfinite_rows.csv")
    assert set(NF.method) == {"raw", "ts", "vs"} and (NF.n_test_dropped == 1).all()
    assert (NF.n_test == 60).all() and (NF.test_idx.astype(str) == "7").all()
    probs = np.load(res / "probs" / f"{run.name}__raw.npz")["probs"]
    assert probs.shape == (59, C) and np.isfinite(probs).all()
