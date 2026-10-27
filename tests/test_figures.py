import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n1lib import figures as F  # noqa: E402


def test_parse_pm():
    assert F.parse_pm("0.8615 ± 0.0068") == (0.8615, 0.0068)
    v, s = F.parse_pm("0.9886")
    assert v == 0.9886 and np.isnan(s)
    assert all(np.isnan(F.parse_pm("–")))
    assert F.parse_pm("34.2 ± 1.7") == (34.2, 1.7)


def test_tables_only_figures(tmp_path):
    tab = tmp_path / "res" / "tables"
    tab.mkdir(parents=True)
    pd.DataFrame([{"Model": "ResNet101", "Kumar et al. 2026 (reported, single split)": "0.9886",
                   "Naive + aug-before-split acc": "0.90", "Naive + aug-before-split macro-F1": "0.89",
                   "Naive random split acc": "0.89", "Naive random split macro-F1": "0.88",
                   "Leakage-aware grouped 5-fold CV acc": "0.86 ± 0.01",
                   "Leakage-aware grouped 5-fold CV macro-F1": "0.85 ± 0.01", "Optimism gap (pp)": "4.0"}]
                 ).to_csv(tab / "tab2_optimism_gap.csv", index=False)
    rows = []
    for sc in ("LAC", "APS", "RAPS"):
        for a in (0.05, 0.1):
            for cc in ("no", "yes"):
                rows.append({"Model": "ResNet101", "Score": sc, "α": a, "Class-conditional": cc,
                             "Coverage": f"{1 - a:.3f} ± 0.01", "Target": 1 - a, "Avg set size": "1.5 ± 0.1",
                             "Singleton rate": "0.7 ± 0.02", "Worst-class coverage": "0.8 ± 0.03"})
    pd.DataFrame(rows).to_csv(tab / "tab5_conformal.csv", index=False)
    pd.DataFrame([{"Model": "ResNet101", "Full-coverage acc": "0.86 ± 0.01", "AURC": "0.03", "E-AURC": "0.02",
                   "Acc@80% (MSP)": "0.93 ± 0.01", "Acc@90% (MSP)": "0.90 ± 0.01", "AURC raw": "0.03",
                   "Singleton coverage (APS α=0.1)": "0.66", "Singleton acc (APS α=0.1)": "0.95 ± 0.01",
                   "Human review load (%)": "34.0 ± 2.0"}]).to_csv(tab / "tab6_selective_prediction.csv", index=False)
    out = tmp_path / "figs"
    w = F.make_all(tmp_path / "res", out, only=["fig1", "fig3", "fig5", "fig6"], draft=True, log=lambda *a: None)
    for k in ("fig1", "fig3", "fig5", "fig6"):
        assert (out / f"{F.FIGNAMES[k]}.pdf").exists(), k
        assert (out / f"{F.FIGNAMES[k]}.tif").exists(), k
