#!/usr/bin/env python
"""Single resumable entry point.

    python run_all.py --config configs/full.yaml               # everything, resumable
    python run_all.py --config configs/full.yaml --estimate    # print runtime estimate only
    python run_all.py --config configs/full.yaml --benchmark   # measure GPU throughput (2-3 min)
    python run_all.py --config configs/full.yaml --stages eval analyze   # re-run post-hoc stages only
    python run_all.py --config configs/full.yaml --blocks A_main_realwaste

Stages: prepare -> train -> eval -> cross -> gradcam -> analyze. Finished training runs
(outputs/runs/<run_id>/done.json) are skipped; an interrupted run resumes from its last epoch
checkpoint. Progress: logs/progress.csv (one row per epoch / run event) and logs/run_all.log.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STAGES = ["prepare", "train", "eval", "cross", "gradcam", "analyze"]


def acquire_lock(log_dir: Path):
    """OS-level lock released automatically if the process dies (no stale locks)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    f = open(log_dir / "run_all.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"Another run_all.py is already running with log dir {log_dir}. Exiting.")
        sys.exit(2)
    return f


def main():
    for _s in (sys.stdout, sys.stderr):  # Windows consoles/redirects may not be UTF-8
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--blocks", nargs="+", default=None, help="only these experiment blocks")
    ap.add_argument("--max-runs", type=int, default=None, help="stop after N newly trained runs")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="retry runs listed as failed")
    a = ap.parse_args()

    from n1lib.common import Progress, expand_runs, load_config, python_info
    from n1lib.runtime import benchmark, estimate_run_seconds, estimate_table, load_throughput

    cfg = load_config(ROOT / a.config if not Path(a.config).is_absolute() else a.config)
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir = log_dir if log_dir.is_absolute() else ROOT / log_dir
    runs = expand_runs(cfg)
    if a.blocks:
        runs = [r for r in runs if r.block in a.blocks]

    if a.benchmark:
        models = sorted({r.model for r in runs})
        print("Benchmarking training throughput:")
        benchmark(models, cfg, log_dir)
        a.estimate = True
    if a.estimate:
        t, total_h = estimate_table(runs, cfg, load_throughput(log_dir))
        import pandas as pd
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(t.to_string(index=False))
        print(f"\nTOTAL: {len(runs)} runs, estimated {total_h:.1f} GPU-hours (+ ~0.5 h prepare/eval/analysis)")
        log_dir.mkdir(parents=True, exist_ok=True)
        t.to_csv(log_dir / "runtime_estimate.csv", index=False)
        return

    lock = acquire_lock(log_dir)  # noqa: F841
    P = Progress(log_dir)
    P.info(f"=== run_all start config={a.config} stages={a.stages} env={python_info()}")

    if "prepare" in a.stages:
        from n1lib.prepare import prepare_dataset
        need = {}
        for r in runs:
            d = need.setdefault(r.dataset, dict(pre={"none"}, naive={0}))
            d["pre"].add(r.preprocess)
            if r.protocol == "naive_aug":
                d["naive"].add(r.naive_aug_copies)
        for t in (cfg.get("crossdata") or {}).get("targets", {}):
            need.setdefault(t, dict(pre={"none"}, naive={0}))
        for ds, d in need.items():
            P.log(stage="prepare", run_id=ds, status="start")
            prepare_dataset(cfg, ds, sorted(d["pre"]), sorted(d["naive"]), log=P.info)
            P.log(stage="prepare", run_id=ds, status="done")

    if "train" in a.stages:
        from n1lib.train import is_done, run_dir, train_run
        import torch
        measured = load_throughput(log_dir)
        todo = [r for r in runs if not is_done(cfg, r)]
        failed_file = log_dir / "failed_runs.txt"
        failed = set(failed_file.read_text().split()) if failed_file.exists() and not a.retry_failed else set()
        todo = [r for r in todo if r.run_id not in failed]
        est = {r.run_id: estimate_run_seconds(r, cfg, measured) for r in todo}
        P.info(f"train: {len(runs)} runs in config, {len(runs) - len(todo)} done/skipped, {len(todo)} to do, "
               f"prior estimate {sum(est.values()) / 3600:.1f} h")
        spent_est, spent_real, n_new = 0.0, 0.0, 0
        for i, r in enumerate(todo):
            if a.max_runs is not None and n_new >= a.max_runs:
                break
            t0 = time.time()
            P.log(stage="train", run_id=r.run_id, run_index=i + 1, n_runs=len(todo), status="start")
            info, err = None, None
            for attempt in range(2):
                try:
                    info = train_run(cfg, r, P, i + 1, len(todo))
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    r.train = dict(r.train, batch_size=max(8, r.train["batch_size"] // 2))
                    P.info(f"OOM in {r.run_id}; retrying with batch_size={r.train['batch_size']}")
                    ck = run_dir(cfg, r) / "ckpt.pt"
                    if ck.exists():
                        ck.unlink()
                    err = "OOM"
                except KeyboardInterrupt:
                    P.log(stage="train", run_id=r.run_id, status="interrupted")
                    raise
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    P.info(traceback.format_exc())
                    break
            dt = time.time() - t0
            if info is None:
                P.log(stage="train", run_id=r.run_id, run_index=i + 1, n_runs=len(todo), status="FAILED", note=err)
                with open(failed_file, "a") as f:
                    f.write(r.run_id + "\n")
                continue
            n_new += 1
            spent_est += est[r.run_id]
            spent_real += dt
            ratio = spent_real / max(1.0, spent_est)
            remaining = sum(est[x.run_id] for x in todo[i + 1:]) * ratio
            P.log(stage="train", run_id=r.run_id, run_index=i + 1, n_runs=len(todo), status="done",
                  test_acc=round(info["test_acc_raw"], 4), run_elapsed_s=round(dt, 1),
                  eta_h=round(remaining / 3600, 2), note=f"speed_vs_estimate={ratio:.2f}")

    if "eval" in a.stages:
        from n1lib.evaluate import evaluate_all
        P.log(stage="eval", status="start")
        evaluate_all(cfg, log=P.info)
        P.log(stage="eval", status="done")
    if "cross" in a.stages and cfg.get("crossdata"):
        from n1lib.crossdata import run_crossdata
        try:
            run_crossdata(cfg, log=P.info)
            P.log(stage="cross", status="done")
        except Exception:  # noqa: BLE001
            P.info("cross-dataset stage failed:\n" + traceback.format_exc())
    if "gradcam" in a.stages:
        from n1lib.gradcam import make_gradcam_figure
        try:
            make_gradcam_figure(cfg, Path(cfg["out_dir"]) / "figures", log=P.info)
            P.log(stage="gradcam", status="done")
        except Exception:  # noqa: BLE001
            P.info("gradcam stage failed:\n" + traceback.format_exc())
    if "analyze" in a.stages:
        from n1lib.analyze import analyze
        analyze(cfg, log=P.info)
        P.log(stage="analyze", status="done")
    P.info("=== run_all finished")


if __name__ == "__main__":
    main()
