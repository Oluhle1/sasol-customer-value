"""Refit the early_history opportunity model for the December 2025 test cutoff.

Pre-registered under outputs/sprint16/proposal_december_blend.json.
Uses the frozen monthly training procedure (run_monthly.predict) on the
early-history panel (16 monthly origins, includes Jun/Jul/Aug 2024).
Saves outputs/sprint16/2025-12-01_early_history.npy.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))

OCT = ROOT / "outputs" / "oct10"
OUT = ROOT / "outputs" / "sprint16"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    from run_monthly import predict

    log("loading early_history panel")
    panel = pickle.loads((OCT / "early_history_panel.pkl").read_bytes())
    log(f"monthly_cutoffs: {panel['monthly_cutoffs']}")
    cutoff = "2025-12-01"
    log(f"fitting early_history for {cutoff}")
    t0 = time.time()
    p = predict(panel, cutoff, threads=1)
    log(f"done in {time.time() - t0:.0f}s shape={p.shape} rowsum={p.sum(axis=1).mean():.6f}")
    frame = panel["frames"][cutoff]
    assert p.shape == (len(frame), len(panel["classes"]))
    assert np.isfinite(p).all()
    path = OUT / f"{cutoff}_early_history.npy"
    np.save(path, p)
    log(f"saved {path}")
    dist = np.bincount(np.argmax(p, axis=1), minlength=17)
    for c, n in zip(panel["classes"], dist):
        log(f"  argmax {c}: {n}")


if __name__ == "__main__":
    main()
