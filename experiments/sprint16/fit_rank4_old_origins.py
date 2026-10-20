"""Fit rank4 p4 arrays for old origins (2024-12 .. 2025-03) on the early panel.

Enables an 8-origin pooled tuning set for the r4full decode vector:
Dec-2024 (seasonal analog), Jan, Feb, Mar (early panel), Apr/May/Jul/Aug (cached).
June/September remain untouched evaluation folds.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))

from fit_rank4_early import fit_rank4  # noqa: E402

OCT = ROOT / "outputs" / "oct10"

OLD = ["2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"]


def main():
    panel = pickle.loads((OCT / "early_history_panel.pkl").read_bytes())
    for cutoff in OLD:
        p4 = fit_rank4(panel, cutoff, (7, 23))
        print(f"{cutoff} p4 shape {p4.shape} rowsum {p4.sum(axis=1).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
