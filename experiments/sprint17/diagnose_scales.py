"""Unlabeled scales for the E2 grid. No labels, no June, no September."""
import json
import pickle

import numpy as np

from fit_weekly_rank4 import DEST, OCT, load_p17, load_production_p4
from evaluate_saved import CLASSES, rank4_combine

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]


def main():
    vR = np.asarray(json.loads((DEST.parent / "sprint16" / "pool_r4full_vector.json").read_text())["vR"])
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    rec, bg = [], []
    for cutoff in POOL:
        frame = panel["frames"][cutoff]
        ids = frame.index.astype(str).to_numpy()
        adjusted = rank4_combine(load_p17(cutoff, ids), load_production_p4(cutoff, ids), "full") * vR
        pred = np.asarray(CLASSES)[np.argmax(adjusted, axis=1)]
        chosen = pred == "Inactivity"
        rec.append(frame.loc[chosen, "fuel_recency"].to_numpy(dtype=float))
        bg.append(frame.loc[chosen, "fuel_bgnbd_z"].to_numpy(dtype=float))
    rec = np.concatenate(rec)
    bg = np.concatenate(bg)
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("predicted_inactivity", len(rec))
    print("recency", {q: round(float(np.quantile(rec, q)), 2) for q in qs})
    print("bgnbd", {q: round(float(np.quantile(bg, q)), 3) for q in qs})


if __name__ == "__main__":
    main()
