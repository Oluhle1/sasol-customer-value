"""Pool-only check that the weekly block is aligned and separates the big three.

Does not fit a model and does not read June or September.
"""
import json
import pickle

import numpy as np
import pandas as pd

from fit_weekly_rank4 import (
    DEST, OCT, WEEKLY, feature_names, load_production_p4, load_p17, load_proposal,
)
from evaluate_saved import CLASSES, rank4_combine

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
FUEL = "Existing-category growth: Fuel"
GROUPS = ["Inactivity", "Stable", FUEL]


def main():
    proposal = load_proposal()
    names = feature_names(proposal)
    vR = np.asarray(json.loads((DEST.parent / "sprint16" / "pool_r4full_vector.json").read_text())["vR"])
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    rows = []
    for cutoff in POOL:
        frame = panel["frames"][cutoff]
        ids = frame.index.astype(str).to_numpy()
        y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
        payload = np.load(WEEKLY / f"{cutoff}.npz", allow_pickle=False)
        x = payload["X"]
        got = payload["names"].astype(str).tolist()
        gap = x[:, got.index("wk_fuel_runrate_gap")]
        last4 = x[:, got.index("wk_fuel_r_last4")]
        shape = x[:, got.index("wk_fuel_r_shape")]
        q1 = frame["q1_fuel_r"].to_numpy()
        # Weeks 0-12 cover about one quarter. Compare their signed-log mean with log quarterly rand.
        week_cols = [got.index(f"wk{k:02d}_fuel_r") for k in range(13)]
        week_level = x[:, week_cols].mean(axis=1)
        corr = float(np.corrcoef(week_level, np.log1p(np.clip(q1, 0, None)))[0, 1])
        p17 = load_p17(cutoff, ids)
        p4 = load_production_p4(cutoff, ids)
        adjusted = rank4_combine(p17, p4, "full") * vR
        order = np.argsort(-adjusted, axis=1)
        pred = np.asarray(CLASSES)[order[:, 0]]
        margin = adjusted[np.arange(len(y)), order[:, 0]] - adjusted[np.arange(len(y)), order[:, 1]]
        wrong = pred != y
        pair = {}
        for a in GROUPS:
            for b in GROUPS:
                if a >= b:
                    continue
                mask = wrong & (((y == a) & (pred == b)) | ((y == b) & (pred == a)))
                pair[f"{a[:6]}-{b[:6]}"] = int(mask.sum())
        summary = {"cutoff": cutoff, "rows": int(len(y)), "week_q1_corr": round(corr, 4), "errors": int(wrong.sum())}
        summary.update(pair)
        slices = {
            "fuel_as_stable": (y == FUEL) & (pred == "Stable"),
            "stable_as_fuel": (y == "Stable") & (pred == FUEL),
            "fuel_as_inact": (y == FUEL) & (pred == "Inactivity"),
            "inact_as_fuel": (y == "Inactivity") & (pred == FUEL),
        }
        for key, mask in slices.items():
            summary[f"{key}_n"] = int(mask.sum())
            summary[f"{key}_shape"] = round(float(shape[mask].mean()), 4) if mask.any() else None
            summary[f"{key}_last4"] = round(float(last4[mask].mean()), 4) if mask.any() else None
            summary[f"{key}_margin"] = round(float(np.median(margin[mask])), 4) if mask.any() else None
        for name, column in [("gap", gap), ("last4", last4), ("shape", shape)]:
            for label in GROUPS:
                mask = y == label
                summary[f"{name}_{label.split()[-1][:6]}"] = round(float(column[mask].mean()), 4)
        rows.append(summary)
        print(json.dumps(summary), flush=True)
    pd.DataFrame(rows).to_csv(DEST / "weekly_signal_pool.csv", index=False)


if __name__ == "__main__":
    main()
