"""Which current features still separate production errors on the pool.

Discovery is April and May. July and August only confirm the same direction.
June and September are not read.
"""
import json
import pickle

import numpy as np
import pandas as pd

from fit_weekly_rank4 import DEST, OCT, load_p17, load_production_p4
from evaluate_saved import CLASSES, rank4_combine

DISCOVER = ["2025-04-01", "2025-05-01"]
CONFIRM = ["2025-07-01", "2025-08-01"]
FUEL = "Existing-category growth: Fuel"


def collect(panel, cutoffs, vR):
    frames = []
    for cutoff in cutoffs:
        frame = panel["frames"][cutoff]
        ids = frame.index.astype(str).to_numpy()
        y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
        adjusted = rank4_combine(load_p17(cutoff, ids), load_production_p4(cutoff, ids), "full") * vR
        pred = np.asarray(CLASSES)[np.argmax(adjusted, axis=1)]
        block = frame[panel["feature_cols"]].copy()
        block["y"] = y
        block["pred"] = pred
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def cohens_d(frame, feature, left, right):
    a = frame.loc[left, feature].to_numpy(dtype=float)
    b = frame.loc[right, feature].to_numpy(dtype=float)
    if len(a) < 30 or len(b) < 30:
        return None
    pooled = np.sqrt(((a.std() ** 2) + (b.std() ** 2)) / 2.0)
    if pooled < 1e-8:
        return None
    return float((a.mean() - b.mean()) / pooled)


def main():
    vR = np.asarray(json.loads((DEST.parent / "sprint16" / "pool_r4full_vector.json").read_text())["vR"])
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    discover = collect(panel, DISCOVER, vR)
    confirm = collect(panel, CONFIRM, vR)
    contrasts = {
        "fuel_called_stable_vs_true_stable": (
            (discover.y.eq(FUEL) & discover.pred.eq("Stable")),
            (discover.y.eq("Stable") & discover.pred.eq("Stable")),
            (confirm.y.eq(FUEL) & confirm.pred.eq("Stable")),
            (confirm.y.eq("Stable") & confirm.pred.eq("Stable")),
        ),
        "stable_called_fuel_vs_true_fuel": (
            (discover.y.eq("Stable") & discover.pred.eq(FUEL)),
            (discover.y.eq(FUEL) & discover.pred.eq(FUEL)),
            (confirm.y.eq("Stable") & confirm.pred.eq(FUEL)),
            (confirm.y.eq(FUEL) & confirm.pred.eq(FUEL)),
        ),
        "fuel_called_inactive_vs_true_inactive": (
            (discover.y.eq(FUEL) & discover.pred.eq("Inactivity")),
            (discover.y.eq("Inactivity") & discover.pred.eq("Inactivity")),
            (confirm.y.eq(FUEL) & confirm.pred.eq("Inactivity")),
            (confirm.y.eq("Inactivity") & confirm.pred.eq("Inactivity")),
        ),
    }
    rows = []
    for name, (d_left, d_right, c_left, c_right) in contrasts.items():
        scored = []
        for feature in panel["feature_cols"]:
            effect = cohens_d(discover, feature, d_left, d_right)
            if effect is None:
                continue
            scored.append((abs(effect), effect, feature))
        scored.sort(reverse=True)
        print(f"\n{name} discover n {int(d_left.sum())} vs {int(d_right.sum())}", flush=True)
        for _mag, effect, feature in scored[:12]:
            confirm_effect = cohens_d(confirm, feature, c_left, c_right)
            same = confirm_effect is not None and effect * confirm_effect > 0 and abs(confirm_effect) >= 0.15
            confirm_text = "na" if confirm_effect is None else f"{confirm_effect:+.3f}"
            print(f"  {feature:28s} d_disc {effect:+.3f} d_conf {confirm_text} {'KEEP' if same else ''}", flush=True)
            rows.append({
                "contrast": name, "feature": feature,
                "d_discover": effect, "d_confirm": confirm_effect, "same_direction": bool(same),
            })
    pd.DataFrame(rows).to_csv(DEST / "residual_contrasts_pool.csv", index=False)


if __name__ == "__main__":
    main()
