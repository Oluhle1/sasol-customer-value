"""E10: CLV blend-weight tuning (pre-registered in proposal_clv_ensemble.json).

Read-only over cached arrays; no model fitting. Tunes on the 4 pool origins,
evaluates June/September, and (only on pass) reports the December blends.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
from evaluate_saved import OCT, OUT  # noqa: E402

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
ORIGINS = POOL + EVAL

FUEL_MEMBERS = ["CLV_fuel_category_values", "CLV_fuel_two_part", "fuel_residual", "original_fuel_recipe"]
NF_MEMBERS = ["CLV_nonfuel_category_values", "CLV_nonfuel_two_part"]
FUEL_ANCHOR = np.array([0.40, 0.35, 0.25, 0.00])
NF_ANCHOR = np.array([0.0, 1.0])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_members(origin, members):
    return [np.load(OCT / f"{origin}_{m}.npy").astype(float) for m in members]


def simplex_grid(k, step=0.05):
    n = int(round(1 / step))
    out = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(prefix + [remaining])
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, slots - 1)

    rec([], n, k)
    return [np.array(g, dtype=float) / n for g in out]


def main():
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    labels = {c: panel["labels"][c].reindex(panel["frames"][c].index) for c in ORIGINS}

    # sanity: reproduce the shipped December columns from the member arrays
    baseline = pd.read_csv(OUT / "baseline_0309411451.csv")
    ef = OUT / "dec_fuel_from_members.npy"
    if (OCT / "2025-12-01_CLV_fuel_category_values.npy").exists():
        cv = np.load(OCT / "2025-12-01_CLV_fuel_category_values.npy").astype(float)
        tp = np.load(OCT / "2025-12-01_CLV_fuel_two_part.npy").astype(float)
        fr = np.load(OCT / "2025-12-01_fuel_residual.npy").astype(float)
        dec_fuel = 0.40 * cv + 0.35 * tp + 0.25 * fr
        print("Dec fuel reproduction diff (max abs):", float(np.max(np.abs(dec_fuel - baseline.CLV_fuel.to_numpy()))))
        nfc = np.load(OCT / "2025-12-01_CLV_nonfuel_two_part.npy").astype(float)
        print("Dec nonfuel (two_part) diff vs shipped:", float(np.max(np.abs(nfc - baseline.CLV_nonfuel.to_numpy()))))

    results = {}
    for tag, members, anchor in [("fuel", FUEL_MEMBERS, FUEL_ANCHOR), ("nonfuel", NF_MEMBERS, NF_ANCHOR)]:
        target = "CLV_fuel" if tag == "fuel" else "CLV_nonfuel"
        arrs = {c: load_members(c, members) for c in ORIGINS}
        ys = {c: labels[c][target].to_numpy() for c in ORIGINS}

        def pooled(w):
            return float(np.mean([rmse(sum(wi * a for wi, a in zip(w, arrs[c])), ys[c]) for c in POOL]))

        def evalfold(w):
            return {c: rmse(sum(wi * a for wi, a in zip(w, arrs[c])), ys[c]) for c in EVAL}

        anchor_scores = {c: rmse(sum(wi * a for wi, a in zip(anchor, arrs[c])), ys[c]) for c in POOL}
        anchor_pooled = float(np.mean(list(anchor_scores.values())))
        best_w, best_pooled = anchor.copy(), anchor_pooled
        grid = simplex_grid(len(members))
        for w in grid:
            p = pooled(w)
            if p < best_pooled - 1e-9:
                best_pooled, best_w = p, w
        gains = {c: rmse(sum(wi * a for wi, a in zip(best_w, arrs[c])), ys[c]) - anchor_scores[c] for c in POOL}
        fold_gains = {c: evalfold(best_w)[c] - rmse(sum(wi * a for wi, a in zip(anchor, arrs[c])), ys[c]) for c in EVAL}
        print(f"\n{tag}: anchor {[round(float(x),3) for x in anchor]} pooled {anchor_pooled:.6f}")
        print(f"{tag}: best   {[round(float(x),3) for x in best_w]} pooled {best_pooled:.6f}")
        print(f"{tag} tuning gains (negative = better):", {c: round(v, 6) for c, v in gains.items()})
        print(f"{tag} fold gains (negative = better):", {c: round(v, 6) for c, v in fold_gains.items()})
        g1 = (anchor_pooled - best_pooled) >= 0.0010 and max(gains.values()) <= 0.0010
        g2 = g1 and all(-v >= 0.0005 for v in fold_gains.values())
        print(f"{tag} G1 {g1}  G2 {g2}")
        results[tag] = {
            "members": members, "anchor": [float(x) for x in anchor], "best_weights": [float(x) for x in best_w],
            "anchor_pooled": anchor_pooled, "best_pooled": best_pooled,
            "pool_gains": {c: float(v) for c, v in gains.items()},
            "fold_gains": {c: float(v) for c, v in fold_gains.items()},
            "G1_pass": bool(g1), "G2_pass": bool(g2),
        }
    (OUT / "clv_ensemble_result.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "clv_ensemble_result.json")


if __name__ == "__main__":
    main()
