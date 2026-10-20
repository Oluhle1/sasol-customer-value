"""Headroom diagnostic on saved October 10 predictions.

Does not fit a candidate and does not tune on June or September.
Fuel blend weights, when searched, are chosen only on April and May.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "oct10"
DEST = ROOT / "outputs" / "top10_round"
DEST.mkdir(parents=True, exist_ok=True)

CLASSES = json.loads((OUT / "category_dynamics_decision.json").read_text())["classes"]
MULT = np.asarray(json.loads((OUT / "category_dynamics_decision.json").read_text())["multipliers"], dtype=float)
FUEL_I = CLASSES.index("Existing-category growth: Fuel")
INACT_I = CLASSES.index("Inactivity")
STABLE_I = CLASSES.index("Stable")
ADOPT = [i for i, name in enumerate(CLASSES) if name.startswith("New category adoption")]


def weighted_f1(y, pred):
    labels = sorted(set(y) | set(pred))
    return float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def opportunity_oracle(panel, cutoff):
    proba = np.load(OUT / f"{cutoff}_category_dynamics.npy")
    y = panel["labels"][cutoff].reindex(panel["frames"][cutoff].index).Opportunity.to_numpy()
    assert len(y) == len(proba)
    adjusted = proba * MULT
    order = np.argsort(-adjusted, axis=1)
    pred = np.asarray(CLASSES)[order[:, 0]]
    second = np.asarray(CLASSES)[order[:, 1]]
    true_index = pd.Series(np.arange(len(CLASSES)), index=CLASSES).reindex(y).to_numpy()
    rank = (order == true_index[:, None]).argmax(axis=1)
    base = weighted_f1(y, pred)
    top2 = pred.copy()
    top2[second == y] = y[second == y]
    three = pred.copy()
    mask3 = np.isin(y, [CLASSES[FUEL_I], CLASSES[INACT_I], CLASSES[STABLE_I]]) & np.isin(
        pred, [CLASSES[FUEL_I], CLASSES[INACT_I], CLASSES[STABLE_I]]
    )
    three[mask3] = y[mask3]
    adopt = pred.copy()
    adopt[np.isin(true_index, ADOPT)] = y[np.isin(true_index, ADOPT)]
    fuel_inact = pred.copy()
    swap = np.isin(y, [CLASSES[FUEL_I], CLASSES[INACT_I]]) & np.isin(pred, [CLASSES[FUEL_I], CLASSES[INACT_I]])
    fuel_inact[swap] = y[swap]
    fuel_stable = pred.copy()
    swap2 = np.isin(y, [CLASSES[FUEL_I], CLASSES[STABLE_I]]) & np.isin(pred, [CLASSES[FUEL_I], CLASSES[STABLE_I]])
    fuel_stable[swap2] = y[swap2]
    margin = adjusted[np.arange(len(y)), order[:, 0]] - adjusted[np.arange(len(y)), order[:, 1]]
    wrong = pred != y
    return {
        "cutoff": cutoff,
        "rows": int(len(y)),
        "baseline_f1": base,
        "top2_oracle_f1": weighted_f1(y, top2),
        "top2_gain": weighted_f1(y, top2) - base,
        "fuel_inactivity_swap_oracle_f1": weighted_f1(y, fuel_inact),
        "fuel_inactivity_gain": weighted_f1(y, fuel_inact) - base,
        "fuel_stable_swap_oracle_f1": weighted_f1(y, fuel_stable),
        "fuel_stable_gain": weighted_f1(y, fuel_stable) - base,
        "three_class_oracle_f1": weighted_f1(y, three),
        "three_class_gain": weighted_f1(y, three) - base,
        "adoption_oracle_f1": weighted_f1(y, adopt),
        "adoption_gain": weighted_f1(y, adopt) - base,
        "true_rank1_share": float((rank == 0).mean()),
        "true_rank2_share": float((rank == 1).mean()),
        "true_rank3plus_share": float((rank >= 2).mean()),
        "wrong_margin_median": float(np.median(margin[wrong])) if wrong.any() else None,
        "wrong_margin_p25": float(np.quantile(margin[wrong], 0.25)) if wrong.any() else None,
        "errors_margin_below_0.05": int(((margin < 0.05) & wrong).sum()),
        "errors": int(wrong.sum()),
    }


def load_fuel_parts(cutoff):
    direct = np.load(OUT / f"{cutoff}_CLV_fuel_category_values.npy")
    hurdle = np.load(OUT / f"{cutoff}_CLV_fuel_two_part.npy")
    residual = np.load(OUT / f"{cutoff}_fuel_residual.npy")
    return direct, hurdle, residual


def fuel_router(panel):
    bins = [0, 7, 30, 90, 10000]
    labels = ["0-7", "7-30", "30-90", "90+"]
    grid = [(a / 10, b / 10, c / 10) for a in range(11) for b in range(11 - a) for c in [10 - a - b]]
    rows = []
    parts = {}
    for cutoff in ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_fuel.to_numpy()
        direct, hurdle, residual = load_fuel_parts(cutoff)
        assert len(y) == len(direct) == len(hurdle) == len(residual)
        recency = frame["fuel_recency"].to_numpy()
        bucket = pd.cut(recency, bins, labels=labels, right=True).astype(str)
        parts[cutoff] = (y, direct, hurdle, residual, bucket)
        current = 0.40 * direct + 0.35 * hurdle + 0.25 * residual
        oracle = np.minimum(np.minimum((y - direct) ** 2, (y - hurdle) ** 2), (y - residual) ** 2)
        rows.append({
            "cutoff": cutoff,
            "current_rmse": rmse(y, current),
            "best_single_direct": rmse(y, direct),
            "best_single_hurdle": rmse(y, hurdle),
            "best_single_residual": rmse(y, residual),
            "row_oracle_rmse": float(np.sqrt(oracle.mean())),
        })
    tune_y, tune_d, tune_h, tune_r, tune_b = [], [], [], [], []
    for cutoff in ["2025-04-01", "2025-05-01"]:
        y, d, h, r, b = parts[cutoff]
        tune_y.append(y); tune_d.append(d); tune_h.append(h); tune_r.append(r); tune_b.append(b)
    tune_y = np.concatenate(tune_y); tune_d = np.concatenate(tune_d)
    tune_h = np.concatenate(tune_h); tune_r = np.concatenate(tune_r); tune_b = np.concatenate(tune_b)
    chosen = {}
    for bucket in labels:
        mask = tune_b == bucket
        best, best_w = 1e9, (0.4, 0.35, 0.25)
        for w in grid:
            pred = w[0] * tune_d[mask] + w[1] * tune_h[mask] + w[2] * tune_r[mask]
            err = rmse(tune_y[mask], pred)
            if err < best:
                best, best_w = err, w
        chosen[bucket] = {"weights": best_w, "april_may_rmse": best, "n": int(mask.sum())}
    eval_rows = []
    for cutoff in ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]:
        y, d, h, r, b = parts[cutoff]
        current = 0.40 * d + 0.35 * h + 0.25 * r
        routed = np.empty_like(current)
        for bucket, spec in chosen.items():
            w = spec["weights"]
            mask = b == bucket
            routed[mask] = w[0] * d[mask] + w[1] * h[mask] + w[2] * r[mask]
        eval_rows.append({
            "cutoff": cutoff,
            "current_rmse": rmse(y, current),
            "routed_rmse": rmse(y, routed),
            "gain": rmse(y, current) - rmse(y, routed),
            "rows": int(len(y)),
        })
    return {"component_ceiling": rows, "weights_fit_on_april_may": chosen, "later_months": eval_rows}


def main():
    panel = pickle.loads((OUT / "category_dynamics_panel.pkl").read_bytes())
    opp = [opportunity_oracle(panel, c) for c in ["2025-06-01", "2025-09-01"]]
    fuel = fuel_router(panel)
    report = {"opportunity_oracles": opp, "fuel_router": fuel}
    (DEST / "headroom.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
