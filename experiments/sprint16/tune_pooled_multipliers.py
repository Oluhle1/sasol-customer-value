"""Pooled coordinate-ascent calibration of the frozen decision multiplier grid.

Protocol: outputs/sprint16/proposal_pooled_multipliers.json (pre-registered).
Tuning origins exclude the June/September evaluation folds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from evaluate_saved import CLASSES, MULT, OCT, TOP10, OUT, rank4_combine, load_rank4, load_panel

POOL = ["2024-12-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL_FOLDS = ["2025-06-01", "2025-09-01"]

GROUPS = {
    "Existing-category growth: Fuel": [0.7, 0.85, 1.0, 1.15, 1.3],
    "Stable": [0.6, 0.7, 0.8, 0.85, 0.95, 1.0],
    "Inactivity": [0.8, 0.9, 1.0, 1.1],
    "Existing-category growth: Beverages": [1.0, 1.3, 1.6, 2.0],
    "Existing-category growth: Food & beverage": [1.0, 1.5, 2.0, 3.0, 4.0],
    "Existing-category growth: Other": [1.0, 1.3, 1.6, 2.0],
    "ADOPTION": [1.5, 2.0, 2.5, 3.0, 4.0],
}


def group_indices(name):
    if name == "ADOPTION":
        return [i for i, c in enumerate(CLASSES) if c.startswith("New category adoption")]
    return [CLASSES.index(name)]


def wf1(y, probs, mult):
    return float(f1_score(y, np.asarray(CLASSES)[np.argmax(probs * mult, axis=1)], average="weighted", zero_division=0))


def load_origin(cutoff):
    panel = load_panel()
    if cutoff == "2024-12-01":
        z = np.load(sorted((OCT.parent / "next_round").glob("b838f9b6679e05b7_2024-12-01_Opportunity.npz"))[0], allow_pickle=True)
        p = np.asarray(z["prediction"], dtype=float)
        ids = z["ids"].astype(str)
    else:
        npz = TOP10 / "probs" / f"{cutoff}_category_dynamics.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=False)
            p = z["proba"].astype(float)
            ids = z["ids"].astype(str)
        else:
            frame = panel["frames"][cutoff]
            p = np.load(OCT / f"{cutoff}_category_dynamics.npy").astype(float)
            assert len(p) == len(frame)
            ids = frame.index.astype(str).to_numpy()
    lab = panel["labels"][cutoff]
    y = lab.reindex(ids)["Opportunity"].to_numpy()
    assert len(y) == len(p) and pd.notna(y).all(), cutoff
    return p, y


def main():
    data = {}
    for c in POOL:
        p, y = load_origin(c)
        data[c] = (p, y)
        print(f"pool origin {c}: rows={len(y)}", flush=True)

    accepted = MULT.copy()
    base_scores = {c: wf1(y, p, accepted) for c, (p, y) in data.items()}
    print("accepted pooled mean:", round(np.mean(list(base_scores.values())), 6))
    print({c: round(v, 5) for c, v in base_scores.items()})

    def objective(mult):
        return float(np.mean([wf1(y, p, mult) for p, y in data.values()]))

    mult = accepted.copy()
    best_obj = objective(mult)
    history = []
    for pas in range(3):
        improved = False
        for gname, grid in GROUPS.items():
            idx = group_indices(gname)
            cands = []
            for v in grid:
                trial = mult.copy()
                trial[idx] = v
                obj = objective(trial)
                tie = abs(v - accepted[idx[0]])
                cands.append((obj, -tie, -abs(v), v, trial))
            cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            obj, _, _, v, trial = cands[0]
            cur_v = mult[idx[0]]
            if obj > best_obj + 1e-12:
                mult = trial
                best_obj = obj
                improved = True
                history.append({"pass": pas, "group": gname, "from": float(cur_v), "to": float(v), "pooled": obj})
                print(f"pass {pas} {gname}: {cur_v} -> {v}  pooled={obj:.6f}", flush=True)
        if not improved:
            break

    print("tuned:", [round(float(x), 3) for x in mult])
    print("tuned pooled mean:", round(best_obj, 6))

    # Evaluation folds (never in objective): catdyn decode
    rows = []
    tuned_eval = {}
    accepted_eval = {}
    for c in EVAL_FOLDS:
        p, y = load_origin(c)
        tuned_eval[c] = wf1(y, p, mult)
        accepted_eval[c] = wf1(y, p, accepted)
        rows.append({"origin": c, "accepted": accepted_eval[c], "tuned": tuned_eval[c], "gain": tuned_eval[c] - accepted_eval[c]})
        print(c, "accepted", round(accepted_eval[c], 6), "tuned", round(tuned_eval[c], 6), "gain", round(tuned_eval[c] - accepted_eval[c], 6))

    g1 = all(r["gain"] >= 0 for r in rows) and any(r["gain"] > 0 for r in rows)

    # G2: same comparison with the registered 2:2:1 blend
    from evaluate_saved import load_proba, load_labels
    blend_rows = []
    for c in EVAL_FOLDS:
        ids, yy = load_labels(c)
        n = len(ids)
        p_cat = load_proba("category_dynamics", c, n)
        p_early = load_proba("early_history", c, n)
        r4 = load_rank4(c, n)
        blend = 0.4 * p_cat + 0.4 * p_early + 0.2 * rank4_combine(p_cat, r4, "full")
        rec = {"origin": c, "blend_accepted": wf1(yy, blend, accepted), "blend_tuned": wf1(yy, blend, mult)}
        rec["gain"] = rec["blend_tuned"] - rec["blend_accepted"]
        blend_rows.append(rec)
        print(c, "blend accepted", round(rec["blend_accepted"], 6), "blend tuned", round(rec["blend_tuned"], 6), "gain", round(rec["gain"], 6))
    g2 = all(r["gain"] >= 0 for r in blend_rows) and any(r["gain"] > 0 for r in blend_rows)

    result = {
        "pool": POOL,
        "accepted_pooled_mean": float(np.mean(list(base_scores.values()))),
        "base_scores": base_scores,
        "tuned_multipliers": [float(x) for x in mult],
        "tuned_pooled_mean": best_obj,
        "history": history,
        "eval_folds_catdyn": rows,
        "eval_folds_blend": blend_rows,
        "G1_pass": bool(g1),
        "G2_pass": bool(g2),
        "accepted_pass": bool(g1 and g2),
    }
    (OUT / "pooled_multiplier_decision.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(OUT / "pooled_multiplier_eval_catdyn.csv", index=False)
    pd.DataFrame(blend_rows).to_csv(OUT / "pooled_multiplier_eval_blend.csv", index=False)
    print("\nG1", g1, "G2", g2)
    print("wrote", OUT / "pooled_multiplier_decision.json")


if __name__ == "__main__":
    main()
