"""Adoption subgroup tuning: split the 11 adoption classes into frequent/rare.

Pool-tuning protocol (Apr/May/Jul/Aug only). Start from vR; coordinate search
over ADOPT_HI, ADOPT_LO, then a re-pass over the original vR groups.
Folds evaluated afterward, never tuned.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
from evaluate_saved import CLASSES, OUT, rank4_combine, load_labels, load_proba, load_rank4  # noqa: E402

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]

HI = ["New category adoption: Other", "New category adoption: Food & beverage",
      "New category adoption: Lubricants", "New category adoption: Beverages"]
LO = [c for c in CLASSES if c.startswith("New category adoption") and c not in HI]

GROUPS = {
    "ADOPT_HI": (HI, [2.5, 3.0, 3.5, 4.0, 4.5]),
    "ADOPT_LO": (LO, [1.5, 2.0, 2.5, 3.0, 3.5]),
    "Stable": (["Stable"], [0.7, 0.8, 0.85, 0.9, 0.95]),
    "Inactivity": (["Inactivity"], [0.6, 0.7, 0.8, 0.9, 1.0]),
    "Existing-category growth: Fuel": (["Existing-category growth: Fuel"], [0.85, 1.0, 1.15, 1.3]),
    "Existing-category growth: Beverages": (["Existing-category growth: Beverages"], [1.6, 2.0, 2.4, 2.8]),
    "Existing-category growth: Food & beverage": (["Existing-category growth: Food & beverage"], [1.5, 2.0, 2.5, 3.0]),
    "Existing-category growth: Other": (["Existing-category growth: Other"], [1.0, 1.1, 1.3, 1.6]),
}


def idx_of(names):
    return [CLASSES.index(c) for c in names]


def wf1(y, probs, mult):
    return float(f1_score(y, np.asarray(CLASSES)[np.argmax(probs * mult, axis=1)], average="weighted", zero_division=0))


def load_origin(cutoff):
    ids, y = load_labels(cutoff)
    n = len(ids)
    p17 = load_proba("category_dynamics", cutoff, n)
    p4 = load_rank4(cutoff, n)
    assert p17 is not None and p4 is not None
    return rank4_combine(p17, p4, "full"), y


def main():
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    data = {c: load_origin(c) for c in POOL}
    base_scores = {c: wf1(y, p, vR) for c, (p, y) in data.items()}
    print("vR pool:", round(float(np.mean(list(base_scores.values()))), 6),
          {c: round(v, 6) for c, v in base_scores.items()}, flush=True)

    def objective(mult):
        return float(np.mean([wf1(y, p, mult) for p, y in data.values()]))

    mult = vR.copy()
    best = objective(mult)
    history = []
    for pas in range(3):
        improved = False
        for gname, (names, grid) in GROUPS.items():
            idx = idx_of(names)
            cands = []
            for v in grid:
                trial = mult.copy()
                trial[idx] = v
                cands.append((objective(trial), -abs(v - vR[idx[0]]), -abs(v), v, trial))
            cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            obj, _, _, v, trial = cands[0]
            if obj > best + 1e-12:
                history.append({"pass": pas, "group": gname, "from": float(mult[idx[0]]), "to": float(v), "pooled": obj})
                mult, best = trial, obj
                improved = True
                print(f"pass {pas} {gname}: {history[-1]['from']} -> {v} pooled={obj:.6f}", flush=True)
        if not improved:
            break

    print("tuned:", [round(float(x), 3) for x in mult])
    print("tuned pooled:", round(best, 6))
    scores = {c: wf1(y, p, mult) for c, (p, y) in data.items()}
    print({c: round(scores[c] - base_scores[c], 6) for c in POOL}, flush=True)

    results = {
        "base_vR_pooled": float(np.mean(list(base_scores.values()))),
        "tuned_vector": [float(x) for x in mult],
        "tuned_pooled": best,
        "pool_gains": {c: scores[c] - base_scores[c] for c in POOL},
        "history": history,
        "eval_folds": [],
    }
    for c in EVAL:
        ids, y = load_labels(c)
        n = len(ids)
        prod = rank4_combine(load_proba("category_dynamics", c, n), load_rank4(c, n), "full")
        row = {"origin": c, "vR": wf1(y, prod, vR), "tuned": wf1(y, prod, mult)}
        row["gain"] = row["tuned"] - row["vR"]
        results["eval_folds"].append(row)
        print(row, flush=True)

    (OUT / "adoption_split_search.json").write_text(json.dumps(results, indent=2))
    np.save(OUT / "vector_adopt_split.npy", mult)
    print("wrote", OUT / "adoption_split_search.json")


if __name__ == "__main__":
    main()
