"""Cross-origin transfer diagnostic for decision multipliers (exploratory).

Complement to the rejected pooled-tuning experiment (proposal_pooled_multipliers.json).
All four vectors are compared on every origin. June/September are reported for
transparency; they were never used in any objective here.

Variants:
  accepted : the frozen category-dynamics vector
  uniform  : all ones
  vD       : coordinate-ascent tuned on 2024-12 alone (target-season analog)
  v5       : coordinate-ascent tuned on 2025-03/04/05/07/08 (pool without December)
"""
import json

import numpy as np
import pandas as pd

from evaluate_saved import CLASSES, MULT, OCT, TOP10, OUT, load_panel
from tune_pooled_multipliers import GROUPS, group_indices, wf1

POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
DEC24 = "2024-12-01"
EVAL_FOLDS = ["2025-06-01", "2025-09-01"]
ALL = [DEC24] + POOL + EVAL_FOLDS


def load_all():
    panel = load_panel()
    data = {}
    for c in ALL:
        if c == "2024-12-01":
            z = np.load(sorted((OCT.parent / "next_round").glob("b838f9b6679e05b7_2024-12-01_Opportunity.npz"))[0], allow_pickle=True)
            p = np.asarray(z["prediction"], dtype=float)
            ids = z["ids"].astype(str)
        else:
            npz = TOP10 / "probs" / f"{c}_category_dynamics.npz"
            if npz.exists():
                z = np.load(npz, allow_pickle=False)
                p = z["proba"].astype(float)
                ids = z["ids"].astype(str)
            else:
                frame = panel["frames"][c]
                p = np.load(OCT / f"{c}_category_dynamics.npy").astype(float)
                ids = frame.index.astype(str).to_numpy()
        y = panel["labels"][c].reindex(ids)["Opportunity"].to_numpy()
        assert len(y) == len(p) and pd.notna(y).all(), c
        data[c] = (p, y)
    return data


def tune(data, origins, init=None):
    def objective(mult):
        return float(np.mean([wf1(data[c][1], data[c][0], mult) for c in origins]))

    mult = (MULT.copy() if init is None else init.copy())
    best = objective(mult)
    for _ in range(3):
        improved = False
        for gname, grid in GROUPS.items():
            idx = group_indices(gname)
            for v in grid:
                trial = mult.copy()
                trial[idx] = v
                obj = objective(trial)
                if obj > best + 1e-12:
                    mult, best = trial, obj
                    improved = True
        if not improved:
            break
    return mult, best


def main():
    data = load_all()
    print("origins:", {c: len(data[c][1]) for c in ALL})

    vD, objD = tune(data, [DEC24])
    v5, obj5 = tune(data, POOL)
    vectors = {
        "accepted": MULT.copy(),
        "uniform": np.ones(17),
        "vD": vD,
        "v5": v5,
    }
    rows = []
    for name, vec in vectors.items():
        row = {"vector": name, "values": [round(float(x), 3) for x in vec]}
        for c in ALL:
            p, y = data[c]
            row[c] = round(wf1(y, p, vec), 6)
        row["pool5_mean"] = round(np.mean([row[c] for c in POOL]), 6)
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(OUT / "multiplier_transfer.csv", index=False)
    (OUT / "multiplier_transfer.json").write_text(json.dumps({"rows": rows, "vD_objective": objD, "v5_objective": obj5}, indent=2))
    print("wrote", OUT / "multiplier_transfer.json")


if __name__ == "__main__":
    main()
