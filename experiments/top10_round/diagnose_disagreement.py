"""How often another saved model already has the label the main head ranks second.

Diagnostic only. It does not choose a submission.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "oct10"
DEST = ROOT / "outputs" / "top10_round"
decision = json.loads((OUT / "category_dynamics_decision.json").read_text())
classes = np.asarray(decision["classes"])
mult = np.asarray(decision["multipliers"], dtype=float)
panel = __import__("pickle").loads((OUT / "category_dynamics_panel.pkl").read_bytes())


def labels_of(proba):
    return classes[np.argmax(proba * mult, axis=1)]


def main():
    rows = []
    for cutoff in ["2025-06-01", "2025-09-01"]:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
        main_p = np.load(OUT / f"{cutoff}_category_dynamics.npy")
        order = np.argsort(-(main_p * mult), axis=1)
        rank1 = classes[order[:, 0]]
        rank2 = classes[order[:, 1]]
        others = {}
        for name in ["monthly", "cat", "group_specialist", "early_history", "station_context"]:
            path = OUT / f"{cutoff}_{name}.npy"
            if not path.exists():
                continue
            proba = np.load(path)
            if proba.ndim != 2 or proba.shape[1] != 17:
                print(cutoff, name, "skip shape", proba.shape)
                continue
            others[name] = labels_of(proba)
        base = float(f1_score(y, rank1, average="weighted", zero_division=0))
        fixable = (rank1 != y) & (rank2 == y)
        print(cutoff, "fixable_second", int(fixable.sum()), "of", len(y))
        for name, pred in others.items():
            agree_fix = fixable & (pred == y)
            damage = (rank1 == y) & (pred != y)
            vote = rank1.copy()
            vote[pred == rank2] = rank2[pred == rank2]
            vote_f1 = float(f1_score(y, vote, average="weighted", zero_division=0))
            rows.append({
                "cutoff": cutoff,
                "other": name,
                "baseline_f1": base,
                "fixable_second": int(fixable.sum()),
                "other_has_true_on_fixable": int(agree_fix.sum()),
                "other_breaks_correct_rank1": int(damage.sum()),
                "vote_when_other_equals_rank2_f1": vote_f1,
                "vote_gain": vote_f1 - base,
            })
            print(rows[-1])
    pd.DataFrame(rows).to_csv(DEST / "disagreement.csv", index=False)


if __name__ == "__main__":
    main()
