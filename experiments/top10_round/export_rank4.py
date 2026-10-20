"""Export the full four-class ranking even though its mean-gain gate failed.

The measured comparison is already in rank_comparison.csv. This script does
not refit the evaluation models or change the recorded decision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "top10_round"))
import run_rank4 as rank

DEST = rank.DEST


def class_table(y, pred, classes):
    rows = []
    for name in classes:
        yt = (y == name).astype(int)
        yp = (pred == name).astype(int)
        rows.append({
            "label": name,
            "support": int(yt.sum()),
            "predicted": int(yp.sum()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        })
    return pd.DataFrame(rows)


def main():
    classes, multipliers = rank.load_decision()
    panel = __import__("pickle").loads((rank.OCT / "category_dynamics_panel.pkl").read_bytes())
    cutoff = "2025-09-01"
    frame = panel["frames"][cutoff]
    y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
    p17 = rank.load_proba(cutoff, frame.index)
    p4 = np.load(DEST / f"{cutoff}_rank4.npz")["proba"]
    base = rank.labels_from(p17, classes, multipliers)
    full = rank.labels_from(rank.combine(p17, p4, classes, "full"), classes, multipliers)
    before = class_table(y, base, classes).rename(columns={"predicted": "baseline_predicted", "f1": "baseline_f1"})
    after = class_table(y, full, classes).rename(columns={"predicted": "rank_predicted", "f1": "rank_f1"})
    table = before.merge(after.drop(columns=["support"]), on="label")
    table["f1_change"] = table.rank_f1 - table.baseline_f1
    table.to_csv(DEST / "september_class_changes.csv", index=False)
    name, changes = rank.export_rank(panel, classes, multipliers, "full")
    audit = {
        "experimental": True,
        "gate_passed": False,
        "reason": "Five of six origins improved and September gained 0.003848 weighted F1, but the mean of June through September was 0.001121, below the pre-registered 0.004.",
        "file": name,
        "opportunity_changes": changes,
        "clv_unchanged": True,
    }
    (DEST / "experimental_export.json").write_text(json.dumps(audit, indent=2))
    print(table.sort_values("f1_change", ascending=False).to_string(index=False))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
