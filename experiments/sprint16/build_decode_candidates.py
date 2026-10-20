"""Emit December exports for decode-only candidates (current product + v5, catdyn + v5).

Same probabilities as existing candidates where applicable; only the decode
vector changes. Files are probe candidates, not promoted by any local gate.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_saved import CLASSES, MULT, OCT, TOP10, OUT, rank4_combine, load_panel

BASELINE = OUT / "baseline_0309411451.csv"
TEST = "2025-12-01"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v5():
    payload = json.loads((OUT / "multiplier_transfer.json").read_text())
    for row in payload["rows"]:
        if row["vector"] == "v5":
            return np.array(row["values"], dtype=float)
    raise RuntimeError("v5 vector missing")


def main():
    v5 = load_v5()
    panel = load_panel()
    frame = panel["frames"][TEST]
    ids = frame.index.astype(str).to_numpy()
    p_cat = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
    r4 = np.load(TOP10 / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
    assert np.array_equal(r4["ids"].astype(str), ids)
    r4full = rank4_combine(p_cat, r4["proba"].astype(float), "full")

    baseline = pd.read_csv(BASELINE, dtype={"ID": str})
    current = baseline.Opportunity.to_numpy()
    candidate_a = pd.read_csv(OUT / "candidate_blend_2025-12-01.csv", dtype={"ID": str}).Opportunity.to_numpy()

    audit = {}
    for name, probs, vec in [("r4full_v5", r4full, v5), ("catdyn_v5", p_cat, v5)]:
        pred = np.asarray(CLASSES)[np.argmax(probs * vec, axis=1)]
        out = baseline.copy()
        out["Opportunity"] = pred
        path = OUT / f"candidate_{name}_2025-12-01.csv"
        out.to_csv(path, index=False)
        audit[name] = {
            "file": path.name,
            "sha256": sha256(path),
            "vector": [float(x) for x in vec],
            "changes_vs_baseline": int((pred != current).sum()),
            "changes_vs_candidate_blend": int((pred != candidate_a).sum()),
            "label_distribution": {c: int((pred == c).sum()) for c in CLASSES if (pred == c).sum()},
        }
        print(f"{name}: {audit[name]['changes_vs_baseline']} vs baseline, {audit[name]['changes_vs_candidate_blend']} vs blend candidate")

    (OUT / "decode_candidate_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "label_distribution"} for k, v in audit.items()}, indent=2))


if __name__ == "__main__":
    main()
