"""Export December probes with the pool-tuned, product-matched vector vR.

vR was tuned by coordinate ascent on four non-evaluation pool origins
(2025-04/05/07/08) using the r4full product, then evaluated (not tuned) on the
June/September folds. Pre-registration: proposal_vR.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_saved import CLASSES, OCT, TOP10, OUT, rank4_combine, load_panel

BASELINE = OUT / "baseline_0309411451.csv"
TEST = "2025-12-01"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    panel = load_panel()
    frame = panel["frames"][TEST]
    ids = frame.index.astype(str).to_numpy()
    p_cat = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
    p_early = np.load(OUT / f"{TEST}_early_history.npy").astype(float)
    r4 = np.load(TOP10 / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
    assert np.array_equal(r4["ids"].astype(str), ids)
    r4full = rank4_combine(p_cat, r4["proba"].astype(float), "full")
    blend = 0.4 * p_cat + 0.4 * p_early + 0.2 * r4full

    baseline = pd.read_csv(BASELINE, dtype={"ID": str})
    current = baseline.Opportunity.to_numpy()
    shipped = pd.read_csv(OUT / "candidate_blend_2025-12-01.csv", dtype={"ID": str}).Opportunity.to_numpy()

    products = {
        "blend_vR": blend,
        "r4full_vR": r4full,
        "catdyn_vR": p_cat,
    }
    gains = {
        "blend_vR": {"2025-06-01": 0.001792, "2025-09-01": 0.005154},
        "r4full_vR": {"2025-06-01": 0.001472, "2025-09-01": 0.005774},
        "catdyn_vR": {"2025-06-01": 0.005195, "2025-09-01": 0.004792},
    }
    audit = {}
    for name, probs in products.items():
        pred = np.asarray(CLASSES)[np.argmax(probs * vR, axis=1)]
        out = baseline.copy()
        out["Opportunity"] = pred
        path = OUT / f"candidate_{name}_2025-12-01.csv"
        out.to_csv(path, index=False)
        audit[name] = {
            "file": path.name,
            "sha256": sha256(path),
            "vector": [float(x) for x in vR],
            "changes_vs_baseline": int((pred != current).sum()),
            "changes_vs_shipped_blend": int((pred != shipped).sum()),
            "label_distribution": {c: int((pred == c).sum()) for c in CLASSES if (pred == c).sum()},
            "eval_gains_vs_accepted": gains[name],
        }
        print(f"{name}: {audit[name]['changes_vs_baseline']} vs baseline, {audit[name]['changes_vs_shipped_blend']} vs shipped blend")
    (OUT / "vR_candidate_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "label_distribution"} for k, v in audit.items()}, indent=2))


if __name__ == "__main__":
    main()
