"""Export the vR2 and rank4-early probes (December).

- r4full_vR2: same r4full product, Stable multiplier 0.85 -> 0.90 (extended grid).
- rank4early_vR: product 0.6 * r4full + 0.4 * early_history, vector vR.
Both were evaluated on June/September after the vectors/products were fixed on
non-evaluation origins.
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
    vR2 = np.array(json.load(open(OUT / "pool_r4full_vector_vR2.json"))["vR2"])
    panel = load_panel()
    frame = panel["frames"][TEST]
    ids = frame.index.astype(str).to_numpy()
    p_cat = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
    p_early = np.load(OUT / f"{TEST}_early_history.npy").astype(float)
    r4 = np.load(TOP10 / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
    assert np.array_equal(r4["ids"].astype(str), ids)
    r4full = rank4_combine(p_cat, r4["proba"].astype(float), "full")

    baseline = pd.read_csv(BASELINE, dtype={"ID": str})
    current = baseline.Opportunity.to_numpy()
    best = pd.read_csv(OUT / "candidate_r4full_vR_2025-12-01.csv", dtype={"ID": str}).Opportunity.to_numpy()

    entries = {
        "r4full_vR2": (r4full, vR2, {"2025-06-01": 0.001012, "2025-09-01": 0.000698}, "vs r4full+vR (public 0.313217559)"),
        "rank4early_vR": (0.6 * r4full + 0.4 * p_early, vR, {"2025-06-01": 0.003494, "2025-09-01": 0.000179}, "vs r4full+vR (public 0.313217559)"),
    }
    audit = {}
    for name, (probs, vec, gains, comparator) in entries.items():
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
            "changes_vs_r4full_vR": int((pred != best).sum()),
            "eval_gains_vs_r4full_vR": gains,
            "comparator": comparator,
            "label_distribution": {c: int((pred == c).sum()) for c in CLASSES if (pred == c).sum()},
        }
        print(f"{name}: {audit[name]['changes_vs_baseline']} vs baseline, {audit[name]['changes_vs_r4full_vR']} vs r4full+vR")
    (OUT / "vR2_probe_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "label_distribution"} for k, v in audit.items()}, indent=2))


if __name__ == "__main__":
    main()
