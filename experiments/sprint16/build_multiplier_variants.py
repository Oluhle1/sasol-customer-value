"""Emit December blend exports under alternative decode vectors (probe files).

Same probability blend as candidate_blend_2025-12-01.csv; only the multiplier
vector changes. Files are clearly audited and are NOT submissions until the
user chooses to upload one.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from evaluate_saved import CLASSES, MULT, OCT, TOP10, OUT, rank4_combine, load_panel  # noqa: E402

BASELINE = OUT / "baseline_0309411451.csv"
TEST = "2025-12-01"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_vectors():
    payload = json.loads((OUT / "multiplier_transfer.json").read_text())
    out = {}
    for row in payload["rows"]:
        out[row["vector"]] = np.array(row["values"], dtype=float)
    return out


def main():
    panel = load_panel()
    frame = panel["frames"][TEST]
    ids = frame.index.astype(str).to_numpy()
    p_cat = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
    p_early = np.load(OUT / f"{TEST}_early_history.npy").astype(float)
    r4 = np.load(TOP10 / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
    assert np.array_equal(r4["ids"].astype(str), ids)
    blend = 0.4 * p_cat + 0.4 * p_early + 0.2 * rank4_combine(p_cat, r4["proba"].astype(float), "full")

    vectors = load_vectors()
    baseline = pd.read_csv(BASELINE, dtype={"ID": str})
    base_pred = baseline.Opportunity.to_numpy()
    audit = {}
    for name, vec in vectors.items():
        if name == "accepted":
            continue
        pred = np.asarray(CLASSES)[np.argmax(blend * vec, axis=1)]
        out = baseline.copy()
        out["Opportunity"] = pred
        path = OUT / f"candidate_blend_mult_{name}.csv"
        out.to_csv(path, index=False)
        audit[name] = {
            "file": path.name,
            "sha256": sha256(path),
            "vector": [float(x) for x in vec],
            "vs_baseline_changes": int((pred != base_pred).sum()),
            "vs_candidate_blend_changes": None,
            "label_distribution": {c: int((pred == c).sum()) for c in CLASSES if (pred == c).sum()},
        }
        print(f"{name}: {audit[name]['vs_baseline_changes']} changes vs baseline -> {path.name}")

    acc = np.asarray(CLASSES)[np.argmax(blend * MULT, axis=1)]
    for name, rec in audit.items():
        pred = np.asarray(CLASSES)[np.argmax(blend * np.array(rec["vector"]), axis=1)]
        rec["vs_candidate_blend_changes"] = int((pred != acc).sum())
    (OUT / "multiplier_variant_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "label_distribution"} for k, v in audit.items()}, indent=2))


if __name__ == "__main__":
    main()
