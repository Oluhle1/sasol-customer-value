"""Build the December blend candidate submission.

Recipe (pre-registered in outputs/sprint16/proposal_december_blend.json):
  blend = 0.4 * catdyn + 0.4 * early_history + 0.2 * catdyn+rank4_full
  decode = argmax(blend * accepted_multipliers)

Guards:
  - rank4_full decode of December catdyn probs must reproduce the current
    baseline Opportunity labels exactly (pipeline replication check);
  - CLV columns are copied byte-for-byte from the baseline CSV;
  - 5488 rows, official order, finite labels.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
from evaluate_saved import CLASSES, MULT, OCT, TOP10, OUT, rank4_combine, load_panel  # noqa: E402

BASELINE = OUT / "baseline_0309411451.csv"
EXPECTED_BASELINE_SHA = "ee559dacc990459bfe5603812515893d52e4357cb79eadad667fe226c2932a88"
TEST = "2025-12-01"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_table(y_true, y_pred):
    rows = []
    for name in CLASSES:
        yt = (y_true == name).astype(int)
        yp = (y_pred == name).astype(int)
        rows.append({
            "label": name,
            "support": int(yt.sum()),
            "predicted": int(yp.sum()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        })
    return pd.DataFrame(rows)


def main():
    assert sha256(BASELINE) == EXPECTED_BASELINE_SHA, "baseline hash mismatch"
    panel = load_panel()
    frame = panel["frames"][TEST]
    ids = frame.index.astype(str).to_numpy()

    p_cat = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
    p_early = np.load(OUT / f"{TEST}_early_history.npy").astype(float)
    r4 = np.load(TOP10 / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
    assert np.array_equal(r4["ids"].astype(str), ids), "rank4 ids mismatch"
    p4 = r4["proba"].astype(float)
    for name, p in [("catdyn", p_cat), ("early", p_early)]:
        assert p.shape == (len(ids), 17) and np.isfinite(p).all(), name

    r4full = rank4_combine(p_cat, p4, "full")
    replication = np.asarray(CLASSES)[np.argmax(r4full * MULT, axis=1)]

    baseline = pd.read_csv(BASELINE, dtype={"ID": str})
    assert len(baseline) == len(ids) and np.array_equal(baseline.ID.to_numpy(), ids)
    current = baseline.Opportunity.to_numpy()
    exact = int((replication == current).sum())
    print(f"replication check: {exact}/{len(ids)} labels identical", flush=True)
    if exact != len(ids):
        diff = pd.DataFrame({"ID": ids, "baseline": current, "replication": replication})
        diff = diff[diff.baseline != diff.replication]
        diff.to_csv(OUT / "replication_mismatch.csv", index=False)
        raise RuntimeError(f"replication failed on {len(diff)} rows; see replication_mismatch.csv")

    blend = 0.4 * p_cat + 0.4 * p_early + 0.2 * r4full
    pred = np.asarray(CLASSES)[np.argmax(blend * MULT, axis=1)]

    changed = pred != current
    print(f"blend changes {int(changed.sum())} labels vs baseline", flush=True)
    out = baseline.copy()
    out["Opportunity"] = pred
    assert out.Opportunity.isin(CLASSES).all()
    assert np.array_equal(out.ID.to_numpy(), ids)
    assert np.array_equal(out.CLV_fuel.to_numpy(), baseline.CLV_fuel.to_numpy())
    assert np.array_equal(out.CLV_nonfuel.to_numpy(), baseline.CLV_nonfuel.to_numpy())

    export = OUT / "candidate_blend_2025-12-01.csv"
    out.to_csv(export, index=False)

    moves = pd.crosstab(
        pd.Series(current, name="from"), pd.Series(pred, name="to")
    ).stack().reset_index().rename(columns={0: "count"})
    moves = moves[moves["count"] > 0].sort_values("count", ascending=False)
    moves.to_csv(OUT / "candidate_blend_moves.csv", index=False)

    audit = {
        "experimental": True,
        "recipe": "0.4 catdyn + 0.4 early_history + 0.2 catdyn+rank4_full, accepted multipliers",
        "baseline": BASELINE.name,
        "baseline_sha256": EXPECTED_BASELINE_SHA,
        "export": export.name,
        "export_sha256": sha256(export),
        "rows": len(out),
        "opportunity_changes": int(changed.sum()),
        "label_distribution": {c: int((pred == c).sum()) for c in CLASSES if (pred == c).sum()},
        "baseline_distribution": {c: int((current == c).sum()) for c in CLASSES if (current == c).sum()},
        "replication_labels_identical": exact,
        "clv_columns_unchanged": True,
        "eval_gains_accepted_multipliers": {"2025-06-01": 0.002586, "2025-09-01": 0.002570},
    }
    (OUT / "candidate_blend_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k not in ("label_distribution", "baseline_distribution")}, indent=2))
    print("\ntop label moves:")
    print(moves.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
