"""Coordinate-ascent tuning of the r4full decode vector on 8 non-eval origins.

Tuning origins (June/September excluded):
  2024-12 .. 2025-05, 2025-07, 2025-08
Old origins (<= 2025-03) need inner windows before Sep-2024 -> early panel
rank4 fits (outputs/sprint16/rank4_early/) + next_round b838 catdyn probs.
Apr/May/Jul/Aug use the production rank4 cache and top10 catdyn probs.

Outputs: vR8 (all 8 origins) and vR7 (drop 2024-12) vectors, per-origin F1
tables, and June/Sept evaluation-fold scores for both (production r4full).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))

import evaluate_saved as es  # noqa: E402
from evaluate_saved import CLASSES, MULT, OCT, OUT, rank4_combine, load_proba, load_rank4  # noqa: E402

NEXT = ROOT / "outputs" / "next_round"
R4E = OUT / "rank4_early"

OLD = ["2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01"]
POOL8 = OLD + ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]

GROUPS = {
    "Existing-category growth: Fuel": [0.7, 0.85, 1.0, 1.15, 1.3],
    "Stable": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
    "Inactivity": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "Existing-category growth: Beverages": [1.3, 1.6, 2.0, 2.4, 2.8],
    "Existing-category growth: Food & beverage": [1.0, 1.5, 2.0, 2.5, 3.0],
    "Existing-category growth: Other": [1.0, 1.1, 1.3, 1.4, 1.6],
    "ADOPTION": [2.0, 2.5, 3.0, 3.5, 4.0],
}


def group_indices(name):
    if name == "ADOPTION":
        return [i for i, c in enumerate(CLASSES) if c.startswith("New category adoption")]
    return [CLASSES.index(name)]


def wf1(y, probs, mult):
    return float(f1_score(y, np.asarray(CLASSES)[np.argmax(probs * mult, axis=1)], average="weighted", zero_division=0))


def load_old(cutoff, panel):
    z = np.load(NEXT / f"b838f9b6679e05b7_{cutoff}_Opportunity.npz", allow_pickle=True)
    p17 = np.asarray(z["prediction"], dtype=float)
    ids = z["ids"].astype(str)
    if ids.dtype == object and hasattr(ids[0], "decode"):
        try:
            ids = np.array([x.decode() if isinstance(x, bytes) else x for x in ids])
        except Exception:
            pass
    z4 = np.load(R4E / f"{cutoff}_rank4_early_2.npz", allow_pickle=True)
    p4_ids = z4["ids"].astype(str)
    pos = pd.Series(np.arange(len(p4_ids)), index=p4_ids).loc[ids].to_numpy()
    p4 = z4["proba"].astype(float)[pos]
    y = panel["labels"][cutoff].reindex(ids)["Opportunity"].to_numpy()
    assert pd.notna(y).all() and len(y) == len(p17)
    return rank4_combine(p17, p4, "full"), y


def load_pool(cutoff, panel):
    ids, y = es.load_labels(cutoff)
    n = len(ids)
    p17 = load_proba("category_dynamics", cutoff, n)
    p4 = load_rank4(cutoff, n)
    assert p17 is not None and p4 is not None
    return rank4_combine(p17, p4, "full"), y


def objective(data, mult, origins):
    return float(np.mean([wf1(y, p, mult) for c, (p, y) in data.items() if c in origins]))


def coordinate_ascent(data, origins, start):
    mult = start.copy()
    best = objective(data, mult, origins)
    base = start.copy()
    history = [{"start": best}]
    for pas in range(3):
        improved = False
        for gname, grid in GROUPS.items():
            idx = group_indices(gname)
            cands = []
            for v in grid:
                trial = mult.copy()
                trial[idx] = v
                tie = abs(v - base[idx[0]])
                cands.append((objective(data, trial, origins), -tie, -abs(v), v, trial))
            cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            obj, _, _, v, trial = cands[0]
            if obj > best + 1e-12:
                history.append({"pass": pas, "group": gname, "from": float(mult[idx[0]]), "to": float(v), "pooled": obj})
                mult, best = trial, obj
                improved = True
        if not improved:
            break
    return mult, best, history


def main():
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    panel = pickle.loads((OCT / "early_history_panel.pkl").read_bytes())
    _oct_cache = {}

    def _oct_panel():
        if "p" not in _oct_cache:
            _oct_cache["p"] = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
        return _oct_cache["p"]

    es.load_panel = _oct_panel
    data = {}
    for c in POOL8:
        if c in OLD:
            prod, y = load_old(c, panel)
        else:
            prod, y = load_pool(c, panel)
        data[c] = (prod, y)
        print(f"origin {c}: rows={len(y)}", flush=True)

    base_scores = {c: wf1(y, p, vR) for c, (p, y) in data.items()}
    print("vR pooled-8 mean:", round(float(np.mean(list(base_scores.values()))), 6))
    print({c: round(v, 5) for c, v in base_scores.items()}, flush=True)

    results = {}
    for name, origins in [("vR8", POOL8), ("vR7", POOL8[1:])]:
        mult, best, hist = coordinate_ascent(data, origins, vR)
        scores = {c: wf1(y, p, mult) for c, (p, y) in data.items()}
        results[name] = {
            "vector": [float(x) for x in mult],
            "pooled": best,
            "scores": {c: round(v, 6) for c, v in scores.items()},
            "gains_vs_vR": {c: round(scores[c] - base_scores[c], 6) for c in scores},
            "history": hist,
        }
        print(f"\n{name}: {results[name]['vector']}")
        print(f"{name} pooled-{len(origins)} mean: {best:.6f}")
        for c in POOL8:
            print(f"  {c}: {scores[c]:.6f} ({scores[c]-base_scores[c]:+.6f})", flush=True)

    print("\nEvaluation folds (production r4full, not used for tuning):")
    eval_rows = []
    for c in EVAL:
        ids, y = es.load_labels(c)
        n = len(ids)
        prod = rank4_combine(load_proba("category_dynamics", c, n), load_rank4(c, n), "full")
        row = {"origin": c, "vR": wf1(y, prod, vR)}
        for name in ["vR7", "vR8"]:
            row[name] = wf1(y, prod, np.array(results[name]["vector"]))
        eval_rows.append(row)
        print(row, flush=True)
        results.setdefault("eval_folds", []).append(row)

    results["base_vR"] = {"vector": [float(x) for x in vR], "scores": {c: round(v, 6) for c, v in base_scores.items()}}
    (OUT / "pool_r4full_vector_vr8.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(eval_rows).to_csv(OUT / "tune_vR8_eval_folds.csv", index=False)
    print("\nwrote", OUT / "pool_r4full_vector_vr8.json")


if __name__ == "__main__":
    main()
