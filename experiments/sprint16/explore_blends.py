"""Explore equal-weight and weighted blends of saved Opportunity probabilities.

Read-only; evaluates every subset of up to 4 components on June and September.
The rank4 four-class output is included as a synthetic 17-class component.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from evaluate_saved import CLASSES, MULT, OUT, load_labels, load_proba, weighted_f1, rank4_combine, load_rank4

ORIGINS = ["2025-06-01", "2025-09-01"]
MODELS = ["category_dynamics", "early_history", "monthly", "monthly_stacked", "station_context", "lgb", "cat"]


def build_components():
    comp = {}
    y = {}
    for c in ORIGINS:
        ids, yy = load_labels(c)
        y[c] = yy
        p17 = load_proba("category_dynamics", c, len(ids))
        comp.setdefault("catdyn", {})[c] = p17
        for m in MODELS:
            if m == "category_dynamics":
                continue
            p = load_proba(m, c, len(ids))
            if p is not None:
                comp.setdefault(m, {})[c] = p
        r4 = load_rank4(c, len(ids))
        if r4 is not None:
            for mode in ("full", "within"):
                comp.setdefault(f"r4{mode}", {})[c] = rank4_combine(p17, r4, mode)
    return comp, y


def main():
    comp, y = build_components()
    names = sorted(comp.keys())
    print("components:", names)
    base = {c: weighted_f1(np.argmax(comp["catdyn"][c] * MULT, axis=1), y[c]) for c in ORIGINS}
    print("catdyn:", {c: round(v, 6) for c, v in base.items()})

    results = []
    for k in (1, 2, 3, 4):
        for subset in itertools.combinations(names, k):
            if "catdyn" not in subset:
                continue
            gains = {}
            for c in ORIGINS:
                p = sum(comp[m][c] for m in subset) / len(subset)
                pred = np.argmax(p * MULT, axis=1)
                gains[c] = weighted_f1(pred, y[c]) - base[c]
            results.append({"subset": subset, "june": gains[ORIGINS[0]], "sept": gains[ORIGINS[1]], "min": min(gains.values())})
    res = sorted(results, key=lambda r: -r["min"])
    print("\nTop equal-weight subsets by min gain:")
    for r in res[:12]:
        print(f"  {r['min']:+.5f}  june {r['june']:+.5f} sept {r['sept']:+.5f}  {'+'.join(r['subset'])}")

    # Weighted variants around the promising 2/3-way blends: integer weights 1..4
    print("\nWeighted 2-3 way searches (weights 1..4, catdyn anchored):")
    best = []
    for k in (2, 3):
        for subset in itertools.combinations([m for m in names if m != "catdyn"], k - 1):
            subset = ("catdyn",) + subset
            for w in itertools.product(range(1, 5), repeat=len(subset)):
                if w[0] != max(w):
                    continue
                gains = {}
                for c in ORIGINS:
                    p = sum(wi * comp[m][c] for wi, m in zip(w, subset)) / sum(w)
                    pred = np.argmax(p * MULT, axis=1)
                    gains[c] = weighted_f1(pred, y[c]) - base[c]
                best.append({"subset": subset, "w": w, "june": gains[ORIGINS[0]], "sept": gains[ORIGINS[1]], "min": min(gains.values())})
    best = sorted(best, key=lambda r: -r["min"])
    for r in best[:12]:
        print(f"  {r['min']:+.5f}  june {r['june']:+.5f} sept {r['sept']:+.5f}  {list(zip(r['subset'], r['w']))}")

    json.dump(
        {"equal_weight": [{**r, "subset": list(r["subset"])} for r in res], "weighted": [{**r, "subset": list(r["subset"])} for r in best]},
        (OUT / "blend_search.json").open("w"), indent=1,
    )
    print("\nwrote", OUT / "blend_search.json")


if __name__ == "__main__":
    main()
