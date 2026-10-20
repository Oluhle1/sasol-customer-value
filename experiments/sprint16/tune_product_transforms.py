"""Product-transform search on the 4-origin pool (no model refits).

Same protocol family as the vR tuning: candidate transforms keep June/Sept as
untouched evaluation folds; the decode vector stays vR.

  condg : sharpen the p17 conditional inside the Other mass, cond^g
  massg : temperature on the p4 four-class masses, p4^g
  temp  : temperature on the final product, prod^(1/T)

Selection: balanced pooled gain on Apr/May/Jul/Aug; folds reported afterward.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
from evaluate_saved import CLASSES, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1  # noqa: E402

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
BIG = {CLASSES.index(c) for c in ["Inactivity", "Stable", "Existing-category growth: Fuel"]}
OTHER = np.array([i for i in range(17) if i not in BIG])


def r4_condg(p17, p4, g):
    base = p17[:, OTHER]
    cond = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-12)
    cond = np.power(np.maximum(cond, 1e-12), g)
    cond = cond / cond.sum(axis=1, keepdims=True)
    out = np.zeros_like(p17)
    out[:, CLASSES.index("Inactivity")] = p4[:, 0]
    out[:, CLASSES.index("Stable")] = p4[:, 1]
    out[:, CLASSES.index("Existing-category growth: Fuel")] = p4[:, 2]
    out[:, OTHER] = p4[:, 3:4] * cond
    return out / out.sum(axis=1, keepdims=True)


def r4_massg(p17, p4, g):
    p4g = np.power(np.maximum(p4, 1e-12), g)
    p4g = p4g / p4g.sum(axis=1, keepdims=True)
    return rank4_combine(p17, p4g, "full")


def load_origin(cutoff):
    ids, y = load_labels(cutoff)
    n = len(ids)
    p17 = load_proba("category_dynamics", cutoff, n)
    p4 = load_rank4(cutoff, n)
    assert p17 is not None and p4 is not None
    return p17, p4, y


def main():
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    data = {c: load_origin(c) for c in POOL}
    print("loaded", {c: len(d[2]) for c, d in data.items()}, flush=True)

    def pooled(fn):
        return float(np.mean([weighted_f1(np.argmax(fn(p17, p4) * vR, axis=1), y) for p17, p4, y in data.values()]))

    base = pooled(lambda p17, p4: rank4_combine(p17, p4, "full"))
    print(f"base r4full+vR pooled: {base:.6f}", flush=True)

    out = {"base_pooled": base, "searches": {}}
    for name, fn in [("condg", r4_condg), ("massg", r4_massg)]:
        grid = [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2]
        rows = []
        for g in grid:
            s = pooled(lambda p17, p4, g=g: fn(p17, p4, g))
            rows.append({"g": g, "pooled": s, "gain": s - base})
            print(f"{name} g={g}: {s:.6f} ({s-base:+.6f})", flush=True)
        out["searches"][name] = rows

    temp_grid = [0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    rows = []
    for t in temp_grid:
        def ft(p17, p4, t=t):
            p = rank4_combine(p17, p4, "full")
            q = np.power(np.maximum(p, 1e-12), 1.0 / t)
            return q / q.sum(axis=1, keepdims=True)
        s = pooled(ft)
        rows.append({"T": t, "pooled": s, "gain": s - base})
        print(f"temp T={t}: {s:.6f} ({s-base:+.6f})", flush=True)
    out["searches"]["temp"] = rows

    (OUT / "product_transform_search.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUT / "product_transform_search.json")


if __name__ == "__main__":
    main()
