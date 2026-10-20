"""Evaluate saved Opportunity probability arrays and blends on historical origins.

Read-only over existing outputs. No model fitting. All decoding uses the
frozen category-dynamics decision multipliers unless stated otherwise.
"""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OCT = ROOT / "outputs" / "oct10"
TOP10 = ROOT / "outputs" / "top10_round"
OUT = ROOT / "outputs" / "sprint16"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Existing-category growth: Beverages",
    "Existing-category growth: Food & beverage",
    "Existing-category growth: Fuel",
    "Existing-category growth: Other",
    "Inactivity",
    "New category adoption: Beverages",
    "New category adoption: Braai & Ice",
    "New category adoption: Breads & rolls",
    "New category adoption: Confectionary",
    "New category adoption: Food & beverage",
    "New category adoption: Groceries",
    "New category adoption: Ice-cream",
    "New category adoption: Lubricants",
    "New category adoption: Other",
    "New category adoption: Snacking",
    "New category adoption: Tobacco",
    "Stable",
]
MULT = np.array([1.3, 2.0, 1.0, 1.3, 1.0] + [2.5] * 11 + [0.85])

MODELS = [
    "category_dynamics",
    "monthly",
    "monthly_stacked",
    "lgb",
    "cat",
    "compact_opportunity",
    "early_history",
    "station_context",
    "recent_weighting",
]


def load_panel():
    return pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())


def load_labels(cutoff):
    panel = load_panel()
    frame = panel["frames"][cutoff]
    y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
    return frame.index.astype(str).to_numpy(), y


def load_proba(model, cutoff, n):
    if model == "category_dynamics":
        npz = TOP10 / "probs" / f"{cutoff}_category_dynamics.npz"
        if npz.exists():
            p = np.load(npz, allow_pickle=False)["proba"]
            if len(p) == n:
                return p
    path = OCT / f"{cutoff}_{model}.npy"
    if not path.exists():
        return None
    p = np.load(path, allow_pickle=True)
    if p.ndim != 2 or p.shape[1] != 17 or len(p) != n:
        return None
    return p.astype(float)


def weighted_f1(pred_idx, y, classes=CLASSES):
    pred = np.asarray(classes)[pred_idx]
    return float(f1_score(y, pred, average="weighted", zero_division=0))


def rank4_combine(p17, p4, mode):
    index = {name: i for i, name in enumerate(CLASSES)}
    big = [index["Inactivity"], index["Stable"], index["Existing-category growth: Fuel"]]
    other = [i for i in range(17) if i not in big]
    if mode == "full":
        base = p17[:, other]
        cond = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-12)
        out = np.zeros_like(p17)
        out[:, index["Inactivity"]] = p4[:, 0]
        out[:, index["Stable"]] = p4[:, 1]
        out[:, index["Existing-category growth: Fuel"]] = p4[:, 2]
        out[:, other] = p4[:, 3:4] * cond
    elif mode == "within":
        mass = p17[:, big].sum(axis=1, keepdims=True)
        cond = p4[:, :3] / np.maximum(p4[:, :3].sum(axis=1, keepdims=True), 1e-12)
        out = p17.copy()
        out[:, big] = mass * cond
    else:
        raise ValueError(mode)
    return out / out.sum(axis=1, keepdims=True)


def load_rank4(cutoff, n):
    path = TOP10 / "rank4" / f"{cutoff}_rank4.npz"
    if not path.exists():
        return None
    p = np.load(path, allow_pickle=False)["proba"]
    return p if len(p) == n else None


def main():
    origins = [c for c in ["2025-06-01", "2025-09-01"]]
    data = {}
    for cutoff in origins:
        ids, y = load_labels(cutoff)
        models = {}
        for m in MODELS:
            p = load_proba(m, cutoff, len(ids))
            if p is not None:
                models[m] = p
        r4 = load_rank4(cutoff, len(ids))
        data[cutoff] = {"ids": ids, "y": y, "models": models, "rank4": r4}

    rows = []
    for cutoff, d in data.items():
        for m, p in d["models"].items():
            pred = np.argmax(p * MULT, axis=1)
            rows.append({"origin": cutoff, "model": m, "f1": weighted_f1(pred, d["y"])})
        if d["rank4"] is not None:
            for mode in ("full", "within"):
                base = d["models"]["category_dynamics"]
                pred = np.argmax(rank4_combine(base, d["rank4"], mode) * MULT, axis=1)
                rows.append({"origin": cutoff, "model": f"catdyn+rank4_{mode}", "f1": weighted_f1(pred, d["y"])})
    table = pd.DataFrame(rows).pivot(index="model", columns="origin", values="f1")
    table["mean"] = table.mean(axis=1)
    table = table.sort_values("mean", ascending=False)
    print(table.round(5).to_string())
    table.to_csv(OUT / "individual_models.csv")

    base_models = sorted(set.intersection(*[set(d["models"].keys()) for d in data.values()]))
    print("\ncommon models:", base_models)
    base_f1 = {c: weighted_f1(np.argmax(data[c]["models"]["category_dynamics"] * MULT, axis=1), data[c]["y"]) for c in origins}
    print("catdyn baseline", {c: round(v, 5) for c, v in base_f1.items()})

    # Greedy forward selection on equal-weight probability averages.
    selected = ["category_dynamics"]
    current = {c: data[c]["models"]["category_dynamics"].copy() for c in origins}
    best_min_gain = -np.inf
    history = []
    remaining = [m for m in base_models if m != "category_dynamics"]
    while remaining:
        best = None
        for m in remaining:
            gains = {}
            for c in origins:
                blended = (current[c] * len(selected) + data[c]["models"][m]) / (len(selected) + 1)
                pred = np.argmax(blended * MULT, axis=1)
                gains[c] = weighted_f1(pred, data[c]["y"]) - base_f1[c]
            key = min(gains.values())
            if best is None or key > best[0]:
                best = (key, m, gains)
        key, m, gains = best
        if key <= max(0.0, best_min_gain) and key <= 0:
            break
        # accept while at least one origin does not regress and min gain improves
        if key <= 0 and best_min_gain > 0:
            break
        selected.append(m)
        remaining.remove(m)
        for c in origins:
            current[c] = (current[c] * (len(selected) - 1) + data[c]["models"][m]) / len(selected)
        best_min_gain = key
        history.append({"added": m, "min_gain": key, "gains": gains, "selected": list(selected)})
        print("greedy add", m, {c: round(g, 5) for c, g in gains.items()}, "min", round(key, 5))
    (OUT / "greedy_history.json").write_text(json.dumps(history, indent=2))

    # Evaluate chosen blend + rank4 layer
    if len(selected) > 1:
        print("\nselected:", selected)
        for c in origins:
            blended = current[c]
            pred = np.argmax(blended * MULT, axis=1)
            print(c, "blend f1", round(weighted_f1(pred, data[c]["y"]), 6), "gain", round(weighted_f1(pred, data[c]["y"]) - base_f1[c], 6))
            r4 = data[c]["rank4"]
            if r4 is not None:
                for mode in ("full", "within"):
                    pred2 = np.argmax(rank4_combine(blended, r4, mode) * MULT, axis=1)
                    print(c, f"blend+rank4_{mode}", round(weighted_f1(pred2, data[c]["y"]), 6))


if __name__ == "__main__":
    main()
