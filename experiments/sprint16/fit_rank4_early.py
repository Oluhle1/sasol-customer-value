"""Fit rank4 (four-class Inactivity/Stable/Fuel/Other) variants on the early panel.

Variants evaluated on June/September (vR and accepted vectors):
  early2 : early panel, seeds (7, 23)   [same seeds as the production rank4]
  early4 : early panel, seeds (7, 23, 11, 37)
  ens24  : mean of production p4 (top10_round cache) and early2
If a variant improves both folds over the production r4full+vR, fit it for
December and export a probe.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
import feature_pipeline as fp  # noqa: E402

from evaluate_saved import MULT, TOP10, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1  # noqa: E402

OCT = ROOT / "outputs" / "oct10"
DEST = OUT / "rank4_early"
DEST.mkdir(parents=True, exist_ok=True)

RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def fit_rank4(panel, cutoff, seeds):
    path = DEST / f"{cutoff}_rank4_early_{len(seeds)}.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"].astype(float)
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        origin_frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(origin_frame.index).Opportunity.to_numpy()
        blocks.append(pd.DataFrame({
            "target": collapse(labels),
            "cutoff": origin,
            "w": weight_of(origin),
        }, index=origin_frame.index).join(origin_frame[panel["feature_cols"]]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    features = panel["feature_cols"]
    params = dict(
        n_estimators=1200, learning_rate=0.04, num_leaves=47, min_child_samples=35,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2,
        deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=1,
    )
    total = np.zeros((len(frame), 4))
    for seed in seeds:
        t0 = time.time()
        inner = LGBMClassifier(**dict(params, random_state=seed))
        inner.fit(
            train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
            eval_set=[(train.loc[va, features], train.loc[va, "target"])],
            callbacks=[early_stopping(60, verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMClassifier(**dict(params, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train.target, sample_weight=train.w)
        total += fp.align_proba(full, frame[features], RANK)
        print(f"  rank4-early {cutoff} seed {seed} rounds {rounds} in {time.time()-t0:.0f}s", flush=True)
    proba = total / len(seeds)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def main():
    import pickle
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    panel = pickle.loads((OCT / "early_history_panel.pkl").read_bytes())

    rows = []
    for cutoff in ["2025-06-01", "2025-09-01"]:
        ids, y = load_labels(cutoff)
        n = len(ids)
        assert list(panel["frames"][cutoff].index.astype(str)) == list(ids)
        p_cat = load_proba("category_dynamics", cutoff, n)
        p4_std = load_rank4(cutoff, n)
        r4_std = rank4_combine(p_cat, p4_std, "full")
        p4_e2 = fit_rank4(panel, cutoff, (7, 23))
        p4_e4 = fit_rank4(panel, cutoff, (7, 23, 11, 37))
        r4_e2 = rank4_combine(p_cat, p4_e2, "full")
        r4_e4 = rank4_combine(p_cat, p4_e4, "full")
        r4_ens = rank4_combine(p_cat, (p4_std + p4_e2) / 2, "full")
        for name, prod in [("std", r4_std), ("early2", r4_e2), ("early4", r4_e4), ("ens24", r4_ens)]:
            rows.append({
                "origin": cutoff, "variant": name,
                "f1_vR": round(weighted_f1(np.argmax(prod * vR, axis=1), y), 6),
                "f1_accepted": round(weighted_f1(np.argmax(prod * MULT, axis=1), y), 6),
            })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(OUT / "rank4_early_comparison.csv", index=False)


if __name__ == "__main__":
    main()
