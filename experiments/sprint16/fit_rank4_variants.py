"""Rank4 model variants on the regular panel, validated on the 4-origin pool.

Variants (all evaluated as r4full product decoded with vR):
  std4  : production params, seeds (11, 37) added to the cached (7, 23) mean
          -> 4-seed average (pure variance reduction)
  capbig: num_leaves 127, min_child 20, lr 0.03, colsample 0.5, reg 2.0
  capsma: num_leaves 31, min_child 60, lr 0.05, reg 1.5
Folds (June/Sept) reported only for variants with a balanced pooled gain.
"""
import json
import pickle
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

from evaluate_saved import (  # noqa: E402
    CLASSES, MULT, OCT, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
DEST = OUT / "rank4_variants"
DEST.mkdir(parents=True, exist_ok=True)

RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"

STD = dict(n_estimators=1200, learning_rate=0.04, num_leaves=47, min_child_samples=35,
           subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2)
CAPBIG = dict(n_estimators=1200, learning_rate=0.03, num_leaves=127, min_child_samples=20,
              subsample=0.85, subsample_freq=1, colsample_bytree=0.50, reg_lambda=2.0)
CAPSMA = dict(n_estimators=1200, learning_rate=0.05, num_leaves=31, min_child_samples=60,
              subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.5)
COMMON = dict(deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=1)


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def fit_variant(panel, cutoff, params, seeds, name):
    path = DEST / f"{cutoff}_{name}_{len(seeds)}.npz"
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
    total = np.zeros((len(frame), 4))
    for seed in seeds:
        t0 = time.time()
        inner = LGBMClassifier(**dict(params, **COMMON, random_state=seed))
        inner.fit(
            train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
            eval_set=[(train.loc[va, features], train.loc[va, "target"])],
            callbacks=[early_stopping(60, verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMClassifier(**dict(params, **COMMON, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train.target, sample_weight=train.w)
        total += fp.align_proba(full, frame[features], RANK)
        print(f"  {name} {cutoff} seed {seed} rounds {rounds} in {time.time()-t0:.0f}s", flush=True)
    proba = total / len(seeds)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def main():
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    print("panel loaded", flush=True)

    p4 = {}
    for c in POOL:
        n = len(panel["frames"][c])
        s2 = load_rank4(c, n)
        assert s2 is not None
        f11 = fit_variant(panel, c, STD, (11,), "std_seed11")
        f37 = fit_variant(panel, c, STD, (37,), "std_seed37")
        p4[c] = {
            "std2": s2,
            "std4": (2 * s2 + f11 + f37) / 4,
            "capbig": fit_variant(panel, c, CAPBIG, (7, 23), "capbig"),
            "capsma": fit_variant(panel, c, CAPSMA, (7, 23), "capsma"),
        }

    rows = []
    for c in POOL:
        ids, y = load_labels(c)
        n = len(ids)
        p17 = load_proba("category_dynamics", c, n)
        assert p17 is not None and len(ids) == len(panel["frames"][c])
        for name, arr in p4[c].items():
            prod = rank4_combine(p17, arr, "full")
            rows.append({"origin": c, "variant": name,
                         "f1_vR": round(weighted_f1(np.argmax(prod * vR, axis=1), y), 6)})
    table = pd.DataFrame(rows).pivot(index="variant", columns="origin", values="f1_vR")
    table["mean"] = table.mean(axis=1)
    print(table.round(6).to_string())

    base = table.loc["std2"]
    summary = {}
    for name in table.index:
        gains = {c: float(table.loc[name, c] - base[c]) for c in POOL}
        summary[name] = {"pooled_gain": float(np.mean(list(gains.values()))), "per_origin": gains}
        print(name, "pooled gain vs std2:", round(summary[name]["pooled_gain"], 6),
              {c: round(g, 6) for c, g in gains.items()}, flush=True)

    candidates = [n for n, s in summary.items() if n != "std2" and s["pooled_gain"] >= 0.002
                  and min(s["per_origin"].values()) > -0.002]
    print("fold candidates:", candidates, flush=True)
    fold_rows = []
    if candidates:
        for c in EVAL:
            ids, y = load_labels(c)
            n = len(ids)
            p17 = load_proba("category_dynamics", c, n)
            std2 = load_rank4(c, n)
            fits = {"std2": std2}
            if "std4" in candidates:
                f11 = fit_variant(panel, c, STD, (11,), "std_seed11")
                f37 = fit_variant(panel, c, STD, (37,), "std_seed37")
                fits["std4"] = (2 * std2 + f11 + f37) / 4
            for name in candidates:
                if name in fits:
                    continue
                fits[name] = fit_variant(panel, c, {"capbig": CAPBIG, "capsma": CAPSMA}[name], (7, 23), name)
            for name, arr in fits.items():
                prod = rank4_combine(p17, arr, "full")
                fold_rows.append({"origin": c, "variant": name,
                                  "f1_vR": round(weighted_f1(np.argmax(prod * vR, axis=1), y), 6)})
        ft = pd.DataFrame(fold_rows).pivot(index="variant", columns="origin", values="f1_vR")
        print(ft.round(6).to_string())

    (OUT / "rank4_variant_search.json").write_text(json.dumps({
        "pool_table": table.round(6).to_dict(),
        "summary": summary,
        "fold_candidates": candidates,
        "fold_table": (pd.DataFrame(fold_rows).pivot(index="variant", columns="origin", values="f1_vR").round(6).to_dict() if fold_rows else {}),
    }, indent=2))
    table.to_csv(OUT / "rank4_variant_pool.csv")
    print("wrote", OUT / "rank4_variant_search.json")


if __name__ == "__main__":
    main()
