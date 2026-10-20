"""Sprint 17 E3 — big-three head reparameterisation on the production panel.

Two pre-registered variants replace the 4-way softmax rank4 head inside the
r4full product while everything else (features, protocol, decode vR, CLV) is
frozen:

  big3_softmax : 3-class LGBM softmax trained on big-three rows only
                 (Other no longer competes for the same trees).
  binary       : two BCE heads (Inactivity vs rest, Fuel vs rest) with
                 pS = clip(1 - pI - pF, 0) — heads choose their own
                 feature interactions; a dedicated Stable head is implied
                 by the cascade identity.

Head protocol copies sprint16 `fit_rank4_variants.fit_variant` exactly:
stacked monthly cutoffs <= origin-3m, recency weights 0.70+0.05*(months-9),
early stopping 60 on the previous month, full refit at best rounds, seeds
7/23 averaged. The "other" column of the r4full product stays the cached
std2 rank4 p4[:,3] resplit by category-dynamics conditionals.

Gates (proposal_binary_cascade.json):
  screen pool Apr/May/Jul/Aug vs cached std2 r4full+vR: mean >= +0.004,
  every origin >= -0.002. One pooled-vR coordinate retune allowed per
  variant, then re-screen. Folds June/Sept: each >= +0.0015, mean >= +0.004.
No December fit unless all gates pass.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))

import feature_pipeline as fp  # noqa: E402
from evaluate_saved import (  # noqa: E402
    CLASSES, OUT, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
DEST = OUT / "cascade"
DEST.mkdir(parents=True, exist_ok=True)

INACT, STAB, FUEL_LABEL = "Inactivity", "Stable", "Existing-category growth: Fuel"
BIG17 = [CLASSES.index(INACT), CLASSES.index(STAB), CLASSES.index(FUEL_LABEL)]
OTHER17 = [i for i in range(17) if i not in BIG17]

STD = dict(n_estimators=1200, learning_rate=0.04, num_leaves=47, min_child_samples=35,
           subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2)
COMMON = dict(deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=1)
SEEDS = (7, 23)

SCREEN = {"mean_min": 0.004, "origin_min": -0.002}
FOLDS = {"origin_min": 0.0015, "mean_min": 0.004}


def big3_index(labels):
    out = np.full(len(labels), -1, dtype=np.int8)
    for k, name in enumerate((INACT, STAB, FUEL_LABEL)):
        out[labels == name] = k
    return out


def fit_head(cutoff, kind, member, panel, seeds=SEEDS):
    """Average over seeds of heads for `cutoff`, shapes (n,3) softmax / (n,1) binary."""
    slug = {"Inactivity": "inact", "Existing-category growth: Fuel": "fuel"}.get(member, "big3")
    path = DEST / f"{cutoff}_{kind}_{slug}.npz"
    frame = panel["frames"][cutoff]
    n_out = 3 if kind == "big3_softmax" else 1
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"].astype(float)
    features = panel["feature_cols"]
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        fr = panel["frames"][origin]
        lab = panel["labels"][origin].reindex(fr.index).Opportunity.to_numpy()
        b3 = big3_index(lab)
        ts = pd.Timestamp(origin)
        w = 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)
        if kind == "big3_softmax":
            keep = b3 >= 0
            target = b3[keep]
        else:
            keep = np.ones(len(fr), dtype=bool)
            target = (lab == member).astype(int)
        blocks.append(pd.DataFrame({"target": target, "cutoff": origin, "w": w},
                                   index=fr.index[keep]).join(fr.loc[fr.index[keep], features]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    total = np.zeros((len(frame), n_out))
    rounds_seen = []
    for seed in seeds:
        t0 = time.time()
        params = dict(STD, random_state=seed)
        if kind == "big3_softmax":
            inner = LGBMClassifier(**{**params, **COMMON})
            inner.fit(train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
                      eval_set=[(train.loc[va, features], train.loc[va, "target"])],
                      callbacks=[early_stopping(60, verbose=False)])
            rounds = max(int(inner.best_iteration_ or 40), 40)
            full = LGBMClassifier(**{**params, **COMMON, "n_estimators": rounds})
            full.fit(train[features], train.target, sample_weight=train.w)
            raw = full.predict_proba(frame[features])
            if list(full.classes_) != [0, 1, 2]:
                raise RuntimeError(f"unexpected big3 head classes_ {full.classes_}")
            total += raw
        else:
            inner = LGBMClassifier(objective="binary", **{**params, **COMMON})
            inner.fit(train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
                      eval_set=[(train.loc[va, features], train.loc[va, "target"])],
                      callbacks=[early_stopping(60, verbose=False)])
            rounds = max(int(inner.best_iteration_ or 40), 40)
            full = LGBMClassifier(objective="binary", **{**params, **COMMON, "n_estimators": rounds})
            full.fit(train[features], train.target, sample_weight=train.w)
            if list(full.classes_) != [0, 1]:
                raise RuntimeError(f"unexpected binary head classes_ {full.classes_}")
            total[:, 0] += full.predict_proba(frame[features])[:, 1]
        rounds_seen.append(rounds)
        print(f"  {kind}/{member} {cutoff} seed {seed} rounds {rounds} in {time.time()-t0:.0f}s", flush=True)
    proba = total / len(seeds)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def heads_for(cutoff, variant, panel):
    n = len(panel["frames"][cutoff])
    if variant == "big3_softmax":
        p3 = fit_head(cutoff, "big3_softmax", "big3", panel)
        return p3
    pi = fit_head(cutoff, "binary", INACT, panel)[:, 0]
    pf = fit_head(cutoff, "binary", FUEL_LABEL, panel)[:, 0]
    ps = np.clip(1.0 - pi - pf, 0.0, 1.0)
    return np.column_stack([pi, ps, pf])


def product(p17, p4std, p3, n):
    """r4full recombination with the big-3 block replaced by p3.

    out[big3] rescale to (1 - p4_other); out[other] = p4_other * cond(catdyn).
    """
    out = np.zeros((n, 17))
    base = p17[:, OTHER17]
    cond = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-12)
    p4other = p4std[:, 3]
    s = p3.sum(axis=1, keepdims=True)
    p3n = p3 / np.maximum(s, 1e-12)
    out[:, BIG17] = p3n * (1.0 - p4other)[:, None]
    out[:, OTHER17] = cond * p4other[:, None]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def origin_scores(variant, cutoff, panel, mult):
    ids, y = load_labels(cutoff)
    n = len(ids)
    p17 = load_proba("category_dynamics", cutoff, n)
    p4std = load_rank4(cutoff, n)
    assert p17 is not None and p4std is not None
    p3 = heads_for(cutoff, variant, panel)
    prod = product(p17, p4std, p3, n)
    return float(weighted_f1(np.argmax(prod * mult, axis=1), y)), ids, y


def coordinate_retune(products, origins, start):
    """Coordinate ascent of the decode vector on the pool products only."""
    groups = {
        "Fuel": [BIG17[2]],
        "Stable": [BIG17[1]],
        "Inactivity": [BIG17[0]],
        "growth_bev": [CLASSES.index("Existing-category growth: Beverages")],
        "growth_fnb": [CLASSES.index("Existing-category growth: Food & beverage")],
        "growth_other": [CLASSES.index("Existing-category growth: Other")],
        "ADOPTION": [i for i in OTHER17 if CLASSES[i].startswith("New category adoption")],
    }
    grids = {
        "Fuel": [0.7, 0.85, 1.0, 1.15, 1.3],
        "Stable": [0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
        "Inactivity": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "growth_bev": [1.3, 1.6, 2.0, 2.4, 2.8],
        "growth_fnb": [1.0, 1.5, 2.0, 2.5, 3.0],
        "growth_other": [1.0, 1.1, 1.3, 1.4, 1.6],
        "ADOPTION": [2.0, 2.5, 3.0, 3.5, 4.0],
    }
    def obj(mult):
        return float(np.mean([weighted_f1(np.argmax(p * mult, axis=1), y)
                              for c, (p, y) in products.items() if c in origins]))
    mult = np.asarray(start, dtype=float).copy()
    best = obj(mult)
    for _ in range(3):
        improved = False
        for gname, idx in groups.items():
            for v in grids[gname]:
                trial = mult.copy()
                trial[idx] = v
                val = obj(trial)
                if val > best + 1e-12:
                    mult, best = trial, val
                    improved = True
        if not improved:
            break
    return mult, best


def main():
    t_start = time.time()
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"], dtype=float)
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    print("panel loaded", flush=True)

    baseline = {}
    for c in POOL + EVAL:
        ids, y = load_labels(c)
        n = len(ids)
        p17 = load_proba("category_dynamics", c, n)
        p4std = load_rank4(c, n)
        prod = rank4_combine(p17, p4std, "full")
        baseline[c] = weighted_f1(np.argmax(prod * vR, axis=1), y)
    print("baseline r4full+vR:", {c: round(b, 6) for c, b in baseline.items()}, flush=True)

    decision = {"baseline_pool": {c: float(b) for c, b in baseline.items() if c in POOL},
                "baseline_eval": {c: float(b) for c, b in baseline.items() if c in EVAL}}
    summary = {}

    for variant in ("big3_softmax", "binary"):
        t_v = time.time()
        print(f"\n### screening variant {variant}", flush=True)
        prods_cycle = {}
        scores = {}
        for c in POOL:
            ids, y = load_labels(c)
            n = len(ids)
            p17 = load_proba("category_dynamics", c, n)
            p4std = load_rank4(c, n)
            prod = product(p17, p4std, heads_for(c, variant, panel), n)
            f1 = weighted_f1(np.argmax(prod * vR, axis=1), y)
            prods_cycle[c] = (prod, y)
            scores[c] = f1
            print(f"  {variant} {c}: {f1:.6f}  gain {f1 - baseline[c]:+.6f}", flush=True)
        gains = {c: scores[c] - baseline[c] for c in POOL}
        res = {"scores": scores, "gains": gains,
               "mean_gain": float(np.mean(list(gains.values()))),
               "min_gain": float(np.min(list(gains.values())))}
        passed = res["mean_gain"] >= SCREEN["mean_min"] and res["min_gain"] >= SCREEN["origin_min"]

        if not passed:
            mult_fb, pooled_fb = coordinate_retune(prods_cycle, POOL, vR)
            gains_fb = {c: weighted_f1(np.argmax(prods_cycle[c][0] * mult_fb, axis=1), prods_cycle[c][1]) - baseline[c]
                        for c in POOL}
            res["retune"] = {"vector": [float(x) for x in mult_fb],
                             "pooled_mean_f1": pooled_fb,
                             "gains": gains_fb,
                             "mean_gain": float(np.mean(list(gains_fb.values()))),
                             "min_gain": float(np.min(list(gains_fb.values())))}
            passed = res["retune"]["mean_gain"] >= SCREEN["mean_min"] and res["retune"]["min_gain"] >= SCREEN["origin_min"]
            mult_used = mult_fb
            print(f"  retune gain {res['retune']['mean_gain']:+.6f} min {res['retune']['min_gain']:+.6f} -> pass={passed}", flush=True)
        else:
            mult_used = vR
        res["passed_screen"] = passed
        res["decode"] = [float(x) for x in mult_used]
        res["seconds"] = round(time.time() - t_v, 1)
        summary[variant] = res

        if not passed:
            print(f"  variant {variant}: SCREEN FAILED (mean {res['mean_gain']:+.6f})", flush=True)
            continue

        # Folds — evaluation only, no tuning on them.
        scores = {}
        for c in EVAL:
            f1, *_ = origin_scores(variant, c, panel, mult_used)
            scores[c] = f1
            print(f"  {variant} EVAL {c}: {f1:.6f}  gain {f1 - baseline[c]:+.6f}", flush=True)
        gains = {c: scores[c] - baseline[c] for c in EVAL}
        fold_ok = all(g >= FOLDS["origin_min"] for g in gains.values()) and \
            float(np.mean(list(gains.values()))) >= FOLDS["mean_min"]
        res["fold_scores"] = scores
        res["fold_gains"] = gains
        res["passed_folds"] = fold_ok
        print(f"  variant {variant}: FOLDS {'PASS' if fold_ok else 'FAIL'} {gains}", flush=True)

        if not fold_ok:
            continue

    # December: only for variants that passed both gates; if more than one,
    # keep the one with the higher min-eval fold gain.
    candidates = {v: r for v, r in summary.items()
                  if r.get("passed_screen") and r.get("passed_folds")}
    if candidates:
        best_variant = max(candidates, key=lambda v: min(candidates[v]["fold_gains"].values()))
        mult_used = np.array(candidates[best_variant]["decode"])
        t_d = time.time()
        print(f"\n### December fit for {best_variant}", flush=True)
        frame = panel["frames"][TEST]
        n = len(frame)
        p17 = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
        r4 = np.load(ROOT / "outputs" / "top10_round" / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
        p4std = r4["proba"].astype(float)
        p3 = heads_for(TEST, best_variant, panel)
        prod = product(p17, p4std, p3, n)
        pred = np.asarray(CLASSES)[np.argmax(prod * mult_used, axis=1)]
        baseline_csv = pd.read_csv(OUT / "baseline_0309411451.csv", dtype={"ID": str})
        assert list(baseline_csv["ID"].astype(str)) == list(frame.index.astype(str))
        out = baseline_csv.copy()
        out["Opportunity"] = pred
        path = OUT.parent / "sprint17" / "candidate_e3_2025-12-01.csv"
        out.to_csv(path, index=False)
        changes = int((out["Opportunity"].to_numpy() != baseline_csv["Opportunity"].to_numpy()).sum())
        print(f"  wrote {path.name}: {changes} label changes vs current best, in {time.time()-t_d:.0f}s", flush=True)
        decision["december"] = {"variant": best_variant, "file": str(path), "changes_vs_best": changes,
                                "decode": [float(x) for x in mult_used]}

    decision["variants"] = summary
    decision["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    decision["total_seconds"] = round(time.time() - t_start, 1)
    (OUT / "binary_cascade_decision.json").write_text(json.dumps(decision, indent=2))
    print("\nwrote", OUT / "binary_cascade_decision.json", flush=True)


if __name__ == "__main__":
    main()
