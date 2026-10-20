"""Sprint 17 E5 — XGBoost rank4 member on the production frames.

Pre-registration: experiments/sprint17/PROPOSAL_E5.md and
outputs/sprint17/proposal_xgb_rank4.json. Gates come from the JSON and are
never relaxed. Frames, weights and splits mirror run_rank4.fit_rank exactly;
only the learner changes (XGBoost depth-wise growth instead of LGBM leaf-wise).
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))

from evaluate_saved import (  # noqa: E402
    CLASSES, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

OUT17 = ROOT / "outputs" / "sprint17"
DEST = OUT17 / "xgb_rank4"
DEST.mkdir(parents=True, exist_ok=True)
PROPOSAL = OUT17 / "proposal_xgb_rank4.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
CODE = {name: i for i, name in enumerate(RANK)}
FUEL_LABEL = "Existing-category growth: Fuel"

INACT = CLASSES.index("Inactivity")
STAB = CLASSES.index("Stable")
FUEL17 = CLASSES.index(FUEL_LABEL)
BIG17 = [INACT, STAB, FUEL17]
OTHER17 = [i for i in range(17) if i not in BIG17]

STD_XGB = dict(
    objective="multi:softprob", n_estimators=1200, learning_rate=0.04, max_depth=6,
    min_child_weight=35, subsample=0.85, colsample_bytree=0.70, reg_lambda=1.2,
    tree_method="hist", n_jobs=4, verbosity=0, random_state=7,
)
SEEDS = (7, 23)


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def make_xgb(seed, n_estimators=1200, early=None):
    params = dict(STD_XGB, random_state=seed, n_estimators=n_estimators)
    if early is not None:
        params["early_stopping_rounds"] = early
    return XGBClassifier(**params)


def fit_with_es(clf, X, y, w, Xva, yva):
    try:
        clf.fit(X, y, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
    except TypeError:
        clf.fit(X, y, sample_weight=w, eval_set=[(Xva, yva)], verbose=False,
                early_stopping_rounds=60)
    return clf


def fit_member(cutoff, panel):
    path = DEST / f"{cutoff}_rank4_xgb.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"]
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        origin_frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(origin_frame.index).Opportunity.to_numpy()
        codes = np.array([CODE[t] for t in collapse(labels)], dtype=int)
        blocks.append(pd.DataFrame({
            "target": codes, "cutoff": origin, "w": weight_of(origin),
        }, index=origin_frame.index).join(origin_frame[panel["feature_cols"]]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if not (pd.to_datetime(train.loc[tr, "cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("xgb training crosses the inner cutoff")
    features = panel["feature_cols"]
    X = train[features].to_numpy(dtype=np.float32)
    y = train["target"].to_numpy()
    w = train["w"].to_numpy(dtype=float)
    Xf = frame[features].to_numpy(dtype=np.float32)
    total = np.zeros((len(frame), 4))
    for seed in SEEDS:
        t0 = time.time()
        inner = fit_with_es(make_xgb(seed, early=60), X[tr], y[tr], w[tr], X[va], y[va])
        best = getattr(inner, "best_iteration", None)
        rounds = max(int(best) + 1 if best is not None else 40, 40)
        full = make_xgb(seed, n_estimators=rounds)
        full.fit(X, y, sample_weight=w, verbose=False)
        if list(full.classes_) != [0, 1, 2, 3]:
            raise RuntimeError(f"unexpected xgb classes_ {full.classes_}")
        total += full.predict_proba(Xf)
        log(f"xgb rank4 {cutoff} seed {seed} rounds {rounds} train {len(train)} in {time.time()-t0:.0f}s")
    proba = total / len(SEEDS)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def coordinate_retune(products, origins, start):
    groups = {
        "Fuel": [FUEL17],
        "Stable": [STAB],
        "Inactivity": [INACT],
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
    started = time.time()
    proposal = json.loads(PROPOSAL.read_text())
    screen, folds_gate = proposal["screen_gate"], proposal["eval_gate"]
    vR = np.array(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    log("panel loaded")

    baseline = {}
    for cutoff in POOL + EVAL:
        ids, y = load_labels(cutoff)
        n = len(ids)
        p17, p4 = load_proba("category_dynamics", cutoff, n), load_rank4(cutoff, n)
        baseline[cutoff] = weighted_f1(np.argmax(rank4_combine(p17, p4, "full") * vR, axis=1), y)
    for cutoff, expected in proposal["baseline"]["pool_origin_f1"].items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}")
    log(f"baseline r4full+vR: { {c: round(b, 6) for c, b in baseline.items()} }")

    decision = {
        "proposal_sha256": PROPOSAL_SHA,
        "baseline": {c: float(b) for c, b in baseline.items()},
        "variants": {},
    }
    summary = {}

    for variant in proposal["variants"]:
        t_v = time.time()
        log(f"### screening variant {variant}")
        prods_cycle, scores = {}, {}
        for cutoff in POOL:
            ids, y = load_labels(cutoff)
            n = len(ids)
            p17 = load_proba("category_dynamics", cutoff, n)
            p4std = load_rank4(cutoff, n)
            p4xgb = fit_member(cutoff, panel)
            p4 = p4xgb if variant == "xgb" else 0.5 * p4std + 0.5 * p4xgb
            prod = rank4_combine(p17, p4, "full")
            f1 = weighted_f1(np.argmax(prod * vR, axis=1), y)
            prods_cycle[cutoff] = (prod, y)
            scores[cutoff] = f1
            log(f"  {variant} {cutoff}: {f1:.6f}  gain {f1 - baseline[cutoff]:+.6f}")
        gains = {c: scores[c] - baseline[c] for c in POOL}
        res = {
            "scores": scores, "gains": gains,
            "mean_gain": float(np.mean(list(gains.values()))),
            "min_gain": float(np.min(list(gains.values()))),
        }
        passed = res["mean_gain"] >= screen["mean_gain_min"] and res["min_gain"] >= screen["per_origin_min"]
        if not passed:
            mult_fb, pooled_fb = coordinate_retune(prods_cycle, POOL, vR)
            gains_fb = {c: weighted_f1(np.argmax(prods_cycle[c][0] * mult_fb, axis=1), prods_cycle[c][1]) - baseline[c] for c in POOL}
            res["retune"] = {
                "vector": [float(x) for x in mult_fb],
                "pooled_mean_f1": pooled_fb,
                "gains": gains_fb,
                "mean_gain": float(np.mean(list(gains_fb.values()))),
                "min_gain": float(np.min(list(gains_fb.values()))),
            }
            passed = res["retune"]["mean_gain"] >= screen["mean_gain_min"] and res["retune"]["min_gain"] >= screen["per_origin_min"]
            mult_used = mult_fb
            log(f"  retune gain {res['retune']['mean_gain']:+.6f} min {res['retune']['min_gain']:+.6f} -> pass={passed}")
        else:
            mult_used = vR
        res["passed_screen"] = bool(passed)
        res["decode"] = [float(x) for x in mult_used]
        res["seconds"] = round(time.time() - t_v, 1)
        summary[variant] = res

        if not passed:
            log(f"  variant {variant}: SCREEN FAILED (mean {res['mean_gain']:+.6f})")
            continue

        fold_scores, fold_gains = {}, {}
        for cutoff in EVAL:
            ids, y = load_labels(cutoff)
            n = len(ids)
            p17 = load_proba("category_dynamics", cutoff, n)
            p4std = load_rank4(cutoff, n)
            p4xgb = fit_member(cutoff, panel)
            p4 = p4xgb if variant == "xgb" else 0.5 * p4std + 0.5 * p4xgb
            prod = rank4_combine(p17, p4, "full")
            f1 = weighted_f1(np.argmax(prod * mult_used, axis=1), y)
            fold_scores[cutoff] = f1
            fold_gains[cutoff] = f1 - baseline[cutoff]
            log(f"  {variant} EVAL {cutoff}: {f1:.6f}  gain {fold_gains[cutoff]:+.6f}")
        fold_ok = all(g >= folds_gate["per_origin_min"] for g in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        res["fold_scores"], res["fold_gains"], res["passed_folds"] = fold_scores, fold_gains, bool(fold_ok)
        log(f"  variant {variant}: FOLDS {'PASS' if fold_ok else 'FAIL'}")

    candidates = {v: r for v, r in summary.items() if r.get("passed_screen") and r.get("passed_folds")}
    if candidates:
        best_variant = max(candidates, key=lambda v: min(candidates[v]["fold_gains"].values()))
        mult_used = np.array(candidates[best_variant]["decode"])
        log(f"### December fit for {best_variant}")
        frame = panel["frames"][TEST]
        ids, _ = load_labels(TEST)
        n = len(ids)
        p17 = load_proba("category_dynamics", TEST, n)
        p4std = load_rank4(TEST, n)
        p4xgb = fit_member(TEST, panel)
        p4 = p4xgb if best_variant == "xgb" else 0.5 * p4std + 0.5 * p4xgb
        prod = rank4_combine(p17, p4, "full")
        pred = np.asarray(CLASSES)[np.argmax(prod * mult_used, axis=1)]
        baseline_csv = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
        if list(baseline_csv["ID"].astype(str)) != list(frame.index.astype(str)):
            raise RuntimeError("December frame order does not match the source submission")
        out = baseline_csv.copy()
        out["Opportunity"] = pred
        if not np.array_equal(out["CLV_fuel"].to_numpy(), baseline_csv["CLV_fuel"].to_numpy()):
            raise RuntimeError("fuel column changed during export")
        if not np.array_equal(out["CLV_nonfuel"].to_numpy(), baseline_csv["CLV_nonfuel"].to_numpy()):
            raise RuntimeError("nonfuel column changed during export")
        path = ROOT / "submissions" / "sasol_sprint17_xgbmix.csv"
        out.to_csv(path, index=False)
        changes = int((out["Opportunity"].to_numpy() != baseline_csv["Opportunity"].to_numpy()).sum())
        (OUT17 / "xgbmix_audit.json").write_text(json.dumps({
            "source": "submissions/sasol_sprint16_r4full_vR.csv",
            "file": "submissions/sasol_sprint17_xgbmix.csv",
            "variant": best_variant,
            "decode": [float(x) for x in mult_used],
            "changes": changes,
            "rows": int(len(out)),
        }, indent=2))
        decision["exported"] = {"file": path.name, "variant": best_variant, "changes_vs_best": changes}
        log(f"exported {path.name} with {changes} label changes")

    decision["variants"] = summary
    decision["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "xgb_rank4_decision.json").write_text(json.dumps(decision, indent=2))
    log("wrote xgb_rank4_decision.json")


if __name__ == "__main__":
    main()
