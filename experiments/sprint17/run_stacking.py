"""Sprint 17 E7 — conditional stacking meta-classifier.

Pre-registration: experiments/sprint17/PROPOSAL_E7.md and
outputs/sprint17/proposal_stacking.json. Training is strictly causal: to
predict origin O the meta trains only on origins X with X + 3m <= O.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))

from evaluate_saved import (  # noqa: E402
    CLASSES, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

OUT17 = ROOT / "outputs" / "sprint17"
PROPOSAL = OUT17 / "proposal_stacking.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
CACHED = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01",
          "2025-07-01", "2025-08-01", "2025-09-01"]
CONTEXT = ["n_tx", "recency_days", "fuel_z_mean", "shop_z_mean", "crossbuy_q1"]
META_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.2, n_jobs=4, verbosity=-1,
)
SEEDS = (7, 23)
CODE = {name: i for i, name in enumerate(CLASSES)}


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def load_rank4_local(cutoff, n):
    p = load_rank4(cutoff, n)
    if p is not None:
        return p
    path = OUT17 / "price_rank4" / f"{cutoff}_rank4_base.npz"
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if len(payload["proba"]) == n:
            return payload["proba"]
    raise RuntimeError(f"rank4 cache missing for {cutoff}")


def member_matrix(cutoff, panel):
    ids, y = load_labels(cutoff)
    n = len(ids)
    frame = panel["frames"][cutoff]
    catdyn = load_proba("category_dynamics", cutoff, n)
    rank4 = load_rank4_local(cutoff, n)
    xgb_path = OUT17 / "xgb_rank4" / f"{cutoff}_rank4_xgb.npz"
    xgb = None
    if xgb_path.exists():
        payload = np.load(xgb_path, allow_pickle=True)
        if len(payload["proba"]) == n:
            xgb = payload["proba"]
    cols = {}
    for i in range(17):
        cols[f"catdyn_{i}"] = catdyn[:, i]
    for i in range(4):
        cols[f"rank4_{i}"] = rank4[:, i]
        cols[f"xgb_{i}"] = xgb[:, i] if xgb is not None else np.full(n, np.nan)
    ctx = frame[CONTEXT].to_numpy(dtype=float)
    for i, name in enumerate(CONTEXT):
        cols[f"ctx_{name}"] = ctx[:, i]
    matrix = pd.DataFrame(cols, index=pd.Index(ids))
    target = np.array([CODE[v] for v in y], dtype=int)
    return matrix, target, y


def train_origins_for(target, available):
    limit = pd.Timestamp(target) - pd.DateOffset(months=3)
    return [c for c in available if pd.Timestamp(c) <= limit]


def fit_meta(train_x, train_y):
    proba = np.zeros((len(train_x["_predict"]), len(CLASSES)))
    for seed in SEEDS:
        clf = LGBMClassifier(**dict(META_PARAMS, random_state=seed))
        clf.fit(train_x["_train"], train_y)
        proba += clf.predict_proba(train_x["_predict"])
    return proba / len(SEEDS)


def main():
    started = time.time()
    proposal = json.loads(PROPOSAL.read_text())
    screen, folds_gate = proposal["screen_gate"], proposal["eval_gate"]
    vR = np.array(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    log("panel loaded")

    blocks, baseline, ytrue = {}, {}, {}
    for cutoff in CACHED:
        m, t, y = member_matrix(cutoff, panel)
        blocks[cutoff] = (m, t)
        if cutoff in POOL + EVAL:
            ids, _ = load_labels(cutoff)
            prod = rank4_combine(load_proba("category_dynamics", cutoff, len(ids)),
                                 load_rank4_local(cutoff, len(ids)), "full")
            ytrue[cutoff] = y
            baseline[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
    for cutoff, expected in proposal["baseline"]["known_pool_f1"].items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}")
    log(f"baseline r4full+vR: { {c: round(b, 6) for c, b in baseline.items()} }")

    decision = {"proposal_sha256": PROPOSAL_SHA, "baseline": {c: float(b) for c, b in baseline.items()}}
    screen_origins = proposal["screen_gate"]["origins"]
    scores, gains = {}, {}
    for cutoff in screen_origins:
        train_set = train_origins_for(cutoff, CACHED)
        if not train_set:
            raise RuntimeError(f"no eligible training origins for {cutoff}")
        train_m = pd.concat([blocks[c][0] for c in train_set])
        train_y = np.concatenate([blocks[c][1] for c in train_set])
        proba = fit_meta({"_train": train_m, "_predict": blocks[cutoff][0]}, train_y)
        scores[cutoff] = weighted_f1(np.argmax(proba * vR, axis=1), ytrue[cutoff])
        gains[cutoff] = scores[cutoff] - baseline[cutoff]
        log(f"  stack {cutoff} (train {train_set}): {scores[cutoff]:.6f}  gain {gains[cutoff]:+.6f}")
    mean_gain = float(np.mean(list(gains.values())))
    min_gain = float(np.min(list(gains.values())))
    passed = mean_gain >= screen["mean_gain_min"] and min_gain >= screen["per_origin_min"]
    decision["screen"] = {"origins": screen_origins, "scores": scores, "gains": gains,
                          "mean_gain": mean_gain, "min_gain": min_gain, "passed": bool(passed)}
    log(f"  screen mean {mean_gain:+.6f} min {min_gain:+.6f} pass={passed}")

    if passed:
        fold_scores, fold_gains = {}, {}
        for cutoff in EVAL:
            train_set = train_origins_for(cutoff, CACHED)
            train_m = pd.concat([blocks[c][0] for c in train_set])
            train_y = np.concatenate([blocks[c][1] for c in train_set])
            proba = fit_meta({"_train": train_m, "_predict": blocks[cutoff][0]}, train_y)
            fold_scores[cutoff] = weighted_f1(np.argmax(proba * vR, axis=1), ytrue[cutoff])
            fold_gains[cutoff] = fold_scores[cutoff] - baseline[cutoff]
            log(f"  stack EVAL {cutoff} (train {train_set}): {fold_scores[cutoff]:.6f}  gain {fold_gains[cutoff]:+.6f}")
        folds_ok = all(g >= folds_gate["per_origin_min"] for g in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        decision["folds"] = {"scores": fold_scores, "gains": fold_gains, "passed": bool(folds_ok)}
        log(f"  folds pass={folds_ok}")
        if folds_ok:
            test_m, _, _ = member_matrix(TEST, panel)
            train_set = train_origins_for(TEST, CACHED)
            train_m = pd.concat([blocks[c][0] for c in train_set])
            train_y = np.concatenate([blocks[c][1] for c in train_set])
            proba = fit_meta({"_train": train_m, "_predict": test_m}, train_y)
            pred = np.asarray(CLASSES)[np.argmax(proba * vR, axis=1)]
            frame = panel["frames"][TEST]
            source = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
            if list(source["ID"].astype(str)) != list(frame.index.astype(str)):
                raise RuntimeError("December order does not match the source submission")
            out = source.copy()
            out["Opportunity"] = pred
            for col in ("CLV_fuel", "CLV_nonfuel"):
                if not np.array_equal(out[col].to_numpy(), source[col].to_numpy()):
                    raise RuntimeError(f"{col} changed during export")
            path = ROOT / "submissions" / "sasol_sprint17_stacking.csv"
            out.to_csv(path, index=False)
            changes = int((out["Opportunity"].to_numpy() != source["Opportunity"].to_numpy()).sum())
            (OUT17 / "stacking_audit.json").write_text(json.dumps({
                "source": "submissions/sasol_sprint16_r4full_vR.csv",
                "file": "submissions/sasol_sprint17_stacking.csv",
                "train_origins": train_set, "changes": changes, "rows": int(len(out)),
            }, indent=2))
            decision["exported"] = {"file": path.name, "changes_vs_best": changes}
            log(f"exported {path.name} with {changes} label changes")
    else:
        log("screen failed; folds and December were not fit, nothing exported")

    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "stacking_decision.json").write_text(json.dumps(decision, indent=2))
    log("wrote stacking_decision.json")


if __name__ == "__main__":
    main()
