"""Sprint 17 E10 — rank4 head refit with the frozen 27-column embedding block.

Pre-registration: experiments/sprint17/PROPOSAL_E10.md and
outputs/sprint17/proposal_item_embeddings.json. Gates and the frozen feature
list are read from that JSON. Fit protocol is the production rank4 recipe
(run_rank4.fit_rank); only the feature matrix changes (305 -> 332 columns).
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))

import feature_pipeline as fp  # noqa: E402
from evaluate_saved import (  # noqa: E402
    CLASSES, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

OUT17 = ROOT / "outputs" / "sprint17"
DEST = OUT17 / "embed_rank4"
DEST.mkdir(parents=True, exist_ok=True)
PROPOSAL = OUT17 / "proposal_item_embeddings.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
EMB = pickle.loads((OUT17 / "embed_features.pkl").read_bytes())

POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"

PARAMS = dict(
    n_estimators=1200, learning_rate=0.04, num_leaves=47, min_child_samples=35,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2,
    n_jobs=4, deterministic=True, force_col_wise=True, verbosity=-1,
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


def matrix(frame, panel, extra, origin):
    cols = panel["feature_cols"]
    out = frame[cols]
    if extra:
        out = out.join(EMB[origin])
    return out


def fit_rank(panel, cutoff, extra):
    path = DEST / f"{cutoff}_rank4_{'ext' if extra else 'base'}.npz"
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
        block = matrix(origin_frame, panel, extra, origin).copy()
        block["target"] = collapse(labels)
        block["cutoff"] = origin
        block["w"] = weight_of(origin)
        blocks.append(block)
    train = pd.concat(blocks)
    features = [c for c in train.columns if c not in ("target", "cutoff", "w")]
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if not (pd.to_datetime(train.loc[tr, "cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("rank training crosses the inner cutoff")
    predict_frame = matrix(frame, panel, extra, cutoff)
    total = np.zeros((len(frame), 4))
    for seed in SEEDS:
        t0 = time.time()
        inner = LGBMClassifier(**dict(PARAMS, random_state=seed))
        inner.fit(
            train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
            eval_set=[(train.loc[va, features], train.loc[va, "target"])],
            callbacks=[early_stopping(60, verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMClassifier(**dict(PARAMS, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train.target, sample_weight=train.w)
        total += fp.align_proba(full, predict_frame, RANK)
        log(f"rank4 {'ext' if extra else 'base'} {cutoff} seed {seed} rounds {rounds} train {len(train)} in {time.time()-t0:.0f}s")
    proba = total / len(SEEDS)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def main():
    started = time.time()
    proposal = json.loads(PROPOSAL.read_text())
    screen, folds_gate = proposal["screen_gate"], proposal["eval_gate"]
    vR = np.array(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    log("panel loaded")

    baseline, catdyn_cache = {}, {}
    for cutoff in POOL + EVAL:
        ids, y = load_labels(cutoff)
        n = len(ids)
        p17 = load_proba("category_dynamics", cutoff, n)
        catdyn_cache[cutoff] = p17
        p4 = load_rank4(cutoff, n)
        if p4 is None:
            if cutoff != "2025-03-01":
                raise RuntimeError(f"production rank4 cache missing for {cutoff}")
            log("fitting production baseline head for 2025-03-01")
            p4 = fit_rank(panel, cutoff, extra=False)
        prod = rank4_combine(p17, p4, "full")
        baseline[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
    for cutoff, expected in proposal["baseline"]["pool_origin_f1_known"].items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}")
    log(f"baseline r4full+vR: { {c: round(b, 6) for c, b in baseline.items()} }")

    decision = {
        "proposal_sha256": PROPOSAL_SHA,
        "baseline": {c: float(b) for c, b in baseline.items()},
    }
    log("### screening extended head on pool")
    scores, gains = {}, {}
    for cutoff in POOL:
        ids, y = load_labels(cutoff)
        p4 = fit_rank(panel, cutoff, extra=True)
        prod = rank4_combine(catdyn_cache[cutoff], p4, "full")
        scores[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
        gains[cutoff] = scores[cutoff] - baseline[cutoff]
        log(f"  ext {cutoff}: {scores[cutoff]:.6f}  gain {gains[cutoff]:+.6f}")
    mean_gain = float(np.mean(list(gains.values())))
    min_gain = float(np.min(list(gains.values())))
    passed = mean_gain >= screen["mean_gain_min"] and min_gain >= screen["per_origin_min"]
    decision["screen"] = {"scores": scores, "gains": gains, "mean_gain": mean_gain,
                          "min_gain": min_gain, "passed": bool(passed)}
    log(f"  screen mean {mean_gain:+.6f} min {min_gain:+.6f} pass={passed}")

    if passed:
        fold_scores, fold_gains = {}, {}
        for cutoff in EVAL:
            ids, y = load_labels(cutoff)
            p4 = fit_rank(panel, cutoff, extra=True)
            prod = rank4_combine(catdyn_cache[cutoff], p4, "full")
            fold_scores[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
            fold_gains[cutoff] = fold_scores[cutoff] - baseline[cutoff]
            log(f"  ext EVAL {cutoff}: {fold_scores[cutoff]:.6f}  gain {fold_gains[cutoff]:+.6f}")
        folds_ok = all(g >= folds_gate["per_origin_min"] for g in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        decision["folds"] = {"scores": fold_scores, "gains": fold_gains, "passed": bool(folds_ok)}
        log(f"  folds pass={folds_ok}")
        if folds_ok:
            frame = panel["frames"][TEST]
            ids, _ = load_labels(TEST)
            p17 = load_proba("category_dynamics", TEST, len(ids))
            p4 = fit_rank(panel, TEST, extra=True)
            prod = rank4_combine(p17, p4, "full")
            pred = np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)]
            source = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
            if list(source["ID"].astype(str)) != list(frame.index.astype(str)):
                raise RuntimeError("December frame order does not match the source submission")
            out = source.copy()
            out["Opportunity"] = pred
            for col in ("CLV_fuel", "CLV_nonfuel"):
                if not np.array_equal(out[col].to_numpy(), source[col].to_numpy()):
                    raise RuntimeError(f"{col} changed during export")
            path = ROOT / "submissions" / "sasol_sprint17_embfeat.csv"
            out.to_csv(path, index=False)
            changes = int((out["Opportunity"].to_numpy() != source["Opportunity"].to_numpy()).sum())
            (OUT17 / "embfeat_audit.json").write_text(json.dumps({
                "source": "submissions/sasol_sprint16_r4full_vR.csv",
                "file": "submissions/sasol_sprint17_embfeat.csv",
                "changes": changes,
                "rows": int(len(out)),
                "decode": "frozen vR",
            }, indent=2))
            decision["exported"] = {"file": path.name, "changes_vs_best": changes}
            log(f"exported {path.name} with {changes} label changes")
    else:
        log("screen failed; folds and December were not fit, nothing exported")

    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "embed_rank4_decision.json").write_text(json.dumps(decision, indent=2))
    log(f"wrote embed_rank4_decision.json in {decision['seconds']}s")


if __name__ == "__main__":
    main()
