"""Sprint 17 E8 — EM prior correction of the r4full product (zero fit).

Pre-registration: experiments/sprint17/PROPOSAL_E8.md and
outputs/sprint17/proposal_em_prior.json. Gates are read from the JSON.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))

from evaluate_saved import (  # noqa: E402
    CLASSES, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

OUT17 = ROOT / "outputs" / "sprint17"
PROPOSAL = OUT17 / "proposal_em_prior.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"


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


def prior_from_labels(cutoff):
    ids, y = load_labels(cutoff)
    counts = pd.Series(y).value_counts().reindex(CLASSES).fillna(0).to_numpy(dtype=float)
    priors = counts / counts.sum()
    priors = np.maximum(priors, 1.0 / len(CLASSES))
    return priors / priors.sum()


def em_adjust(p, target, max_iter=500, tol=1e-7):
    q = p.mean(axis=0)
    for _ in range(max_iter):
        w = p * (target / np.clip(q, 1e-12, None))
        w /= w.sum(axis=1, keepdims=True)
        q_new = w.mean(axis=0)
        if np.max(np.abs(q_new - q)) < tol:
            return w
        q = q_new
    return w


def main():
    started = time.time()
    proposal = json.loads(PROPOSAL.read_text())
    screen, folds_gate = proposal["screen_gate"], proposal["eval_gate"]
    vR = np.array(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)

    baseline, products, ytrue = {}, {}, {}
    for cutoff in POOL + EVAL:
        ids, y = load_labels(cutoff)
        n = len(ids)
        prod = rank4_combine(load_proba("category_dynamics", cutoff, n), load_rank4_local(cutoff, n), "full")
        products[cutoff] = prod
        ytrue[cutoff] = y
        baseline[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
    for cutoff, expected in {"2025-04-01": 0.5006451168766207, "2025-05-01": 0.5060866657571113,
                             "2025-07-01": 0.5031954596858046, "2025-08-01": 0.5059436172953636}.items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}")
    log(f"baseline r4full+vR: { {c: round(b, 6) for c, b in baseline.items()} }")

    decision = {"proposal_sha256": PROPOSAL_SHA, "baseline": {c: float(b) for c, b in baseline.items()}}
    scores, gains = {}, {}
    for cutoff in POOL:
        prior_cutoff = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        target = prior_from_labels(prior_cutoff)
        w = em_adjust(products[cutoff], target)
        scores[cutoff] = weighted_f1(np.argmax(w * vR, axis=1), ytrue[cutoff])
        gains[cutoff] = scores[cutoff] - baseline[cutoff]
        log(f"  em {cutoff} (prior {prior_cutoff}): {scores[cutoff]:.6f}  gain {gains[cutoff]:+.6f}")
    mean_gain = float(np.mean(list(gains.values())))
    min_gain = float(np.min(list(gains.values())))
    passed = mean_gain >= screen["mean_gain_min"] and min_gain >= screen["per_origin_min"]
    decision["screen"] = {"scores": scores, "gains": gains, "mean_gain": mean_gain, "min_gain": min_gain, "passed": bool(passed)}
    log(f"  screen mean {mean_gain:+.6f} min {min_gain:+.6f} pass={passed}")

    if passed:
        fold_scores, fold_gains = {}, {}
        for cutoff in EVAL:
            prior_cutoff = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
            w = em_adjust(products[cutoff], prior_from_labels(prior_cutoff))
            fold_scores[cutoff] = weighted_f1(np.argmax(w * vR, axis=1), ytrue[cutoff])
            fold_gains[cutoff] = fold_scores[cutoff] - baseline[cutoff]
            log(f"  em EVAL {cutoff}: {fold_scores[cutoff]:.6f}  gain {fold_gains[cutoff]:+.6f}")
        folds_ok = all(g >= folds_gate["per_origin_min"] for g in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        decision["folds"] = {"scores": fold_scores, "gains": fold_gains, "passed": bool(folds_ok)}
        log(f"  folds pass={folds_ok}")
        if folds_ok:
            ids, _ = load_labels(TEST)
            n = len(ids)
            prod = rank4_combine(load_proba("category_dynamics", TEST, n), load_rank4_local(TEST, n), "full")
            target = prior_from_labels("2025-09-01")
            w = em_adjust(prod, target)
            pred = np.asarray(CLASSES)[np.argmax(w * vR, axis=1)]
            source = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
            frame_ids = load_labels(TEST)[0]
            if list(source["ID"].astype(str)) != list(frame_ids):
                raise RuntimeError("December order does not match the source submission")
            out = source.copy()
            out["Opportunity"] = pred
            for col in ("CLV_fuel", "CLV_nonfuel"):
                if not np.array_equal(out[col].to_numpy(), source[col].to_numpy()):
                    raise RuntimeError(f"{col} changed during export")
            path = ROOT / "submissions" / "sasol_sprint17_emprior.csv"
            out.to_csv(path, index=False)
            changes = int((out["Opportunity"].to_numpy() != source["Opportunity"].to_numpy()).sum())
            (OUT17 / "emprior_audit.json").write_text(json.dumps({
                "source": "submissions/sasol_sprint16_r4full_vR.csv",
                "file": "submissions/sasol_sprint17_emprior.csv",
                "changes": changes, "rows": int(len(out)),
            }, indent=2))
            decision["exported"] = {"file": path.name, "changes_vs_best": changes}
            log(f"exported {path.name} with {changes} label changes")
    else:
        log("screen failed; folds not scored, nothing exported")

    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "em_prior_decision.json").write_text(json.dumps(decision, indent=2))
    log("wrote em_prior_decision.json")


if __name__ == "__main__":
    main()
