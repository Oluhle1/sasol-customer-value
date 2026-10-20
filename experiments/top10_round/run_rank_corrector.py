"""Second-rank Opportunity corrector.

The accepted category-dynamics classifier stays the probability head.
This model may only replace its top label with its second label.
It is trained on March, April and May 2025 out-of-fold scores.
June through September are evaluation origins and are not used to fit
or to choose the decision rule.

Gate, fixed before the evaluation labels are scored:
- June and September weighted F1 each improve;
- mean gain across June, July, August and September is at least 0.004;
- no evaluation origin loses more than 0.002;
- at least 3 of the 4 evaluation origins improve.

If the gate passes, the same hyperparameters are refit on March through
September out-of-fold scores and applied to the December forecast.
That refit does not change the gate.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
from run_monthly import predict  # noqa: E402

OCT = ROOT / "outputs" / "oct10"
DEST = ROOT / "outputs" / "top10_round"
DEST.mkdir(parents=True, exist_ok=True)
PROGRESS = DEST / "progress.log"

TRAIN_ORIGINS = ["2025-03-01", "2025-04-01", "2025-05-01"]
EVAL_ORIGINS = ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
REFIT_ORIGINS = TRAIN_ORIGINS + EVAL_ORIGINS
NEEDED = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
NUMERIC = [
    "fuel_recency",
    "fuel_overdue",
    "fuel_growth_gap",
    "recency_days",
    "fuel_z_q1",
    "shop_z_q1",
    "dyn_fuel_recency",
    "dyn_fuel_change",
    "dyn_fuel_purchases",
    "dyn_tx_0_14_n",
    "dyn_fuel_tx_0_14_n",
]
PARAMS = dict(
    n_estimators=60,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    random_state=7,
    n_jobs=1,
    verbosity=-1,
)


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def finish(status, message):
    log(f"{status}: {message}")
    raise SystemExit(0 if status == "DONE" else 1)


def weighted_f1(y, pred):
    return float(f1_score(y, pred, average="weighted", zero_division=0))


def load_decision():
    payload = json.loads((OCT / "category_dynamics_decision.json").read_text())
    classes = payload["classes"]
    multipliers = np.asarray(payload["multipliers"], dtype=float)
    if len(classes) != 17 or len(multipliers) != 17:
        finish("FAILED", "category-dynamics decision does not have 17 classes")
    return classes, multipliers


def proba_path(cutoff):
    existing = OCT / f"{cutoff}_category_dynamics.npy"
    if existing.exists() and cutoff not in NEEDED:
        return existing
    return DEST / "probs" / f"{cutoff}_category_dynamics.npz"


def save_proba(cutoff, proba, ids):
    path = DEST / "probs" / f"{cutoff}_category_dynamics.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, proba=proba, ids=np.asarray(ids).astype(str))
    return path


def load_proba(cutoff, ids):
    path = proba_path(cutoff)
    if path.suffix == ".npy":
        proba = np.load(path)
        if len(proba) != len(ids):
            finish("FAILED", f"{cutoff} saved probabilities have {len(proba)} rows, frame has {len(ids)}")
        return proba
    payload = np.load(path, allow_pickle=False)
    if not np.array_equal(payload["ids"].astype(str), np.asarray(ids).astype(str)):
        finish("FAILED", f"{cutoff} probability IDs do not match the feature frame")
    return payload["proba"]


def design(proba, frame, classes, multipliers):
    missing = [name for name in NUMERIC if name not in frame.columns]
    if missing:
        finish("FAILED", "missing features: " + ", ".join(missing))
    adjusted = proba * multipliers
    order = np.argsort(-adjusted, axis=1)
    rank1 = order[:, 0]
    rank2 = order[:, 1]
    rows = np.arange(len(proba))
    p1 = adjusted[rows, rank1]
    p2 = adjusted[rows, rank2]
    margin = p1 - p2
    share = p1 / np.maximum(p1 + p2, 1e-12)
    numbers = frame[NUMERIC].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        finish("FAILED", "non-finite corrector features")
    eye = np.eye(len(classes), dtype=float)
    matrix = np.column_stack([p1, p2, margin, share, numbers, eye[rank1], eye[rank2]])
    return matrix, rank1, rank2


def outcomes(y, rank1, rank2, classes):
    index = pd.Series(np.arange(len(classes)), index=classes)
    true = index.reindex(y).to_numpy()
    if np.isnan(true.astype(float)).any():
        finish("FAILED", "unexpected opportunity label")
    true = true.astype(int)
    target = np.full(len(y), 2, dtype=int)
    target[true == rank1] = 0
    target[true == rank2] = 1
    return target


def apply_rule(proba_class, classes, rank1, rank2):
    flip = (proba_class[:, 1] > proba_class[:, 0]) & (proba_class[:, 1] > proba_class[:, 2])
    chosen = rank1.copy()
    chosen[flip] = rank2[flip]
    return np.asarray(classes)[chosen], flip


def fit_corrector(panel, origins, classes, multipliers):
    matrices, targets = [], []
    for cutoff in origins:
        frame = panel["frames"][cutoff]
        proba = load_proba(cutoff, frame.index)
        matrix, rank1, rank2 = design(proba, frame, classes, multipliers)
        y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
        matrices.append(matrix)
        targets.append(outcomes(y, rank1, rank2, classes))
    model = LGBMClassifier(**PARAMS)
    model.fit(np.vstack(matrices), np.concatenate(targets))
    return model


def score_origin(panel, model, cutoff, classes, multipliers):
    frame = panel["frames"][cutoff]
    proba = load_proba(cutoff, frame.index)
    matrix, rank1, rank2 = design(proba, frame, classes, multipliers)
    y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
    base = np.asarray(classes)[rank1]
    pred, flip = apply_rule(model.predict_proba(matrix), classes, rank1, rank2)
    base_f1 = weighted_f1(y, base)
    new_f1 = weighted_f1(y, pred)
    return {
        "cutoff": cutoff,
        "rows": int(len(y)),
        "baseline_f1": base_f1,
        "candidate_f1": new_f1,
        "gain": new_f1 - base_f1,
        "flips": int(flip.sum()),
        "flip_share": float(flip.mean()),
    }


def main():
    PROGRESS.write_text("")
    proposal = {
        "hypothesis": "When the accepted classifier's true class is its second label, a model using only that ranking and past-only customer state can choose between the two labels without refitting the head.",
        "difference_from_rejected_experiments": "The October 14 rule model replaced the probability head with logistic regression. This corrector cannot change any label outside the head's top two and cannot alter class probabilities.",
        "train_origins": TRAIN_ORIGINS,
        "eval_origins": EVAL_ORIGINS,
        "features": ["adjusted_p1", "adjusted_p2", "margin", "p1_share"] + NUMERIC + ["rank1_one_hot", "rank2_one_hot"],
        "model": PARAMS,
        "decision": "Flip only when P(second is true) exceeds both P(first is true) and P(neither).",
        "gate": "June and September each improve; mean of four evaluation gains >= 0.004; no loss worse than -0.002; at least 3 of 4 improve.",
        "deployment_if_accepted": "Refit the same model on March-September out-of-fold scores, then apply it to the saved December category-dynamics probabilities. Preserve both CLV columns from the 0.309164264 submission.",
    }
    (DEST / "proposal.json").write_text(json.dumps(proposal, indent=2))
    log("proposal saved")
    classes, multipliers = load_decision()
    log("loading category panel")
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    if "monthly_cutoffs" not in panel:
        finish("FAILED", "category panel has no monthly cutoffs")
    for cutoff in TRAIN_ORIGINS + EVAL_ORIGINS + ["2025-12-01"]:
        if cutoff not in panel["frames"]:
            finish("FAILED", f"{cutoff} is missing from the category panel")
    june = panel["frames"]["2025-06-01"]
    june_proba = load_proba("2025-06-01", june.index)
    june_y = panel["labels"]["2025-06-01"].reindex(june.index).Opportunity.to_numpy()
    june_pred = np.asarray(classes)[np.argmax(june_proba * multipliers, axis=1)]
    june_f1 = weighted_f1(june_y, june_pred)
    if abs(june_f1 - 0.5095271507692096) > 1e-12:
        finish("FAILED", f"June baseline F1 {june_f1} does not match the archived category model")
    log(f"June baseline reproduced at {june_f1:.10f}")
    for cutoff in NEEDED:
        path = DEST / "probs" / f"{cutoff}_category_dynamics.npz"
        if path.exists():
            log(f"reuse {cutoff} probabilities")
            continue
        log(f"fitting out-of-fold head for {cutoff}")
        started = time.time()
        proba = predict(panel, cutoff, threads=4)
        save_proba(cutoff, proba, panel["frames"][cutoff].index)
        log(f"saved {cutoff} in {time.time() - started:.0f}s")
    log("fitting corrector on March-May")
    model = fit_corrector(panel, TRAIN_ORIGINS, classes, multipliers)
    rows = [score_origin(panel, model, cutoff, classes, multipliers) for cutoff in EVAL_ORIGINS]
    gains = [row["gain"] for row in rows]
    by_cutoff = {row["cutoff"]: row["gain"] for row in rows}
    accepted = (
        by_cutoff["2025-06-01"] > 0
        and by_cutoff["2025-09-01"] > 0
        and float(np.mean(gains)) >= 0.004
        and min(gains) >= -0.002
        and sum(gain > 0 for gain in gains) >= 3
    )
    decision = {"accepted": bool(accepted), "rows": rows, "mean_gain": float(np.mean(gains))}
    (DEST / "rank_corrector_decision.json").write_text(json.dumps(decision, indent=2))
    log("evaluation " + json.dumps(decision))
    if not accepted:
        finish("DONE", "corrector failed the pre-registered gate; baseline submission unchanged")
    log("gate passed; refitting on March-September")
    final = fit_corrector(panel, REFIT_ORIGINS, classes, multipliers)
    frame = panel["frames"]["2025-12-01"]
    proba = load_proba("2025-12-01", frame.index)
    matrix, rank1, rank2 = design(proba, frame, classes, multipliers)
    pred, flip = apply_rule(final.predict_proba(matrix), classes, rank1, rank2)
    baseline_path = ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv"
    baseline = pd.read_csv(baseline_path, dtype={"ID": str})
    mapped = pd.Series(pred, index=frame.index.astype(str))
    if not baseline.ID.isin(mapped.index).all() or mapped.loc[baseline.ID].isna().any():
        finish("FAILED", "December IDs do not cover the baseline submission")
    out = baseline.copy()
    out["Opportunity"] = mapped.loc[out.ID].to_numpy()
    if list(out.columns) != ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]:
        finish("FAILED", "submission columns changed")
    if not out.Opportunity.isin(classes).all():
        finish("FAILED", "prediction outside the 17 official labels")
    changed = int((out.Opportunity.to_numpy() != baseline.Opportunity.to_numpy()).sum())
    output = ROOT / "submissions" / "sasol_top10_rank_corrector.csv"
    out.to_csv(output, index=False)
    audit = {
        "accepted": True,
        "rows": int(len(out)),
        "opportunity_changes": changed,
        "december_flips": int(flip.sum()),
        "clv_unchanged": bool(
            np.array_equal(out.CLV_fuel.to_numpy(), baseline.CLV_fuel.to_numpy())
            and np.array_equal(out.CLV_nonfuel.to_numpy(), baseline.CLV_nonfuel.to_numpy())
        ),
        "evaluation": rows,
    }
    (DEST / "submission_audit.json").write_text(json.dumps(audit, indent=2))
    finish("DONE", f"wrote {output.name} with {changed} Opportunity changes")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
