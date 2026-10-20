"""Re-rank Inactivity, Stable and Fuel growth with a dedicated four-class model.

The accepted 17-class probabilities still choose the label inside the
remaining classes. Two combination rules are fixed before evaluation:

- full: the four-class probabilities set the mass of Inactivity, Stable,
  Fuel growth and everything else. The 17-class model only splits the
  everything-else mass.
- within: the 17-class model keeps its total mass on the three large
  classes. The four-class model only reallocates that mass.

Gate for each rule, fixed before scoring:
June and September weighted F1 each improve, the mean gain on June,
July, August and September is at least 0.004, at least 5 of the 6
April-September origins improve, and no origin loses more than 0.002.
If both rules pass, the larger six-origin mean gain is exported.
Both CLV columns stay on the 0.309164264 submission.

The lapsed-fuel model then replaces fuel predictions only where the last
fuel purchase was more than 30 days earlier. It must improve June and
September RMSE, with a six-origin mean gain of at least 0.001 and no
loss worse than 0.002.
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
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
import feature_pipeline as fp  # noqa: E402

OCT = ROOT / "outputs" / "oct10"
DEST = ROOT / "outputs" / "top10_round" / "rank4"
DEST.mkdir(parents=True, exist_ok=True)
PROGRESS = DEST / "progress.log"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"
BIG = ["Inactivity", "Stable", FUEL_LABEL]
EVAL = ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
JUNE_BASE = 0.5095271507692096
SEP_BASE = 0.5165568275346171


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def weighted_f1(y, pred):
    return float(f1_score(y, pred, average="weighted", zero_division=0))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def load_decision():
    payload = json.loads((OCT / "category_dynamics_decision.json").read_text())
    classes = payload["classes"]
    multipliers = np.asarray(payload["multipliers"], dtype=float)
    if len(classes) != 17 or len(multipliers) != 17:
        raise RuntimeError("expected 17 classes")
    return classes, multipliers


def load_proba(cutoff, ids):
    npz = ROOT / "outputs" / "top10_round" / "probs" / f"{cutoff}_category_dynamics.npz"
    npy = OCT / f"{cutoff}_category_dynamics.npy"
    if npz.exists():
        payload = np.load(npz, allow_pickle=False)
        if not np.array_equal(payload["ids"].astype(str), np.asarray(ids).astype(str)):
            raise RuntimeError(f"{cutoff} probability ids do not match the frame")
        return payload["proba"]
    proba = np.load(npy)
    if len(proba) != len(ids):
        raise RuntimeError(f"{cutoff} probability length does not match the frame")
    return proba


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def marginal(p17, classes):
    index = {name: i for i, name in enumerate(classes)}
    out = np.column_stack([
        p17[:, index["Inactivity"]],
        p17[:, index["Stable"]],
        p17[:, index[FUEL_LABEL]],
    ])
    other = 1.0 - out.sum(axis=1)
    return np.column_stack([out, other])


def combine(p17, p4, classes, mode):
    index = {name: i for i, name in enumerate(classes)}
    big = [index[name] for name in BIG]
    other = [i for i, name in enumerate(classes) if name not in BIG]
    if mode == "full":
        base = p17[:, other]
        cond = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-12)
        out = np.zeros_like(p17)
        out[:, index["Inactivity"]] = p4[:, 0]
        out[:, index["Stable"]] = p4[:, 1]
        out[:, index[FUEL_LABEL]] = p4[:, 2]
        out[:, other] = p4[:, 3:4] * cond
    elif mode == "within":
        mass = p17[:, big].sum(axis=1, keepdims=True)
        cond = p4[:, :3] / np.maximum(p4[:, :3].sum(axis=1, keepdims=True), 1e-12)
        out = p17.copy()
        out[:, big] = mass * cond
    else:
        raise RuntimeError(mode)
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise RuntimeError("combined probabilities are empty")
    return out / total


def labels_from(proba, classes, multipliers):
    return np.asarray(classes)[np.argmax(proba * multipliers, axis=1)]


def fit_rank(panel, cutoff, classes):
    path = DEST / f"{cutoff}_rank4.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=False)
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
        blocks.append(pd.DataFrame({
            "target": collapse(labels),
            "cutoff": origin,
            "w": weight_of(origin),
        }, index=origin_frame.index).join(origin_frame[panel["feature_cols"]]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if not (pd.to_datetime(train.loc[tr, "cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("rank training crosses the inner cutoff")
    features = panel["feature_cols"]
    params = dict(
        n_estimators=1200, learning_rate=0.04, num_leaves=47, min_child_samples=35,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2,
        n_jobs=4, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    total = np.zeros((len(frame), 4))
    for seed in (7, 23):
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
        log(f"rank4 {cutoff} seed {seed} rounds {rounds} train {len(train)}")
    proba = total / 2
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def score_rank(panel, classes, multipliers):
    rows = []
    cache = {}
    for cutoff in EVAL:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
        p17 = load_proba(cutoff, frame.index)
        p4 = fit_rank(panel, cutoff, classes)
        base = labels_from(p17, classes, multipliers)
        base_f1 = weighted_f1(y, base)
        record = {"cutoff": cutoff, "rows": int(len(y)), "baseline_f1": base_f1}
        for mode in ("full", "within"):
            pred = labels_from(combine(p17, p4, classes, mode), classes, multipliers)
            f1 = weighted_f1(y, pred)
            record[f"{mode}_f1"] = f1
            record[f"{mode}_gain"] = f1 - base_f1
            record[f"{mode}_changes"] = int((pred != base).sum())
        rows.append(record)
        cache[cutoff] = (y, base_f1)
        log(json.dumps(record))
        pd.DataFrame(rows).to_csv(DEST / "rank_comparison.csv", index=False)
    decisions = {}
    for mode in ("full", "within"):
        gains = {row["cutoff"]: row[f"{mode}_gain"] for row in rows}
        screen = [gains["2025-06-01"], gains["2025-09-01"]]
        later = [gains[c] for c in ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]]
        accepted = (
            min(screen) > 0
            and float(np.mean(later)) >= 0.004
            and sum(g > 0 for g in gains.values()) >= 5
            and min(gains.values()) >= -0.002
        )
        decisions[mode] = {"accepted": bool(accepted), "mean_six": float(np.mean(list(gains.values())))}
    chosen = None
    passing = [mode for mode, item in decisions.items() if item["accepted"]]
    if passing:
        chosen = max(passing, key=lambda mode: decisions[mode]["mean_six"])
    return rows, decisions, chosen


def export_rank(panel, classes, multipliers, mode):
    cutoff = fp.TEST_CUTOFF
    p4 = fit_rank(panel, cutoff, classes)
    frame = panel["frames"][cutoff]
    p17 = load_proba(cutoff, frame.index)
    pred = labels_from(combine(p17, p4, classes, mode), classes, multipliers)
    baseline = pd.read_csv(ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv", dtype={"ID": str})
    mapped = pd.Series(pred, index=frame.index.astype(str))
    out = baseline.copy()
    out["Opportunity"] = mapped.loc[out.ID].to_numpy()
    if out.Opportunity.isna().any() or not out.Opportunity.isin(classes).all():
        raise RuntimeError("invalid opportunity export")
    if not np.array_equal(out.CLV_fuel.to_numpy(), baseline.CLV_fuel.to_numpy()):
        raise RuntimeError("fuel column changed during opportunity export")
    output = ROOT / "submissions" / "sasol_top10_rank4.csv"
    out.to_csv(output, index=False)
    changed = int((out.Opportunity.to_numpy() != baseline.Opportunity.to_numpy()).sum())
    log(f"wrote {output.name} mode {mode} changes {changed}")
    return output.name, changed


def fit_lapsed(panel, cutoff):
    path = DEST / f"{cutoff}_lapsed_fuel.npy"
    if path.exists():
        return np.load(path)
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        frame = panel["frames"][origin]
        keep = frame.fuel_recency.to_numpy() > 30
        labels = panel["labels"][origin].reindex(frame.index).CLV_fuel.to_numpy()
        block = frame.loc[keep, panel["feature_cols"]].copy()
        block["target"] = labels[keep]
        block["cutoff"] = origin
        block["w"] = weight_of(origin)
        blocks.append(block)
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if tr.sum() < 500 or va.sum() < 100:
        raise RuntimeError(f"lapsed training split is too small at {cutoff}")
    features = panel["feature_cols"]
    params = dict(
        n_estimators=1000, learning_rate=0.04, num_leaves=15, min_child_samples=80,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.2,
        n_jobs=4, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    preds = []
    predict_frame = panel["frames"][cutoff]
    for seed in (7, 21):
        inner = LGBMRegressor(**dict(params, random_state=seed))
        inner.fit(
            train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
            eval_set=[(train.loc[va, features], train.loc[va, "target"])],
            callbacks=[early_stopping(60, verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMRegressor(**dict(params, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train.target, sample_weight=train.w)
        preds.append(np.maximum(full.predict(predict_frame[features]), 0))
        log(f"lapsed fuel {cutoff} seed {seed} rounds {rounds} train {len(train)}")
    pred = np.mean(preds, axis=0)
    np.save(path, pred)
    return pred


def control_fuel(cutoff):
    return (
        0.40 * np.load(OCT / f"{cutoff}_CLV_fuel_category_values.npy")
        + 0.35 * np.load(OCT / f"{cutoff}_CLV_fuel_two_part.npy")
        + 0.25 * np.load(OCT / f"{cutoff}_fuel_residual.npy")
    )


def score_lapsed(panel):
    rows = []
    for cutoff in EVAL:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_fuel.to_numpy()
        base = control_fuel(cutoff)
        specialist = fit_lapsed(panel, cutoff)
        lapsed = frame.fuel_recency.to_numpy() > 30
        pred = base.copy()
        pred[lapsed] = specialist[lapsed]
        row = {
            "cutoff": cutoff,
            "rows": int(len(y)),
            "lapsed_rows": int(lapsed.sum()),
            "control_rmse": rmse(y, base),
            "candidate_rmse": rmse(y, pred),
        }
        row["gain"] = row["control_rmse"] - row["candidate_rmse"]
        rows.append(row)
        log(json.dumps(row))
        pd.DataFrame(rows).to_csv(DEST / "lapsed_comparison.csv", index=False)
    gains = [row["gain"] for row in rows]
    screen = [row["gain"] for row in rows if row["cutoff"] in ("2025-06-01", "2025-09-01")]
    accepted = min(screen) > 0 and float(np.mean(screen)) >= 0.001 and sum(g > 0 for g in gains) >= 5 and min(gains) >= -0.002 and float(np.mean(gains)) >= 0.001
    return rows, bool(accepted)


def export_lapsed(panel, opportunity_file):
    cutoff = fp.TEST_CUTOFF
    frame = panel["frames"][cutoff]
    specialist = fit_lapsed(panel, cutoff)
    base = control_fuel(cutoff)
    lapsed = frame.fuel_recency.to_numpy() > 30
    pred = base.copy()
    pred[lapsed] = specialist[lapsed]
    source = ROOT / "submissions" / opportunity_file if opportunity_file else ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv"
    baseline = pd.read_csv(source, dtype={"ID": str})
    original = pd.read_csv(ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv", dtype={"ID": str})
    mapped = pd.Series(pred, index=frame.index.astype(str))
    out = baseline.copy()
    out["CLV_fuel"] = mapped.loc[out.ID].to_numpy()
    if not np.array_equal(out.CLV_nonfuel.to_numpy(), original.CLV_nonfuel.to_numpy()):
        raise RuntimeError("nonfuel column changed")
    if out.CLV_fuel.isna().any() or not np.isfinite(out.CLV_fuel).all() or (out.CLV_fuel < 0).any():
        raise RuntimeError("invalid fuel export")
    output = ROOT / "submissions" / "sasol_top10_rank4_lapsed.csv"
    out.to_csv(output, index=False)
    changed = int(np.sum(np.abs(out.CLV_fuel.to_numpy() - original.CLV_fuel.to_numpy()) > 1e-12))
    log(f"wrote {output.name} fuel changes {changed}")


def main():
    PROGRESS.write_text("")
    proposal = {
        "rank_model": "Four-class LightGBM, same 305 features and the same 47-leaf recipe as the accepted classifier.",
        "rules": ["full", "within"],
        "rank_gate": "June and September each improve; mean of June-September monthly gains >= 0.004; at least 5 of 6 April-September origins improve; no loss worse than 0.002.",
        "lapsed_model": "15-leaf regressor trained only on customers with fuel recency above 30 days. It replaces only those customers.",
        "lapsed_gate": "June and September RMSE each improve with mean gain >= 0.001; at least 5 of 6 origins improve; no loss worse than 0.002.",
    }
    (DEST / "proposal.json").write_text(json.dumps(proposal, indent=2))
    classes, multipliers = load_decision()
    log("loading panel")
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    june = panel["frames"]["2025-06-01"]
    p17 = load_proba("2025-06-01", june.index)
    rebuilt = combine(p17, marginal(p17, classes), classes, "full")
    if np.max(np.abs(rebuilt - p17)) > 1e-8:
        raise RuntimeError("full combination does not reproduce the 17-class probabilities")
    rebuilt_b = combine(p17, marginal(p17, classes), classes, "within")
    if np.max(np.abs(rebuilt_b - p17)) > 1e-8:
        raise RuntimeError("within combination does not reproduce the 17-class probabilities")
    y = panel["labels"]["2025-06-01"].reindex(june.index).Opportunity.to_numpy()
    june_f1 = weighted_f1(y, labels_from(p17, classes, multipliers))
    if abs(june_f1 - JUNE_BASE) > 1e-12:
        raise RuntimeError(f"June baseline {june_f1} does not match the archived model")
    log(f"June baseline reproduced at {june_f1:.10f}")
    rows, decisions, chosen = score_rank(panel, classes, multipliers)
    summary = {"rank_rows": rows, "rank_decisions": decisions, "chosen_rank_rule": chosen}
    opportunity_file = None
    if chosen:
        opportunity_file, changes = export_rank(panel, classes, multipliers, chosen)
        summary["opportunity_file"] = opportunity_file
        summary["opportunity_changes"] = changes
    else:
        log("rank rules failed the gate")
    lapsed_rows, lapsed_ok = score_lapsed(panel)
    summary["lapsed_rows"] = lapsed_rows
    summary["lapsed_accepted"] = lapsed_ok
    if lapsed_ok:
        export_lapsed(panel, opportunity_file)
    else:
        log("lapsed fuel failed the gate")
    (DEST / "decision.json").write_text(json.dumps(summary, indent=2))
    log("DONE: " + json.dumps({"rank": chosen, "lapsed": lapsed_ok}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
