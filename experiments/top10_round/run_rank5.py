"""Extend the four-class ranker that scored 0.309411451 on the public board.

That file keeps the accepted CLV columns and changes Opportunity with the
full rule: a four-class model sets the mass of Inactivity, Stable, Fuel
growth and the other 14 labels, and the 17-class model splits those 14.
The September gain sat inside those 14 labels. Two extensions are fixed
before either one is scored.

- rank5: a five-class model sets Inactivity, Stable, Fuel growth, the other
  three growth labels, and the 11 adoption labels. The 17-class model
  only splits inside those last two groups.
- remainder: the saved four-class masses stay as they are. A model trained
  only on customers whose true label is one of the 14 smaller labels
  replaces the 17-class split of that mass.

Gate for each rule, fixed before scoring. Against the four-class full
rule: June and September weighted F1 each improve, the mean gain on June
through September is at least 0.001, at least 5 of the 6 April-September
origins improve, and no origin loses more than 0.002. June and September
must also stay above the original 17-class model. If both pass, the
larger six-origin mean gain is exported. Both CLV columns stay on the
0.309411451 file.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
sys.path.insert(0, str(ROOT / "experiments" / "top10_round"))
import feature_pipeline as fp  # noqa: E402
import run_rank4 as r4  # noqa: E402

DEST = ROOT / "outputs" / "top10_round" / "rank5"
DEST.mkdir(parents=True, exist_ok=True)
PROGRESS = DEST / "progress.log"
EVAL = ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
LATER = ["2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
RANK5 = ["Inactivity", "Stable", "Fuel", "Growth", "Adoption"]
JUNE_BASE = 0.5095271507692096
SEP_BASE = 0.5165568275346171
JUNE_RANK4 = 0.5096298579050775
SEP_RANK4 = 0.5204046870140537
PARAMS = dict(
    n_estimators=1200, learning_rate=0.04, num_leaves=47, min_child_samples=35,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2,
    n_jobs=4, deterministic=True, force_col_wise=True, verbosity=-1,
)


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def groups(classes):
    growth = [name for name in classes if name.startswith("Existing-category growth:") and name != r4.FUEL_LABEL]
    adopt = [name for name in classes if name.startswith("New category adoption:")]
    other = [name for name in classes if name not in r4.BIG]
    if growth != [
        "Existing-category growth: Beverages",
        "Existing-category growth: Food & beverage",
        "Existing-category growth: Other",
    ]:
        raise RuntimeError(f"unexpected growth labels: {growth}")
    if len(adopt) != 11 or len(other) != 14 or set(growth + adopt) != set(other):
        raise RuntimeError("label groups do not partition the 17 classes")
    return growth, adopt, other


def collapse5(labels, growth, adopt):
    out = np.full(len(labels), None, dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == r4.FUEL_LABEL] = "Fuel"
    out[np.isin(labels, growth)] = "Growth"
    out[np.isin(labels, adopt)] = "Adoption"
    if np.any(pd.isna(out)):
        raise RuntimeError("a label fell outside the five groups")
    return out


def conditional(block):
    cond = block / np.maximum(block.sum(axis=1, keepdims=True), 1e-12)
    return cond


def combine5(p17, p5, classes, growth, adopt):
    index = {name: i for i, name in enumerate(classes)}
    out = np.zeros_like(p17)
    out[:, index["Inactivity"]] = p5[:, 0]
    out[:, index["Stable"]] = p5[:, 1]
    out[:, index[r4.FUEL_LABEL]] = p5[:, 2]
    growth_at = [index[name] for name in growth]
    adopt_at = [index[name] for name in adopt]
    out[:, growth_at] = p5[:, 3:4] * conditional(p17[:, growth_at])
    out[:, adopt_at] = p5[:, 4:5] * conditional(p17[:, adopt_at])
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise RuntimeError("five-class combination is empty")
    return out / total


def combine_remainder(p4, prem, classes, other):
    index = {name: i for i, name in enumerate(classes)}
    out = np.zeros((len(p4), len(classes)))
    out[:, index["Inactivity"]] = p4[:, 0]
    out[:, index["Stable"]] = p4[:, 1]
    out[:, index[r4.FUEL_LABEL]] = p4[:, 2]
    out[:, [index[name] for name in other]] = p4[:, 3:4] * conditional(prem)
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise RuntimeError("remainder combination is empty")
    return out / total


def marginal5(p17, classes, growth, adopt):
    index = {name: i for i, name in enumerate(classes)}
    growth_mass = p17[:, [index[name] for name in growth]].sum(axis=1)
    adopt_mass = p17[:, [index[name] for name in adopt]].sum(axis=1)
    return np.column_stack([
        p17[:, index["Inactivity"]],
        p17[:, index["Stable"]],
        p17[:, index[r4.FUEL_LABEL]],
        growth_mass,
        adopt_mass,
    ])


def load_cached(path, ids, width):
    payload = np.load(path, allow_pickle=True)
    saved = np.asarray(payload["ids"]).astype(str)
    if not np.array_equal(saved, np.asarray(ids).astype(str)):
        raise RuntimeError(f"{path.name} ids do not match the frame")
    proba = np.asarray(payload["proba"], dtype=float)
    if proba.shape != (len(ids), width):
        raise RuntimeError(f"{path.name} shape {proba.shape}")
    return proba


def fit_classifier(panel, cutoff, name, targets, rows_ok):
    path = DEST / f"{cutoff}_{name}.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        return load_cached(path, frame.index, len(targets))
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        origin_frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(origin_frame.index).Opportunity.to_numpy()
        keep = rows_ok(labels)
        if int(keep.sum()) == 0:
            continue
        blocks.append(pd.DataFrame({
            "target": labels[keep],
            "cutoff": origin,
            "w": r4.weight_of(origin),
        }, index=origin_frame.index[keep]).join(origin_frame.loc[keep, panel["feature_cols"]]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if tr.sum() < 500 or va.sum() < 100:
        raise RuntimeError(f"{name} split is too small at {cutoff}: {tr.sum()} {va.sum()}")
    if not (pd.to_datetime(train.loc[tr, "cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError(f"{name} training crosses the inner cutoff")
    if train.loc[tr, "target"].nunique() < 2 or train.loc[va, "target"].nunique() < 2:
        raise RuntimeError(f"{name} split does not contain two classes at {cutoff}")
    features = panel["feature_cols"]
    total = np.zeros((len(frame), len(targets)))
    for seed in (7, 23):
        inner = LGBMClassifier(**dict(PARAMS, random_state=seed))
        inner.fit(
            train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
            eval_set=[(train.loc[va, features], train.loc[va, "target"])],
            callbacks=[early_stopping(60, verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMClassifier(**dict(PARAMS, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train.target, sample_weight=train.w)
        total += fp.align_proba(full, frame[features], targets)
        log(f"{name} {cutoff} seed {seed} rounds {rounds} train {len(train)}")
    proba = total / 2
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def fit_rank5(panel, cutoff, growth, adopt):
    path = DEST / f"{cutoff}_rank5.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        return load_cached(path, frame.index, 5)
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        origin_frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(origin_frame.index).Opportunity.to_numpy()
        blocks.append(pd.DataFrame({
            "target": collapse5(labels, growth, adopt),
            "cutoff": origin,
            "w": r4.weight_of(origin),
        }, index=origin_frame.index).join(origin_frame[panel["feature_cols"]]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if not (pd.to_datetime(train.loc[tr, "cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("rank5 training crosses the inner cutoff")
    features = panel["feature_cols"]
    total = np.zeros((len(frame), 5))
    for seed in (7, 23):
        inner = LGBMClassifier(**dict(PARAMS, random_state=seed))
        inner.fit(
            train.loc[tr, features], train.loc[tr, "target"], sample_weight=train.loc[tr, "w"],
            eval_set=[(train.loc[va, features], train.loc[va, "target"])],
            callbacks=[early_stopping(60, verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMClassifier(**dict(PARAMS, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train.target, sample_weight=train.w)
        total += fp.align_proba(full, frame[features], RANK5)
        log(f"rank5 {cutoff} seed {seed} rounds {rounds} train {len(train)}")
    proba = total / 2
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    return proba


def fit_remainder(panel, cutoff, other):
    return fit_classifier(
        panel, cutoff, "remainder", other,
        rows_ok=lambda labels: np.isin(labels, other),
    )


def passes(gains, candidate_f1):
    screen = [gains["2025-06-01"], gains["2025-09-01"]]
    later = [gains[cutoff] for cutoff in LATER]
    return bool(
        min(screen) > 0
        and float(np.mean(later)) >= 0.001
        and sum(gain > 0 for gain in gains.values()) >= 5
        and min(gains.values()) >= -0.002
        and candidate_f1["2025-06-01"] > JUNE_BASE
        and candidate_f1["2025-09-01"] > SEP_BASE
    )


def class_table(y, reference, candidate, classes):
    rows = []
    for name in classes:
        rows.append({
            "label": name,
            "support": int((y == name).sum()),
            "rank4_predicted": int((reference == name).sum()),
            "rank4_f1": float(f1_score(y == name, reference == name, zero_division=0)),
            "candidate_predicted": int((candidate == name).sum()),
            "candidate_f1": float(f1_score(y == name, candidate == name, zero_division=0)),
        })
    table = pd.DataFrame(rows)
    table["f1_change"] = table.candidate_f1 - table.rank4_f1
    return table


def score(panel, classes, multipliers, growth, adopt, other):
    rows = []
    september = {}
    for cutoff in EVAL:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).Opportunity.to_numpy()
        p17 = r4.load_proba(cutoff, frame.index)
        p4 = load_cached(r4.DEST / f"{cutoff}_rank4.npz", frame.index, 4)
        p5 = fit_rank5(panel, cutoff, growth, adopt)
        prem = fit_remainder(panel, cutoff, other)
        base = r4.labels_from(p17, classes, multipliers)
        rank4_labels = r4.labels_from(r4.combine(p17, p4, classes, "full"), classes, multipliers)
        rank5_labels = r4.labels_from(combine5(p17, p5, classes, growth, adopt), classes, multipliers)
        remainder_labels = r4.labels_from(combine_remainder(p4, prem, classes, other), classes, multipliers)
        base_f1 = r4.weighted_f1(y, base)
        rank4_f1 = r4.weighted_f1(y, rank4_labels)
        record = {
            "cutoff": cutoff,
            "rows": int(len(y)),
            "baseline_f1": base_f1,
            "rank4_f1": rank4_f1,
        }
        for name, pred in (("rank5", rank5_labels), ("remainder", remainder_labels)):
            f1 = r4.weighted_f1(y, pred)
            record[f"{name}_f1"] = f1
            record[f"{name}_gain"] = f1 - rank4_f1
            record[f"{name}_changes"] = int((pred != rank4_labels).sum())
        rows.append(record)
        if cutoff == "2025-09-01":
            september = {
                "rank5": class_table(y, rank4_labels, rank5_labels, classes),
                "remainder": class_table(y, rank4_labels, remainder_labels, classes),
            }
        log(json.dumps(record))
        pd.DataFrame(rows).to_csv(DEST / "rank_comparison.csv", index=False)
    decisions = {}
    f1_maps = {}
    for name in ("rank5", "remainder"):
        gains = {row["cutoff"]: row[f"{name}_gain"] for row in rows}
        f1_maps[name] = {row["cutoff"]: row[f"{name}_f1"] for row in rows}
        decisions[name] = {
            "accepted": passes(gains, f1_maps[name]),
            "mean_six": float(np.mean(list(gains.values()))),
            "mean_later": float(np.mean([gains[cutoff] for cutoff in LATER])),
        }
    passing = [name for name, item in decisions.items() if item["accepted"]]
    chosen = max(passing, key=lambda name: decisions[name]["mean_six"]) if passing else None
    return rows, decisions, chosen, september


def export(panel, classes, multipliers, growth, adopt, other, chosen):
    cutoff = fp.TEST_CUTOFF
    frame = panel["frames"][cutoff]
    p17 = r4.load_proba(cutoff, frame.index)
    if chosen == "rank5":
        p5 = fit_rank5(panel, cutoff, growth, adopt)
        pred = r4.labels_from(combine5(p17, p5, classes, growth, adopt), classes, multipliers)
        filename = "sasol_top10_rank5.csv"
    elif chosen == "remainder":
        p4 = load_cached(r4.DEST / f"{cutoff}_rank4.npz", frame.index, 4)
        prem = fit_remainder(panel, cutoff, other)
        pred = r4.labels_from(combine_remainder(p4, prem, classes, other), classes, multipliers)
        filename = "sasol_top10_remainder.csv"
    else:
        raise RuntimeError(chosen)
    parent = pd.read_csv(ROOT / "submissions" / "sasol_top10_rank4.csv", dtype={"ID": str})
    original = pd.read_csv(ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv", dtype={"ID": str})
    if not np.array_equal(parent.ID.to_numpy(), original.ID.to_numpy()):
        raise RuntimeError("rank4 ids differ from the accepted file")
    if np.max(np.abs(parent.CLV_fuel.to_numpy() - original.CLV_fuel.to_numpy())) > 1e-12:
        raise RuntimeError("rank4 fuel column differs from the accepted file")
    if np.max(np.abs(parent.CLV_nonfuel.to_numpy() - original.CLV_nonfuel.to_numpy())) > 1e-12:
        raise RuntimeError("rank4 nonfuel column differs from the accepted file")
    mapped = pd.Series(pred, index=frame.index.astype(str))
    out = parent.copy()
    out["Opportunity"] = mapped.loc[out.ID].to_numpy()
    if out.Opportunity.isna().any() or not out.Opportunity.isin(classes).all():
        raise RuntimeError("invalid opportunity export")
    if np.max(np.abs(out.CLV_fuel.to_numpy() - original.CLV_fuel.to_numpy())) > 1e-12:
        raise RuntimeError("fuel column changed")
    if np.max(np.abs(out.CLV_nonfuel.to_numpy() - original.CLV_nonfuel.to_numpy())) > 1e-12:
        raise RuntimeError("nonfuel column changed")
    output = ROOT / "submissions" / filename
    out.to_csv(output, index=False)
    changed_parent = int((out.Opportunity.to_numpy() != parent.Opportunity.to_numpy()).sum())
    changed_original = int((out.Opportunity.to_numpy() != original.Opportunity.to_numpy()).sum())
    log(f"wrote {filename} mode {chosen} changes_vs_rank4 {changed_parent} changes_vs_original {changed_original}")
    return filename, changed_parent, changed_original


def main():
    PROGRESS.write_text("")
    proposal = {
        "parent_public_score": 0.309411451,
        "parent_file": "submissions/sasol_top10_rank4.csv",
        "rules": ["rank5", "remainder"],
        "rank5": "Five-class LightGBM sets Inactivity, Stable, Fuel growth, other growth and adoption. The 17-class model splits inside other growth and inside adoption.",
        "remainder": "Saved four-class masses stay fixed. A 14-class LightGBM, trained only on rows whose true label is outside Inactivity, Stable and Fuel growth, splits that remaining mass.",
        "recipe": "Same 305 features, 47 leaves, min_child 35, learning rate 0.04, seeds 7 and 23, and the same purged inner split as the four-class model.",
        "gate": "Against the four-class full rule: June and September each improve; mean of June-September gains >= 0.001; at least 5 of 6 April-September origins improve; no loss worse than 0.002. June and September also stay above the original 17-class model. The larger six-origin mean gain is exported if both pass.",
    }
    (DEST / "proposal.json").write_text(json.dumps(proposal, indent=2))
    classes, multipliers = r4.load_decision()
    growth, adopt, other = groups(classes)
    log("loading panel")
    panel = __import__("pickle").loads((r4.OCT / "category_dynamics_panel.pkl").read_bytes())
    june = panel["frames"]["2025-06-01"]
    p17 = r4.load_proba("2025-06-01", june.index)
    y = panel["labels"]["2025-06-01"].reindex(june.index).Opportunity.to_numpy()
    june_f1 = r4.weighted_f1(y, r4.labels_from(p17, classes, multipliers))
    if abs(june_f1 - JUNE_BASE) > 1e-12:
        raise RuntimeError(f"June baseline {june_f1} does not match the archived model")
    rebuilt = combine5(p17, marginal5(p17, classes, growth, adopt), classes, growth, adopt)
    if np.max(np.abs(rebuilt - p17)) > 1e-8:
        raise RuntimeError("five-class combination does not reproduce the 17-class probabilities")
    p4 = load_cached(r4.DEST / "2025-06-01_rank4.npz", june.index, 4)
    rank4_labels = r4.labels_from(r4.combine(p17, p4, classes, "full"), classes, multipliers)
    june_rank4 = r4.weighted_f1(y, rank4_labels)
    if abs(june_rank4 - JUNE_RANK4) > 1e-12:
        raise RuntimeError(f"June rank4 {june_rank4} does not match the saved full rule")
    other_from_17 = conditional(p17[:, [classes.index(name) for name in other]])
    restored = combine_remainder(p4, other_from_17, classes, other)
    if np.max(np.abs(restored - r4.combine(p17, p4, classes, "full"))) > 1e-8:
        raise RuntimeError("remainder combination does not reproduce the four-class full rule")
    log(f"June baseline {june_f1:.10f} and rank4 {june_rank4:.10f} reproduced")
    rows, decisions, chosen, september = score(panel, classes, multipliers, growth, adopt, other)
    sep = next(row for row in rows if row["cutoff"] == "2025-09-01")
    if abs(sep["rank4_f1"] - SEP_RANK4) > 1e-12:
        raise RuntimeError(f"September rank4 {sep['rank4_f1']} does not match the saved full rule")
    for name, table in september.items():
        table.to_csv(DEST / f"september_{name}_class_changes.csv", index=False)
    summary = {"rows": rows, "decisions": decisions, "chosen": chosen}
    if chosen:
        filename, changed_parent, changed_original = export(
            panel, classes, multipliers, growth, adopt, other, chosen,
        )
        summary["file"] = filename
        summary["changes_vs_rank4"] = changed_parent
        summary["changes_vs_original"] = changed_original
    else:
        log("both rules failed the gate")
    (DEST / "decision.json").write_text(json.dumps(summary, indent=2))
    log("DONE: " + json.dumps({"chosen": chosen}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
