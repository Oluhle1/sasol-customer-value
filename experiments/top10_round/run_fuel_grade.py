"""Fuel-grade history added to the accepted small fuel blend.

Petrol and diesel litres are past-only. The model recipe, seeds, blend
weights and validation gate stay the same as the 0.309164264 fuel model.
April through September decide acceptance. December is fit only after that.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
import feature_pipeline as fp  # noqa: E402

OCT = ROOT / "outputs" / "oct10"
DEST = ROOT / "outputs" / "top10_round" / "fuel_grade"
DEST.mkdir(parents=True, exist_ok=True)
PROGRESS = DEST / "progress.log"
FEATURES = [
    "grade_diesel_share",
    "grade_diesel_share_q1",
    "grade_last_diesel",
    "grade_diesel_recency",
    "grade_petrol_recency",
]
EVAL = ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with PROGRESS.open("a") as handle:
        handle.write(line + "\n")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def weight(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def load_fuel():
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "train.csv",
        usecols=["DateTimeZA", "ID", "LineItemType", "ItemCategoryLevel2", "ItemCategoryLevel3", "Quantity"],
        dtype={"ID": str, "LineItemType": str, "ItemCategoryLevel2": str, "ItemCategoryLevel3": str},
    )
    fuel = raw.loc[raw.LineItemType.eq("Sale_Fuel") & raw.ItemCategoryLevel2.eq("Fuel"),
                   ["DateTimeZA", "ID", "ItemCategoryLevel3", "Quantity"]].copy()
    fuel["time"] = pd.to_datetime(fuel.DateTimeZA, format="mixed")
    fuel["qty"] = pd.to_numeric(fuel.Quantity, errors="coerce").fillna(0).clip(lower=0)
    fuel["diesel"] = fuel.ItemCategoryLevel3.eq("Diesel")
    return fuel[["ID", "time", "qty", "diesel"]]


def add_grade(panel, fuel):
    for cutoff, frame in panel["frames"].items():
        t = pd.Timestamp(cutoff)
        history = fuel.loc[fuel.time.lt(t)]
        ids = frame.index.astype(str)
        total = history.groupby("ID").qty.sum()
        diesel = history.loc[history.diesel].groupby("ID").qty.sum()
        share = (diesel / total.replace(0, np.nan)).reindex(ids).fillna(0).to_numpy()
        recent = history.loc[history.time.ge(t - pd.DateOffset(months=3))]
        total_q1 = recent.groupby("ID").qty.sum()
        diesel_q1 = recent.loc[recent.diesel].groupby("ID").qty.sum()
        share_q1 = (diesel_q1 / total_q1.replace(0, np.nan)).reindex(ids).fillna(0).to_numpy()
        last = history.sort_values("time").drop_duplicates("ID", keep="last").set_index("ID")
        last_diesel = last.diesel.astype(float).reindex(ids).fillna(0).to_numpy()
        diesel_last = history.loc[history.diesel].groupby("ID").time.max()
        petrol_last = history.loc[~history.diesel].groupby("ID").time.max()
        diesel_recency = (t - diesel_last).dt.total_seconds().div(86400).reindex(ids).fillna(999).to_numpy()
        petrol_recency = (t - petrol_last).dt.total_seconds().div(86400).reindex(ids).fillna(999).to_numpy()
        extra = pd.DataFrame({
            "grade_diesel_share": share,
            "grade_diesel_share_q1": share_q1,
            "grade_last_diesel": last_diesel,
            "grade_diesel_recency": diesel_recency,
            "grade_petrol_recency": petrol_recency,
        }, index=frame.index)
        if not np.isfinite(extra.to_numpy()).all():
            raise RuntimeError(f"non-finite grade features at {cutoff}")
        panel["frames"][cutoff] = pd.concat([frame, extra], axis=1)
        log(f"grade features {cutoff} diesel share {float(np.mean(share)):.4f}")
    panel["feature_cols"] = list(panel["feature_cols"]) + FEATURES
    return panel


def stack(panel, cutoff):
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        block = panel["frames"][origin]
        target = panel["labels"][origin].reindex(block.index).CLV_fuel
        blocks.append(pd.DataFrame({
            "target": target.to_numpy(),
            "anchor": block.fuel_z_q1.to_numpy(),
            "cutoff": origin,
            "w": weight(origin),
        }, index=block.index).join(block[panel["feature_cols"]]))
    train = pd.concat(blocks)
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    if not (pd.to_datetime(train.loc[tr, "cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("training outcomes cross the inner cutoff")
    return train, tr, va, panel["feature_cols"]


def fit_mean(train, tr, va, features, predict_frame, target, seeds, min_child):
    params = dict(
        n_estimators=1000, learning_rate=0.04, num_leaves=15, min_child_samples=min_child,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.2,
        n_jobs=2, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    preds = []
    for seed in seeds:
        inner = LGBMRegressor(**dict(params, random_state=seed))
        inner.fit(train.loc[tr, features], train.loc[tr, target], sample_weight=train.loc[tr, "w"],
                  eval_set=[(train.loc[va, features], train.loc[va, target])],
                  callbacks=[early_stopping(60, verbose=False)])
        rounds = max(int(inner.best_iteration_), 40)
        full = LGBMRegressor(**dict(params, random_state=seed, n_estimators=rounds))
        full.fit(train[features], train[target], sample_weight=train.w)
        preds.append(full.predict(predict_frame[features]))
        log(f"regressor {target} seed {seed} rounds {rounds}")
    return np.mean(preds, axis=0)


def direct(panel, cutoff):
    path = DEST / f"{cutoff}_direct.npy"
    if path.exists():
        return np.load(path)
    train, tr, va, features = stack(panel, cutoff)
    pred = np.maximum(fit_mean(train, tr, va, features, panel["frames"][cutoff], "target", (7, 21), 100), 0)
    np.save(path, pred)
    return pred


def residual(panel, cutoff):
    path = DEST / f"{cutoff}_residual.npy"
    if path.exists():
        return np.load(path)
    train, tr, va, features = stack(panel, cutoff)
    train = train.copy()
    train["target"] = train["target"] - train["anchor"]
    delta = fit_mean(train, tr, va, features, panel["frames"][cutoff], "target", (7, 21), 100)
    pred = np.maximum(panel["frames"][cutoff].fuel_z_q1.to_numpy() + delta, 0)
    np.save(path, pred)
    return pred


def hurdle(panel, cutoff):
    path = DEST / f"{cutoff}_hurdle.npy"
    if path.exists():
        return np.load(path)
    train, tr, va, features = stack(panel, cutoff)
    positive = train.target.gt(0).to_numpy()
    params = dict(
        n_estimators=1000, learning_rate=0.04, num_leaves=15, min_child_samples=50,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.2,
        n_jobs=2, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    outputs = []
    frame = panel["frames"][cutoff]
    for seed in (7, 21):
        clf = LGBMClassifier(**dict(params, random_state=seed))
        clf.fit(train.loc[tr, features], positive[tr].astype(int), sample_weight=train.loc[tr, "w"],
                eval_set=[(train.loc[va, features], positive[va].astype(int))],
                callbacks=[early_stopping(60, verbose=False)])
        rounds = max(int(clf.best_iteration_), 40)
        clf = LGBMClassifier(**dict(params, random_state=seed, n_estimators=rounds))
        clf.fit(train[features], positive.astype(int), sample_weight=train.w)
        probability = clf.predict_proba(frame[features])[:, 1]
        reg = LGBMRegressor(**dict(params, random_state=seed))
        reg.fit(train.loc[tr & positive, features], train.loc[tr & positive, "target"],
                sample_weight=train.loc[tr & positive, "w"],
                eval_set=[(train.loc[va & positive, features], train.loc[va & positive, "target"])],
                callbacks=[early_stopping(60, verbose=False)])
        value_rounds = max(int(reg.best_iteration_), 40)
        reg = LGBMRegressor(**dict(params, random_state=seed, n_estimators=value_rounds))
        reg.fit(train.loc[positive, features], train.loc[positive, "target"], sample_weight=train.loc[positive, "w"])
        outputs.append(probability * np.maximum(reg.predict(frame[features]), 0))
        log(f"hurdle {cutoff} seed {seed} purchase {rounds} value {value_rounds}")
    pred = np.mean(outputs, axis=0)
    np.save(path, pred)
    return pred


def candidate(panel, cutoff):
    return 0.40 * direct(panel, cutoff) + 0.35 * hurdle(panel, cutoff) + 0.25 * residual(panel, cutoff)


def control(cutoff):
    return (
        0.40 * np.load(OCT / f"{cutoff}_CLV_fuel_category_values.npy")
        + 0.35 * np.load(OCT / f"{cutoff}_CLV_fuel_two_part.npy")
        + 0.25 * np.load(OCT / f"{cutoff}_fuel_residual.npy")
    )


def main():
    PROGRESS.write_text("")
    proposal = {
        "hypothesis": "Diesel and petrol customers with the same recent volume have different future fuel totals. Grade history is absent from the 305-feature model.",
        "features": FEATURES,
        "recipe": "Unchanged 15-leaf direct, hurdle and residual models, blend 0.40/0.35/0.25.",
        "gate": "June and September each beat the saved blend, mean gain >= 0.001. At least 5 of April-September improve, mean gain >= 0.001, no loss worse than 0.002.",
    }
    (DEST / "proposal.json").write_text(json.dumps(proposal, indent=2))
    log("loading panel and fuel lines")
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    panel = add_grade(panel, load_fuel())
    rows = []
    for cutoff in EVAL:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_fuel.to_numpy()
        new = candidate(panel, cutoff)
        old = control(cutoff)
        if len(new) != len(y) or len(old) != len(y):
            raise RuntimeError(f"row mismatch at {cutoff}")
        row = {"cutoff": cutoff, "control_rmse": rmse(y, old), "candidate_rmse": rmse(y, new)}
        row["gain"] = row["control_rmse"] - row["candidate_rmse"]
        rows.append(row)
        log(json.dumps(row))
        pd.DataFrame(rows).to_csv(DEST / "comparison.csv", index=False)
    gains = {row["cutoff"]: row["gain"] for row in rows}
    screen = [gains["2025-06-01"], gains["2025-09-01"]]
    accepted = (
        min(screen) > 0
        and float(np.mean(screen)) >= 0.001
        and sum(row["gain"] > 0 for row in rows) >= 5
        and min(row["gain"] for row in rows) >= -0.002
        and float(np.mean([row["gain"] for row in rows])) >= 0.001
    )
    decision = {"accepted": bool(accepted), "rows": rows}
    (DEST / "decision.json").write_text(json.dumps(decision, indent=2))
    if not accepted:
        log("DONE: fuel grade failed the gate; baseline unchanged")
        return
    forecast = candidate(panel, fp.TEST_CUTOFF)
    baseline = pd.read_csv(ROOT / "submissions" / "sasol_oct10_small_fuel_values.csv", dtype={"ID": str})
    mapped = pd.Series(forecast, index=panel["frames"][fp.TEST_CUTOFF].index.astype(str))
    out = baseline.copy()
    out["CLV_fuel"] = mapped.loc[out.ID].to_numpy()
    if out.CLV_fuel.isna().any() or (out.CLV_fuel < 0).any() or not np.isfinite(out.CLV_fuel).all():
        raise RuntimeError("invalid fuel predictions")
    if not np.array_equal(out.CLV_nonfuel.to_numpy(), baseline.CLV_nonfuel.to_numpy()):
        raise RuntimeError("nonfuel column changed")
    if not out.Opportunity.equals(baseline.Opportunity):
        raise RuntimeError("opportunity column changed")
    output = ROOT / "submissions" / "sasol_top10_fuel_grade.csv"
    out.to_csv(output, index=False)
    changed = int(np.sum(np.abs(out.CLV_fuel.to_numpy() - baseline.CLV_fuel.to_numpy()) > 1e-12))
    log(f"DONE: wrote {output.name} with {changed} fuel changes")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
