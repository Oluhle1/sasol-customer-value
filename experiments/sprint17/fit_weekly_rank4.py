"""Sprint 17 E1. Weekly fuel-rand block on the production rank4 head.

Reads the pre-registered gate from outputs/sprint17/proposal_weekly_rank4.json.
June and September are fit only after the pool screen passes. A December
submission is written only after the evaluation gate passes.
"""
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sasol-mpl")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
import feature_pipeline as fp  # noqa: E402
from evaluate_saved import CLASSES, rank4_combine, weighted_f1  # noqa: E402

PROPOSAL = ROOT / "outputs" / "sprint17" / "proposal_weekly_rank4.json"
DEST = ROOT / "outputs" / "sprint17"
WEEKLY = DEST / "weekly"
OCT = ROOT / "outputs" / "oct10"
TOP10 = ROOT / "outputs" / "top10_round"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"
BIG3 = ["Inactivity", "Stable", FUEL_LABEL]
WEEK_SEC = 7 * 86400
SCALE = 1_000_000


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slog(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def feature_names(proposal):
    names = []
    for week in range(int(proposal["n_weeks"])):
        for series in proposal["weekly_series"]:
            names.append(f"wk{week:02d}_{series}")
    names.extend(proposal["scalars"])
    return names


def week_index(cutoff, times, n_weeks):
    """Week 0 is the most recent. -1 means outside the window."""
    cutoff = pd.Timestamp(cutoff)
    delta = (cutoff - pd.Series(pd.to_datetime(times))).dt.total_seconds().to_numpy()
    week = np.full(len(delta), -1, dtype=np.int16)
    keep = (delta > 0) & (delta <= n_weeks * WEEK_SEC)
    week[keep] = ((delta[keep] - 1e-6) // WEEK_SEC).astype(np.int16)
    if ((week[keep] < 0) | (week[keep] >= n_weeks)).any():
        raise RuntimeError("week index fell outside the registered window")
    return week


def self_test_weeks():
    cutoff = "2025-04-01"
    times = [
        "2025-03-31 23:00:00",
        "2025-03-25 00:00:00",
        "2025-03-24 23:00:00",
        "2024-12-10 00:00:00",
        "2024-12-09 23:00:00",
        "2025-04-01 00:00:00",
    ]
    got = week_index(cutoff, times, 16)
    expect = [0, 0, 1, 15, -1, -1]
    if got.tolist() != expect:
        raise RuntimeError(f"week window self-test failed: {got.tolist()} != {expect}")


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def load_proposal():
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    if proposal["id"] != "sprint17_weekly_rank4":
        raise RuntimeError("unexpected proposal id")
    if proposal["weekly_series"] != ["fuel_r", "fuel_l", "fuel_n", "shop_r"]:
        raise RuntimeError("weekly series differ from the registered block")
    if proposal["scalars"] != [
        "wk_fuel_r_last4",
        "wk_fuel_r_prev4",
        "wk_fuel_r_shape",
        "wk_fuel_runrate_gap",
        "wk_fuel_weeks_last8",
        "wk_shop_r_last4",
        "wk_active_last4",
    ]:
        raise RuntimeError("scalar names differ from the registered block")
    return proposal


def load_panel():
    return pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())


def load_p17(cutoff, ids):
    npz = TOP10 / "probs" / f"{cutoff}_category_dynamics.npz"
    if npz.exists():
        # Older caches stored ID strings as object arrays.
        payload = np.load(npz, allow_pickle=True)
        if "ids" in payload.files and np.array_equal(payload["ids"].astype(str), ids):
            proba = payload["proba"].astype(float)
            if proba.shape != (len(ids), 17):
                raise RuntimeError(f"{cutoff} category probabilities have a bad shape")
            return proba
    path = OCT / f"{cutoff}_category_dynamics.npy"
    if not path.exists():
        raise RuntimeError(f"missing category probabilities for {cutoff}")
    proba = np.load(path).astype(float)
    if proba.shape != (len(ids), 17):
        raise RuntimeError(f"{cutoff} category npy does not match the frame")
    return proba


def load_production_p4(cutoff, ids):
    path = TOP10 / "rank4" / f"{cutoff}_rank4.npz"
    payload = np.load(path, allow_pickle=True)
    if not np.array_equal(payload["ids"].astype(str), ids):
        raise RuntimeError(f"{cutoff} production rank4 ids do not match the frame")
    proba = payload["proba"].astype(float)
    if proba.shape != (len(ids), 4):
        raise RuntimeError(f"{cutoff} production rank4 has a bad shape")
    return proba


def decode_f1(p17, p4, vR, y):
    product = rank4_combine(p17, p4, "full")
    pred_idx = np.argmax(product * vR, axis=1)
    pred = np.asarray(CLASSES)[pred_idx]
    return weighted_f1(pred_idx, y), pred


def big3_errors(y, pred):
    wrong = pred != y
    within = wrong & np.isin(y, BIG3) & np.isin(pred, BIG3)
    return {"errors": int(wrong.sum()), "errors_within_big3": int(within.sum()), "rows": int(len(y))}


def extract_lines():
    cfg, included, shop, _classes = fp.load_config()
    log("loading transactions")
    raw = fp.lr.load_transactions(fp.DOWNLOADS / "train.csv")
    log(f"transaction rows {len(raw)}")
    core = raw.loc[~raw.ItemName.isin(cfg["adjustment_items"])]
    fuel = core.loc[core.ItemCategoryLevel2.eq("Fuel"), [
        "ID", "TransactionId", "_time", "_amount", "_quantity", "LineItemType",
    ]].copy()
    shop_lines = core.loc[core.ItemCategoryLevel2.isin(shop), [
        "ID", "TransactionId", "_time", "_amount", "_quantity",
    ]].copy()
    fuel["ID"] = fuel["ID"].astype(str)
    shop_lines["ID"] = shop_lines["ID"].astype(str)
    fuel_rows = int(len(fuel))
    sale_rows = int(fuel.LineItemType.eq("Sale_Fuel").sum())
    log(f"fuel category rows {fuel_rows}, Sale_Fuel rows {sale_rows}")
    return fuel, shop_lines


def _align_wide(table, ids, n_weeks):
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(int)
    table = table.reindex(index=pd.Index(ids.astype(str)), fill_value=0)
    table = table.reindex(columns=np.arange(n_weeks), fill_value=0)
    return table.to_numpy(dtype=float)


def _sum_wide(frame, value, ids, n_weeks):
    if frame.empty:
        return np.zeros((len(ids), n_weeks), dtype=float)
    grouped = frame.groupby(["ID", "week"], sort=False)[value].sum()
    return _align_wide(grouped.unstack(fill_value=0), ids, n_weeks)


def _count_wide(frame, ids, n_weeks):
    if frame.empty:
        return np.zeros((len(ids), n_weeks), dtype=float)
    grouped = frame.groupby(["ID", "week"], sort=False)["TransactionId"].nunique()
    return _align_wide(grouped.unstack(fill_value=0), ids, n_weeks)


def matrix_for_cutoff(fuel, shop, cutoff, ids, n_weeks):
    t = pd.Timestamp(cutoff)
    fuel_week = week_index(t, fuel["_time"], n_weeks)
    shop_week = week_index(t, shop["_time"], n_weeks)
    fuel_keep = np.flatnonzero(fuel_week >= 0)
    shop_keep = np.flatnonzero(shop_week >= 0)
    fuel_w = fuel.iloc[fuel_keep].copy()
    shop_w = shop.iloc[shop_keep].copy()
    fuel_w["week"] = fuel_week[fuel_keep].astype(int)
    shop_w["week"] = shop_week[shop_keep].astype(int)
    if len(fuel_w) and not (fuel_w["_time"] < t).all():
        raise RuntimeError(f"{cutoff} weekly fuel includes the cutoff")
    if len(shop_w) and not (shop_w["_time"] < t).all():
        raise RuntimeError(f"{cutoff} weekly shop includes the cutoff")

    fuel_rand = _sum_wide(fuel_w, "_amount", ids, n_weeks) / SCALE
    if np.abs(fuel_rand).sum() == 0:
        raise RuntimeError(f"{cutoff} weekly fuel rand is empty")
    sale = fuel_w.loc[fuel_w.LineItemType.eq("Sale_Fuel")]
    fuel_litres = _sum_wide(sale, "_quantity", ids, n_weeks) / SCALE
    fuel_qual = fuel_w.loc[fuel_w["_quantity"].gt(0) & fuel_w["_amount"].gt(0)]
    fuel_n = _count_wide(fuel_qual, ids, n_weeks)
    shop_rand = _sum_wide(shop_w, "_amount", ids, n_weeks) / SCALE
    shop_qual = shop_w.loc[shop_w["_quantity"].gt(0) & shop_w["_amount"].gt(0)]
    shop_n = _count_wide(shop_qual, ids, n_weeks)

    past_fuel = fuel.loc[fuel["_time"] < t]
    ever_counts = past_fuel.groupby("ID").size()
    ever_counts.index = ever_counts.index.astype(str)
    ever = ever_counts.reindex(pd.Index(ids.astype(str)), fill_value=0).to_numpy() > 0
    if not ever.any():
        raise RuntimeError(f"{cutoff} matched no fuel history; ID types may differ")
    q1_start = t - pd.DateOffset(months=3)
    q1 = past_fuel.loc[past_fuel["_time"] >= q1_start]
    q1_rand = q1.groupby("ID")["_amount"].sum()
    q1_rand.index = q1_rand.index.astype(str)
    q1_rand = q1_rand.reindex(pd.Index(ids.astype(str)), fill_value=0).to_numpy() / SCALE
    horizon = float((t + pd.DateOffset(months=3) - t).days)
    last4 = fuel_rand[:, 0:4].sum(axis=1)
    prev4 = fuel_rand[:, 4:8].sum(axis=1)
    need = np.maximum(q1_rand * 1.25, q1_rand + 20.0)
    run = last4 * (horizon / 28.0)
    gap = (run - need) / np.maximum(np.abs(need), 1.0)
    gap = np.where(ever, gap, 0.0)
    weeks_last8 = (fuel_n[:, 0:8] > 0).sum(axis=1) / 8.0
    active = ((fuel_n[:, 0:4].sum(axis=1) > 0) | (shop_n[:, 0:4].sum(axis=1) > 0)).astype(float)

    columns = []
    for week in range(n_weeks):
        columns.extend([
            slog(fuel_rand[:, week]),
            slog(fuel_litres[:, week]),
            np.log1p(fuel_n[:, week]),
            slog(shop_rand[:, week]),
        ])
    columns.extend([
        slog(last4),
        slog(prev4),
        slog(last4) - slog(prev4),
        gap,
        weeks_last8,
        slog(shop_rand[:, 0:4].sum(axis=1)),
        active,
    ])
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{cutoff} weekly features are not finite")
    return matrix


def build_weekly(panel, proposal):
    WEEKLY.mkdir(parents=True, exist_ok=True)
    names = feature_names(proposal)
    n_weeks = int(proposal["n_weeks"])
    pending = []
    for cutoff, frame in panel["frames"].items():
        path = WEEKLY / f"{cutoff}.npz"
        ids = frame.index.astype(str).to_numpy()
        if not path.exists():
            pending.append(cutoff)
            continue
        try:
            payload = np.load(path, allow_pickle=False)
            same_ids = np.array_equal(payload["ids"].astype(str), ids)
            same_names = payload["names"].astype(str).tolist() == names
            same_shape = payload["X"].shape == (len(ids), len(names))
        except ValueError:
            same_ids = same_names = same_shape = False
        if not (same_ids and same_names and same_shape):
            pending.append(cutoff)
    if not pending:
        log("weekly cache is complete")
        return
    fuel, shop = extract_lines()
    for cutoff in pending:
        frame = panel["frames"][cutoff]
        ids = frame.index.astype(str).to_numpy()
        if max(len(str(value)) for value in ids) > 64:
            raise RuntimeError(f"{cutoff} has an ID longer than the cache width")
        t0 = time.time()
        matrix = matrix_for_cutoff(fuel, shop, cutoff, ids, n_weeks)
        np.savez_compressed(
            WEEKLY / f"{cutoff}.npz",
            X=matrix.astype(np.float32),
            ids=np.asarray(ids, dtype="U64"),
            names=np.asarray(names, dtype="U64"),
        )
        log(f"weekly {cutoff} shape {matrix.shape} in {time.time() - t0:.1f}s")
    log("FEATURES_DONE")


def load_weekly(cutoff, ids, names):
    payload = np.load(WEEKLY / f"{cutoff}.npz", allow_pickle=False)
    if not np.array_equal(payload["ids"].astype(str), ids):
        raise RuntimeError(f"{cutoff} weekly ids do not match the frame")
    if payload["names"].astype(str).tolist() != names:
        raise RuntimeError(f"{cutoff} weekly names do not match the proposal")
    return payload["X"].astype(float)


def design_matrix(panel, cutoff, names):
    frame = panel["frames"][cutoff]
    ids = frame.index.astype(str).to_numpy()
    base = frame[panel["feature_cols"]].to_numpy(dtype=float)
    extra = load_weekly(cutoff, ids, names)
    matrix = np.hstack([base, extra])
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{cutoff} design matrix is not finite")
    return ids, matrix


def fit_rank4(panel, cutoff, names, proposal):
    path = DEST / "rank4" / f"{cutoff}_weekly_rank4.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = panel["frames"][cutoff]
    ids = frame.index.astype(str).to_numpy()
    if path.exists():
        payload = np.load(path, allow_pickle=False)
        if np.array_equal(payload["ids"].astype(str), ids):
            log(f"rank4 cache hit {cutoff}")
            importance = payload["importance"].astype(float) if "importance" in payload.files else None
            return payload["proba"].astype(float), importance
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks_x = []
    blocks_y = []
    blocks_w = []
    blocks_c = []
    feature_count = None
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        origin_ids, matrix = design_matrix(panel, origin, names)
        labels = panel["labels"][origin].reindex(panel["frames"][origin].index).Opportunity
        if labels.isna().any():
            raise RuntimeError(f"{origin} is missing labels")
        feature_count = matrix.shape[1]
        blocks_x.append(matrix)
        blocks_y.append(collapse(labels.to_numpy()))
        blocks_w.append(np.full(len(origin_ids), weight_of(origin)))
        blocks_c.append(np.full(len(origin_ids), origin))
    x = np.vstack(blocks_x)
    y = np.concatenate(blocks_y)
    w = np.concatenate(blocks_w)
    c = np.concatenate(blocks_c)
    tr = c <= limit
    va = c == latest
    if not tr.any() or not va.any():
        raise RuntimeError(f"{cutoff} has an empty purged split")
    if not (pd.to_datetime(pd.Series(c[tr])) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError(f"{cutoff} training crosses the inner cutoff")
    head = proposal["head"]
    params = dict(
        n_estimators=int(head["n_estimators_cap"]),
        learning_rate=float(head["learning_rate"]),
        num_leaves=int(head["num_leaves"]),
        min_child_samples=int(head["min_child_samples"]),
        subsample=float(head["subsample"]),
        subsample_freq=int(head["subsample_freq"]),
        colsample_bytree=float(head["colsample_bytree"]),
        reg_lambda=float(head["reg_lambda"]),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=int(head["n_jobs"]),
    )
    _ids, predict_x = design_matrix(panel, cutoff, names)
    if predict_x.shape[1] != feature_count:
        raise RuntimeError(f"{cutoff} feature count changed")
    total = np.zeros((len(ids), 4))
    importances = []
    for seed in head["seeds"]:
        t0 = time.time()
        inner = LGBMClassifier(**dict(params, random_state=int(seed)))
        inner.fit(
            x[tr], y[tr], sample_weight=w[tr],
            eval_set=[(x[va], y[va])],
            callbacks=[early_stopping(int(head["early_stopping_rounds"]), verbose=False)],
        )
        rounds = max(int(inner.best_iteration_ or 40), 40)
        full = LGBMClassifier(**dict(params, random_state=int(seed), n_estimators=rounds))
        full.fit(x, y, sample_weight=w)
        raw = full.predict_proba(predict_x)
        aligned = np.zeros((len(ids), 4))
        index = {label: i for i, label in enumerate(full.classes_)}
        for column, label in enumerate(RANK):
            if label in index:
                aligned[:, column] = raw[:, index[label]]
        denom = aligned.sum(axis=1, keepdims=True)
        if np.any(denom <= 0):
            raise RuntimeError(f"{cutoff} seed {seed} predicted an empty class distribution")
        total += aligned / denom
        importances.append(full.feature_importances_.astype(float))
        log(f"rank4 {cutoff} seed {seed} rounds {rounds} train {len(y)} in {time.time() - t0:.0f}s")
    proba = total / len(head["seeds"])
    importance = np.mean(importances, axis=0)
    np.savez(path, proba=proba, ids=np.asarray(ids, dtype="U64"), importance=importance)
    return proba, importance


def weekly_importance_share(importance, names, base_count):
    if importance is None:
        return None
    weekly = float(importance[base_count:].sum())
    total = float(importance.sum())
    return weekly / total if total else 0.0


def score_origin(panel, cutoff, p4, vR):
    frame = panel["frames"][cutoff]
    ids = frame.index.astype(str).to_numpy()
    y = panel["labels"][cutoff].reindex(frame.index).Opportunity
    if y.isna().any():
        raise RuntimeError(f"{cutoff} is missing labels")
    y = y.to_numpy()
    p17 = load_p17(cutoff, ids)
    p4_old = load_production_p4(cutoff, ids)
    base_f1, base_pred = decode_f1(p17, p4_old, vR, y)
    new_f1, new_pred = decode_f1(p17, p4, vR, y)
    return {
        "cutoff": cutoff,
        "rows": int(len(y)),
        "baseline_f1": base_f1,
        "weekly_f1": new_f1,
        "gain": new_f1 - base_f1,
        "baseline_big3": big3_errors(y, base_pred),
        "weekly_big3": big3_errors(y, new_pred),
    }


def screen_passes(rows, proposal):
    gains = [row["gain"] for row in rows]
    rule = proposal["screen"]
    return (
        float(np.mean(gains)) >= float(rule["mean_gain_at_least"])
        and min(gains) >= float(rule["min_origin_gain_at_least"])
    )


def eval_passes(rows, proposal):
    gains = [row["gain"] for row in rows]
    rule = proposal["export_gate"]
    return (
        min(gains) >= float(rule["each_fold_gain_at_least"])
        and float(np.mean(gains)) >= float(rule["mean_fold_gain_at_least"])
    )


def export_december(panel, p4, vR, eval_rows):
    source = ROOT / proposal_path_clv(vR)
    frame = panel["frames"]["2025-12-01"]
    ids = frame.index.astype(str).to_numpy()
    p17 = load_p17("2025-12-01", ids)
    pred = np.asarray(CLASSES)[np.argmax(rank4_combine(p17, p4, "full") * vR, axis=1)]
    submitted = pd.read_csv(source, dtype={"ID": str})
    if list(submitted.columns) != ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]:
        raise RuntimeError("CLV source columns are not the submission schema")
    if submitted["ID"].duplicated().any() or len(submitted) != 5488:
        raise RuntimeError("CLV source is not the 5488-row submission")
    mapped = pd.Series(pred, index=ids)
    labels = mapped.reindex(submitted["ID"]).to_numpy()
    if pd.isna(labels).any():
        raise RuntimeError("December predictions do not cover the submission ids")
    if not set(labels) <= set(CLASSES):
        raise RuntimeError("December predictions contain an unknown label")
    out = submitted.copy()
    changed = int((out["Opportunity"].to_numpy() != labels).sum())
    out["Opportunity"] = labels
    values = out[["CLV_fuel", "CLV_nonfuel"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("CLV columns are not finite and nonnegative")
    if not np.array_equal(values, submitted[["CLV_fuel", "CLV_nonfuel"]].to_numpy(dtype=float)):
        raise RuntimeError("export changed a CLV column")
    path = ROOT / "submissions" / "sasol_sprint17_weekly_r4.csv"
    out.to_csv(path, index=False)
    audit = {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "source": "submissions/sasol_sprint16_r4full_vR.csv",
        "source_sha256": sha256(source),
        "rows": int(len(out)),
        "opportunity_changes_vs_source": changed,
        "eval_gains": {row["cutoff"]: row["gain"] for row in eval_rows},
        "label_counts": {label: int((labels == label).sum()) for label in CLASSES if (labels == label).sum()},
    }
    (DEST / "export_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log(f"EXPORT {path.name} changes {changed}")
    return audit


def proposal_path_clv(_vR):
    return "submissions/sasol_sprint16_r4full_vR.csv"


def main():
    self_test_weeks()
    proposal = load_proposal()
    names = feature_names(proposal)
    vR = np.asarray(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    if len(vR) != 17:
        raise RuntimeError("vR is not a 17-class vector")
    if (ROOT / proposal_path_clv(vR)).name != Path(proposal["clv_source"]).name:
        raise RuntimeError("CLV source differs from the proposal")
    panel = load_panel()
    base_count = len(panel["feature_cols"])
    if base_count != 305:
        raise RuntimeError(f"expected 305 base features, found {base_count}")
    log(f"proposal {sha256(PROPOSAL)[:12]} features {base_count}+{len(names)}")
    build_weekly(panel, proposal)

    pool_rows = []
    shares = []
    for cutoff in proposal["pool"]:
        proba, importance = fit_rank4(panel, cutoff, names, proposal)
        share = weekly_importance_share(importance, names, base_count)
        if share is not None:
            shares.append(share)
        row = score_origin(panel, cutoff, proba, vR)
        row["weekly_importance_share"] = share
        pool_rows.append(row)
        log(f"POOL {cutoff} gain {row['gain']:+.6f} baseline {row['baseline_f1']:.6f}")
    passed = screen_passes(pool_rows, proposal)
    decision = {
        "proposal_sha256": sha256(PROPOSAL),
        "screen_passed": passed,
        "screen_rule": proposal["screen"],
        "pool_mean_gain": float(np.mean([row["gain"] for row in pool_rows])),
        "pool_min_gain": float(min(row["gain"] for row in pool_rows)),
        "weekly_importance_share_mean": float(np.mean(shares)) if shares else None,
        "pool": pool_rows,
        "eval": None,
        "exported": False,
    }
    (DEST / "weekly_rank4_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    log(
        f"{'SCREEN_PASS' if passed else 'SCREEN_FAIL'} "
        f"mean {decision['pool_mean_gain']:+.6f} min {decision['pool_min_gain']:+.6f}"
    )
    if not passed:
        log("DONE")
        return

    eval_rows = []
    for cutoff in proposal["eval"]:
        proba, _importance = fit_rank4(panel, cutoff, names, proposal)
        row = score_origin(panel, cutoff, proba, vR)
        eval_rows.append(row)
        log(f"EVAL {cutoff} gain {row['gain']:+.6f} baseline {row['baseline_f1']:.6f}")
    accepted = eval_passes(eval_rows, proposal)
    decision["eval_passed"] = accepted
    decision["eval_mean_gain"] = float(np.mean([row["gain"] for row in eval_rows]))
    decision["eval_min_gain"] = float(min(row["gain"] for row in eval_rows))
    decision["eval"] = eval_rows
    if accepted:
        decision["export"] = export_december(
            panel, fit_rank4(panel, "2025-12-01", names, proposal)[0], vR, eval_rows,
        )
        decision["exported"] = True
        log("EVAL_PASS")
    else:
        log(
            f"EVAL_FAIL mean {decision['eval_mean_gain']:+.6f} "
            f"min {decision['eval_min_gain']:+.6f}"
        )
    (DEST / "weekly_rank4_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
