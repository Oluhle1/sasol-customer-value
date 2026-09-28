"""Walk-forward CLV and opportunity models for the Sasol recruitment challenge.

Methodology (implemented, not decorative):
- Labels are produced only by the published label_rules.py / label_config.json.
- Features use history strictly before each cutoff. The test cutoff is
  2025-12-01, inferred from public_history_end_exclusive / outcome_end_exclusive.
- Customer targets follow a hurdle-aware gradient boosting regression on the
  official transformed targets z = ln(1+y)/s (LightGBM; Ke et al. 2017),
  which is the quantity RMSE is computed on.
- Opportunity is a pooled category-level purchase-and-spend model whose
  predictions are passed through the published deterministic label policy,
  blended with a customer-level multiclass model. This encodes the
  growth/adoption rules instead of treating the 17 labels as arbitrary classes.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reference.label_rules import load_transactions, raw_labels, transform  # noqa: E402

CONFIG = json.loads((ROOT / "configs" / "label_config.json").read_text())
SCALE = 1_000_000
S_FUEL = CONFIG["normalization"]["CLV_fuel"]
S_NON = CONFIG["normalization"]["CLV_nonfuel"]
CATS = sorted(c for c, v in CONFIG["category_mapping"].items() if v["included"])
LABELS = CONFIG["opportunity_labels"]
ADJUST = set(CONFIG["adjustment_items"])
CUTOFFS = list(CONFIG["calibration_cutoffs"])  # outcomes fully inside history
VALID_CUTOFF = "2025-09-01"
TEST_CUTOFF = CONFIG["public_history_end_exclusive"]  # 2025-12-01
GROWTH_MIN = float(CONFIG["growth_min_rand"])
GROWTH_REL = float(CONFIG["growth_min_relative"])
ID = "ID"


def slug(name: str) -> str:
    return (
        name.lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def to_z(y, s: float) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0, None)) / s


def from_z(z, s: float) -> np.ndarray:
    return np.expm1(np.clip(np.asarray(z, dtype=float), 0, None) * s)


def family(label: str) -> str:
    if label == "Inactivity":
        return "Inactivity"
    if label == "Stable":
        return "Stable"
    if label.startswith("Existing-category growth"):
        return "Growth"
    if label.startswith("New category adoption"):
        return "Adoption"
    return "Other"


def prepare_frame(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["qty"] = out["_quantity"] / SCALE
    out["amt"] = out["_amount"] / SCALE
    out["in_scope"] = out.ItemCategoryLevel2.isin(CATS) & ~out.ItemName.isin(ADJUST)
    out["fuel_sale"] = out.ItemCategoryLevel2.eq("Fuel") & out.LineItemType.eq("Sale_Fuel")
    out["qual"] = out.in_scope & (out["_quantity"] > 0) & (out["_amount"] > 0)
    out["nonfuel"] = out.in_scope & ~out.ItemCategoryLevel2.eq("Fuel")
    out["TransactionTotalInclAmount"] = pd.to_numeric(out["TransactionTotalInclAmount"], errors="coerce")
    return out


def window(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    return frame[(frame._time >= start) & (frame._time < end)]


def _reindex_ids(series: pd.Series, ids: pd.Index) -> pd.Series:
    return series.reindex(ids).fillna(0)


def customer_table(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """One row per eligible customer. History is strictly before cutoff."""
    t = pd.Timestamp(cutoff)
    hist = frame[frame._time < t]
    n_txn = hist.groupby(ID).TransactionId.nunique()
    ids = n_txn[n_txn >= CONFIG["minimum_historical_transactions"]].index.sort_values()
    hist = hist[hist.ID.isin(ids)]

    q_end = t
    edges = [t - pd.DateOffset(months=3 * k) for k in range(0, 5)]
    # edges[0]=T, edges[1]=T-3m, ... edges[4]=T-12m
    # q1 = [T-3m, T), q4 = [T-12m, T-9m) which is the year-ago analogue of [T, T+3m)

    baskets = (
        hist[[ID, "TransactionId", "_time", "LocationId", "TransactionTotalInclAmount"]]
        .sort_values("_time")
        .groupby([ID, "TransactionId"], sort=False)
        .agg(
            time=("_time", "min"),
            loc=("LocationId", "first"),
            total=("TransactionTotalInclAmount", "first"),
        )
        .reset_index()
    )
    baskets["dow"] = baskets.time.dt.dayofweek
    baskets["hour"] = baskets.time.dt.hour
    fuel_lines = hist[hist.fuel_sale]
    fuel_b = fuel_lines.groupby([ID, "TransactionId"]).agg(litres=("qty", "sum"), famt=("amt", "sum"))
    shop_lines = hist[hist.nonfuel]
    shop_b = shop_lines.groupby([ID, "TransactionId"]).agg(samt=("amt", "sum"))
    baskets = baskets.merge(fuel_b, on=[ID, "TransactionId"], how="left").merge(
        shop_b, on=[ID, "TransactionId"], how="left"
    )
    baskets["litres"] = baskets.litres.fillna(0)
    baskets["famt"] = baskets.famt.fillna(0)
    baskets["samt"] = baskets.samt.fillna(0)
    baskets["has_fuel"] = baskets.litres > 0
    baskets["has_shop"] = baskets.samt > 0

    baskets = baskets.sort_values([ID, "time"])
    baskets["prev"] = baskets.groupby(ID).time.shift()
    baskets["gap"] = (baskets.time - baskets.prev).dt.total_seconds() / 86400.0

    g = baskets.groupby(ID)
    first = g.time.min()
    last = g.time.max()
    recency = (t - last).dt.total_seconds() / 86400.0
    tenure = ((t - first).dt.total_seconds() / 86400.0).clip(lower=1)
    gap_mean = g.gap.mean()
    gap_std = g.gap.std()
    n_loc = g.loc.nunique()
    # primary location share
    loc_counts = baskets.groupby([ID, "loc"]).size().rename("c").reset_index()
    top = loc_counts.sort_values("c", ascending=False).groupby(ID).head(1).set_index(ID).c
    primary_share = top / n_txn.reindex(ids)

    weekend = baskets.dow.ge(5).groupby(baskets[ID]).mean()
    morning = baskets.hour.between(5, 9).groupby(baskets[ID]).mean()
    evening = baskets.hour.between(15, 19).groupby(baskets[ID]).mean()

    def q_mask(i: int) -> pd.Series:
        return (baskets.time >= edges[i]) & (baskets.time < edges[i - 1])

    feat = pd.DataFrame(index=ids)
    feat.index.name = ID
    feat["n_txn"] = n_txn.reindex(ids).astype(float)
    feat["recency_days"] = recency.reindex(ids)
    feat["tenure_days"] = tenure.reindex(ids)
    feat["txn_per_30d"] = feat.n_txn / (feat.tenure_days / 30.0)
    feat["gap_mean"] = gap_mean.reindex(ids).fillna(feat.tenure_days)
    feat["gap_std"] = gap_std.reindex(ids).fillna(0)
    feat["gap_cv"] = feat.gap_std / feat.gap_mean.clip(lower=0.5)
    feat["overdue_ratio"] = feat.recency_days / feat.gap_mean.clip(lower=0.5)
    feat["n_locations"] = n_loc.reindex(ids).astype(float)
    feat["primary_loc_share"] = primary_share.reindex(ids).fillna(1)
    feat["weekend_share"] = weekend.reindex(ids).fillna(0)
    feat["morning_share"] = morning.reindex(ids).fillna(0)
    feat["evening_share"] = evening.reindex(ids).fillna(0)
    feat["aov"] = g.total.mean().reindex(ids).fillna(0)
    month = t.month
    feat["sin_m"] = np.sin(2 * np.pi * month / 12)
    feat["cos_m"] = np.cos(2 * np.pi * month / 12)

    for i, name in enumerate(["q1", "q2", "q3", "q4"], start=1):
        m = q_mask(i)
        sub = baskets[m]
        sg = sub.groupby(ID)
        feat[f"n_txn_{name}"] = _reindex_ids(sg.size(), ids)
        feat[f"fuel_l_{name}"] = _reindex_ids(sg.litres.sum(), ids)
        feat[f"fuel_b_{name}"] = _reindex_ids(sg.has_fuel.sum(), ids)
        feat[f"nonfuel_{name}"] = _reindex_ids(sg.samt.sum(), ids)
        feat[f"shop_b_{name}"] = _reindex_ids(sg.has_shop.sum(), ids)

    feat["fuel_l_life"] = _reindex_ids(g.litres.sum(), ids)
    feat["nonfuel_life"] = _reindex_ids(g.samt.sum(), ids)
    feat["fuel_b_life"] = _reindex_ids(g.has_fuel.sum(), ids)
    feat["shop_b_life"] = _reindex_ids(g.has_shop.sum(), ids)
    both = (baskets.has_fuel & baskets.has_shop).groupby(baskets[ID]).sum()
    feat["attachment"] = _reindex_ids(both, ids) / feat.fuel_b_life.clip(lower=1)
    feat["litres_per_fill"] = feat.fuel_l_life / feat.fuel_b_life.clip(lower=1)
    feat["rand_per_shop"] = feat.nonfuel_life / feat.shop_b_life.clip(lower=1)
    feat["trend_fuel"] = feat.fuel_l_q1 / (feat.fuel_l_q2.abs() + 5)
    feat["trend_nonfuel"] = feat.nonfuel_q1 / (feat.nonfuel_q2.abs() + 5)
    feat["log_fuel_q1"] = np.log1p(feat.fuel_l_q1.clip(lower=0))
    feat["log_nonfuel_q1"] = np.log1p(feat.nonfuel_q1.clip(lower=0))
    feat["log_fuel_yoy"] = np.log1p(feat.fuel_l_q4.clip(lower=0))
    feat["log_nonfuel_yoy"] = np.log1p(feat.nonfuel_q4.clip(lower=0))

    # days since last fuel / shop basket
    last_fuel = baskets.loc[baskets.has_fuel].groupby(ID).time.max()
    last_shop = baskets.loc[baskets.has_shop].groupby(ID).time.max()
    feat["days_since_fuel"] = (t - last_fuel).dt.total_seconds().div(86400).reindex(ids)
    feat["days_since_shop"] = (t - last_shop).dt.total_seconds().div(86400).reindex(ids)
    feat["never_fuel"] = feat.days_since_fuel.isna().astype(float)
    feat["never_shop"] = feat.days_since_shop.isna().astype(float)
    feat["days_since_fuel"] = feat.days_since_fuel.fillna(feat.tenure_days + 30)
    feat["days_since_shop"] = feat.days_since_shop.fillna(feat.tenure_days + 30)

    # category breadth and per-category history used by the label policy
    scope = hist[hist.in_scope]
    qual = scope[scope.qual]
    ever = qual.groupby([ID, "ItemCategoryLevel2"]).TransactionId.nunique().unstack()
    ever = ever.reindex(index=ids, columns=CATS).fillna(0)
    feat["n_cat_ever"] = (ever > 0).sum(axis=1).astype(float)
    q1_scope = window(scope, edges[1], edges[0])
    q1_qual = q1_scope[q1_scope.qual]
    ever_q1 = q1_qual.groupby([ID, "ItemCategoryLevel2"]).TransactionId.nunique().unstack()
    ever_q1 = ever_q1.reindex(index=ids, columns=CATS).fillna(0)
    feat["n_cat_q1"] = (ever_q1 > 0).sum(axis=1).astype(float)
    spend_q1 = q1_scope.groupby([ID, "ItemCategoryLevel2"]).amt.sum().unstack()
    spend_q1 = spend_q1.reindex(index=ids, columns=CATS).fillna(0)
    life_b = ever  # qualifying baskets
    for c in CATS:
        s = slug(c)
        feat[f"ever_{s}"] = (ever[c] > 0).astype(np.float32)
        feat[f"sq1_{s}"] = spend_q1[c].astype(np.float32)
        feat[f"blife_{s}"] = life_b[c].astype(np.float32)

    feat["cutoff"] = cutoff
    feat["eligible"] = True
    return feat.reset_index()


def outcome_category_panel(frame: pd.DataFrame, cutoff: str, ids: pd.Index) -> dict[str, pd.DataFrame]:
    """Future category spend (rands) and qualifying basket counts. None if no future in file."""
    t = pd.Timestamp(cutoff)
    end = t + pd.DateOffset(months=3)
    future = window(frame, t, end)
    future = future[future.in_scope & future.ID.isin(ids)]
    if future.empty and end > frame._time.max() + pd.Timedelta(days=1):
        return {}
    spend = future.groupby([ID, "ItemCategoryLevel2"]).amt.sum().unstack()
    spend = spend.reindex(index=ids, columns=CATS).fillna(0.0)
    qual = future[future.qual]
    counts = qual.groupby([ID, "ItemCategoryLevel2"]).TransactionId.nunique().unstack()
    counts = counts.reindex(index=ids, columns=CATS).fillna(0.0)
    return {"spend": spend, "counts": counts}


def build_panel_row(cust: pd.DataFrame, spend_q: dict | None, counts_future, spend_future) -> pd.DataFrame:
    """Long customer×category table. spend_q holds observed q1/q2 spend matrices aligned to cust order."""
    ids = cust[ID].to_numpy()
    n = len(cust)
    k = len(CATS)
    base_cols = [
        "n_txn",
        "recency_days",
        "tenure_days",
        "gap_mean",
        "overdue_ratio",
        "n_cat_ever",
        "n_txn_q1",
        "n_txn_q2",
        "fuel_l_q1",
        "fuel_l_q2",
        "nonfuel_q1",
        "nonfuel_q2",
        "trend_fuel",
        "trend_nonfuel",
        "weekend_share",
        "attachment",
        "days_since_fuel",
        "days_since_shop",
        "never_fuel",
        "sin_m",
        "cos_m",
        "log_fuel_q1",
        "log_nonfuel_q1",
        "log_fuel_yoy",
    ]
    base = np.repeat(cust[base_cols].to_numpy(dtype=np.float32), k, axis=0)
    ever = np.column_stack([cust[f"ever_{slug(c)}"].to_numpy(dtype=np.float32) for c in CATS]).reshape(-1, 1)
    sq1 = np.column_stack([cust[f"sq1_{slug(c)}"].to_numpy(dtype=np.float32) for c in CATS]).reshape(-1, 1)
    blife = np.column_stack([cust[f"blife_{slug(c)}"].to_numpy(dtype=np.float32) for c in CATS]).reshape(-1, 1)
    # q2 category spend if provided
    if spend_q is None:
        sq2 = np.zeros((n * k, 1), dtype=np.float32)
    else:
        sq2 = spend_q.reshape(-1, 1).astype(np.float32)
    cat_code = np.tile(np.arange(k, dtype=np.int16), n).reshape(-1, 1)
    X = np.hstack([base, ever, sq1, sq2, blife, cat_code.astype(np.float32)])
    out = {"X": X, "cat": cat_code.ravel(), "n": n, "k": k, "ids": ids}
    if counts_future is not None:
        out["y_buy"] = (counts_future.to_numpy() >= 1).astype(np.int8).reshape(-1)
        out["y_spend"] = spend_future.to_numpy(dtype=np.float32).reshape(-1)
    return out


def q2_spend_matrix(frame: pd.DataFrame, cutoff: str, ids: pd.Index) -> np.ndarray:
    t = pd.Timestamp(cutoff)
    start = t - pd.DateOffset(months=6)
    end = t - pd.DateOffset(months=3)
    scope = window(frame, start, end)
    scope = scope[scope.in_scope & scope.ID.isin(ids)]
    spend = scope.groupby([ID, "ItemCategoryLevel2"]).amt.sum().unstack()
    spend = spend.reindex(index=ids, columns=CATS).fillna(0.0)
    return spend.to_numpy(dtype=np.float32)


def apply_policy(ever, prev_spend, p_buy, spend_hat, buy_threshold: float) -> np.ndarray:
    """Vectorised label_rules opportunity policy. Amounts in rands."""
    future_buy = p_buy >= buy_threshold
    future_spend = np.asarray(spend_hat, dtype=float)
    delta = future_spend - prev_spend
    grow = ever & future_buy & (delta >= GROWTH_MIN) & (future_spend > np.clip(prev_spend, 0, None) * (1 + GROWTH_REL))
    adopt = (~ever) & future_buy & (future_spend > 0)
    av = np.where(adopt, future_spend, 0.0)
    gv = np.where(grow, delta, 0.0)
    am = av.max(axis=1)
    gm = gv.max(axis=1)
    is_adopt = (am > 0) & (am >= gm)
    is_growth = (gm > 0) & ~is_adopt
    cat_index = np.array(CATS)
    labels = np.full(len(ever), "Stable", dtype=object)
    if is_adopt.any():
        cat = cat_index[av.argmax(axis=1)]
        labels[is_adopt] = [
            "New category adoption: " + CONFIG["category_mapping"][c]["New category adoption"] for c in cat[is_adopt]
        ]
    if is_growth.any():
        cat = cat_index[gv.argmax(axis=1)]
        labels[is_growth] = [
            "Existing-category growth: " + CONFIG["category_mapping"][c]["Existing-category growth"]
            for c in cat[is_growth]
        ]
    inactive = ~future_buy.any(axis=1)
    labels[inactive] = "Inactivity"
    return labels


def wide_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop = {
        ID, "cutoff", "eligible",
        "CLV_fuel", "CLV_nonfuel", "Opportunity",
        "fuel_litres", "nonfuel_rand",
    }
    cols = [c for c in df.columns if c not in drop]
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, cols


def fit_regressor(X, y, Xv=None, yv=None, seed=7):
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
    )
    if Xv is not None:
        model.fit(X, y, eval_set=[(Xv, yv)], eval_metric="rmse", callbacks=_early())
    else:
        model.fit(X, y)
    return model


def fit_classifier(X, y, Xv=None, yv=None, seed=7):
    model = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        class_weight=None,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
    )
    if Xv is not None:
        model.fit(X, y, eval_set=[(Xv, yv)], eval_metric="multi_logloss", callbacks=_early())
    else:
        model.fit(X, y)
    return model


def _early():
    from lightgbm import early_stopping, log_evaluation

    return [early_stopping(60, verbose=False), log_evaluation(0)]


def best_iter(model, default=400) -> int:
    bi = getattr(model, "best_iteration_", None)
    if bi is None or bi <= 0:
        return default
    return int(bi)


def history_months(frame: pd.DataFrame) -> list[dict]:
    scope_fuel = frame[frame.fuel_sale]
    scope_shop = frame[frame.nonfuel]
    months = frame._time.dt.to_period("M").astype(str)
    fuel = scope_fuel.groupby(scope_fuel._time.dt.to_period("M")).qty.sum()
    shop = scope_shop.groupby(scope_shop._time.dt.to_period("M")).amt.sum()
    baskets = frame.groupby(frame._time.dt.to_period("M")).TransactionId.nunique()
    rows = []
    for p in sorted(set(months.unique())):
        key = pd.Period(p, freq="M")
        rows.append(
            {
                "month": str(key),
                "fuel_litres": round(float(fuel.get(key, 0.0)), 1),
                "nonfuel_rand": round(float(shop.get(key, 0.0)), 1),
                "baskets": int(baskets.get(key, 0)),
            }
        )
    return rows


def importance(model, names: list[str], k=12) -> list[dict]:
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:k]
    total = float(gain.sum()) or 1.0
    return [{"feature": names[i], "share": round(float(gain[i]) / total, 4)} for i in order]


def hist_compare(actual, pred, bins=10) -> dict:
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    lo = 0.0
    hi = float(np.quantile(np.concatenate([actual, pred]), 0.98))
    edges = np.linspace(lo, max(hi, 0.1), bins + 1)
    ah, _ = np.histogram(actual, bins=edges)
    ph, _ = np.histogram(pred, bins=edges)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(bins)]
    return {"bins": labels, "actual": ah.astype(int).tolist(), "predicted": ph.astype(int).tolist()}



# ============================================================================
# Phase 06R-G0: nested temporal replication of the exact 0.301425001 pipeline
# ============================================================================

PHASE04_DIR = ROOT / "data" / "processed" / "phase04"
SUPERVISED_PATH = PHASE04_DIR / "supervised_historical.csv"
FEATURE_MANIFEST_PATH = PHASE04_DIR / "model_feature_manifest.csv"
CLASS_MAP_PATH = PHASE04_DIR / "opportunity_class_map.csv"
BASELINE_CONFIG_PATH = ROOT / "configs" / "lightgbm_baseline.json"
FINAL_SPEC_PATH = ROOT / "configs" / "final_model_spec.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase06" / "g0_nested_replication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# F1 cannot be evaluated without either using the outer validation labels for
# G0 tuning or changing G0, because there is no earlier labelled snapshot to
# serve as the internal holdout. F2-F4 permit an honest nested simulation.
OUTER_SPECS = [
    {
        "fold_id": "F2",
        "inner_train": ["2024-09-01"],
        "inner_valid": "2024-12-01",
        "outer_train": ["2024-09-01", "2024-12-01"],
        "outer_valid": "2025-03-01",
    },
    {
        "fold_id": "F3",
        "inner_train": ["2024-09-01", "2024-12-01"],
        "inner_valid": "2025-03-01",
        "outer_train": ["2024-09-01", "2024-12-01", "2025-03-01"],
        "outer_valid": "2025-06-01",
    },
    {
        "fold_id": "F4",
        "inner_train": ["2024-09-01", "2024-12-01", "2025-03-01"],
        "inner_valid": "2025-06-01",
        "outer_train": ["2024-09-01", "2024-12-01", "2025-03-01", "2025-06-01"],
        "outer_valid": "2025-09-01",
    },
]


def rescale_opportunity(proba_m, names, g_mult, a_mult):
    adj = proba_m.copy()
    for i, name in enumerate(names):
        if name.startswith("Existing-category growth"):
            adj[:, i] *= g_mult
        elif name.startswith("New category adoption"):
            adj[:, i] *= a_mult
    return np.asarray(names, dtype=object)[adj.argmax(axis=1)]


def predict_z_exact(model_direct, model_hurdle_clf, model_hurdle_reg, X, use_hurdle, name):
    direct = np.clip(model_direct.predict(X), 0, None)
    if not use_hurdle:
        return direct
    hurdle = model_hurdle_clf.predict_proba(X)[:, 1] * np.clip(
        model_hurdle_reg.predict(X), 0, None
    )
    w = 0.3
    if "hurdle" in name:
        tail = name.split("hurdle", 1)[1].strip().split()
        if tail:
            try:
                w = float(tail[0])
            except ValueError:
                w = 0.3
    return (1 - w) * direct + w * hurdle


def persist_weight(name: str) -> float:
    if "persist" not in name:
        return 0.0
    tail = name.split("persist", 1)[1].strip().split()
    try:
        return float(tail[0])
    except (IndexError, ValueError):
        return 0.0


def build_historical_snapshots(frame):
    cust_by_cutoff = {}
    panel_by_cutoff = {}

    print("Building G0 customer/panel snapshots...", flush=True)
    for cutoff in CUTOFFS:
        print(f"  {cutoff}", flush=True)
        cust = customer_table(frame, cutoff)
        ids = pd.Index(cust[ID])
        official_raw = raw_labels(raw_global, cutoff, CONFIG)
        official = transform(official_raw, CONFIG)
        official_raw = official_raw.set_index(ID).loc[ids].reset_index()
        official = official.set_index(ID).loc[ids].reset_index()
        cust = cust.merge(
            official[[ID, "CLV_fuel", "CLV_nonfuel", "Opportunity"]],
            on=ID,
            how="left",
            validate="one_to_one",
        )
        cust = cust.merge(
            official_raw[[ID, "fuel_litres", "nonfuel_rand"]],
            on=ID,
            how="left",
            validate="one_to_one",
        )
        fut = outcome_category_panel(frame, cutoff, ids)
        q2 = q2_spend_matrix(frame, cutoff, ids)
        panel = build_panel_row(cust, q2, fut["counts"], fut["spend"])
        panel["cutoff"] = cutoff
        cust_by_cutoff[cutoff] = cust
        panel_by_cutoff[cutoff] = panel

    return cust_by_cutoff, panel_by_cutoff


def concat_customers(cust_by_cutoff, cutoffs):
    return pd.concat([cust_by_cutoff[c] for c in cutoffs], ignore_index=True)


def select_g0_recipe(frame, cust_by_cutoff, panel_by_cutoff, train_cutoffs, valid_cutoff):
    """Run the original G0 model-selection logic on one internal holdout."""
    train = concat_customers(cust_by_cutoff, train_cutoffs)
    valid = cust_by_cutoff[valid_cutoff].copy()
    both = pd.concat([train, valid], ignore_index=True)
    X_both, feat_names = wide_matrix(both)
    Xtr = X_both.iloc[: len(train)].copy()
    Xva = X_both.iloc[len(train) :].copy()

    yf_tr = train["CLV_fuel"].to_numpy()
    yf_va = valid["CLV_fuel"].to_numpy()
    yn_tr = train["CLV_nonfuel"].to_numpy()
    yn_va = valid["CLV_nonfuel"].to_numpy()
    yo_tr = train["Opportunity"].to_numpy()
    yo_va = valid["Opportunity"].to_numpy()

    # Direct regressors: exact G0 settings.
    reg_f = fit_regressor(Xtr, yf_tr, Xva, yf_va, seed=7)
    reg_n = fit_regressor(Xtr, yn_tr, Xva, yn_va, seed=11)
    pred_f = np.clip(reg_f.predict(Xva), 0, None)
    pred_n = np.clip(reg_n.predict(Xva), 0, None)

    # Hurdle components: exact G0 settings.
    pos_f = yf_tr > 1e-8
    pos_n = yn_tr > 1e-8
    hurdle_f_clf = LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
        random_state=7, verbosity=-1, n_jobs=4,
    )
    hurdle_n_clf = LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
        random_state=11, verbosity=-1, n_jobs=4,
    )
    hurdle_f_clf.fit(
        Xtr, pos_f.astype(int),
        eval_set=[(Xva, (yf_va > 1e-8).astype(int))], callbacks=_early(),
    )
    hurdle_n_clf.fit(
        Xtr, pos_n.astype(int),
        eval_set=[(Xva, (yn_va > 1e-8).astype(int))], callbacks=_early(),
    )
    reg_f_pos = fit_regressor(Xtr[pos_f], yf_tr[pos_f], seed=7)
    reg_n_pos = fit_regressor(Xtr[pos_n], yn_tr[pos_n], seed=11)
    hurdle_f = hurdle_f_clf.predict_proba(Xva)[:, 1] * np.clip(reg_f_pos.predict(Xva), 0, None)
    hurdle_n = hurdle_n_clf.predict_proba(Xva)[:, 1] * np.clip(reg_n_pos.predict(Xva), 0, None)

    best_f, best_wf, best_name_f = pred_f, rmse(yf_va, pred_f), "direct"
    best_n, best_wn, best_name_n = pred_n, rmse(yn_va, pred_n), "direct"
    for w in (0.15, 0.3, 0.5, 0.7):
        bf = (1 - w) * pred_f + w * hurdle_f
        bn = (1 - w) * pred_n + w * hurdle_n
        rf, rn = rmse(yf_va, bf), rmse(yn_va, bn)
        if rf < best_wf:
            best_f, best_wf, best_name_f = bf, rf, f"blend hurdle {w}"
        if rn < best_wn:
            best_n, best_wn, best_name_n = bn, rn, f"blend hurdle {w}"

    persist_f = to_z(valid["fuel_l_q1"], S_FUEL)
    persist_n = to_z(valid["nonfuel_q1"].clip(lower=0), S_NON)
    for w in (0.1, 0.2, 0.35):
        bf = (1 - w) * best_f + w * persist_f
        bn = (1 - w) * best_n + w * persist_n
        rf, rn = rmse(yf_va, bf), rmse(yn_va, bn)
        if rf < best_wf:
            best_f, best_wf, best_name_f = bf, rf, best_name_f + f" + persist {w}"
        if rn < best_wn:
            best_n, best_wn, best_name_n = bn, rn, best_name_n + f" + persist {w}"

    # Direct Opportunity classifier plus the exact G0 reweight/prior search.
    clf = fit_classifier(Xtr, yo_tr, Xva, yo_va, seed=7)
    proba = clf.predict_proba(Xva)
    class_list = list(clf.classes_)
    multi_pred = clf.predict(Xva)
    multi_f1 = f1_score(yo_va, multi_pred, average="weighted", zero_division=0)

    clf_bal = LGBMClassifier(
        n_estimators=max(best_iter(clf), 80), learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.85,
        colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced",
        random_state=19, verbosity=-1, n_jobs=4,
    )
    clf_bal.fit(Xtr, yo_tr)
    candidates = [
        ("unweighted", clf, proba, class_list),
        ("balanced", clf_bal, clf_bal.predict_proba(Xva), list(clf_bal.classes_)),
    ]

    best_prior = (1.0, 1.0)
    best_kind = "unweighted"
    for kind, model_k, proba_k, names_k in candidates:
        raw_pred = np.asarray(names_k, dtype=object)[proba_k.argmax(axis=1)]
        raw_score = f1_score(yo_va, raw_pred, average="weighted", zero_division=0)
        if raw_score > multi_f1:
            multi_f1 = raw_score
            multi_pred = raw_pred
            proba = proba_k
            class_list = names_k
            clf = model_k
            best_kind = kind
            best_prior = (1.0, 1.0)
        for g_mult in (1.0, 1.2, 1.45, 1.7):
            for a_mult in (1.0, 2.0, 4.0, 7.0):
                if g_mult == 1.0 and a_mult == 1.0:
                    continue
                pred = rescale_opportunity(proba_k, names_k, g_mult, a_mult)
                score = f1_score(yo_va, pred, average="weighted", zero_division=0)
                if score > multi_f1:
                    multi_f1 = score
                    multi_pred = pred
                    proba = proba_k
                    class_list = names_k
                    clf = model_k
                    best_kind = kind
                    best_prior = (g_mult, a_mult)

    # Exact G0 structural category models.
    Xc_tr = np.vstack([panel_by_cutoff[c]["X"] for c in train_cutoffs])
    yb_tr = np.concatenate([panel_by_cutoff[c]["y_buy"] for c in train_cutoffs])
    ys_tr = np.nan_to_num(
        np.concatenate([panel_by_cutoff[c]["y_spend"] for c in train_cutoffs]), nan=0.0
    )
    panel_va = panel_by_cutoff[valid_cutoff]
    Xc_va = panel_va["X"]
    yb_va = panel_va["y_buy"]

    buy_model = LGBMClassifier(
        n_estimators=700, learning_rate=0.06, num_leaves=47,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=7, verbosity=-1, n_jobs=4,
        scale_pos_weight=float((yb_tr == 0).sum() / max((yb_tr == 1).sum(), 1)),
    )
    buy_model.fit(Xc_tr, yb_tr, eval_set=[(Xc_va, yb_va)], callbacks=_early())

    ys_tr_c = np.log1p(np.clip(ys_tr, 0, None))
    ys_va = np.log1p(np.clip(np.nan_to_num(panel_va["y_spend"], nan=0.0), 0, None))
    spend_model = LGBMRegressor(
        n_estimators=700, learning_rate=0.06, num_leaves=47,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=11, verbosity=-1, n_jobs=4,
    )
    spend_model.fit(Xc_tr, ys_tr_c, eval_set=[(Xc_va, ys_va)], callbacks=_early())

    def structural_labels(panel, buy_threshold, spend_scale):
        p_buy = buy_model.predict_proba(panel["X"])[:, 1].reshape(panel["n"], panel["k"])
        spend = (
            np.clip(np.expm1(spend_model.predict(panel["X"])), 0, None) * spend_scale
        ).reshape(panel["n"], panel["k"])
        n_base = 24
        ever = panel["X"][:, n_base].reshape(panel["n"], panel["k"]) > 0.5
        prev = panel["X"][:, n_base + 1].reshape(panel["n"], panel["k"]).astype(float)
        return apply_policy(ever, prev, p_buy, spend, buy_threshold)

    best_thr, best_scale, best_sf1, best_spred = 0.35, 1.0, -1.0, None
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        for scale in (0.85, 1.0, 1.25, 1.6, 2.0):
            pred = structural_labels(panel_va, thr, scale)
            score = f1_score(yo_va, pred, average="weighted", zero_division=0)
            if score > best_sf1:
                best_thr, best_scale, best_sf1, best_spred = thr, scale, score, pred

    inact_i = class_list.index("Inactivity") if "Inactivity" in class_list else None
    best_blend_f1, best_blend_pred, best_rule = multi_f1, multi_pred, "multiclass"
    if best_spred is not None and inact_i is not None:
        p_inact = proba[:, inact_i]
        top = proba.max(axis=1)
        for t_low in (0.28, 0.35, 0.42):
            for t_force in (0.55, 0.65, 0.75):
                pred = multi_pred.copy()
                unsure = top < t_low
                pred[unsure] = best_spred[unsure]
                force = (p_inact >= t_force) & (best_spred == "Inactivity")
                pred[force] = "Inactivity"
                score = f1_score(yo_va, pred, average="weighted", zero_division=0)
                if score > best_blend_f1:
                    best_blend_f1 = score
                    best_blend_pred = pred
                    best_rule = f"blend unsure<{t_low} force_inact>={t_force}"

    # The original final scoring path uses structural outright if it beat the
    # selected multiclass/blend on the internal holdout.
    if best_sf1 > best_blend_f1 + 1e-6:
        best_blend_f1 = best_sf1
        best_blend_pred = best_spred
        best_rule = f"structural thr={best_thr} scale={best_scale}"

    return {
        "feat_names": feat_names,
        "fuel_approach": best_name_f,
        "nonfuel_approach": best_name_n,
        "best_kind": best_kind,
        "growth_mult": float(best_prior[0]),
        "adoption_mult": float(best_prior[1]),
        "buy_threshold": float(best_thr),
        "spend_scale": float(best_scale),
        "opportunity_rule": best_rule,
        "n_f": max(best_iter(reg_f), 80),
        "n_n": max(best_iter(reg_n), 80),
        "n_c": max(best_iter(clf), 80),
        "n_b": max(best_iter(buy_model), 80),
        "n_s": max(best_iter(spend_model), 80),
        "n_hf": max(best_iter(hurdle_f_clf), 50),
        "n_hn": max(best_iter(hurdle_n_clf), 50),
        "n_hf_reg": max(best_iter(reg_f_pos), 40),
        "n_hn_reg": max(best_iter(reg_n_pos), 40),
        "inner_fuel_rmse": float(best_wf),
        "inner_nonfuel_rmse": float(best_wn),
        "inner_opportunity_f1": float(best_blend_f1),
        "inner_multiclass_f1": float(multi_f1),
        "inner_structural_f1": float(best_sf1),
    }


def fit_and_score_g0_outer(frame, cust_by_cutoff, panel_by_cutoff, recipe, train_cutoffs, valid_cutoff):
    train = concat_customers(cust_by_cutoff, train_cutoffs)
    valid = cust_by_cutoff[valid_cutoff].copy()

    X_train_all, names = wide_matrix(train)
    X_valid_all, names_valid = wide_matrix(valid)
    if names != names_valid:
        raise AssertionError("G0 feature columns differ between outer train and validation.")
    if names != recipe["feat_names"]:
        raise AssertionError("G0 feature columns drifted between inner selection and outer refit.")

    yf = train["CLV_fuel"].to_numpy()
    yn = train["CLV_nonfuel"].to_numpy()
    yo = train["Opportunity"].to_numpy()

    final_f = LGBMRegressor(
        n_estimators=recipe["n_f"], learning_rate=0.05, num_leaves=31,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.75,
        reg_lambda=1.0, random_state=7, verbosity=-1, n_jobs=4,
    )
    final_n = LGBMRegressor(
        n_estimators=recipe["n_n"], learning_rate=0.05, num_leaves=31,
        min_child_samples=40, subsample=0.85, colsample_bytree=0.75,
        reg_lambda=1.0, random_state=11, verbosity=-1, n_jobs=4,
    )
    final_c = LGBMClassifier(
        n_estimators=recipe["n_c"], learning_rate=0.05, num_leaves=31,
        min_child_samples=20 if recipe["best_kind"] == "balanced" else 25,
        subsample=0.85, colsample_bytree=0.7, reg_lambda=1.0,
        class_weight="balanced" if recipe["best_kind"] == "balanced" else None,
        random_state=19 if recipe["best_kind"] == "balanced" else 7,
        verbosity=-1, n_jobs=4,
    )
    final_f.fit(X_train_all, yf)
    final_n.fit(X_train_all, yn)
    final_c.fit(X_train_all, yo)

    # Structural models on all outer-training snapshots.
    Xc_all = np.vstack([panel_by_cutoff[c]["X"] for c in train_cutoffs])
    yb_all = np.concatenate([panel_by_cutoff[c]["y_buy"] for c in train_cutoffs])
    ys_all = np.log1p(
        np.clip(
            np.nan_to_num(
                np.concatenate([panel_by_cutoff[c]["y_spend"] for c in train_cutoffs]),
                nan=0.0,
            ),
            0,
            None,
        )
    )
    final_buy = LGBMClassifier(
        n_estimators=recipe["n_b"], learning_rate=0.06, num_leaves=47,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=7, verbosity=-1, n_jobs=4,
        scale_pos_weight=float((yb_all == 0).sum() / max((yb_all == 1).sum(), 1)),
    )
    final_spend = LGBMRegressor(
        n_estimators=recipe["n_s"], learning_rate=0.06, num_leaves=47,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=11, verbosity=-1, n_jobs=4,
    )
    final_buy.fit(Xc_all, yb_all)
    final_spend.fit(Xc_all, ys_all)

    use_hurdle_f = "hurdle" in recipe["fuel_approach"]
    use_hurdle_n = "hurdle" in recipe["nonfuel_approach"]
    final_hf = final_hn = final_hf_reg = final_hn_reg = None
    if use_hurdle_f or use_hurdle_n:
        final_hf = LGBMClassifier(
            n_estimators=recipe["n_hf"], learning_rate=0.05, num_leaves=31,
            min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
            random_state=7, verbosity=-1, n_jobs=4,
        )
        final_hn = LGBMClassifier(
            n_estimators=recipe["n_hn"], learning_rate=0.05, num_leaves=31,
            min_child_samples=30, subsample=0.85, colsample_bytree=0.8,
            random_state=11, verbosity=-1, n_jobs=4,
        )
        final_hf.fit(X_train_all, (yf > 1e-8).astype(int))
        final_hn.fit(X_train_all, (yn > 1e-8).astype(int))
        final_hf_reg = LGBMRegressor(
            n_estimators=recipe["n_hf_reg"], learning_rate=0.05, num_leaves=31,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.75,
            reg_lambda=1.0, random_state=7, verbosity=-1, n_jobs=4,
        )
        final_hn_reg = LGBMRegressor(
            n_estimators=recipe["n_hn_reg"], learning_rate=0.05, num_leaves=31,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.75,
            reg_lambda=1.0, random_state=11, verbosity=-1, n_jobs=4,
        )
        final_hf_reg.fit(X_train_all[yf > 1e-8], yf[yf > 1e-8])
        final_hn_reg.fit(X_train_all[yn > 1e-8], yn[yn > 1e-8])

    zf = predict_z_exact(
        final_f, final_hf, final_hf_reg, X_valid_all, use_hurdle_f, recipe["fuel_approach"]
    )
    zn = predict_z_exact(
        final_n, final_hn, final_hn_reg, X_valid_all, use_hurdle_n, recipe["nonfuel_approach"]
    )
    wf_p = persist_weight(recipe["fuel_approach"])
    wn_p = persist_weight(recipe["nonfuel_approach"])
    if wf_p:
        zf = (1 - wf_p) * zf + wf_p * to_z(valid["fuel_l_q1"].to_numpy(), S_FUEL)
    if wn_p:
        zn = (1 - wn_p) * zn + wn_p * to_z(valid["nonfuel_q1"].clip(lower=0).to_numpy(), S_NON)
    zf = np.clip(zf, 0, None)
    zn = np.clip(zn, 0, None)

    # Multiclass + prior scaling.
    multi_p = final_c.predict_proba(X_valid_all)
    names_f = list(final_c.classes_)
    multi_pred = rescale_opportunity(
        multi_p, names_f, recipe["growth_mult"], recipe["adoption_mult"]
    )

    # Structural prediction at the outer origin.
    panel_v = panel_by_cutoff[valid_cutoff]
    p_buy = final_buy.predict_proba(panel_v["X"])[:, 1].reshape(panel_v["n"], panel_v["k"])
    spend = (
        np.clip(np.expm1(final_spend.predict(panel_v["X"])), 0, None) * recipe["spend_scale"]
    ).reshape(panel_v["n"], panel_v["k"])
    n_base = 24
    ever = panel_v["X"][:, n_base].reshape(panel_v["n"], panel_v["k"]) > 0.5
    prev = panel_v["X"][:, n_base + 1].reshape(panel_v["n"], panel_v["k"]).astype(float)
    struct_pred = apply_policy(ever, prev, p_buy, spend, recipe["buy_threshold"])

    opp = multi_pred.copy()
    rule = recipe["opportunity_rule"]
    if rule.startswith("blend") and "Inactivity" in names_f:
        t_low = float(rule.split("unsure<", 1)[1].split()[0])
        t_force = float(rule.split("force_inact>=", 1)[1])
        p_inact = multi_p[:, names_f.index("Inactivity")]
        top = multi_p.max(axis=1)
        unsure = top < t_low
        opp[unsure] = struct_pred[unsure]
        force = (p_inact >= t_force) & (struct_pred == "Inactivity")
        opp[force] = "Inactivity"
    elif rule.startswith("structural"):
        opp = struct_pred

    yfv = valid["CLV_fuel"].to_numpy()
    ynv = valid["CLV_nonfuel"].to_numpy()
    yov = valid["Opportunity"].to_numpy()

    return {
        "validation_rows": len(valid),
        "fuel_rmse": rmse(yfv, zf),
        "nonfuel_rmse": rmse(ynv, zn),
        "opportunity_f1": float(f1_score(yov, opp, average="weighted", zero_division=0)),
        "prediction_frame": pd.DataFrame(
            {
                "ID": valid[ID].to_numpy(),
                "cutoff": valid_cutoff,
                "CLV_fuel_actual": yfv,
                "CLV_fuel_prediction": zf,
                "CLV_nonfuel_actual": ynv,
                "CLV_nonfuel_prediction": zn,
                "Opportunity_actual": yov,
                "Opportunity_prediction": opp,
            }
        ),
    }


def make_s0_regression_parameters(base_config, final_spec, task):
    p = dict(base_config["common_parameters"])
    p.update(base_config["regression"])
    p.pop("prediction_floor", None)
    p["num_leaves"] = int(final_spec[task]["num_leaves"])
    p["min_data_in_leaf"] = int(final_spec[task]["min_data_in_leaf"])
    seed = int(base_config["random_seed"])
    p.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )
    return p


def make_s0_opportunity_parameters(base_config, final_spec):
    p = dict(base_config["common_parameters"])
    p.update(base_config["opportunity"])
    for key in ["class_weighting", "decision_rule", "validation_metric"]:
        p.pop(key, None)
    p["num_leaves"] = int(final_spec["opportunity"]["num_leaves"])
    p["min_data_in_leaf"] = int(final_spec["opportunity"]["min_data_in_leaf"])
    seed = int(base_config["random_seed"])
    p.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )
    return p


def s0_weights(labels, num_classes):
    labels = np.asarray(labels, dtype="int64")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise AssertionError("S0 outer training population is missing an Opportunity class.")
    raw = np.sqrt(len(labels) / (num_classes * counts))
    normalizer = raw[labels].mean()
    normalized = raw / normalizer
    weights = normalized[labels]
    if not np.isclose(weights.mean(), 1.0, atol=1e-12):
        raise AssertionError("S0 Opportunity sample weights do not normalize to 1.")
    return weights


def fit_and_score_s0_outer(supervised, selected_features, class_map, base_config, final_spec, train_cutoffs, valid_cutoff):
    train = supervised[supervised["cutoff"].isin(pd.to_datetime(train_cutoffs))].copy()
    valid = supervised[supervised["cutoff"].eq(pd.Timestamp(valid_cutoff))].copy()
    Xtr = train[selected_features]
    Xva = valid[selected_features]

    out = {"validation_rows": len(valid)}
    for task, target in [("fuel", "CLV_fuel"), ("nonfuel", "CLV_nonfuel")]:
        p = make_s0_regression_parameters(base_config, final_spec, task)
        dtrain = lgb.Dataset(Xtr, label=train[target].to_numpy(), feature_name=selected_features)
        model = lgb.train(
            p,
            dtrain,
            num_boost_round=int(final_spec[task]["num_boost_round"]),
            callbacks=[lgb.log_evaluation(period=0)],
        )
        pred = np.maximum(
            model.predict(Xva, num_iteration=int(final_spec[task]["num_boost_round"])), 0.0
        )
        out[f"{task}_rmse"] = rmse(valid[target].to_numpy(), pred)

    num_classes = len(class_map)
    ytr = train["Opportunity_class_id"].to_numpy(dtype="int64")
    yva = valid["Opportunity_class_id"].to_numpy(dtype="int64")
    weights = s0_weights(ytr, num_classes)
    p = make_s0_opportunity_parameters(base_config, final_spec)
    dtrain = lgb.Dataset(
        Xtr, label=ytr, weight=weights, feature_name=selected_features
    )
    model = lgb.train(
        p,
        dtrain,
        num_boost_round=int(final_spec["opportunity"]["num_boost_round"]),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    proba = model.predict(
        Xva, num_iteration=int(final_spec["opportunity"]["num_boost_round"])
    )
    pred = np.argmax(proba, axis=1).astype("int64")
    out["opportunity_f1"] = float(
        f1_score(yva, pred, average="weighted", zero_division=0)
    )
    return out


def target_summary(metrics, target, lower_is_better):
    weights = metrics["validation_rows"] / metrics["validation_rows"].sum()
    s0_col = f"s0_{target}"
    g0_col = f"g0_{target}"
    s0_w = float(np.sum(metrics[s0_col] * weights))
    g0_w = float(np.sum(metrics[g0_col] * weights))
    if lower_is_better:
        fold_wins = int((metrics[g0_col] < metrics[s0_col]).sum())
        weighted_win = g0_w < s0_w
        delta = s0_w - g0_w  # positive = G0 improvement
    else:
        fold_wins = int((metrics[g0_col] > metrics[s0_col]).sum())
        weighted_win = g0_w > s0_w
        delta = g0_w - s0_w  # positive = G0 improvement
    return {
        "target": target,
        "folds_won_by_g0": fold_wins,
        "s0_row_weighted": s0_w,
        "g0_row_weighted": g0_w,
        "g0_improvement_directional": delta,
        "row_weighted_improved": bool(weighted_win),
        "target_win": bool(weighted_win and fold_wins >= 2),
    }


def main():
    global raw_global

    print("=" * 100)
    print("PHASE 06R-G0 — NESTED TEMPORAL REPLICATION")
    print("=" * 100)
    print()
    print("Important: F1 is intentionally excluded.")
    print("G0 needs a prior labelled holdout to select its internal recipe.")
    print("F2-F4 are evaluated without using the outer validation outcomes for tuning.")
    print()

    raw_global = load_transactions(str(ROOT / "data" / "raw" / "train.csv"))
    frame = prepare_frame(raw_global)
    cust_by_cutoff, panel_by_cutoff = build_historical_snapshots(frame)

    supervised = pd.read_csv(
        SUPERVISED_PATH, dtype={"ID": str}, parse_dates=["cutoff"]
    )
    manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    class_map = pd.read_csv(CLASS_MAP_PATH).sort_values("class_id")
    base_config = json.loads(BASELINE_CONFIG_PATH.read_text())
    final_spec = json.loads(FINAL_SPEC_PATH.read_text())
    selected_features = manifest.loc[
        manifest["selected_for_baseline"].eq(1), "feature"
    ].tolist()
    if len(selected_features) != 276:
        raise AssertionError("Expected 276 frozen S0 features.")
    if len(class_map) != 17:
        raise AssertionError("Expected 17 Opportunity classes.")

    metric_rows = []
    recipe_rows = []
    prediction_frames = []

    for spec in OUTER_SPECS:
        print()
        print("#" * 100)
        print(
            f"{spec['fold_id']} | inner tune={spec['inner_valid']} | "
            f"outer evaluate={spec['outer_valid']}"
        )
        print("#" * 100)

        recipe = select_g0_recipe(
            frame,
            cust_by_cutoff,
            panel_by_cutoff,
            spec["inner_train"],
            spec["inner_valid"],
        )

        g0 = fit_and_score_g0_outer(
            frame,
            cust_by_cutoff,
            panel_by_cutoff,
            recipe,
            spec["outer_train"],
            spec["outer_valid"],
        )

        s0 = fit_and_score_s0_outer(
            supervised,
            selected_features,
            class_map,
            base_config,
            final_spec,
            spec["outer_train"],
            spec["outer_valid"],
        )

        # ID-set reconciliation between the two representations.
        g0_ids = set(cust_by_cutoff[spec["outer_valid"]][ID].astype(str))
        s0_ids = set(
            supervised.loc[
                supervised["cutoff"].eq(pd.Timestamp(spec["outer_valid"])), "ID"
            ].astype(str)
        )
        if g0_ids != s0_ids:
            raise AssertionError(
                f"{spec['fold_id']}: S0/G0 outer validation customer populations differ."
            )
        if g0["validation_rows"] != s0["validation_rows"]:
            raise AssertionError(f"{spec['fold_id']}: validation row counts differ.")

        metric_rows.append(
            {
                "fold_id": spec["fold_id"],
                "inner_valid": spec["inner_valid"],
                "outer_valid": spec["outer_valid"],
                "validation_rows": g0["validation_rows"],
                "s0_fuel_rmse": s0["fuel_rmse"],
                "g0_fuel_rmse": g0["fuel_rmse"],
                "fuel_g0_minus_s0": g0["fuel_rmse"] - s0["fuel_rmse"],
                "s0_nonfuel_rmse": s0["nonfuel_rmse"],
                "g0_nonfuel_rmse": g0["nonfuel_rmse"],
                "nonfuel_g0_minus_s0": g0["nonfuel_rmse"] - s0["nonfuel_rmse"],
                "s0_opportunity_f1": s0["opportunity_f1"],
                "g0_opportunity_f1": g0["opportunity_f1"],
                "opportunity_g0_minus_s0": g0["opportunity_f1"] - s0["opportunity_f1"],
            }
        )

        recipe_rows.append(
            {
                "fold_id": spec["fold_id"],
                "inner_valid": spec["inner_valid"],
                "outer_valid": spec["outer_valid"],
                "fuel_approach": recipe["fuel_approach"],
                "nonfuel_approach": recipe["nonfuel_approach"],
                "opportunity_kind": recipe["best_kind"],
                "growth_mult": recipe["growth_mult"],
                "adoption_mult": recipe["adoption_mult"],
                "buy_threshold": recipe["buy_threshold"],
                "spend_scale": recipe["spend_scale"],
                "opportunity_rule": recipe["opportunity_rule"],
                "n_f": recipe["n_f"],
                "n_n": recipe["n_n"],
                "n_c": recipe["n_c"],
                "n_b": recipe["n_b"],
                "n_s": recipe["n_s"],
                "inner_fuel_rmse": recipe["inner_fuel_rmse"],
                "inner_nonfuel_rmse": recipe["inner_nonfuel_rmse"],
                "inner_opportunity_f1": recipe["inner_opportunity_f1"],
            }
        )

        pf = g0["prediction_frame"].copy()
        pf.insert(0, "fold_id", spec["fold_id"])
        prediction_frames.append(pf)

        print(
            f"{spec['fold_id']} OUTER | "
            f"Fuel S0={s0['fuel_rmse']:.6f} G0={g0['fuel_rmse']:.6f} | "
            f"Nonfuel S0={s0['nonfuel_rmse']:.6f} G0={g0['nonfuel_rmse']:.6f} | "
            f"Opp S0={s0['opportunity_f1']:.6f} G0={g0['opportunity_f1']:.6f}"
        )
        print(
            f"Selected recipe | fuel={recipe['fuel_approach']} | "
            f"nonfuel={recipe['nonfuel_approach']} | "
            f"opp={recipe['best_kind']} growth×{recipe['growth_mult']} "
            f"adoption×{recipe['adoption_mult']} | {recipe['opportunity_rule']}"
        )

    metrics = pd.DataFrame(metric_rows)
    recipes = pd.DataFrame(recipe_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    summary = pd.DataFrame(
        [
            target_summary(metrics, "fuel_rmse", lower_is_better=True),
            target_summary(metrics, "nonfuel_rmse", lower_is_better=True),
            target_summary(metrics, "opportunity_f1", lower_is_better=False),
        ]
    )

    # Single pre-registered branch decision:
    # G0 is promoted as the modelling direction only if Opportunity is a target
    # win and at least one of the two regression targets is also a target win.
    win_lookup = dict(zip(summary["target"], summary["target_win"]))
    promoted = bool(
        win_lookup["opportunity_f1"]
        and (win_lookup["fuel_rmse"] or win_lookup["nonfuel_rmse"])
    )

    metrics.to_csv(OUTPUT_DIR / "outer_fold_comparison.csv", index=False)
    recipes.to_csv(OUTPUT_DIR / "selected_recipes.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "g0_outer_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "target_summary.csv", index=False)

    decision = {
        "experiment": "Phase 06R-G0 nested temporal replication",
        "public_same_challenge_evidence": {
            "s0_public_score": 0.289392053,
            "g0_public_score": 0.301425001,
        },
        "outer_folds": ["F2", "F3", "F4"],
        "f1_excluded_reason": (
            "The G0 architecture requires an earlier labelled holdout for internal "
            "recipe selection. No earlier calibration origin exists before F1."
        ),
        "target_win_rule": (
            "Row-weighted metric improves and G0 wins at least 2 of 3 outer folds."
        ),
        "branch_promotion_rule": (
            "Promote G0 only if Opportunity is a target win and at least one "
            "regression target is also a target win."
        ),
        "promote_g0_direction": promoted,
    }
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2))

    print()
    print("=" * 100)
    print("OUTER FOLD COMPARISON")
    print("=" * 100)
    print(metrics.to_string(index=False))
    print()
    print("=" * 100)
    print("TARGET SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))
    print()
    print("=" * 100)
    print("BRANCH DECISION")
    print("=" * 100)
    print("PROMOTE G0 DIRECTION:", promoted)
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
