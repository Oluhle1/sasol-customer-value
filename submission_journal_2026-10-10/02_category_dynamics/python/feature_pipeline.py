"""Portable feature and nested-fit helpers adapted from the user's
Downloads/sasol-fresh/model_clv.py on 2026-10-10.
Only released Sasol transactions and the official label configuration are used.
Leaderboard normalization assumptions are not used to choose models here.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1
from sklearn.metrics import f1_score, mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT = Path(__file__).resolve().parents[2]
DOWNLOADS = PROJECT / "data" / "raw"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from reference import label_rules as lr

CACHE = ROOT / "panel_cache.pkl"
TARGET_BOARD = 0.318678065
# September persistence RMSE, used only to reconstruct the public number.
SCALE_FUEL = 0.6905
SCALE_SHOP = 0.9296

TRAIN_CUTOFFS = [
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
    "2025-09-01",
]
PREV_CUTOFF = "2024-06-01"
TEST_CUTOFF = "2025-12-01"
ALL_LABEL_CUTOFFS = [PREV_CUTOFF] + TRAIN_CUTOFFS

SNAPSHOT_WEIGHT = {
    "2024-09-01": 0.70,
    "2024-12-01": 0.85,
    "2025-03-01": 1.00,
    "2025-06-01": 1.15,
    "2025-09-01": 1.30,
}

# September 2025 counts from a clean label_rules run. If these move, the
# labels are not the official ones and every later number is void.
SEP_COUNTS = {
    "Inactivity": 1436,
    "Existing-category growth: Fuel": 1435,
    "Stable": 1404,
    "Existing-category growth: Other": 278,
    "Existing-category growth: Beverages": 188,
    "Existing-category growth: Food & beverage": 129,
}

HABITUAL = [
    "Tobacco",
    "Beverages",
    "Coffee",
    "Snacking",
    "Food & beverage",
    "Confectionary",
]
OCCASIONAL = [
    "Gas",
    "Lubricants",
    "Braai & Ice",
    "Krispy Kreme",
    "Car Wash",
    "Automotive Products",
]


def yms(ts, start_back, length):
    """Months before a day-1 cutoff. start_back=1, length=3 is the last quarter."""
    end = ts.year * 12 + ts.month - start_back
    first = end - length + 1
    return list(range(first, end + 1))


assert yms(pd.Timestamp("2025-09-01"), 1, 3) == [2025 * 12 + 6, 2025 * 12 + 7, 2025 * 12 + 8]
assert yms(pd.Timestamp("2025-09-01"), 4, 3) == [2025 * 12 + 3, 2025 * 12 + 4, 2025 * 12 + 5]


def year_ago_outcome(ts):
    base = ts.year * 12 + ts.month
    return [base - 12, base - 11, base - 10]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def weighted_f1(y_true, y_pred):
    labels = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))


def board(fuel, shop, f1):
    return float(
        0.3 * (1.0 - fuel / SCALE_FUEL)
        + 0.3 * (1.0 - shop / SCALE_SHOP)
        + 0.4 * f1
    )


def slug(name):
    return (
        str(name)
        .lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "")
    )


def prev_quarter(cutoff):
    return (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")


def reindex_fill(series, ids, fill):
    return series.reindex(ids).fillna(fill)


def safe_div(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros(len(num), dtype=float)
    ok = den != 0
    out[ok] = num[ok] / den[ok]
    return out


def load_config():
    cfg = json.loads((PROJECT / "configs" / "label_config.json").read_text(encoding="utf-8"))
    included = [c for c, v in cfg["category_mapping"].items() if v["included"]]
    shop = [c for c in included if c != "Fuel"]
    classes = list(cfg["opportunity_labels"])
    return cfg, included, shop, classes


def prepare_tables(d, included, shop_cats, adjustments):
    d = d.copy()
    d["qty"] = d["_quantity"] / lr.SCALE
    d["amt"] = d["_amount"] / lr.SCALE
    d["basket"] = pd.to_numeric(d["TransactionTotalInclAmount"], errors="coerce")
    d["ym"] = (d["_time"].dt.year * 12 + d["_time"].dt.month).astype(np.int32)

    tx = (
        d.groupby(["ID", "TransactionId"], sort=False)
        .agg(time=("_time", "min"), location=("LocationId", "first"), basket=("basket", "first"))
        .reset_index()
    )
    tx["ym"] = (tx["time"].dt.year * 12 + tx["time"].dt.month).astype(np.int32)
    tx["dow"] = tx["time"].dt.dayofweek.astype(np.int8)
    tx["hour"] = tx["time"].dt.hour.astype(np.int8)
    tx["weekend"] = (tx["dow"] >= 5).astype(np.int8)
    tx["morning"] = ((tx["hour"] >= 5) & (tx["hour"] < 9)).astype(np.int8)
    tx["evening"] = ((tx["hour"] >= 16) & (tx["hour"] < 20)).astype(np.int8)

    core = d.loc[~d["ItemName"].isin(adjustments)].copy()
    fuel_lines = core.loc[
        core["ItemCategoryLevel2"].eq("Fuel") & core["LineItemType"].eq("Sale_Fuel")
    ]
    fuel_tx = (
        fuel_lines.groupby(["ID", "TransactionId"], sort=False)
        .agg(qty=("qty", "sum"), amt=("amt", "sum"), time=("_time", "min"))
        .reset_index()
    )
    fuel_tx["ym"] = (fuel_tx["time"].dt.year * 12 + fuel_tx["time"].dt.month).astype(np.int32)
    fuel_tx["dow"] = fuel_tx["time"].dt.dayofweek.astype(np.int8)

    shop_lines = core.loc[core["ItemCategoryLevel2"].isin(shop_cats)]
    shop_tx = (
        shop_lines.groupby(["ID", "TransactionId"], sort=False)
        .agg(amt=("amt", "sum"), time=("_time", "min"))
        .reset_index()
    )
    shop_tx["ym"] = (shop_tx["time"].dt.year * 12 + shop_tx["time"].dt.month).astype(np.int32)

    cat = core.loc[core["ItemCategoryLevel2"].isin(included), ["ID", "TransactionId", "ItemCategoryLevel2", "qty", "amt", "ym", "_time"]].copy()
    cat = cat.rename(columns={"ItemCategoryLevel2": "cat"})
    cat_spend = cat.groupby(["ID", "ym", "cat"], sort=False)["amt"].sum().rename("spend").reset_index()
    qual = cat.loc[(cat["qty"] > 0) & (cat["amt"] > 0)]
    cat_tx = (
        qual.groupby(["ID", "ym", "cat"], sort=False)["TransactionId"].nunique().rename("ntx").reset_index()
    )

    priced = fuel_lines.loc[fuel_lines["qty"] > 0, ["ym", "qty", "amt"]].copy()
    priced["unit"] = priced["amt"] / priced["qty"]
    price_ym = priced.groupby("ym")["unit"].median()

    return {
        "tx": tx,
        "fuel_tx": fuel_tx,
        "shop_tx": shop_tx,
        "cat_spend": cat_spend,
        "cat_tx": cat_tx,
        "price_ym": price_ym,
    }


def timing_block(events, t, ids):
    """Recency, age, gaps. events has ID and time, already one row per purchase."""
    h = events.loc[events["time"] < t, ["ID", "time"]].sort_values(["ID", "time"])
    if h.empty:
        z = pd.Series(0.0, index=ids)
        rec = pd.Series(999.0, index=ids)
        return rec, z, z, z, z, z, pd.Series(False, index=ids)
    g = h.groupby("ID", sort=False)["time"]
    last = g.max()
    first = g.min()
    n = g.size()
    h = h.copy()
    h["gap"] = h.groupby("ID", sort=False)["time"].diff().dt.total_seconds() / 86400.0
    gg = h.groupby("ID", sort=False)["gap"]
    gap_mean = gg.mean()
    gap_std = gg.std()
    gap_max = gg.max()
    rec = (t - last).dt.total_seconds() / 86400.0
    age = (t - first).dt.total_seconds() / 86400.0
    span = (last - first).dt.total_seconds() / 86400.0
    never = ~n.index.isin(n.index)  # placeholder, overwritten below
    n = n.reindex(ids)
    never = n.isna() | n.eq(0)
    return (
        reindex_fill(rec, ids, 999.0),
        reindex_fill(age, ids, 0.0),
        reindex_fill(span, ids, 0.0),
        reindex_fill(gap_mean, ids, 0.0),
        reindex_fill(gap_std, ids, 0.0),
        reindex_fill(gap_max, ids, 0.0),
        never.reindex(ids).fillna(True),
        reindex_fill(n, ids, 0.0),
    )


def sum_ym(frame, yms_keep, ids, col):
    sub = frame.loc[frame["ym"].isin(yms_keep)]
    if sub.empty:
        return pd.Series(0.0, index=ids)
    return sub.groupby("ID", sort=False)[col].sum().reindex(ids).fillna(0.0)


def ses_level(monthly, alpha):
    """Simple exponential smoothing. monthly is (n, 6) oldest to newest."""
    level = monthly[:, 0]
    for k in range(1, monthly.shape[1]):
        level = alpha * monthly[:, k] + (1.0 - alpha) * level
    return level


def bgnbd_expected(x, tx, T, horizon=90.0):
    """Expected purchases over the next horizon. Birth is the first purchase.

    Likelihood and conditional expectation are Fader, Hardie and Lee (2005),
    equations for the BG/NBD. a is kept above 1 so the expectation is finite.
    Returns zeros when the fit is not usable.
    """
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)
    keep = np.isfinite(x) & np.isfinite(T) & (T > 0) & (x >= 0)
    if int(keep.sum()) < 200:
        return np.zeros(len(x)), None
    xk, txk, Tk = x[keep], tx[keep], T[keep]
    Tk = np.maximum(Tk, np.maximum(txk, 0.5))
    txk = np.where(xk > 0, np.clip(txk, 0, Tk), 0.0)

    def nll(theta):
        r, alpha, a, b = theta
        log_a = (
            gammaln(r + xk)
            - gammaln(r)
            + r * np.log(alpha)
            - (r + xk) * np.log(alpha + Tk)
            + betaln(a, b + xk)
            - betaln(a, b)
        )
        both = log_a.copy()
        pos = xk > 0
        if pos.any():
            log_b = (
                gammaln(r + xk[pos])
                - gammaln(r)
                + r * np.log(alpha)
                - (r + xk[pos]) * np.log(alpha + txk[pos])
                + betaln(a + 1.0, b + xk[pos] - 1.0)
                - betaln(a, b)
            )
            m = np.maximum(log_a[pos], log_b)
            both[pos] = m + np.log(np.exp(log_a[pos] - m) + np.exp(log_b - m))
        if not np.isfinite(both).all():
            return 1e12
        return float(-both.sum())

    res = minimize(
        nll,
        x0=np.array([0.8, 15.0, 1.4, 2.5]),
        method="L-BFGS-B",
        bounds=[(0.05, 30.0), (0.2, 2000.0), (1.05, 20.0), (0.05, 80.0)],
        options={"maxiter": 80},
    )
    if not np.isfinite(res.fun):
        return np.zeros(len(x)), None
    r, alpha, a, b = (float(v) for v in res.x)
    out = np.zeros(len(x))
    xx, txx, TT = xk, txk, Tk
    ratio = (alpha + TT) / (alpha + TT + horizon)
    z = horizon / (alpha + TT + horizon)
    hyp = np.real(hyp2f1(r + xx, b + xx, a + b + xx - 1.0, z))
    first = (a + b + xx - 1.0) / (a - 1.0)
    second = 1.0 - np.power(ratio, r + xx) * hyp
    numer = first * second
    extra = np.zeros(len(xx))
    pos = xx > 0
    extra[pos] = (a / (b + xx[pos] - 1.0)) * np.power((alpha + TT[pos]) / (alpha + txx[pos]), r + xx[pos])
    denom = 1.0 + extra
    pred = numer / denom
    pred = np.where(np.isfinite(pred), pred, 0.0)
    out[keep] = np.clip(pred, 0.0, 60.0)
    return out, (float(r), float(alpha), float(a), float(b), float(res.fun))


def matrix_from(frame, yms_keep, ids, cats, value, how):
    sub = frame.loc[frame["ym"].isin(yms_keep)]
    empty = pd.DataFrame(0.0, index=ids, columns=cats)
    if sub.empty:
        return empty
    if how == "sum":
        s = sub.groupby(["ID", "cat"], sort=False)[value].sum()
    else:
        s = sub.groupby(["ID", "cat"], sort=False)[value].sum()
    wide = s.unstack(fill_value=0.0)
    return wide.reindex(index=ids, columns=cats, fill_value=0.0).astype(float)


def build_features(pack, cutoff, ids, prev_labels, included, s_fuel, s_shop, classes):
    t = pd.Timestamp(cutoff)
    base = t.year * 12 + t.month
    ids = pd.Index(pd.Series(ids, dtype=str).unique())
    tx = pack["tx"]
    fuel_tx = pack["fuel_tx"]
    shop_tx = pack["shop_tx"]
    h = tx.loc[tx["time"] < t]
    # one row per transaction already
    g = h.groupby("ID", sort=False)
    n_tx = g["TransactionId"].nunique()
    n_loc = g["location"].nunique()
    last = g["time"].max()
    first = g["time"].min()
    weekend_share = g["weekend"].mean()
    morning_share = g["morning"].mean()
    evening_share = g["evening"].mean()
    mean_basket = g["basket"].mean()

    h_sorted = h.sort_values(["ID", "time"])
    h_sorted = h_sorted.copy()
    h_sorted["gap"] = h_sorted.groupby("ID", sort=False)["time"].diff().dt.total_seconds() / 86400.0
    gap_mean = h_sorted.groupby("ID", sort=False)["gap"].mean()
    gap_std = h_sorted.groupby("ID", sort=False)["gap"].std()
    gap_max = h_sorted.groupby("ID", sort=False)["gap"].max()

    loc_counts = h.groupby(["ID", "location"], sort=False).size().rename("c").reset_index()
    loc_counts = loc_counts.sort_values(["ID", "c"], ascending=[True, False])
    primary = loc_counts.drop_duplicates("ID").set_index("ID")
    primary_loc = primary["location"]
    primary_n = primary["c"]

    recency = (t - last).dt.total_seconds() / 86400.0
    age = (t - first).dt.total_seconds() / 86400.0

    fuel_h = fuel_tx.loc[fuel_tx["time"] < t]
    shop_h = shop_tx.loc[shop_tx["time"] < t]
    fuel_n = fuel_h.groupby("ID").size()
    shop_n = shop_h.groupby("ID").size()
    fuel_last = fuel_h.groupby("ID")["time"].max()
    shop_last = shop_h.groupby("ID")["time"].max()
    fuel_first = fuel_h.groupby("ID")["time"].min()
    shop_first = shop_h.groupby("ID")["time"].min()
    fuel_qty_life = fuel_h.groupby("ID")["qty"].sum()
    fuel_amt_life = fuel_h.groupby("ID")["amt"].sum()
    shop_amt_life = shop_h.groupby("ID")["amt"].sum()

    fuel_sorted = fuel_h.sort_values(["ID", "time"]).copy()
    fuel_sorted["gap"] = fuel_sorted.groupby("ID", sort=False)["time"].diff().dt.total_seconds() / 86400.0
    fuel_gap_mean = fuel_sorted.groupby("ID")["gap"].mean()
    fuel_gap_std = fuel_sorted.groupby("ID")["gap"].std()
    # modal weekday share: regularity of the fill-up
    if len(fuel_h):
        dow_counts = fuel_h.groupby(["ID", "dow"]).size()
        modal = dow_counts.groupby("ID").max()
        dow_tot = dow_counts.groupby("ID").sum()
        modal_share = modal / dow_tot
    else:
        modal_share = pd.Series(dtype=float)

    q1 = yms(t, 1, 3)
    q2 = yms(t, 4, 3)
    q3 = yms(t, 7, 3)
    q4 = yms(t, 10, 3)
    months = yms(t, 1, 6)  # oldest -> newest
    life_ym = list(range(int(tx["ym"].min()), base))
    yago = year_ago_outcome(t)
    recent30 = t - pd.Timedelta(days=30)
    cols = {}
    cols["n_tx"] = reindex_fill(n_tx, ids, 0).to_numpy(dtype=float)
    cols["log_n_tx"] = np.log1p(cols["n_tx"])
    cols["n_locations"] = reindex_fill(n_loc, ids, 0).to_numpy(dtype=float)
    cols["primary_share"] = safe_div(reindex_fill(primary_n, ids, 0).to_numpy(), cols["n_tx"])
    cols["recency_days"] = reindex_fill(recency, ids, 999).to_numpy(dtype=float)
    cols["age_days"] = reindex_fill(age, ids, 0).to_numpy(dtype=float)
    cols["gap_mean"] = reindex_fill(gap_mean, ids, 0).to_numpy(dtype=float)
    cols["gap_cv"] = safe_div(reindex_fill(gap_std, ids, 0).to_numpy(), cols["gap_mean"])
    cols["gap_max"] = reindex_fill(gap_max, ids, 0).to_numpy(dtype=float)
    cols["overdue"] = safe_div(cols["recency_days"], cols["gap_mean"])
    cols["recency_over_maxgap"] = safe_div(cols["recency_days"], cols["gap_max"])
    cols["weekend_share"] = reindex_fill(weekend_share, ids, 0).to_numpy(dtype=float)
    cols["morning_share"] = reindex_fill(morning_share, ids, 0).to_numpy(dtype=float)
    cols["evening_share"] = reindex_fill(evening_share, ids, 0).to_numpy(dtype=float)
    cols["mean_basket"] = reindex_fill(mean_basket, ids, 0).to_numpy(dtype=float)
    cols["origin_month"] = float((base - (2024 * 12 + 3)))

    fn = reindex_fill(fuel_n, ids, 0).to_numpy(dtype=float)
    sn = reindex_fill(shop_n, ids, 0).to_numpy(dtype=float)
    cols["fuel_tx"] = fn
    cols["shop_tx"] = sn
    cols["fuel_never"] = (fn == 0).astype(float)
    cols["shop_never"] = (sn == 0).astype(float)
    cols["fuel_recency"] = reindex_fill((t - fuel_last).dt.total_seconds() / 86400.0, ids, 999).to_numpy()
    cols["shop_recency"] = reindex_fill((t - shop_last).dt.total_seconds() / 86400.0, ids, 999).to_numpy()
    cols["fuel_age"] = reindex_fill((t - fuel_first).dt.total_seconds() / 86400.0, ids, 0).to_numpy()
    cols["shop_age"] = reindex_fill((t - shop_first).dt.total_seconds() / 86400.0, ids, 0).to_numpy()
    cols["fuel_qty_life"] = reindex_fill(fuel_qty_life, ids, 0).to_numpy(dtype=float)
    cols["fuel_amt_life"] = reindex_fill(fuel_amt_life, ids, 0).to_numpy(dtype=float)
    cols["shop_amt_life"] = reindex_fill(shop_amt_life, ids, 0).to_numpy(dtype=float)
    cols["log_fuel_qty_life"] = np.log1p(np.clip(cols["fuel_qty_life"], 0, None))
    cols["log_shop_amt_life"] = np.log1p(np.clip(cols["shop_amt_life"], 0, None))
    cols["fuel_per_tx"] = safe_div(cols["fuel_qty_life"], fn)
    cols["fuel_rand_per_tx"] = safe_div(cols["fuel_amt_life"], fn)
    cols["shop_per_tx"] = safe_div(cols["shop_amt_life"], sn)
    cols["fuel_gap_mean"] = reindex_fill(fuel_gap_mean, ids, 0).to_numpy(dtype=float)
    cols["fuel_gap_cv"] = safe_div(reindex_fill(fuel_gap_std, ids, 0).to_numpy(), cols["fuel_gap_mean"])
    cols["fuel_overdue"] = safe_div(cols["fuel_recency"], cols["fuel_gap_mean"])
    cols["fuel_modal_dow"] = reindex_fill(modal_share, ids, 0).to_numpy(dtype=float)

    return _finish_features(
        cols, ids, t, base, pack, included, s_fuel, s_shop, classes,
        prev_labels, primary_loc, q1, q2, q3, q4, months, life_ym, yago, recent30,
        fuel_tx, shop_tx, fn, sn,
    )


def _count_ym(frame, window, ids):
    sub = frame.loc[frame["ym"].isin(window)]
    if sub.empty:
        return np.zeros(len(ids))
    return sub.groupby("ID").size().reindex(ids).fillna(0).to_numpy(dtype=float)


def _finish_features(
    cols, ids, t, base, pack, included, s_fuel, s_shop, classes,
    prev_labels, primary_loc, q1, q2, q3, q4, months, life_ym, yago, recent30,
    fuel_tx, shop_tx, fn, sn,
):
    # Replace the clumsy quarter counts with a direct size.
    fuel_h = fuel_tx.loc[fuel_tx["time"] < t]
    shop_h = shop_tx.loc[shop_tx["time"] < t]
    for name, window in {1: q1, 2: q2, 3: q3, 4: q4}.items():
        cols[f"q{name}_fuel_l"] = sum_ym(fuel_h, window, ids, "qty").to_numpy(dtype=float)
        cols[f"q{name}_fuel_r"] = sum_ym(fuel_h, window, ids, "amt").to_numpy(dtype=float)
        cols[f"q{name}_fuel_n"] = _count_ym(fuel_h, window, ids)
        cols[f"q{name}_shop_r"] = sum_ym(shop_h, window, ids, "amt").to_numpy(dtype=float)
        cols[f"q{name}_shop_n"] = _count_ym(shop_h, window, ids)

    for i, m in enumerate(reversed(months), start=1):
        cols[f"m{i}_fuel_l"] = sum_ym(fuel_h, [m], ids, "qty").to_numpy(dtype=float)
        cols[f"m{i}_shop_r"] = sum_ym(shop_h, [m], ids, "amt").to_numpy(dtype=float)

    cols["yago_fuel_l"] = sum_ym(fuel_h, yago, ids, "qty").to_numpy(dtype=float)
    cols["yago_shop_r"] = sum_ym(shop_h, yago, ids, "amt").to_numpy(dtype=float)
    cols["log_q1_fuel_l"] = np.log1p(np.clip(cols["q1_fuel_l"], 0, None))
    cols["log_q1_shop_r"] = np.log1p(np.clip(cols["q1_shop_r"], 0, None))
    cols["fuel_z_q1"] = np.log1p(np.clip(cols["q1_fuel_l"], 0, None)) / s_fuel
    cols["shop_z_q1"] = np.log1p(np.clip(cols["q1_shop_r"], 0, None)) / s_shop
    cols["fuel_z_q2"] = np.log1p(np.clip(cols["q2_fuel_l"], 0, None)) / s_fuel
    cols["shop_z_q2"] = np.log1p(np.clip(cols["q2_shop_r"], 0, None)) / s_shop
    cols["fuel_z_yago"] = np.log1p(np.clip(cols["yago_fuel_l"], 0, None)) / s_fuel
    cols["shop_z_yago"] = np.log1p(np.clip(cols["yago_shop_r"], 0, None)) / s_shop
    cols["fuel_dz"] = cols["fuel_z_q1"] - cols["fuel_z_q2"]
    cols["shop_dz"] = cols["shop_z_q1"] - cols["shop_z_q2"]

    past_z_f = np.vstack([
        np.log1p(np.clip(cols[f"q{k}_fuel_l"], 0, None)) / s_fuel for k in (1, 2, 3, 4)
    ])
    past_z_s = np.vstack([
        np.log1p(np.clip(cols[f"q{k}_shop_r"], 0, None)) / s_shop for k in (1, 2, 3, 4)
    ])
    # A quarter with no history yet is all zeros; still a real observation of zero
    # only when the customer was already active. Use age to count observed quarters.
    observed = np.vstack([
        (cols["age_days"] >= 90 * k).astype(float) for k in (1, 2, 3, 4)
    ])
    obs_n = observed.sum(0)
    cols["n_past_quarters"] = obs_n
    cols["fuel_z_mean"] = safe_div((past_z_f * observed).sum(0), obs_n)
    cols["shop_z_mean"] = safe_div((past_z_s * observed).sum(0), obs_n)
    # Empirical-Bayes shrink of last quarter toward the customer's own mean.
    k_shrink = 1.5
    cols["fuel_z_eb"] = (cols["fuel_z_q1"] + k_shrink * cols["fuel_z_mean"]) / (1.0 + k_shrink)
    cols["shop_z_eb"] = (cols["shop_z_q1"] + k_shrink * cols["shop_z_mean"]) / (1.0 + k_shrink)

    month_f = np.column_stack([cols[f"m{i}_fuel_l"] for i in range(6, 0, -1)])
    month_s = np.column_stack([cols[f"m{i}_shop_r"] for i in range(6, 0, -1)])
    # ses_level expects oldest first. m6 is oldest, m1 is newest, so reverse the stack.
    month_f = np.column_stack([cols[f"m{i}_fuel_l"] for i in range(6, 0, -1)])
    month_s = np.column_stack([cols[f"m{i}_shop_r"] for i in range(6, 0, -1)])
    cols["fuel_ses_z"] = np.log1p(np.clip(3.0 * ses_level(month_f, 0.4), 0, None)) / s_fuel
    cols["shop_ses_z"] = np.log1p(np.clip(3.0 * ses_level(month_s, 0.4), 0, None)) / s_shop
    cols["fuel_ses_fast_z"] = np.log1p(np.clip(3.0 * ses_level(month_f, 0.7), 0, None)) / s_fuel
    cols["shop_ses_fast_z"] = np.log1p(np.clip(3.0 * ses_level(month_s, 0.7), 0, None)) / s_shop

    # Cadence: Ehrenberg / BG-CNBD-k. Weekly fuel is not memoryless.
    # Horizon is the actual next three calendar months, not a flat 90 days.
    horizon = float((t + pd.DateOffset(months=3) - t).days)
    cols["outcome_days"] = horizon
    overdue_excess = np.maximum(cols["fuel_recency"] - cols["fuel_gap_mean"], 0.0)
    alive = np.exp(-safe_div(overdue_excess, np.maximum(cols["fuel_gap_mean"], 1.0)))
    alive = np.where(cols["fuel_tx"] >= 2, alive, 0.0)
    e_fills = safe_div(np.full(len(ids), horizon), np.maximum(cols["fuel_gap_mean"], 1.0)) * alive
    e_fills = np.where(cols["fuel_tx"] >= 2, e_fills, 0.0)
    cols["fuel_cadence_z"] = np.log1p(np.clip(e_fills * cols["fuel_per_tx"], 0, None)) / s_fuel
    shop_over = np.maximum(cols["shop_recency"] - cols["gap_mean"], 0.0)
    shop_alive = np.exp(-safe_div(shop_over, np.maximum(cols["gap_mean"], 1.0)))
    shop_alive = np.where(cols["shop_tx"] >= 2, shop_alive, 0.0)
    e_shop_tx = safe_div(np.full(len(ids), horizon), np.maximum(cols["gap_mean"], 1.0)) * shop_alive
    cols["shop_cadence_z"] = np.log1p(np.clip(e_shop_tx * cols["shop_per_tx"], 0, None)) / s_shop

    # BG/NBD expected purchases, then gamma-gamma style monetary scaling.
    fuel_span = np.maximum(cols["fuel_age"] - cols["fuel_recency"], 0.0)  # not quite t_x
    # t_x = age - recency, T = age, x = n - 1
    x_f = np.maximum(fn - 1.0, 0.0)
    T_f = cols["fuel_age"]
    tx_f = np.where(x_f > 0, np.maximum(T_f - cols["fuel_recency"], 0.0), 0.0)
    e_f, fit_f = bgnbd_expected(x_f, tx_f, T_f, horizon=horizon)
    cols["fuel_bgnbd_z"] = np.log1p(np.clip(e_f * cols["fuel_per_tx"], 0, None)) / s_fuel
    x_s = np.maximum(sn - 1.0, 0.0)
    T_s = cols["shop_age"]
    tx_s = np.where(x_s > 0, np.maximum(T_s - cols["shop_recency"], 0.0), 0.0)
    e_s, fit_s = bgnbd_expected(x_s, tx_s, T_s, horizon=horizon)
    cols["shop_bgnbd_z"] = np.log1p(np.clip(e_s * cols["shop_per_tx"], 0, None)) / s_shop
    cols["_bgnbd_fuel_fit"] = 0.0  # placeholder removed after
    fit_note = {"fuel": fit_f, "shop": fit_s}

    f30 = fuel_h.loc[fuel_h["time"] >= recent30].groupby("ID")["qty"].sum()
    s30 = shop_h.loc[shop_h["time"] >= recent30].groupby("ID")["amt"].sum()
    n30 = pack["tx"].loc[(pack["tx"]["time"] < t) & (pack["tx"]["time"] >= recent30)].groupby("ID").size()
    cols["fuel_l_30"] = reindex_fill(f30, ids, 0).to_numpy(dtype=float)
    cols["shop_r_30"] = reindex_fill(s30, ids, 0).to_numpy(dtype=float)
    cols["tx_30"] = reindex_fill(n30, ids, 0).to_numpy(dtype=float)
    cols["fuel_accel"] = safe_div(cols["fuel_l_30"], np.maximum(cols["q1_fuel_l"] / 3.0, 1e-6))
    cols["shop_accel"] = safe_div(cols["shop_r_30"], np.maximum(cols["q1_shop_r"] / 3.0, 1e-6))
    life_months = np.maximum(cols["age_days"] / 30.0, 1.0)
    cols["fuel_rate_ratio"] = safe_div(cols["q1_fuel_l"] / 3.0, cols["fuel_qty_life"] / life_months)
    cols["shop_rate_ratio"] = safe_div(cols["q1_shop_r"] / 3.0, np.maximum(cols["shop_amt_life"], 0) / life_months)
    # Recent fill size versus the customer's own tank. A shrinking fill is
    # volume falling even when the visit rate holds (gamma-gamma monetary rate).
    cols["fuel_fill_recent"] = safe_div(cols["q1_fuel_l"], cols["q1_fuel_n"])
    cols["fuel_fill_ratio"] = safe_div(cols["fuel_fill_recent"], np.maximum(cols["fuel_per_tx"], 1e-6))
    cols["fuel_freq_ratio"] = safe_div(cols["q1_fuel_n"], np.maximum(cols["q2_fuel_n"], 0.0))
    cols["shop_ticket_recent"] = safe_div(cols["q1_shop_r"], cols["q1_shop_n"])
    cols["shop_ticket_ratio"] = safe_div(cols["shop_ticket_recent"], np.maximum(cols["shop_per_tx"], 1e-6))
    shop_q = np.column_stack([cols["m1_shop_r"], cols["m2_shop_r"], cols["m3_shop_r"]])
    fuel_qm = np.column_stack([cols["m1_fuel_l"], cols["m2_fuel_l"], cols["m3_fuel_l"]])
    cols["shop_q_cv"] = safe_div(shop_q.std(axis=1), np.abs(shop_q.mean(axis=1)) + 1e-6)
    cols["fuel_q_cv"] = safe_div(fuel_qm.std(axis=1), np.abs(fuel_qm.mean(axis=1)) + 1e-6)
    cols["crossbuy_q1"] = ((cols["q1_fuel_l"] > 0) & (cols["q1_shop_r"] > 0)).astype(float)
    cols["fuel_price_q1"] = safe_div(cols["q1_fuel_r"], cols["q1_fuel_l"])
    price = pack["price_ym"]
    p_prev_idx = price.loc[price.index < base].sort_index()
    p_now = float(p_prev_idx.iloc[-1]) if len(p_prev_idx) else 0.0
    p_3 = float(p_prev_idx.iloc[-4]) if len(p_prev_idx) >= 4 else p_now
    cols["price_index"] = p_now
    cols["price_change"] = (p_now / p_3 - 1.0) if p_3 else 0.0
    cols["price_gap"] = safe_div(cols["fuel_price_q1"], np.full(len(ids), p_now if p_now else 1.0)) - 1.0
    cols["price_gap"] = np.where(cols["q1_fuel_l"] > 0, cols["price_gap"], 0.0)
    # Rand the growth rule needs next quarter, if litres merely repeat at the new price.
    cols["fuel_rand_if_persist"] = np.maximum(cols["q1_fuel_l"], 0) * (p_now if p_now else 0.0)
    need = np.maximum(cols["q1_fuel_r"] * 1.25, cols["q1_fuel_r"] + 20.0)
    cols["fuel_growth_gap"] = safe_div(cols["fuel_rand_if_persist"] - need, np.maximum(np.abs(need), 1.0))
    cols["fuel_growth_gap"] = np.where(fn > 0, cols["fuel_growth_gap"], 0.0)

    # Category breadth. Spend is signed rands; counts are qualifying baskets.
    spend_q1 = matrix_from(pack["cat_spend"], q1, ids, included, "spend", "sum")
    spend_life = matrix_from(pack["cat_spend"], life_ym, ids, included, "spend", "sum")
    tx_q1 = matrix_from(pack["cat_tx"], q1, ids, included, "ntx", "sum")
    tx_life = matrix_from(pack["cat_tx"], life_ym, ids, included, "ntx", "sum")
    for c in included:
        key = slug(c)
        cols[f"cat_{key}_q1"] = np.log1p(np.clip(spend_q1[c].to_numpy(), 0, None))
        cols[f"cat_{key}_ever"] = (tx_life[c].to_numpy() > 0).astype(float)
    pos = spend_q1.clip(lower=0)
    tot = pos.sum(axis=1).to_numpy()
    share = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(share.to_numpy() * np.log(np.clip(share.to_numpy(), 1e-12, None))).sum(axis=1)
    ent = np.where(tot > 0, ent, 0.0)
    cols["cat_entropy"] = ent
    cols["n_cat_ever"] = (tx_life.to_numpy() > 0).sum(axis=1).astype(float)
    cols["n_cat_q1"] = (tx_q1.to_numpy() > 0).sum(axis=1).astype(float)
    cols["n_lapsed"] = ((tx_life.to_numpy() > 0) & (tx_q1.to_numpy() == 0)).sum(axis=1).astype(float)
    cols["n_easy"] = ((spend_q1.to_numpy() > 0) & (spend_q1.to_numpy() <= 40)).sum(axis=1).astype(float)
    hab = [c for c in HABITUAL if c in pos.columns]
    occ = [c for c in OCCASIONAL if c in pos.columns]
    hab_sum = pos[hab].sum(axis=1).to_numpy() if hab else np.zeros(len(ids))
    occ_sum = pos[occ].sum(axis=1).to_numpy() if occ else np.zeros(len(ids))
    cols["habitual_share"] = safe_div(hab_sum, tot)
    cols["occasional_share"] = safe_div(occ_sum, tot)
    cols["log_bev_q1"] = np.log1p(np.clip(spend_q1["Beverages"].to_numpy(), 0, None)) if "Beverages" in spend_q1 else np.zeros(len(ids))
    cols["log_food_q1"] = np.log1p(np.clip(spend_q1["Food & beverage"].to_numpy(), 0, None)) if "Food & beverage" in spend_q1 else np.zeros(len(ids))

    # Peer level at the primary site, leave-one-out, from last-quarter z only.
    site = primary_loc.reindex(ids)
    site_frame = pd.DataFrame({
        "site": site.astype(str).fillna("NA").to_numpy(),
        "fz": cols["fuel_z_q1"],
        "sz": cols["shop_z_q1"],
    })
    gsum_f = site_frame.groupby("site")["fz"].transform("sum")
    gsum_s = site_frame.groupby("site")["sz"].transform("sum")
    gcnt = site_frame.groupby("site")["fz"].transform("count")
    glob_f = float(np.mean(cols["fuel_z_q1"]))
    glob_s = float(np.mean(cols["shop_z_q1"]))
    loo_f = np.where(gcnt > 1, (gsum_f - site_frame["fz"]) / (gcnt - 1), glob_f)
    loo_s = np.where(gcnt > 1, (gsum_s - site_frame["sz"]) / (gcnt - 1), glob_s)
    cols["site_loo_fuel_z"] = np.asarray(loo_f, dtype=float)
    cols["site_loo_shop_z"] = np.asarray(loo_s, dtype=float)
    cols["site_n"] = np.asarray(gcnt, dtype=float)

    prev = prev_labels.reindex(ids).fillna("No prior label")
    levels = ["No prior label"] + list(classes)
    for level in levels:
        cols["prev__" + slug(level)] = (prev.to_numpy() == level).astype(float)
    cols["prev_inactivity"] = (prev.to_numpy() == "Inactivity").astype(float)
    cols["prev_stable"] = (prev.to_numpy() == "Stable").astype(float)
    cols["prev_growth"] = pd.Series(prev.to_numpy()).str.startswith("Existing-category growth").astype(float).to_numpy()
    cols["prev_adopt"] = pd.Series(prev.to_numpy()).str.startswith("New category adoption").astype(float).to_numpy()

    frame = pd.DataFrame(cols, index=ids)
    frame = frame.drop(columns=["_bgnbd_fuel_fit"], errors="ignore")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame.index.name = "ID"
    return frame, fit_note


def build_panel(quick_note=False):
    t0 = time.time()
    cfg, included, shop_cats, classes = load_config()
    print("loading transactions", flush=True)
    d = lr.load_transactions(DOWNLOADS / "train.csv")
    print(f"rows {len(d)} in {time.time() - t0:.1f}s", flush=True)
    pack = prepare_tables(d, included, shop_cats, set(cfg["adjustment_items"]))
    s_fuel = float(cfg["normalization"]["CLV_fuel"])
    s_shop = float(cfg["normalization"]["CLV_nonfuel"])
    labels = {}
    for cutoff in ALL_LABEL_CUTOFFS:
        t1 = time.time()
        raw = lr.raw_labels(d, cutoff, cfg)
        lab = lr.transform(raw, cfg).set_index("ID")
        lab.index = lab.index.astype(str)
        labels[cutoff] = lab
        print(f"labels {cutoff} n={len(lab)} {time.time() - t1:.1f}s", flush=True)
    # Official September audit.
    vc = labels["2025-09-01"]["Opportunity"].value_counts()
    for name, n in SEP_COUNTS.items():
        got = int(vc.get(name, 0))
        if got != n:
            raise SystemExit(f"September label mismatch {name}: got {got} expected {n}")
    adopt = int(vc[vc.index.str.startswith("New category adoption")].sum())
    if adopt != 368:
        raise SystemExit(f"September adoption count {adopt} != 368")
    print("September labels match the official counts", flush=True)

    frames = {}
    fit_notes = {}
    minimum = int(cfg["minimum_historical_transactions"])
    test_ids = pd.read_csv(DOWNLOADS / "test.csv", dtype=str)["ID"].astype(str)
    for cutoff in TRAIN_CUTOFFS + [TEST_CUTOFF]:
        t1 = time.time()
        if cutoff == TEST_CUTOFF:
            ids = test_ids
        else:
            ids = pd.Index(labels[cutoff].index.astype(str))
        prev = labels[prev_quarter(cutoff)]["Opportunity"]
        frame, note = build_features(
            pack, cutoff, ids, prev, included, s_fuel, s_shop, classes
        )
        if cutoff != TEST_CUTOFF:
            # same customers, label order
            if set(frame.index) != set(labels[cutoff].index) or len(frame) != len(labels[cutoff]):
                raise SystemExit(f"feature/label id mismatch at {cutoff}")
        frames[cutoff] = frame
        fit_notes[cutoff] = note
        print(
            f"features {cutoff} n={len(frame)} cols={frame.shape[1]} "
            f"bgnbd_fuel={note['fuel']} bgnbd_shop={note['shop']} {time.time() - t1:.1f}s",
            flush=True,
        )
    feature_cols = [c for c in frames[TRAIN_CUTOFFS[0]].columns]
    for cutoff, frame in frames.items():
        if list(frame.columns) != feature_cols:
            raise SystemExit(f"feature columns differ at {cutoff}")
    panel = {
        "frames": frames,
        "labels": {c: labels[c] for c in TRAIN_CUTOFFS},
        "feature_cols": feature_cols,
        "classes": classes,
        "s_fuel": s_fuel,
        "s_shop": s_shop,
        "test_ids": test_ids.to_numpy(),
    }
    print(f"panel ready in {time.time() - t0:.1f}s features={len(feature_cols)}", flush=True)
    return panel


def stack_train(panel, cutoffs):
    xs = []
    parts = []
    for c in cutoffs:
        f = panel["frames"][c]
        y = panel["labels"][c].reindex(f.index)
        block = f.copy()
        block["CLV_fuel"] = y["CLV_fuel"].to_numpy()
        block["CLV_nonfuel"] = y["CLV_nonfuel"].to_numpy()
        block["Opportunity"] = y["Opportunity"].to_numpy()
        block["cutoff"] = c
        block["w"] = SNAPSHOT_WEIGHT[c]
        parts.append(block)
    train = pd.concat(parts, axis=0)
    return train


def lgb_reg_predict(train, pred, target, seeds, n_cap, quick, clip=True):
    features = [c for c in train.columns if c not in ("CLV_fuel", "CLV_nonfuel", "Opportunity", "cutoff", "w")]
    y = train[target].to_numpy(dtype=float)
    w = train["w"].to_numpy(dtype=float)
    cuts = train["cutoff"].to_numpy()
    latest = sorted(set(cuts))[-1]
    tr = cuts != latest
    va = ~tr
    Xtr, Xva = train.loc[tr, features], train.loc[va, features]
    ytr, yva = y[tr], y[va]
    wtr = w[tr]
    preds = []
    n_used = []
    imps = []
    for seed in seeds:
        n_est = 120 if quick else n_cap
        model = LGBMRegressor(
            n_estimators=n_est,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            force_row_wise=True,
        )
        try:
            model.fit(
                Xtr, ytr,
                sample_weight=wtr,
                eval_set=[(Xva, yva)],
                eval_metric="rmse",
                callbacks=[early_stopping(40 if quick else 60, verbose=False), log_evaluation(0)],
            )
            best = int(model.best_iteration_ or n_est)
        except Exception as exc:
            print(f"regressor early stopping skipped: {exc}", flush=True)
            best = 80 if quick else 400
        best = max(best, 40)
        full = LGBMRegressor(
            n_estimators=best,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            force_row_wise=True,
        )
        full.fit(train[features], y, sample_weight=w)
        preds.append(full.predict(pred[features]))
        n_used.append(best)
        imps.append(full.feature_importances_)
    out = np.mean(preds, axis=0)
    if clip:
        out = np.clip(out, 0, None)
    return out, int(np.mean(n_used)), features, np.mean(imps, axis=0)


def lgb_hurdle_predict(train, pred, target, seed, n_cap, quick):
    """Cragg hurdle. z is zero exactly when the raw total is zero."""
    features = [c for c in train.columns if c not in ("CLV_fuel", "CLV_nonfuel", "Opportunity", "cutoff", "w")]
    y = train[target].to_numpy(dtype=float)
    w = train["w"].to_numpy(dtype=float)
    cuts = train["cutoff"].to_numpy()
    latest = sorted(set(cuts))[-1]
    tr = cuts != latest
    va = ~tr
    pos = y > 1e-8
    n_est = 120 if quick else n_cap
    clf = LGBMClassifier(
        n_estimators=n_est,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    clf.fit(
        train.loc[tr, features], pos[tr].astype(int),
        sample_weight=w[tr],
        eval_set=[(train.loc[va, features], pos[va].astype(int))],
        callbacks=[early_stopping(40 if quick else 50, verbose=False), log_evaluation(0)],
    )
    best_c = max(int(clf.best_iteration_ or n_est), 30)
    clf = LGBMClassifier(
        n_estimators=best_c, learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=seed, n_jobs=1, verbosity=-1,
    )
    clf.fit(train[features], pos.astype(int), sample_weight=w)
    p = clf.predict_proba(pred[features])[:, 1]

    tr_pos = tr & pos
    va_pos = va & pos
    reg = LGBMRegressor(
        n_estimators=n_est, learning_rate=0.04, num_leaves=47, min_child_samples=25,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.2,
        random_state=seed, n_jobs=1, verbosity=-1, force_row_wise=True,
    )
    if tr_pos.sum() > 50 and va_pos.sum() > 20:
        reg.fit(
            train.loc[tr_pos, features], y[tr_pos],
            sample_weight=w[tr_pos],
            eval_set=[(train.loc[va_pos, features], y[va_pos])],
            callbacks=[early_stopping(40 if quick else 60, verbose=False), log_evaluation(0)],
        )
        best_r = max(int(reg.best_iteration_ or n_est), 30)
    else:
        best_r = 80
    reg = LGBMRegressor(
        n_estimators=best_r, learning_rate=0.04, num_leaves=47, min_child_samples=25,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.2,
        random_state=seed, n_jobs=1, verbosity=-1, force_row_wise=True,
    )
    reg.fit(train.loc[pos, features], y[pos], sample_weight=w[pos])
    z = np.clip(reg.predict(pred[features]), 0, None)
    return np.clip(p * z, 0, None)


def cat_predict(train, pred, target, seed, quick):
    from catboost import CatBoostRegressor

    features = [c for c in train.columns if c not in ("CLV_fuel", "CLV_nonfuel", "Opportunity", "cutoff", "w")]
    y = train[target].to_numpy(dtype=float)
    w = train["w"].to_numpy(dtype=float)
    cuts = train["cutoff"].to_numpy()
    latest = sorted(set(cuts))[-1]
    tr = cuts != latest
    va = ~tr
    iters = 200 if quick else 700
    model = CatBoostRegressor(
        iterations=iters,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    model.fit(
        train.loc[tr, features], y[tr],
        sample_weight=w[tr],
        eval_set=(train.loc[va, features], y[va]),
        early_stopping_rounds=40,
        use_best_model=True,
    )
    best = max(int(model.get_best_iteration() or iters), 40)
    full = CatBoostRegressor(
        iterations=best, learning_rate=0.05, depth=6, l2_leaf_reg=4.0,
        loss_function="RMSE", random_seed=seed, verbose=False,
        allow_writing_files=False, thread_count=1,
    )
    full.fit(train[features], y, sample_weight=w)
    return np.clip(full.predict(pred[features]), 0, None)


def value_models(panel, train_cutoffs, pred_cutoff, quick, with_cat):
    train = stack_train(panel, train_cutoffs)
    pred = panel["frames"][pred_cutoff]
    seeds = (7,) if quick else (7, 21)
    shop_seeds = (11,) if quick else (11, 29)
    out = {}
    meta = {}
    for target, seed, key in (
        ("CLV_fuel", seeds, "fuel"),
        ("CLV_nonfuel", shop_seeds, "shop"),
    ):
        t1 = time.time()
        direct, n_used, features, imp = lgb_reg_predict(train, pred, target, seed, 1400, quick)
        hurdle = lgb_hurdle_predict(train, pred, target, seed[0], 1000, quick)
        out[f"{key}_lgb"] = direct
        out[f"{key}_hurdle"] = hurdle
        if with_cat:
            out[f"{key}_cat"] = cat_predict(train, pred, target, seed[0] + 3, quick)
        else:
            out[f"{key}_cat"] = direct.copy()
        # Residual on last-quarter z. The lag coefficient stays 1, which is the
        # persistence the series actually has; the tree only learns the change.
        # Fader and Hardie: the change in a non-contractual series mean-reverts.
        zname = "fuel_z_q1" if key == "fuel" else "shop_z_q1"
        resid_train = train.copy()
        base_tr = resid_train[zname].to_numpy(dtype=float)
        resid_train[target] = resid_train[target].to_numpy(dtype=float) - base_tr
        delta, _, _, _ = lgb_reg_predict(
            resid_train, pred, target, (seed[0],), 1000, quick, clip=False
        )
        out[f"{key}_resid"] = np.clip(pred[zname].to_numpy(dtype=float) + delta, 0, None)
        bname = "fuel_bgnbd_z" if key == "fuel" else "shop_bgnbd_z"
        sname = "fuel_ses_z" if key == "fuel" else "shop_ses_z"
        cname = "fuel_cadence_z" if key == "fuel" else "shop_cadence_z"
        ebname = "fuel_z_eb" if key == "fuel" else "shop_z_eb"
        out[f"{key}_persist"] = np.clip(pred[zname].to_numpy(dtype=float), 0, None)
        out[f"{key}_bgnbd"] = np.clip(pred[bname].to_numpy(dtype=float), 0, None)
        out[f"{key}_ses"] = np.clip(pred[sname].to_numpy(dtype=float), 0, None)
        out[f"{key}_cadence"] = np.clip(pred[cname].to_numpy(dtype=float), 0, None)
        out[f"{key}_eb"] = np.clip(pred[ebname].to_numpy(dtype=float), 0, None)
        meta[key] = {
            "trees": n_used,
            "seconds": round(time.time() - t1, 1),
            "importance": sorted(
                zip(features, imp.tolist()), key=lambda kv: kv[1], reverse=True
            )[:20],
        }
        print(f"value {key} -> {pred_cutoff} trees {n_used} in {meta[key]['seconds']}s", flush=True)
    return out, meta


def align_proba(model, X, classes):
    raw = model.predict_proba(X)
    out = np.zeros((len(X), len(classes)))
    index = {c: i for i, c in enumerate(model.classes_)}
    for j, c in enumerate(classes):
        if c in index:
            out[:, j] = raw[:, index[c]]
    total = out.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return out / total


def opp_multiclass(train, pred, classes, seeds, quick):
    features = [c for c in train.columns if c not in ("CLV_fuel", "CLV_nonfuel", "Opportunity", "cutoff", "w")]
    y = train["Opportunity"].to_numpy()
    w = train["w"].to_numpy(dtype=float)
    cuts = train["cutoff"].to_numpy()
    latest = sorted(set(cuts))[-1]
    tr = cuts != latest
    va = ~tr
    n_est = 150 if quick else 1200
    probas = []
    imps = []
    feat = None
    for seed in seeds:
        model = LGBMClassifier(
            n_estimators=n_est,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        try:
            model.fit(
                train.loc[tr, features], y[tr],
                sample_weight=w[tr],
                eval_set=[(train.loc[va, features], y[va])],
                callbacks=[early_stopping(40 if quick else 60, verbose=False), log_evaluation(0)],
            )
            best = max(int(model.best_iteration_ or n_est), 40)
        except Exception as exc:
            print(f"classifier early stopping skipped: {exc}", flush=True)
            best = 80 if quick else 400
        full = LGBMClassifier(
            n_estimators=best, learning_rate=0.04, num_leaves=47, min_child_samples=35,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.70, reg_lambda=1.2,
            random_state=seed, n_jobs=1, verbosity=-1,
        )
        full.fit(train[features], y, sample_weight=w)
        probas.append(align_proba(full, pred[features], classes))
        imps.append(pd.Series(full.feature_importances_, index=features))
        feat = features
    proba = np.mean(probas, axis=0)
    imp = sum(imps) / len(imps)
    top = list(imp.sort_values(ascending=False).head(15).items())
    return proba, top

