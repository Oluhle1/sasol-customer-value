"""Sprint 17 E6 — build the frozen 23-column price/discount/novelty block.

Reads only data/raw/train.csv and the panel frames for alignment. Windows are
strictly before each cutoff. Feature list is frozen in
outputs/sprint17/proposal_price_features.json; do not edit it here.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT17 = ROOT / "outputs" / "sprint17"
OCT = ROOT / "outputs" / "oct10"
DEST = OUT17 / "price_features.pkl"
REPORT = OUT17 / "price_features_report.json"

CATS = {
    "beverages": "Beverages",
    "food_and_beverage": "Food & beverage",
    "confectionary": "Confectionary",
    "snacking": "Snacking",
    "ice_cream": "Ice-cream",
    "tobacco": "Tobacco",
}

FEATURES = [
    "px_mod_share_90",
    "px_mod_intensity_90",
    "px_mod_intensity_30",
    "px_mod_trend",
    "px_item_entropy_90",
    "px_new_item_share_90",
    "px_fuel_disc_share_90",
    "px_fuel_price_30",
    "px_fuel_price_chg",
    "px_fuel_grades_90",
    "px_fuel_grade_delta",
] + [f"px_new_item_share_{c}" for c in CATS] + [f"px_price_chg_{c}" for c in CATS]


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def load_lines():
    config = json.loads((ROOT / "configs" / "label_config.json").read_text())
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv", dtype=str, keep_default_na=False)
    d = pd.DataFrame({
        "ID": raw["ID"].to_numpy(),
        "time": pd.to_datetime(raw["DateTimeZA"], format="mixed", errors="raise"),
        "item": raw["StockItemId"].to_numpy(),
        "l2": raw["ItemCategoryLevel2"].to_numpy(),
        "type": raw["LineItemType"].to_numpy(),
        "qty": pd.to_numeric(raw["Quantity"], errors="coerce").to_numpy(dtype=float),
        "amount": pd.to_numeric(raw["TotalExclAmount"], errors="coerce").to_numpy(dtype=float),
        "mod": pd.to_numeric(raw["TotalModifierAmount"], errors="coerce").to_numpy(dtype=float),
        "price": pd.to_numeric(raw["UnitSellingExclPrice"], errors="coerce").to_numpy(dtype=float),
    })
    d = d[~raw["ItemName"].isin(config["adjustment_items"]).to_numpy()].reset_index(drop=True)
    d["modflag"] = d["mod"].ne(0).astype(float)
    d["is_fuel"] = ((d["l2"] == "Fuel") & (d["type"] == "Sale_Fuel")).astype(bool)
    d["unit_price"] = np.where(d["qty"] > 0, d["amount"] / d["qty"], np.nan)
    return d


def window(d, start, end):
    return d[(d["time"] >= start) & (d["time"] < end)]


def group_share(frame, flag):
    counts = frame.groupby("ID")[flag].mean()
    return counts


def mod_intensity(frame):
    total = frame.groupby("ID")[["mod", "amount"]].sum()
    return total["mod"] / total["amount"].where(total["amount"] > 0)


def item_entropy(frame):
    counts = frame.groupby(["ID", "item"]).size()
    if counts.empty:
        return pd.Series(dtype=float)
    p = counts / counts.groupby("ID").transform("sum")
    ent = -(p * np.log(p)).groupby("ID").sum()
    k = counts.groupby("ID").size()
    return ent / np.log(k).where(k > 1)


def new_item_share(cur, prev, subset=None):
    if subset is not None:
        cur = cur[cur["l2"] == subset]
        prev = prev[prev["l2"] == subset]
    if cur.empty:
        return pd.Series(dtype=float)
    prev_pairs = prev[["ID", "item"]].drop_duplicates()
    prev_pairs = prev_pairs.assign(seen=1)
    merged = cur[["ID", "item"]].merge(prev_pairs, on=["ID", "item"], how="left")
    merged["seen"] = merged["seen"].fillna(0)
    return merged.groupby("ID")["seen"].apply(lambda s: float((s == 0).mean()))


def price_change(cur, prev, subset=None):
    if subset is not None:
        cur = cur[cur["l2"] == subset]
        prev = prev[prev["l2"] == subset]
    cur = cur[cur["unit_price"].notna()]
    prev = prev[prev["unit_price"].notna()]
    if cur.empty:
        return pd.Series(dtype=float)
    a = cur.groupby("ID")["unit_price"].mean()
    b = prev.groupby("ID")["unit_price"].mean()
    return a / b.where(b > 0) - 1.0


def fuel_change(cur, prev, start30):
    cur_f = cur[cur["is_fuel"]]
    prev_f = prev[prev["is_fuel"]]
    w30 = cur_f[cur_f["time"] >= start30]
    out = pd.DataFrame(index=pd.Index([], name="ID"))
    out["px_fuel_disc_share_90"] = cur_f.groupby("ID")["modflag"].mean()
    out["px_fuel_price_30"] = w30.groupby("ID")["price"].mean()
    a = cur_f.groupby("ID")["price"].mean()
    b = prev_f.groupby("ID")["price"].mean()
    out["px_fuel_price_chg"] = a / b.where(b > 0) - 1.0
    out["px_fuel_grades_90"] = cur_f.groupby("ID")["item"].nunique().astype(float)
    g_prev = prev_f.groupby("ID")["item"].nunique().astype(float)
    out["px_fuel_grade_delta"] = out["px_fuel_grades_90"] - g_prev.reindex(out.index)
    return out


def compute_cutoff(d, cutoff):
    t = pd.Timestamp(cutoff)
    start90, start30, start180 = t - pd.Timedelta(days=90), t - pd.Timedelta(days=30), t - pd.Timedelta(days=180)
    w30 = window(d, start30, t)
    w90 = window(d, start90, t)
    w_prev = window(d, start180, start90)

    cols = {}
    cols["px_mod_share_90"] = group_share(w90, "modflag")
    cols["px_mod_intensity_90"] = mod_intensity(w90)
    cols["px_mod_intensity_30"] = mod_intensity(w30)
    cols["px_item_entropy_90"] = item_entropy(w90)
    cols["px_new_item_share_90"] = new_item_share(w90, w_prev)
    out = pd.DataFrame(cols)
    out["px_mod_trend"] = out["px_mod_intensity_30"] - out["px_mod_intensity_90"]
    out = out.join(fuel_change(w90, w_prev, start30))
    for c, l2 in CATS.items():
        out[f"px_new_item_share_{c}"] = new_item_share(w90, w_prev, subset=l2)
        out[f"px_price_chg_{c}"] = price_change(w90, w_prev, subset=l2)
    return out[FEATURES]


def main():
    started = time.time()
    proposal = json.loads((OUT17 / "proposal_price_features.json").read_text())
    frozen = (
        proposal["feature_block"]["global"] + proposal["feature_block"]["fuel"]
        + [f"px_new_item_share_{c}" for c in proposal["feature_block"]["categories"]]
        + [f"px_price_chg_{c}" for c in proposal["feature_block"]["categories"]]
    )
    if sorted(frozen) != sorted(FEATURES):
        raise RuntimeError("feature list diverged from the frozen proposal")
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    d = load_lines()
    log(f"lines {len(d)}, cutoffs {len(panel['frames'])}")

    tables = {}
    report = {}
    for cutoff, frame in panel["frames"].items():
        table = compute_cutoff(d, cutoff)
        table = table.reindex(frame.index.astype(str))
        tables[cutoff] = table
        report[cutoff] = {
            "rows": int(len(table)),
            "non_null_share": {c: float(table[c].notna().mean()) for c in FEATURES},
        }
        log(f"{cutoff}: rows {len(table)}, non-null min {min(report[cutoff]['non_null_share'].values()):.3f}")

    probe = "2025-04-01"
    mask = d["time"] < pd.Timestamp(probe)
    table_full = compute_cutoff(d, probe)
    table_trunc = compute_cutoff(d[mask].reset_index(drop=True), probe)
    a, b = table_full.reindex(tables[probe].index), table_trunc.reindex(tables[probe].index)
    diff = np.nanmax(np.abs(a.to_numpy() - b.to_numpy())) if len(a) else 0.0
    if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, atol=1e-12):
        raise RuntimeError(f"future leakage detected: max diff {diff}")
    log(f"leakage check passed on {probe} (max diff {diff:.2e})")

    DEST.write_bytes(pickle.dumps(tables))
    report["_leakage_check"] = {"cutoff": probe, "max_diff": float(diff)}
    report["_features"] = FEATURES
    report["_seconds"] = round(time.time() - started, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"wrote {DEST.name} and {REPORT.name} in {report['_seconds']}s")


if __name__ == "__main__":
    main()
