"""Anderson, Kumar, Tomkins and Vassilvitskii (WWW 2014).

Add category-level recency to the frozen 0.30225 decision rule.
The multipliers are not retuned. June and September are both scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHIPPED = Path(r"C:\Users\oluhl\sasol-customer-value\top4_shipped")
sys.path.insert(0, str(SHIPPED))
import train_growth_override as base  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PUBLISHED_SEPTEMBER_F1 = 0.5118


def add_category_recency(frame, cust: pd.DataFrame, cutoff: str, ns) -> pd.DataFrame:
    """Days since last qualifying purchase, and last-quarter qualifying baskets, by category."""
    ID = ns["ID"]
    CATS = ns["CATS"]
    slug = ns["slug"]
    t = pd.Timestamp(cutoff)
    ids = pd.Index(cust[ID])
    hist = frame[(frame._time < t) & frame[ID].isin(ids) & frame.qual]
    last = hist.groupby([ID, "ItemCategoryLevel2"])._time.max().unstack()
    last = last.reindex(index=ids, columns=CATS)
    days = (t - last).apply(lambda column: column.dt.total_seconds() / 86400.0)
    tenure = cust["tenure_days"].to_numpy(dtype=float)
    out = cust.copy()
    start = t - pd.DateOffset(months=3)
    recent = hist[(hist._time >= start) & (hist._time < t)]
    recent_n = recent.groupby([ID, "ItemCategoryLevel2"]).TransactionId.nunique().unstack()
    recent_n = recent_n.reindex(index=ids, columns=CATS).fillna(0)
    for category in CATS:
        name = slug(category)
        raw = days[category].to_numpy(dtype=float)
        missing = np.isnan(raw)
        filled = np.where(missing, tenure + 30.0, raw)
        out[f"recency_{name}"] = filled
        out[f"log_recency_{name}"] = np.log1p(filled)
        out[f"q1n_{name}"] = recent_n[category].to_numpy(dtype=np.float32)
    return out


def build(ns, with_recency: bool):
    print(f"building features recency={with_recency}", flush=True)
    raw = ns["load_transactions"](str(base.GROK / "data" / "train.csv"))
    frame = ns["prepare_frame"](raw)
    parts = []
    for cutoff in base.SNAPSHOT_WEIGHT:
        print(f"  snapshot {cutoff}", flush=True)
        cust = ns["customer_table"](frame, cutoff)
        cust = base.add_history_features(frame, cust, cutoff, ns)
        if with_recency:
            cust = add_category_recency(frame, cust, cutoff, ns)
        official = ns["transform"](ns["raw_labels"](raw, cutoff, ns["CONFIG"]), ns["CONFIG"])
        cust = cust.merge(official[["ID", "Opportunity"]], on="ID", how="left")
        parts.append(cust)
    customers = pd.concat(parts, ignore_index=True)
    customers = base.add_previous_opportunity(customers, list(ns["LABELS"]))
    X, columns = base.matrix(customers)
    history = pd.concat(
        [customers[["ID", "cutoff", "Opportunity"]].reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )
    return history, columns


def score_rule(customers, features, train_cutoffs, valid_cutoff) -> float:
    Xtr, ytr, wtr, Xva, yva = base.slice_fold(customers, features, train_cutoffs, valid_cutoff)
    print(f"  classifiers for {valid_cutoff} features={Xtr.shape[1]}", flush=True)
    models = [base.fit_classifier(Xtr, ytr, wtr, Xva, yva, seed) for seed in (7, 23)]
    proba, classes = base.average_proba(models, Xva)
    pred = base.apply_rule(proba, classes)
    return base.weighted_f1(yva, pred)


def main():
    ns = base.load_base()
    cutoffs = list(base.SNAPSHOT_WEIGHT)
    control, control_features = build(ns, with_recency=False)
    treated, treated_features = build(ns, with_recency=True)
    print(f"control features={len(control_features)} treated={len(treated_features)}", flush=True)

    rows = []
    for name, data, features in (
        ("shipped_features", control, control_features),
        ("category_recency", treated, treated_features),
    ):
        print(name, "June", flush=True)
        june = score_rule(data, features, cutoffs[:3], base.JUNE)
        print(name, "September", flush=True)
        september = score_rule(data, features, cutoffs[:4], base.SEPTEMBER)
        print(f"{name} June {june:.4f} September {september:.4f}", flush=True)
        rows.append({"model": name, "june_f1": june, "september_f1": september})

    control_row, treated_row = rows
    june_gain = treated_row["june_f1"] - control_row["june_f1"]
    september_gain = treated_row["september_f1"] - control_row["september_f1"]
    decision = {
        "paper": "Anderson, Kumar, Tomkins, Vassilvitskii (WWW 2014)",
        "mechanism": "category-level days since last qualifying purchase",
        "multipliers_retuned": False,
        "rows": rows,
        "june_gain": june_gain,
        "september_gain": september_gain,
        "published_september_f1": PUBLISHED_SEPTEMBER_F1,
        "accepted": bool(
            june_gain > 0
            and september_gain > 0
            and treated_row["september_f1"] > PUBLISHED_SEPTEMBER_F1
            and (june_gain + september_gain) / 2 >= 0.01
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    if decision["accepted"]:
        print("Gate passed. A December refit is the next step, not an upload yet.", flush=True)
    else:
        print("Gate did not pass. The 0.30225 file stays in place.", flush=True)


if __name__ == "__main__":
    main()
