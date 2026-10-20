"""Segment bias of the accepted fuel blend, and April-May blends checked later.

The April-May weight search is a candidate only if later months improve.
This script records the search and the later-month results. It does not
write a submission.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "oct10"
DEST = ROOT / "outputs" / "top10_round"
panel = pickle.loads((OUT / "category_dynamics_panel.pkl").read_bytes())
BINS = [0, 7, 30, 90, 10000]
LABELS = ["0-7", "7-30", "30-90", "90+"]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fuel_blend(cutoff):
    d = np.load(OUT / f"{cutoff}_CLV_fuel_category_values.npy")
    h = np.load(OUT / f"{cutoff}_CLV_fuel_two_part.npy")
    r = np.load(OUT / f"{cutoff}_fuel_residual.npy")
    return 0.40 * d + 0.35 * h + 0.25 * r


def shop_parts(cutoff):
    d = np.load(OUT / f"{cutoff}_CLV_nonfuel_category_values.npy")
    h = np.load(OUT / f"{cutoff}_CLV_nonfuel_two_part.npy")
    return d, h


def bucket(frame):
    return pd.cut(frame["fuel_recency"].to_numpy(), BINS, labels=LABELS, right=True).astype(str)


def segment_report():
    rows = []
    for cutoff in ["2025-06-01", "2025-09-01"]:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_fuel.to_numpy()
        p = fuel_blend(cutoff)
        b = bucket(frame)
        yago = frame["fuel_z_yago"].to_numpy()
        q1 = frame["fuel_z_q1"].to_numpy()
        for name in LABELS:
            m = b == name
            rows.append({
                "cutoff": cutoff,
                "bucket": name,
                "n": int(m.sum()),
                "rmse": rmse(y[m], p[m]),
                "bias": float((p[m] - y[m]).mean()),
                "corr": float(np.corrcoef(p[m], y[m])[0, 1]),
                "yago_rmse": rmse(y[m], np.maximum(yago[m], 0)),
                "q1_rmse": rmse(y[m], np.maximum(q1[m], 0)),
                "positive_rate": float((y[m] > 0).mean()),
            })
    return rows


def search_fuel_anchor():
    """Weights for blend, year-ago and last quarter, fit on April-May only."""
    grid = [i / 10 for i in range(11)]
    packs = {}
    for cutoff in ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_fuel.to_numpy()
        packs[cutoff] = (y, fuel_blend(cutoff), np.maximum(frame.fuel_z_yago.to_numpy(), 0), np.maximum(frame.fuel_z_q1.to_numpy(), 0))
    best, best_w = 1e9, (1, 0, 0)
    y = np.concatenate([packs[c][0] for c in ["2025-04-01", "2025-05-01"]])
    a = np.concatenate([packs[c][1] for c in ["2025-04-01", "2025-05-01"]])
    b = np.concatenate([packs[c][2] for c in ["2025-04-01", "2025-05-01"]])
    c = np.concatenate([packs[c][3] for c in ["2025-04-01", "2025-05-01"]])
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1:
                continue
            w0 = 1 - w1 - w2
            err = rmse(y, w0 * a + w1 * b + w2 * c)
            if err < best:
                best, best_w = err, (w0, w1, w2)
    rows = []
    for cutoff, (y, a, b, c) in packs.items():
        if cutoff < "2025-06-01":
            continue
        pred = best_w[0] * a + best_w[1] * b + best_w[2] * c
        rows.append({"cutoff": cutoff, "current": rmse(y, a), "anchored": rmse(y, pred), "gain": rmse(y, a) - rmse(y, pred)})
    return {"weights_april_may": best_w, "april_may_rmse": best, "later": rows}


def search_shop():
    grid = [i / 10 for i in range(11)]
    packs = {}
    for cutoff in ["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_nonfuel.to_numpy()
        d, h = shop_parts(cutoff)
        packs[cutoff] = (y, d, h)
    y = np.concatenate([packs[c][0] for c in ["2025-04-01", "2025-05-01"]])
    d = np.concatenate([packs[c][1] for c in ["2025-04-01", "2025-05-01"]])
    h = np.concatenate([packs[c][2] for c in ["2025-04-01", "2025-05-01"]])
    best, best_w = 1e9, 0.5
    for w in grid:
        err = rmse(y, w * d + (1 - w) * h)
        if err < best:
            best, best_w = err, w
    rows = []
    for cutoff, (y, d, h) in packs.items():
        if cutoff < "2025-06-01":
            continue
        current = 0.5 * d + 0.5 * h
        pred = best_w * d + (1 - best_w) * h
        rows.append({"cutoff": cutoff, "current": rmse(y, current), "retuned": rmse(y, pred), "gain": rmse(y, current) - rmse(y, pred), "direct_weight": best_w})
    return {"direct_weight_april_may": best_w, "later": rows}


def main():
    report = {"segments": segment_report(), "fuel_anchor": search_fuel_anchor(), "shop_weight": search_shop()}
    (DEST / "segment_blend.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
