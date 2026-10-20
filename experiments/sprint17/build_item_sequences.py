"""Sprint 17 E11 — build per-cutoff GRU sequence tensors (last K=64 lines).

Frozen in outputs/sprint17/proposal_gru_seq.json. Outputs one npz per cutoff
under outputs/sprint17/seq_tensors/ plus a report. Runs with system python3.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
OUT17 = ROOT / "outputs" / "sprint17"
OCT = ROOT / "outputs" / "oct10"
DEST = OUT17 / "seq_tensors"
DEST.mkdir(parents=True, exist_ok=True)
REPORT = OUT17 / "seq_tensors_report.json"

K = 64
DIM = 20


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
        "amount": pd.to_numeric(raw["TotalExclAmount"], errors="coerce").to_numpy(dtype=float),
    })
    d = d[~raw["ItemName"].isin(config["adjustment_items"]).to_numpy()].reset_index(drop=True)
    d = d[d["amount"] > 0].reset_index(drop=True)
    d["is_fuel"] = ((d["l2"] == "Fuel") & (d["type"] == "Sale_Fuel")).astype(np.float32)
    return d


def build_tensor(d, cutoff, idx, item_pc, item_pos):
    t = pd.Timestamp(cutoff)
    sub = d[d["time"] < t].sort_values(["ID", "time"], kind="stable")
    sub = sub.groupby("ID", sort=False).tail(K)
    n = len(idx)
    tensor = np.zeros((n, K, DIM), dtype=np.float32)
    if len(sub):
        pos = idx.get_indexer(sub["ID"].to_numpy())
        rows = item_pos.get_indexer(sub["item"].to_numpy())
        ok = (pos >= 0) & (rows >= 0)
        sub, pos, rows = sub[ok], pos[ok], rows[ok]
        cum = sub.groupby("ID", sort=False).cumcount().to_numpy()
        cnt = sub.groupby("ID", sort=False)["ID"].transform("size").to_numpy()
        slot = K - cnt + cum
        tensor[pos, slot, :16] = item_pc[rows]
        tensor[pos, slot, 16] = np.log1p(sub["amount"].to_numpy(dtype=np.float32))
        days = (t - sub["time"]).dt.total_seconds().to_numpy() / 86400.0
        tensor[pos, slot, 17] = np.log1p(days)
        tensor[pos, slot, 18] = sub["is_fuel"].to_numpy(dtype=np.float32)
        tensor[pos, slot, 19] = 1.0
    return tensor


def main():
    started = time.time()
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    payload = np.load(OUT17 / "item_embeddings.npz", allow_pickle=False)
    ids, embs = payload["ids"].astype(str), payload["embs"].astype(np.float64)
    item_pos = pd.Index(ids)
    pca = PCA(n_components=16)
    pca.fit(embs)
    item_pc = ((embs - pca.mean_) @ pca.components_.T).astype(np.float32)
    d = load_lines()
    log(f"lines {len(d)}, items {len(ids)}, cutoffs {len(panel['frames'])}")

    report = {"_pca_explained_variance_ratio_sum16": float(pca.explained_variance_ratio_.sum())}
    for cutoff, frame in panel["frames"].items():
        t0 = time.time()
        idx = frame.index.astype(str)
        tensor = build_tensor(d, cutoff, idx, item_pc, item_pos)
        np.savez_compressed(DEST / f"{cutoff}_seq.npz", tensor=tensor, ids=np.asarray(idx.tolist(), dtype=str))
        filled = float((tensor[:, :, 19] > 0).mean())
        report[cutoff] = {"rows": int(len(tensor)), "filled_share": filled, "seconds": round(time.time() - t0, 1)}
        log(f"{cutoff}: rows {len(tensor)}, filled {filled:.3f} in {report[cutoff]['seconds']}s")

    probe = "2025-04-01"
    idx = panel["frames"][probe].index.astype(str)
    mask = d["time"] < pd.Timestamp(probe)
    a = build_tensor(d, probe, idx, item_pc, item_pos)
    b = build_tensor(d[mask].reset_index(drop=True), probe, idx, item_pc, item_pos)
    if not np.array_equal(a, b):
        diff = float(np.abs(a - b).max())
        raise RuntimeError(f"future leakage detected: max diff {diff}")
    log(f"leakage check passed on {probe}")

    report["_leakage_check"] = {"cutoff": probe, "max_diff": 0.0}
    report["_seconds"] = round(time.time() - started, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"wrote seq_tensors/*.npz and {REPORT.name} in {report['_seconds']}s")


if __name__ == "__main__":
    main()
