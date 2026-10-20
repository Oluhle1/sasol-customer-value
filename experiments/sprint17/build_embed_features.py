"""Sprint 17 E10 — build the frozen 27-column item-embedding block.

Reads data/raw/train.csv lines, outputs/sprint17/item_embeddings.npz and the
panel frames. Lines: adjustment items excluded, TotalExclAmount > 0, strictly
before each cutoff. Feature list is frozen in
outputs/sprint17/proposal_item_embeddings.json; do not edit it here.
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
DEST = OUT17 / "embed_features.pkl"
REPORT = OUT17 / "embed_features_report.json"

DIM = 384
FEATURES = (
    [f"emb_pc{i:02d}" for i in range(16)]
    + [f"emb_recent_pc{i:02d}" for i in range(8)]
    + ["emb_shift_cos", "emb_recent_outlier", "emb_norm"]
)


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def load_lines():
    config = json.loads((ROOT / "configs" / "label_config.json").read_text())
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv", dtype=str, keep_default_na=False)
    d = pd.DataFrame({
        "ID": raw["ID"].to_numpy(),
        "time": pd.to_datetime(raw["DateTimeZA"], format="mixed", errors="raise"),
        "item": raw["StockItemId"].to_numpy(),
        "amount": pd.to_numeric(raw["TotalExclAmount"], errors="coerce").to_numpy(dtype=float),
    })
    d = d[~raw["ItemName"].isin(config["adjustment_items"]).to_numpy()].reset_index(drop=True)
    d = d[d["amount"] > 0].reset_index(drop=True)
    return d


def window_sums(frame, idx, embs, item_pos):
    n = len(idx)
    sums = np.zeros((n, DIM))
    wsum = np.zeros(n)
    if len(frame):
        pos = idx.get_indexer(frame["ID"])
        rows = item_pos.get_indexer(frame["item"])
        ok = (pos >= 0) & (rows >= 0)
        pos, rows = pos[ok], rows[ok]
        w = frame["amount"].to_numpy(dtype=float)[ok]
        for j in range(DIM):
            col = embs[:, j]
            sums[:, j] = np.bincount(pos, weights=w * col[rows], minlength=n)
        wsum = np.bincount(pos, weights=w, minlength=n)
    return sums, wsum


def compute_cutoff(d, cutoff, idx, embs, comp, comp8, pca_mean, item_pos):
    t = pd.Timestamp(cutoff)
    all_frame = d[d["time"] < t]
    rec_frame = d[(d["time"] >= t - pd.Timedelta(days=90)) & (d["time"] < t)]
    s_all, w_all = window_sums(all_frame, idx, embs, item_pos)
    s_rec, w_rec = window_sums(rec_frame, idx, embs, item_pos)

    mean_all = np.full((len(idx), DIM), np.nan)
    np.divide(s_all, w_all[:, None], out=mean_all, where=w_all[:, None] > 0)
    mean_rec = np.full((len(idx), DIM), np.nan)
    np.divide(s_rec, w_rec[:, None], out=mean_rec, where=w_rec[:, None] > 0)

    out = pd.DataFrame(index=idx)
    pc_all = (mean_all - pca_mean) @ comp.T
    pc_rec = (mean_rec - pca_mean) @ comp8.T
    for i in range(16):
        out[f"emb_pc{i:02d}"] = pc_all[:, i]
    for i in range(8):
        out[f"emb_recent_pc{i:02d}"] = pc_rec[:, i]

    dot = np.einsum("ij,ij->i", mean_all, mean_rec)
    na, nr = np.linalg.norm(mean_all, axis=1), np.linalg.norm(mean_rec, axis=1)
    cos = np.full(len(idx), np.nan)
    np.divide(dot, na * nr, out=cos, where=(na > 0) & (nr > 0))
    out["emb_shift_cos"] = 1.0 - cos
    out["emb_norm"] = np.where(na > 0, na, np.nan)

    outlier = np.full(len(idx), np.nan)
    if len(rec_frame):
        pos = idx.get_indexer(rec_frame["ID"])
        rows = item_pos.get_indexer(rec_frame["item"])
        ok = (pos >= 0) & (rows >= 0)
        pos, rows = pos[ok], rows[ok]
        e_cust = mean_all[pos]
        e_item = embs[rows]
        dot_l = np.einsum("ij,ij->i", e_cust, e_item)
        nl = np.linalg.norm(e_cust, axis=1) * np.linalg.norm(e_item, axis=1)
        good = np.isfinite(dot_l) & np.isfinite(nl) & (nl > 0)
        line_val = 1.0 - dot_l[good] / nl[good]
        tot = np.bincount(pos[good], weights=line_val, minlength=len(idx))
        cnt = np.bincount(pos[good], minlength=len(idx))
        np.divide(tot, cnt, out=outlier, where=cnt > 0)
    out["emb_recent_outlier"] = outlier
    return out[FEATURES]


def main():
    started = time.time()
    proposal = json.loads((OUT17 / "proposal_item_embeddings.json").read_text())
    frozen = proposal["feature_block"]["feature_names"]
    if sorted(frozen) != sorted(FEATURES):
        raise RuntimeError("feature list diverged from the frozen proposal")
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    payload = np.load(OUT17 / "item_embeddings.npz", allow_pickle=False)
    ids, embs = payload["ids"].astype(str), payload["embs"].astype(np.float64)
    item_pos = pd.Index(ids)
    pca = PCA(n_components=16)
    pca.fit(embs)
    comp, comp8, pca_mean = pca.components_, pca.components_[:8], pca.mean_
    d = load_lines()
    log(f"lines {len(d)}, items {len(ids)}, cutoffs {len(panel['frames'])}")

    tables = {}
    report = {}
    for cutoff, frame in panel["frames"].items():
        t0 = time.time()
        idx = frame.index.astype(str)
        table = compute_cutoff(d, cutoff, idx, embs, comp, comp8, pca_mean, item_pos)
        tables[cutoff] = table
        report[cutoff] = {
            "rows": int(len(table)),
            "seconds": round(time.time() - t0, 1),
            "non_null_share": {c: float(table[c].notna().mean()) for c in FEATURES},
        }
        log(f"{cutoff}: rows {len(table)}, non-null min {min(report[cutoff]['non_null_share'].values()):.3f} in {report[cutoff]['seconds']}s")

    probe = "2025-04-01"
    idx = tables[probe].index
    mask = d["time"] < pd.Timestamp(probe)
    a = compute_cutoff(d, probe, idx, embs, comp, comp8, pca_mean, item_pos)
    b = compute_cutoff(d[mask].reset_index(drop=True), probe, idx, embs, comp, comp8, pca_mean, item_pos)
    if not np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, atol=1e-12):
        diff = np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))
        raise RuntimeError(f"future leakage detected: max diff {diff}")
    log(f"leakage check passed on {probe}")

    DEST.write_bytes(pickle.dumps(tables))
    report["_leakage_check"] = {"cutoff": probe, "max_diff": 0.0}
    report["_features"] = FEATURES
    report["_pca_explained_variance_ratio_sum16"] = float(pca.explained_variance_ratio_.sum())
    report["_seconds"] = round(time.time() - started, 1)
    REPORT.write_text(json.dumps(report, indent=2))
    log(f"wrote {DEST.name} and {REPORT.name} in {report['_seconds']}s")


if __name__ == "__main__":
    main()
