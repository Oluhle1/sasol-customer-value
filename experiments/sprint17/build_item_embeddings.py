"""Sprint 17 E10 — embed every unique ItemName with all-MiniLM-L6-v2.

Run with the oct14 venv python (torch + transformers):
  /home/oluhle/.venvs/sasol-oct14/bin/python3 -u experiments/sprint17/build_item_embeddings.py

An item vector is the mean of the embeddings of each distinct ItemName ever
seen for that StockItemId (2742 of 6849 ids carry multiple names).
Outputs outputs/sprint17/item_embeddings.npz (ids, embs) plus a report JSON.
Frozen in outputs/sprint17/proposal_item_embeddings.json.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT17 = ROOT / "outputs" / "sprint17"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 64


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(6)
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "train.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["StockItemId", "ItemName"],
    )
    pairs = raw.drop_duplicates().sort_values(["StockItemId", "ItemName"]).reset_index(drop=True)
    names = pairs["ItemName"].tolist()
    log(f"unique ids {pairs['StockItemId'].nunique()} unique name rows {len(names)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    dim = int(model.config.hidden_size)
    name_embs = np.zeros((len(names), dim), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(names), BATCH):
            batch = names[i : i + BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=32, return_tensors="pt")
            out = model(**enc)
            h = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1)
            name_embs[i : i + len(batch)] = e.numpy()
            if i % (BATCH * 40) == 0 or i + BATCH >= len(names):
                log(f"{min(i + BATCH, len(names))}/{len(names)} {time.time() - t0:.0f}s")

    frame = pd.DataFrame(name_embs)
    frame["item"] = pairs["StockItemId"].to_numpy()
    per_id = frame.groupby("item", sort=True).mean()
    ids = np.asarray(per_id.index.tolist(), dtype=str)
    embs = per_id.to_numpy(dtype=np.float32)
    np.savez_compressed(OUT17 / "item_embeddings.npz", ids=ids, embs=embs)
    digest = hashlib.sha256((OUT17 / "item_embeddings.npz").read_bytes()).hexdigest()
    report = {
        "model": MODEL,
        "pooling": "mean over tokens with attention mask, unnormalized; per-id mean over distinct names",
        "n_items": int(len(ids)),
        "n_name_rows": int(len(names)),
        "dim": dim,
        "seconds": round(time.time() - t0, 1),
        "npz_sha256": digest,
    }
    (OUT17 / "item_embeddings_report.json").write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote item_embeddings.npz {embs.shape} sha256 {digest[:16]}")


if __name__ == "__main__":
    main()
