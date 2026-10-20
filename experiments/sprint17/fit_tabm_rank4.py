"""Sprint 17 E9 — TabM-style batch-ensemble MLP member for the rank4 head.

Pre-registration: experiments/sprint17/PROPOSAL_E9.md and
outputs/sprint17/proposal_tabm.json (architecture and gates frozen there).
Run with the venv python (torch CPU): /home/oluhle/.venvs/sasol-oct14/bin/python3
"""
import copy
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))

from evaluate_saved import (  # noqa: E402
    CLASSES, OCT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1,
)

OUT17 = ROOT / "outputs" / "sprint17"
DEST = OUT17 / "tabm_rank4"
DEST.mkdir(parents=True, exist_ok=True)
PROPOSAL = OUT17 / "proposal_tabm.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
RANK_CODE = {name: i for i, name in enumerate(RANK)}
FUEL_LABEL = "Existing-category growth: Fuel"
K = 8
SEED = 7


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


class BatchEnsembleLinear(nn.Module):
    def __init__(self, in_f, out_f, k):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_f)
        nn.init.uniform_(self.bias, -bound, bound)
        self.s = nn.Parameter(torch.ones(k, out_f) + 0.05 * torch.randn(k, out_f))
        self.r = nn.Parameter(torch.ones(k, in_f) + 0.05 * torch.randn(k, in_f))

    def forward(self, x):
        w = self.weight.unsqueeze(0) * self.s.unsqueeze(-1) * self.r.unsqueeze(1)
        b = self.bias.unsqueeze(0) * self.s
        return torch.einsum("bki,koi->bko", x, w) + b


class TabM(nn.Module):
    def __init__(self, in_f, hidden, k, n_classes):
        super().__init__()
        self.k = k
        self.l1 = BatchEnsembleLinear(in_f, hidden, k)
        self.l2 = BatchEnsembleLinear(hidden, hidden, k)
        self.l3 = BatchEnsembleLinear(hidden, n_classes, k)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.k, 1)
        x = F.relu(self.l1(x))
        x = self.drop(F.relu(self.l2(x)))
        return self.l3(x)


def build_split(panel, cutoff):
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    cols = panel["feature_cols"]
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(frame.index).Opportunity.to_numpy()
        block = frame[cols].copy()
        block["_y"] = [RANK_CODE[t] for t in collapse(labels)]
        block["_w"] = weight_of(origin)
        block["_cutoff"] = origin
        blocks.append(block)
    train = pd.concat(blocks)
    tr = train._cutoff.le(limit).to_numpy()
    va = train._cutoff.eq(latest).to_numpy()
    if not (pd.to_datetime(train.loc[tr, "_cutoff"]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("tabm training crosses the inner cutoff")
    predict = panel["frames"][cutoff][cols]
    return train, tr, va, predict, cols


def fit_tabm(panel, cutoff):
    path = DEST / f"{cutoff}_rank4_tabm.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"]
    t0 = time.time()
    train, tr, va, predict, cols = build_split(panel, cutoff)
    mu = train.loc[tr, cols].to_numpy(dtype=np.float32).mean(axis=0)
    sd = train.loc[tr, cols].to_numpy(dtype=np.float32).std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    def standardize(matrix):
        z = np.clip((matrix.to_numpy(dtype=np.float32) - mu) / sd, -8.0, 8.0)
        return torch.from_numpy(z)

    x_tr = standardize(train.loc[tr, cols])
    y_tr = torch.from_numpy(train.loc[tr, "_y"].to_numpy(dtype=np.int64))
    w_tr = torch.from_numpy(train.loc[tr, "_w"].to_numpy(dtype=np.float32))
    x_va = standardize(train.loc[va, cols])
    y_va = torch.from_numpy(train.loc[va, "_y"].to_numpy(dtype=np.int64))
    x_all = standardize(train[cols])
    y_all = torch.from_numpy(train["_y"].to_numpy(dtype=np.int64))
    w_all = torch.from_numpy(train["_w"].to_numpy(dtype=np.float32))
    x_pred = standardize(predict)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = TabM(len(cols), 128, K, len(RANK))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def batch_loss(xb, yb, wb):
        logits = model(xb)
        per = F.cross_entropy(logits.reshape(-1, len(RANK)), yb.repeat(K), reduction="none").view(len(yb), K)
        return (per.mean(dim=1) * wb).sum() / wb.sum()

    def predict_proba(x):
        model.eval()
        with torch.no_grad():
            logits = model(x).mean(dim=1)
            return torch.softmax(logits, dim=1)

    batch = 1024
    best_loss, best_state, patience = float("inf"), None, 8
    for epoch in range(50):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = batch_loss(x_tr[idx], y_tr[idx], w_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_va).mean(dim=1)
            val_loss = float(F.cross_entropy(val_logits, y_va, reduction="mean"))
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 8
        else:
            patience -= 1
        if epoch % 5 == 0 or patience == 0:
            log(f"  tabm {cutoff} epoch {epoch} val_logloss {val_loss:.4f} best {best_loss:.4f} patience {patience} ({time.time()-t0:.0f}s)")
        if patience == 0:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = predict_proba(x_pred).numpy()
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    log(f"tabm {cutoff} best val_logloss {best_loss:.4f} in {time.time()-t0:.0f}s train {len(train)}")
    return proba


def load_rank4_local(cutoff, n):
    p = load_rank4(cutoff, n)
    if p is not None:
        return p
    path = OUT17 / "price_rank4" / f"{cutoff}_rank4_base.npz"
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if len(payload["proba"]) == n:
            return payload["proba"]
    raise RuntimeError(f"rank4 cache missing for {cutoff}")


def main():
    started = time.time()
    proposal = json.loads(PROPOSAL.read_text())
    screen, folds_gate = proposal["screen_gate"], proposal["eval_gate"]
    vR = np.array(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    log(f"panel loaded, torch {torch.__version__}, threads {torch.get_num_threads()}")

    baseline, catdyn_cache, rank4_cache = {}, {}, {}
    for cutoff in POOL + EVAL:
        ids, y = load_labels(cutoff)
        n = len(ids)
        catdyn_cache[cutoff] = load_proba("category_dynamics", cutoff, n)
        rank4_cache[cutoff] = load_rank4_local(cutoff, n)
        prod = rank4_combine(catdyn_cache[cutoff], rank4_cache[cutoff], "full")
        baseline[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
    for cutoff, expected in {"2025-04-01": 0.5006451168766207, "2025-05-01": 0.5060866657571113,
                             "2025-07-01": 0.5031954596858046, "2025-08-01": 0.5059436172953636}.items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}")
    log(f"baseline r4full+vR: { {c: round(b, 6) for c, b in baseline.items()} }")

    decision = {"proposal_sha256": PROPOSAL_SHA, "baseline": {c: float(b) for c, b in baseline.items()}}
    first_fit = None
    scores, gains = {}, {}
    for cutoff in POOL:
        t0 = time.time()
        tabm = fit_tabm(panel, cutoff)
        if first_fit is None:
            first_fit = time.time() - t0
            if first_fit > 15 * 60:
                decision["aborted"] = {"first_fit_seconds": round(first_fit, 1)}
                (OUT17 / "tabm_decision.json").write_text(json.dumps(decision, indent=2))
                log(f"ABORT: March fit took {first_fit:.0f}s > 15 min")
                return
        ids, y = load_labels(cutoff)
        p4 = 0.5 * rank4_cache[cutoff] + 0.5 * tabm
        prod = rank4_combine(catdyn_cache[cutoff], p4, "full")
        scores[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
        gains[cutoff] = scores[cutoff] - baseline[cutoff]
        log(f"  tabm-mix {cutoff}: {scores[cutoff]:.6f}  gain {gains[cutoff]:+.6f}")
    mean_gain = float(np.mean(list(gains.values())))
    min_gain = float(np.min(list(gains.values())))
    passed = mean_gain >= screen["mean_gain_min"] and min_gain >= screen["per_origin_min"]
    decision["screen"] = {"scores": scores, "gains": gains, "mean_gain": mean_gain,
                          "min_gain": min_gain, "passed": bool(passed),
                          "first_fit_seconds": round(first_fit, 1)}
    log(f"  screen mean {mean_gain:+.6f} min {min_gain:+.6f} pass={passed}")

    if passed:
        fold_scores, fold_gains = {}, {}
        for cutoff in EVAL:
            tabm = fit_tabm(panel, cutoff)
            p4 = 0.5 * rank4_cache[cutoff] + 0.5 * tabm
            prod = rank4_combine(catdyn_cache[cutoff], p4, "full")
            ids, y = load_labels(cutoff)
            fold_scores[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
            fold_gains[cutoff] = fold_scores[cutoff] - baseline[cutoff]
            log(f"  tabm-mix EVAL {cutoff}: {fold_scores[cutoff]:.6f}  gain {fold_gains[cutoff]:+.6f}")
        folds_ok = all(g >= folds_gate["per_origin_min"] for g in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        decision["folds"] = {"scores": fold_scores, "gains": fold_gains, "passed": bool(folds_ok)}
        log(f"  folds pass={folds_ok}")
        if folds_ok:
            tabm = fit_tabm(panel, TEST)
            p4 = 0.5 * load_rank4_local(TEST, len(panel["frames"][TEST])) + 0.5 * tabm
            prod = rank4_combine(load_proba("category_dynamics", TEST, len(panel["frames"][TEST])), p4, "full")
            pred = np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)]
            frame = panel["frames"][TEST]
            source = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
            if list(source["ID"].astype(str)) != list(frame.index.astype(str)):
                raise RuntimeError("December order does not match the source submission")
            out = source.copy()
            out["Opportunity"] = pred
            for col in ("CLV_fuel", "CLV_nonfuel"):
                if not np.array_equal(out[col].to_numpy(), source[col].to_numpy()):
                    raise RuntimeError(f"{col} changed during export")
            path = ROOT / "submissions" / "sasol_sprint17_tabmmix.csv"
            out.to_csv(path, index=False)
            changes = int((out["Opportunity"].to_numpy() != source["Opportunity"].to_numpy()).sum())
            (OUT17 / "tabmmix_audit.json").write_text(json.dumps({
                "source": "submissions/sasol_sprint16_r4full_vR.csv",
                "file": "submissions/sasol_sprint17_tabmmix.csv",
                "changes": changes, "rows": int(len(out)),
            }, indent=2))
            decision["exported"] = {"file": path.name, "changes_vs_best": changes}
            log(f"exported {path.name} with {changes} label changes")
    else:
        log("screen failed; folds and December were not fit, nothing exported")

    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "tabm_decision.json").write_text(json.dumps(decision, indent=2))
    log("wrote tabm_decision.json")


if __name__ == "__main__":
    main()
