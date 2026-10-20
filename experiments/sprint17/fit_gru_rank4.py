"""Sprint 17 E11 — GRU sequence member for the rank4 head.

Pre-registration: experiments/sprint17/PROPOSAL_E11.md and
outputs/sprint17/proposal_gru_seq.json (architecture and gates frozen there).
Run with the venv python (torch CPU): /home/oluhle/.venvs/sasol-oct14/bin/python3
"""
import copy
import hashlib
import json
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
DEST = OUT17 / "seq_rank4"
DEST.mkdir(parents=True, exist_ok=True)
SEQ = OUT17 / "seq_tensors"
PROPOSAL = OUT17 / "proposal_gru_seq.json"
PROPOSAL_SHA = hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()
POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
RANK_CODE = {name: i for i, name in enumerate(RANK)}
FUEL_LABEL = "Existing-category growth: Fuel"
SEED = 7
K = 64
CH = 19


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


def load_seq(cutoff, idx):
    payload = np.load(SEQ / f"{cutoff}_seq.npz", allow_pickle=False)
    if not np.array_equal(payload["ids"].astype(str), np.asarray(idx, dtype=str)):
        raise RuntimeError(f"sequence ids do not match frame index for {cutoff}")
    return payload["tensor"]


class GRUNet(nn.Module):
    def __init__(self, in_f, hidden, n_classes):
        super().__init__()
        self.gru = nn.GRU(in_f, hidden, 1, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.drop(h[:, -1]))


def build_split(panel, cutoff):
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    xs, ys, ws, cuts = [], [], [], []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(frame.index).Opportunity.to_numpy()
        xs.append(load_seq(origin, frame.index.astype(str)))
        ys.append(np.array([RANK_CODE[t] for t in collapse(labels)], dtype=np.int64))
        ws.append(np.full(len(frame), weight_of(origin), dtype=np.float32))
        cuts.append(np.full(len(frame), origin, dtype=object))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    cut = np.concatenate(cuts)
    tr = cut <= limit
    va = cut == latest
    if not (pd.to_datetime(cut[tr]) + pd.DateOffset(months=3) <= pd.Timestamp(latest)).all():
        raise RuntimeError("gru training crosses the inner cutoff")
    predict = load_seq(cutoff, panel["frames"][cutoff].index.astype(str))
    return x, y, w, tr, va, predict


def standardizer(x_tr):
    real = x_tr[..., CH] > 0
    lines = x_tr[real][:, :CH]
    mu = lines.mean(axis=0)
    sd = lines.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_standardize(x, mu, sd):
    z = x.copy()
    z[..., :CH] = np.clip((x[..., :CH] - mu) / sd, -8.0, 8.0)
    z[..., :CH] *= x[..., CH:CH + 1]
    return torch.from_numpy(z)


def fit_gru(panel, cutoff):
    path = DEST / f"{cutoff}_rank4_gru.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"]
    t0 = time.time()
    x, y, w, tr, va, predict = build_split(panel, cutoff)
    mu, sd = standardizer(x[tr])
    x_tr = apply_standardize(x[tr], mu, sd)
    y_tr = torch.from_numpy(y[tr])
    w_tr = torch.from_numpy(w[tr])
    x_va = apply_standardize(x[va], mu, sd)
    y_va = torch.from_numpy(y[va])
    x_pred = apply_standardize(predict, mu, sd)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = GRUNet(x.shape[2], 96, len(RANK))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    batch = 256
    best_loss, best_state, patience = float("inf"), None, 6
    epoch = 0
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logits = model(x_tr[idx])
            per = F.cross_entropy(logits, y_tr[idx], reduction="none")
            loss = (per * w_tr[idx]).sum() / w_tr[idx].sum()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(x_va), y_va, reduction="mean"))
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 6
        else:
            patience -= 1
        if epoch % 5 == 0 or patience == 0:
            log(f"  gru {cutoff} epoch {epoch} val_logloss {val_loss:.4f} best {best_loss:.4f} patience {patience} ({time.time()-t0:.0f}s)")
        if patience == 0:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(x_pred), dim=1).numpy()
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy())
    log(f"gru {cutoff} best val_logloss {best_loss:.4f} epochs {epoch + 1} in {time.time()-t0:.0f}s")
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
    for cutoff, expected in proposal["baseline"]["pool_origin_f1_known"].items():
        if abs(baseline[cutoff] - expected) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}")
    log(f"baseline r4full+vR: { {c: round(b, 6) for c, b in baseline.items()} }")

    decision = {"proposal_sha256": PROPOSAL_SHA, "baseline": {c: float(b) for c, b in baseline.items()}}
    first_fit = None
    scores, gains = {}, {}
    for cutoff in POOL:
        t0 = time.time()
        gru = fit_gru(panel, cutoff)
        if first_fit is None:
            first_fit = time.time() - t0
            if first_fit > 20 * 60:
                decision["aborted"] = {"first_fit_seconds": round(first_fit, 1)}
                (OUT17 / "gru_decision.json").write_text(json.dumps(decision, indent=2))
                log(f"ABORT: March fit took {first_fit:.0f}s > 20 min")
                return
        ids, y = load_labels(cutoff)
        p4 = 0.5 * rank4_cache[cutoff] + 0.5 * gru
        prod = rank4_combine(catdyn_cache[cutoff], p4, "full")
        scores[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
        gains[cutoff] = scores[cutoff] - baseline[cutoff]
        log(f"  gru-mix {cutoff}: {scores[cutoff]:.6f}  gain {gains[cutoff]:+.6f}")
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
            gru = fit_gru(panel, cutoff)
            p4 = 0.5 * rank4_cache[cutoff] + 0.5 * gru
            prod = rank4_combine(catdyn_cache[cutoff], p4, "full")
            ids, y = load_labels(cutoff)
            fold_scores[cutoff] = weighted_f1(np.argmax(prod * vR, axis=1), y)
            fold_gains[cutoff] = fold_scores[cutoff] - baseline[cutoff]
            log(f"  gru-mix EVAL {cutoff}: {fold_scores[cutoff]:.6f}  gain {fold_gains[cutoff]:+.6f}")
        folds_ok = all(g >= folds_gate["per_origin_min"] for g in fold_gains.values()) and \
            float(np.mean(list(fold_gains.values()))) >= folds_gate["mean_gain_min"]
        decision["folds"] = {"scores": fold_scores, "gains": fold_gains, "passed": bool(folds_ok)}
        log(f"  folds pass={folds_ok}")
        if folds_ok:
            gru = fit_gru(panel, TEST)
            p4 = 0.5 * load_rank4_local(TEST, len(panel["frames"][TEST])) + 0.5 * gru
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
            path = ROOT / "submissions" / "sasol_sprint17_grumix.csv"
            out.to_csv(path, index=False)
            changes = int((out["Opportunity"].to_numpy() != source["Opportunity"].to_numpy()).sum())
            (OUT17 / "grumix_audit.json").write_text(json.dumps({
                "source": "submissions/sasol_sprint16_r4full_vR.csv",
                "file": "submissions/sasol_sprint17_grumix.csv",
                "changes": changes, "rows": int(len(out)),
            }, indent=2))
            decision["exported"] = {"file": path.name, "changes_vs_best": changes}
            log(f"exported {path.name} with {changes} label changes")
    else:
        log("screen failed; folds and December were not fit, nothing exported")

    decision["seconds"] = round(time.time() - started, 1)
    (OUT17 / "gru_decision.json").write_text(json.dumps(decision, indent=2))
    log("wrote gru_decision.json")


if __name__ == "__main__":
    main()
