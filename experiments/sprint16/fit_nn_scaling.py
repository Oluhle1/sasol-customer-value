"""E9b: neural scaling diagnostic (2 seeds, dropout 0.3) on June/September.

Proposal: outputs/sprint16/proposal_nn_scaling.json
Run with venv python (torch):
  /home/oluhle/.venvs/sasol-oct14/bin/python3 -B -u experiments/sprint16/fit_nn_scaling.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
from evaluate_saved import CLASSES, OCT, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1  # noqa: E402
from fit_multitask_nn import MultiTaskMLP, build_origin_arrays, weighted_ce  # noqa: E402

CACHE = OUT / "nn_cache_v2"
CACHE.mkdir(parents=True, exist_ok=True)
EVAL = ["2025-06-01", "2025-09-01"]
SEEDS = (7, 23)


def train_nn_v2(panel, cutoff, seed):
    path = CACHE / f"{cutoff}_nnv2_s{seed}.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"].astype(float)
    d = build_origin_arrays(panel, cutoff)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    qt.fit(d["X"][d["tr"]])
    X = qt.transform(d["X"]).astype(np.float32)
    Xte = qt.transform(frame[panel["feature_cols"]].to_numpy(dtype=np.float32)).astype(np.float32)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MultiTaskMLP(X.shape[1])
    for m in model.trunk.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.3
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X)
    t17 = torch.from_numpy(d["y17"])
    t4 = torch.from_numpy(d["y4"])
    tw = torch.from_numpy(d["w"])
    tz_f = torch.from_numpy(d["zf"])
    tz_n = torch.from_numpy(d["zn"])
    tr_m = torch.from_numpy(d["tr"])
    va_m = torch.from_numpy(d["va"])

    n = len(X)
    best_va = np.inf
    best_state = None
    patience = 0
    batch = 512
    for epoch in range(80):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            m = tr_m[idx]
            if m.sum() == 0:
                continue
            xb = Xt[idx][m]
            l17, l4, lf, ln = model(xb)
            loss = (weighted_ce(l17, t17[idx][m], tw[idx][m])
                    + 0.3 * weighted_ce(l4, t4[idx][m], tw[idx][m])
                    + 0.1 * torch.nn.functional.mse_loss(lf, tz_f[idx][m])
                    + 0.1 * torch.nn.functional.mse_loss(ln, tz_n[idx][m]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            l17, l4, _, _ = model(Xt[va_m])
            va_loss = float(torch.nn.functional.cross_entropy(l17, t17[va_m])
                            + 0.3 * torch.nn.functional.cross_entropy(l4, t4[va_m]))
        if va_loss < best_va - 1e-5:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        l17, _, _, _ = model(torch.from_numpy(Xte))
        proba = torch.softmax(l17, dim=1).numpy().astype(float)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy(), va_loss=best_va)
    print(f"  nnv2 {cutoff} seed {seed} best_va {best_va:.4f} epochs {epoch+1} rows {n}", flush=True)
    return proba


def main():
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    print("panel loaded", flush=True)
    rows = {}
    for c in EVAL:
        ids, y = load_labels(c)
        n = len(ids)
        p_cat = load_proba("category_dynamics", c, n)
        p4 = load_rank4(c, n)
        nn = np.mean([train_nn_v2(panel, c, s) for s in SEEDS], axis=0)
        vals = {}
        for name, p17 in [("catdyn", p_cat), ("nn", nn), ("nn+catdyn", (nn + p_cat) / 2)]:
            prod = rank4_combine(p17, p4, "full")
            vals[name] = weighted_f1(np.argmax(prod * vR, axis=1), y)
        rows[c] = vals
        print(c, {k: round(v, 6) for k, v in vals.items()}, flush=True)
    table = pd.DataFrame(rows).T
    print(table.round(6).to_string())
    gains = {c: {name: float(table.loc[c, name] - table.loc[c, "catdyn"]) for name in ["nn", "nn+catdyn"]} for c in EVAL}
    best_name, best_min = None, -np.inf
    for name in ["nn", "nn+catdyn"]:
        g = [gains[c][name] for c in EVAL]
        print(name, "fold gains", [round(x, 6) for x in g], flush=True)
        if min(g) >= 0.0010 and min(g) > best_min:
            best_name, best_min = name, min(g)
    out = {"table": table.round(6).to_dict(), "gains": gains, "G1b_pass": best_name is not None,
           "best_integration": best_name}
    (OUT / "nn_scaling_result.json").write_text(json.dumps(out, indent=2))
    print("G1b", "PASS with " + str(best_name) if best_name else "FAIL -> neural family closed; no probe")


if __name__ == "__main__":
    main()
