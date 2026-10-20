"""E9: multi-task MLP as a neural diversity member (pre-registered).

Proposal: outputs/sprint16/proposal_multitask_nn.json
Stages:
  screen : fit Apr/May/Jul/Aug, evaluate NN p17 variants in the r4full+vR path
  folds  : (only if G1 passes) fit June/Sept, G2 check
  december: (only if G2 passes) fit December, export candidate + audit
Caches NN probabilities per origin under outputs/sprint16/nn_cache/.
Run with the venv python that has torch:
  /home/oluhle/.venvs/sasol-oct14/bin/python3 -B -u experiments/sprint16/fit_multitask_nn.py --stage screen
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
from evaluate_saved import CLASSES, OCT, TOP10, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1  # noqa: E402

CACHE = OUT / "nn_cache"
CACHE.mkdir(parents=True, exist_ok=True)

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"

RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"
SEEDS = (7,)

BASELINE_POOL = 0.503968  # catdyn r4full + vR pooled mean (recorded in pool_r4full_vector.json)


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def build_origin_arrays(panel, cutoff):
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    feats = panel["feature_cols"]
    xs, ys, ws, rs = [], [], [], []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(frame.index)
        xs.append(frame[feats].to_numpy(dtype=np.float32))
        ys.append(np.array([CLASSES.index(c) for c in labels.Opportunity]))
        rs.append(np.array([RANK.index(c) for c in collapse(labels.Opportunity.to_numpy())]))
        ws.append(np.full(len(frame), weight_of(origin), dtype=np.float32))
    cut = np.concatenate([np.full(len(x), o, dtype=object) for x, o in
                          zip(xs, [c for c in panel["monthly_cutoffs"] if c <= latest])])
    X = np.concatenate(xs)
    y17 = np.concatenate(ys).astype(np.int64)
    y4 = np.concatenate(rs).astype(np.int64)
    w = np.concatenate(ws)
    tr = cut <= limit
    va = cut == latest
    # auxiliary regression targets (transformed z), standardized on train rows
    zf, zn = [], []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        labels = panel["labels"][origin].reindex(panel["frames"][origin].index)
        zf.append(labels.CLV_fuel.to_numpy(dtype=np.float32))
        zn.append(labels.CLV_nonfuel.to_numpy(dtype=np.float32))
    zf = np.concatenate(zf)
    zn = np.concatenate(zn)
    mu_f, sd_f = zf[tr].mean(), zf[tr].std()
    mu_n, sd_n = zn[tr].mean(), zn[tr].std()
    zf = ((zf - mu_f) / sd_f).astype(np.float32)
    zn = ((zn - mu_n) / sd_n).astype(np.float32)
    return dict(X=X, y17=y17, y4=y4, zf=zf, zn=zn, w=w, tr=tr, va=va)


class MultiTaskMLP(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_features, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        )
        self.head17 = torch.nn.Linear(256, 17)
        self.head4 = torch.nn.Linear(256, 4)
        self.head_f = torch.nn.Linear(256, 1)
        self.head_n = torch.nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head17(h), self.head4(h), self.head_f(h).squeeze(-1), self.head_n(h).squeeze(-1)


def weighted_ce(logits, target, w):
    loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    return (loss * w).sum() / w.sum()


def train_nn(panel, cutoff, seed):
    path = CACHE / f"{cutoff}_nn_s{seed}.npz"
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
    for epoch in range(60):
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
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        l17, _, _, _ = model(torch.from_numpy(Xte))
        proba = torch.softmax(l17, dim=1).numpy().astype(float)
    np.savez(path, proba=proba, ids=frame.index.astype(str).to_numpy(), va_loss=best_va)
    print(f"  nn {cutoff} seed {seed} best_va {best_va:.4f} epochs_done {epoch+1} rows {n}", flush=True)
    return proba


def evaluate_origin(cutoff, panel, seed):
    ids, y = load_labels(cutoff)
    n = len(ids)
    p_cat = load_proba("category_dynamics", cutoff, n)
    p4 = load_rank4(cutoff, n)
    nn = train_nn(panel, cutoff, seed)
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    out = {}
    for name, p17 in [("catdyn", p_cat), ("nn", nn), ("nn+catdyn", (nn + p_cat) / 2)]:
        prod = rank4_combine(p17, p4, "full")
        out[name] = weighted_f1(np.argmax(prod * vR, axis=1), y)
    print(cutoff, {k: round(v, 6) for k, v in out.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screen", choices=["screen", "folds", "december"])
    args = ap.parse_args()
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    print(f"panel loaded; stage {args.stage}", flush=True)

    if args.stage == "screen":
        rows = {}
        for c in POOL:
            rows[c] = evaluate_origin(c, panel, SEEDS[0])
        table = pd.DataFrame(rows).T
        table["mean"] = table.mean(axis=1)
        print(table.round(6).to_string())
        gains = {name: {c: table.loc[c, name] - table.loc[c, "catdyn"] for c in POOL}
                 for name in ["nn", "nn+catdyn"]}
        result = {"table": table.round(6).to_dict(), "gains": {k: {c: float(v) for c, v in g.items()} for k, g in gains.items()},
                  "baseline_ref": BASELINE_POOL}
        best_name, best_g = None, None
        for name, g in gains.items():
            mean_g = float(np.mean(list(g.values())))
            min_g = float(min(g.values()))
            print(name, "mean", round(mean_g, 6), "min", round(min_g, 6), flush=True)
            if mean_g >= 0.0015 and min_g >= -0.0010:
                if best_g is None or mean_g > best_g:
                    best_name, best_g = name, mean_g
        result["G1_pass"] = best_name is not None
        result["best_integration"] = best_name
        (OUT / "nn_screen_result.json").write_text(json.dumps(result, indent=2))
        if best_name is None:
            print("G1 FAIL -> neural family rejected for this round; no folds, no probe.")
        else:
            print(f"G1 PASS with {best_name}; run --stage folds")
    elif args.stage == "folds":
        res = json.loads((OUT / "nn_screen_result.json").read_text())
        assert res.get("G1_pass"), "G1 did not pass; do not run folds"
        name = res["best_integration"]
        rows = {}
        for c in EVAL:
            rows[c] = evaluate_origin(c, panel, SEEDS[0])
        table = pd.DataFrame(rows).T
        print(table.round(6).to_string())
        gains = {c: table.loc[c, name] - table.loc[c, "catdyn"] for c in EVAL}
        print("fold gains vs catdyn:", {c: round(float(v), 6) for c, v in gains.items()}, flush=True)
        g2 = all(float(v) >= 0.0010 for v in gains.values())
        out = {"integration": name, "folds": {c: {k: float(table.loc[c, k]) for k in table.columns} for c in EVAL},
               "gains": {c: float(v) for c, v in gains.items()}, "G2_pass": bool(g2)}
        (OUT / "nn_folds_result.json").write_text(json.dumps(out, indent=2))
        print("G2", "PASS -> run --stage december" if g2 else "FAIL -> no probe")
    else:
        res = json.loads((OUT / "nn_folds_result.json").read_text())
        assert res.get("G2_pass"), "G2 did not pass; do not build December candidate"
        name = res["integration"]
        nn = train_nn(panel, TEST, SEEDS[0])
        ids, _ = load_labels("2025-09-01")  # placeholder for panel access
        frame = panel["frames"][TEST]
        ids_t = frame.index.astype(str).to_numpy()
        p_cat = np.load(OCT / f"{TEST}_category_dynamics.npy").astype(float)
        r4 = np.load(TOP10 / "rank4" / f"{TEST}_rank4.npz", allow_pickle=True)
        assert np.array_equal(r4["ids"].astype(str), ids_t)
        p4 = r4["proba"].astype(float)
        p17 = nn if name == "nn" else (nn + p_cat) / 2
        vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
        prod = rank4_combine(p17, p4, "full")
        pred = np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)]
        baseline = pd.read_csv(OUT / "baseline_0309411451.csv", dtype={"ID": str})
        current = baseline.Opportunity.to_numpy()
        out = baseline.copy()
        out["Opportunity"] = pred
        path = OUT / f"candidate_nn_{name.replace('+','_')}_2025-12-01.csv"
        out.to_csv(path, index=False)
        best_cand = pd.read_csv(OUT / "candidate_r4full_vR_2025-12-01.csv", dtype={"ID": str}).Opportunity.to_numpy()
        print(f"wrote {path}; changes vs baseline {(pred != current).sum()}, vs r4full_vR {(pred != best_cand).sum()}", flush=True)


if __name__ == "__main__":
    main()
