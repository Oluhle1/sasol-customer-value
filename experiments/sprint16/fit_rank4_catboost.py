"""CatBoost rank4 (4-class) as a diversity member averaged with the LGBM p4.

Same blocks/weights/features as the production rank4. Validated on the pool
origins with the r4full product + vR; folds reported only if the pool gain is
balanced. If accepted, a December fit + probe export follows separately.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "oct10"))
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))
import feature_pipeline as fp  # noqa: E402

from evaluate_saved import OCT, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1  # noqa: E402

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
DEST = OUT / "rank4_catboost"
DEST.mkdir(parents=True, exist_ok=True)

RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"

PARAMS = dict(
    loss_function="MultiClass", iterations=900, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, random_seed=0, verbose=0, thread_count=1,
    allow_writing_files=False, bootstrap_type="Bernoulli", subsample=0.85, rsm=0.7,
)


def collapse(labels):
    out = np.full(len(labels), "Other", dtype=object)
    out[labels == "Inactivity"] = "Inactivity"
    out[labels == "Stable"] = "Stable"
    out[labels == FUEL_LABEL] = "Fuel"
    return out


def weight_of(cutoff):
    ts = pd.Timestamp(cutoff)
    return 0.70 + 0.05 * ((ts.year - 2024) * 12 + ts.month - 9)


def fit_cat(panel, cutoff, seeds):
    path = DEST / f"{cutoff}_rank4_cat_{len(seeds)}.npz"
    frame = panel["frames"][cutoff]
    if path.exists():
        payload = np.load(path, allow_pickle=True)
        if np.array_equal(payload["ids"].astype(str), frame.index.astype(str)):
            return payload["proba"].astype(float)
    latest = (pd.Timestamp(cutoff) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    limit = (pd.Timestamp(latest) - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    blocks = []
    for origin in panel["monthly_cutoffs"]:
        if origin > latest:
            continue
        origin_frame = panel["frames"][origin]
        labels = panel["labels"][origin].reindex(origin_frame.index).Opportunity.to_numpy()
        blocks.append(pd.DataFrame({
            "target": collapse(labels),
            "cutoff": origin,
            "w": weight_of(origin),
        }, index=origin_frame.index).join(origin_frame[panel["feature_cols"]]))
    train = pd.concat(blocks)
    tr = train.cutoff.le(limit).to_numpy()
    va = train.cutoff.eq(latest).to_numpy()
    features = panel["feature_cols"]
    total = np.zeros((len(frame), 4))
    for seed in seeds:
        t0 = time.time()
        model = CatBoostClassifier(**dict(PARAMS, random_seed=seed))
        model.fit(
            Pool(train.loc[tr, features], train.loc[tr, "target"], weight=train.loc[tr, "w"]),
            eval_set=Pool(train.loc[va, features], train.loc[va, "target"]),
            early_stopping_rounds=60,
        )
        proba = model.predict_proba(frame[features])
        order = list(model.classes_)
        align = np.zeros((len(frame), 4))
        for j, cls in enumerate(RANK):
            align[:, j] = proba[:, order.index(cls)]
        total += align
        print(f"  cat {cutoff} seed {seed} iters {model.get_best_iteration()} in {time.time()-t0:.0f}s", flush=True)
    out = total / len(seeds)
    np.savez(path, proba=out, ids=frame.index.astype(str).to_numpy())
    return out


def main():
    vR = np.array(json.load(open(OUT / "pool_r4full_vector.json"))["vR"])
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    print("panel loaded", flush=True)

    rows = []
    arrs = {}
    for c in POOL:
        n = len(panel["frames"][c])
        lgb = load_rank4(c, n)
        cat = fit_cat(panel, c, (7, 23))
        arrs[c] = {"lgb": lgb, "catboost": cat, "mean": (lgb + cat) / 2}
        ids, y = load_labels(c)
        p17 = load_proba("category_dynamics", c, n)
        for name, arr in arrs[c].items():
            prod = rank4_combine(p17, arr, "full")
            rows.append({"origin": c, "variant": name,
                         "f1_vR": round(weighted_f1(np.argmax(prod * vR, axis=1), y), 6)})
    table = pd.DataFrame(rows).pivot(index="variant", columns="origin", values="f1_vR")
    table["mean"] = table.mean(axis=1)
    print(table.round(6).to_string())

    gains = {c: float(table.loc["mean", c] - table.loc["lgb", c]) for c in POOL}
    pooled = float(np.mean(list(gains.values())))
    print("mean vs lgb pooled gain:", round(pooled, 6), {c: round(g, 6) for c, g in gains.items()}, flush=True)

    result = {"pool_table": table.round(6).to_dict(), "mean_gain": pooled, "per_origin": gains, "fold_table": {}}
    if pooled >= 0.002 and min(gains.values()) > -0.001:
        fold_rows = []
        for c in EVAL:
            n = len(panel["frames"][c])
            lgb = load_rank4(c, n)
            cat = fit_cat(panel, c, (7, 23))
            ids, y = load_labels(c)
            p17 = load_proba("category_dynamics", c, n)
            for name, arr in {"lgb": lgb, "mean": (lgb + cat) / 2}.items():
                prod = rank4_combine(p17, arr, "full")
                fold_rows.append({"origin": c, "variant": name,
                                  "f1_vR": round(weighted_f1(np.argmax(prod * vR, axis=1), y), 6)})
        ft = pd.DataFrame(fold_rows).pivot(index="variant", columns="origin", values="f1_vR")
        print(ft.round(6).to_string())
        result["fold_table"] = ft.round(6).to_dict()

    (OUT / "rank4_catboost_search.json").write_text(json.dumps(result, indent=2))
    print("wrote", OUT / "rank4_catboost_search.json")


if __name__ == "__main__":
    main()
