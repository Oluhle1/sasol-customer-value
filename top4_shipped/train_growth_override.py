"""Reproduce the 0.30225 Opportunity rule, then test one June-gated override.

The threshold is chosen on June 2025 only. September is scored once.
A submission file is written only when the gate in README.md passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score

GROK = Path(r"C:\Users\oluhl\Downloads\Y2YrnikRKzLjsyEg-grok-workspace")
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
BEST_FILE = Path(r"C:\Users\oluhl\Downloads\submission (1).csv")
SHIPPED_SEPTEMBER_F1 = 0.5118

MULTIPLIER = {
    "Inactivity": 1.0,
    "Stable": 1.0,
    "Existing-category growth: Fuel": 1.17,
    "Existing-category growth: Beverages": 1.3,
    "Existing-category growth: Food & beverage": 4.55,
    "Existing-category growth: Other": 1.3,
}
ADOPTION_MULTIPLIER = 2.5
FUEL_GROWTH = "Existing-category growth: Fuel"
FOOD_GROWTH = "Existing-category growth: Food & beverage"
SNAPSHOT_WEIGHT = {
    "2024-09-01": 0.70,
    "2024-12-01": 0.85,
    "2025-03-01": 1.00,
    "2025-06-01": 1.15,
    "2025-09-01": 1.30,
}
JUNE = "2025-06-01"
SEPTEMBER = "2025-09-01"


def load_base():
    source = (GROK / "ml" / "run_pipeline.py").read_text(encoding="utf-8")
    source = source.replace(
        'sys.path.insert(0, "/workspace/attachments")',
        f'sys.path.insert(0, r"{GROK / "attachments"}")',
    )
    source = source.replace(
        'ROOT = Path("/workspace")',
        f'ROOT = Path(r"{GROK}")',
    )
    source = source.split("\ndef main(")[0]
    namespace = {"__file__": str(GROK / "ml" / "run_pipeline.py")}
    exec(compile(source, "run_pipeline.py", "exec"), namespace)
    return namespace


def multiplier(label: str) -> float:
    if label.startswith("New category adoption"):
        return ADOPTION_MULTIPLIER
    return MULTIPLIER[label]


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def add_history_features(frame, cust: pd.DataFrame, cutoff: str, ns) -> pd.DataFrame:
    """Features named in the handoff that the older pipeline function does not emit."""
    ID = ns["ID"]
    CATS = ns["CATS"]
    slug = ns["slug"]
    t = pd.Timestamp(cutoff)
    ids = pd.Index(cust[ID])
    hist = frame[(frame._time < t) & frame[ID].isin(ids)]
    out = cust.copy()

    counts = hist.groupby([ID, "LocationId"]).size().reset_index(name="n")
    counts = counts.sort_values(["n", "LocationId"], ascending=[False, True])
    primary = counts.groupby(ID, sort=False).head(1).set_index(ID).LocationId
    out["primary_loc"] = primary.reindex(ids).to_numpy()

    scope = hist[hist.in_scope]
    for i in range(1, 7):
        start = t - pd.DateOffset(months=i)
        end = t - pd.DateOffset(months=i - 1)
        month = scope[(scope._time >= start) & (scope._time < end)]
        baskets = month.groupby(ID).TransactionId.nunique()
        fuel = month[month.fuel_sale].groupby(ID)
        shop = month[month.nonfuel].groupby(ID)
        out[f"n_txn_m{i}"] = baskets.reindex(ids).fillna(0).to_numpy()
        out[f"fuel_l_m{i}"] = fuel.qty.sum().reindex(ids).fillna(0).to_numpy()
        out[f"fuel_rand_m{i}"] = fuel.amt.sum().reindex(ids).fillna(0).to_numpy()
        out[f"shop_m{i}"] = shop.amt.sum().reindex(ids).fillna(0).to_numpy()
        out[f"shop_b_m{i}"] = shop.TransactionId.nunique().reindex(ids).fillna(0).to_numpy()

    for quarter, start_k, end_k in (("q1", 3, 0), ("q2", 6, 3), ("q3", 9, 6), ("q4", 12, 9)):
        start = t - pd.DateOffset(months=start_k)
        end = t - pd.DateOffset(months=end_k)
        chunk = scope[(scope._time >= start) & (scope._time < end)]
        fuel = chunk[chunk.fuel_sale].groupby(ID).amt.sum()
        out[f"fuel_rand_{quarter}"] = fuel.reindex(ids).fillna(0).to_numpy()

    ever_cols = [f"ever_{slug(c)}" for c in CATS]
    spend_cols = [f"sq1_{slug(c)}" for c in CATS]
    ever = out[ever_cols].to_numpy()
    spend = out[spend_cols].to_numpy()
    out["n_lapsed_cat"] = ((ever > 0.5) & (spend <= 0)).sum(axis=1).astype(float)
    out["n_easy_cat"] = ((spend > 0) & (spend <= 40)).sum(axis=1).astype(float)

    site = out["primary_loc"].astype(str)
    for column in ("fuel_l_q1", "nonfuel_q1", "n_txn_q1"):
        values = out[column].to_numpy(dtype=float)
        grouped = pd.DataFrame({"v": values, "site": site})
        total = grouped.groupby("site").v.transform("sum")
        count = grouped.groupby("site").v.transform("size")
        out[f"loo_{column}"] = (total - grouped.v) / (count - 1).clip(lower=1)
    return out


def add_previous_opportunity(customers: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    parts = []
    previous = None
    for cutoff in SNAPSHOT_WEIGHT:
        part = customers[customers.cutoff.eq(cutoff)].copy()
        if previous is None:
            prior = pd.Series("missing", index=part.ID)
        else:
            prior = part.ID.map(previous).fillna("missing")
        part["prev_missing"] = prior.eq("missing").astype(np.float32)
        for label in labels:
            part[f"prev_{label}"] = prior.eq(label).astype(np.float32)
        parts.append(part)
        previous = part.set_index("ID").Opportunity
    return pd.concat(parts, ignore_index=True)


def matrix(df: pd.DataFrame):
    drop = {
        "ID",
        "cutoff",
        "eligible",
        "primary_loc",
        "CLV_fuel",
        "CLV_nonfuel",
        "Opportunity",
        "fuel_litres",
        "nonfuel_rand",
    }
    columns = [c for c in df.columns if c not in drop]
    X = df[columns].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, columns


def fit_classifier(X, y, w, Xv, yv, seed: int) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.04,
        num_leaves=47,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_lambda=1.2,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
    )
    model.fit(
        X,
        y,
        sample_weight=w,
        eval_set=[(Xv, yv)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )
    return model


def fit_binary(X, y, w, Xv, yv, seed: int) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.04,
        num_leaves=47,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_lambda=1.2,
        random_state=seed,
        verbosity=-1,
        n_jobs=4,
    )
    model.fit(
        X,
        y,
        sample_weight=w,
        eval_set=[(Xv, yv)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )
    return model


def average_proba(models, X) -> tuple[np.ndarray, list[str]]:
    classes = list(models[0].classes_)
    total = np.zeros((len(X), len(classes)), dtype=float)
    for model in models:
        order = [list(model.classes_).index(name) for name in classes]
        total += model.predict_proba(X)[:, order]
    return total / len(models), classes


def apply_rule(proba, classes) -> np.ndarray:
    adjusted = proba.copy()
    for i, name in enumerate(classes):
        adjusted[:, i] *= multiplier(str(name))
    return np.asarray(classes, dtype=object)[adjusted.argmax(axis=1)]


def positive_probability(models, X) -> np.ndarray:
    total = np.zeros(len(X), dtype=float)
    for model in models:
        classes = list(model.classes_)
        column = classes.index(1) if 1 in classes else classes.index(1.0)
        total += model.predict_proba(X)[:, column]
    return total / len(models)


def override(shipped, p_fuel, p_food, threshold: float) -> np.ndarray:
    pred = shipped.copy()
    stable = pred == "Stable"
    food = stable & (p_food >= threshold) & (p_food >= p_fuel)
    fuel = stable & (p_fuel >= threshold) & ~food
    pred[food] = FOOD_GROWTH
    pred[fuel] = FUEL_GROWTH
    return pred


def choose_threshold(shipped, p_fuel, p_food, y_true) -> dict:
    best = {"threshold": None, "f1": weighted_f1(y_true, shipped), "changed": 0}
    for threshold in np.round(np.arange(0.40, 0.86, 0.05), 2):
        pred = override(shipped, p_fuel, p_food, float(threshold))
        score = weighted_f1(y_true, pred)
        if score > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": score,
                "changed": int((pred != shipped).sum()),
            }
    return best


def slice_fold(customers, features, train_cutoffs, valid_cutoff):
    train = customers.cutoff.isin(train_cutoffs)
    valid = customers.cutoff.eq(valid_cutoff)
    X = customers[features]
    return (
        X.loc[train].to_numpy(),
        customers.loc[train, "Opportunity"].to_numpy(),
        customers.loc[train, "cutoff"].map(SNAPSHOT_WEIGHT).to_numpy(dtype=float),
        X.loc[valid].to_numpy(),
        customers.loc[valid, "Opportunity"].to_numpy(),
    )


def shipped_and_events(customers, features, train_cutoffs, valid_cutoff):
    Xtr, ytr, wtr, Xva, yva = slice_fold(customers, features, train_cutoffs, valid_cutoff)
    print(f"  training shipped classifiers for {valid_cutoff}", flush=True)
    classifiers = [
        fit_classifier(Xtr, ytr, wtr, Xva, yva, seed) for seed in (7, 23)
    ]
    proba, classes = average_proba(classifiers, Xva)
    shipped = apply_rule(proba, classes)
    print(f"  training growth-event models for {valid_cutoff}", flush=True)
    y_fuel = (ytr == FUEL_GROWTH).astype(int)
    y_food = (ytr == FOOD_GROWTH).astype(int)
    y_fuel_v = (yva == FUEL_GROWTH).astype(int)
    y_food_v = (yva == FOOD_GROWTH).astype(int)
    fuel_models = [fit_binary(Xtr, y_fuel, wtr, Xva, y_fuel_v, seed) for seed in (7, 23)]
    food_models = [fit_binary(Xtr, y_food, wtr, Xva, y_food_v, seed) for seed in (7, 23)]
    return {
        "y": yva,
        "shipped": shipped,
        "shipped_f1": weighted_f1(yva, shipped),
        "p_fuel": positive_probability(fuel_models, Xva),
        "p_food": positive_probability(food_models, Xva),
        "classifiers": classifiers,
        "fuel_models": fuel_models,
        "food_models": food_models,
    }


def build_customers(ns):
    print("loading transactions", flush=True)
    raw = ns["load_transactions"](str(GROK / "data" / "train.csv"))
    frame = ns["prepare_frame"](raw)
    parts = []
    for cutoff in SNAPSHOT_WEIGHT:
        print(f"  snapshot {cutoff}", flush=True)
        cust = ns["customer_table"](frame, cutoff)
        cust = add_history_features(frame, cust, cutoff, ns)
        official = ns["transform"](ns["raw_labels"](raw, cutoff, ns["CONFIG"]), ns["CONFIG"])
        cust = cust.merge(official[["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]], on="ID", how="left")
        parts.append(cust)
    customers = pd.concat(parts, ignore_index=True)
    labels = list(ns["LABELS"])
    customers = add_previous_opportunity(customers, labels)
    print("  test snapshot 2025-12-01", flush=True)
    test = ns["customer_table"](frame, "2025-12-01")
    test = add_history_features(frame, test, "2025-12-01", ns)
    september = customers[customers.cutoff.eq(SEPTEMBER)].set_index("ID").Opportunity
    prior = test.ID.map(september).fillna("missing")
    test["prev_missing"] = prior.eq("missing").astype(np.float32)
    for label in labels:
        test[f"prev_{label}"] = prior.eq(label).astype(np.float32)
    test["Opportunity"] = "missing"
    test["CLV_fuel"] = 0.0
    test["CLV_nonfuel"] = 0.0
    X, columns = matrix(pd.concat([customers, test], ignore_index=True))
    history = pd.concat(
        [customers[["ID", "cutoff", "Opportunity"]].reset_index(drop=True), X.iloc[: len(customers)].reset_index(drop=True)],
        axis=1,
    )
    holdout = pd.concat(
        [test[["ID"]].reset_index(drop=True), X.iloc[len(customers):].reset_index(drop=True)],
        axis=1,
    )
    return history, holdout, columns


def main():
    ns = load_base()
    customers, test, features = build_customers(ns)
    print(f"rows={len(customers):,} features={len(features)}", flush=True)
    cutoffs = list(SNAPSHOT_WEIGHT)

    print("June fold", flush=True)
    june = shipped_and_events(customers, features, cutoffs[:3], JUNE)
    chosen = choose_threshold(june["shipped"], june["p_fuel"], june["p_food"], june["y"])
    print(
        f"June shipped {june['shipped_f1']:.4f}  "
        f"chosen threshold {chosen['threshold']} f1 {chosen['f1']:.4f}",
        flush=True,
    )

    print("September fold", flush=True)
    september = shipped_and_events(customers, features, cutoffs[:4], SEPTEMBER)
    if chosen["threshold"] is None:
        september_pred = september["shipped"]
        september_f1 = september["shipped_f1"]
    else:
        september_pred = override(
            september["shipped"],
            september["p_fuel"],
            september["p_food"],
            chosen["threshold"],
        )
        september_f1 = weighted_f1(september["y"], september_pred)
    june_f1 = chosen["f1"]
    print(
        f"September shipped {september['shipped_f1']:.4f}  override {september_f1:.4f}",
        flush=True,
    )

    june_gain = june_f1 - june["shipped_f1"]
    september_gain = september_f1 - september["shipped_f1"]
    decision = {
        "june_shipped_f1": june["shipped_f1"],
        "june_override_f1": june_f1,
        "june_gain": june_gain,
        "september_shipped_f1": september["shipped_f1"],
        "september_override_f1": september_f1,
        "september_gain": september_gain,
        "published_september_f1": SHIPPED_SEPTEMBER_F1,
        "threshold": chosen["threshold"],
        "june_rows_changed": chosen["changed"],
        "accepted": bool(
            chosen["threshold"] is not None
            and june_gain > 0
            and september_f1 > SHIPPED_SEPTEMBER_F1
            and september_gain >= 0
            and (june_gain + september_gain) / 2 >= 0.01
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    if not decision["accepted"]:
        print("Gate did not pass. The 0.30225 file stays in place.", flush=True)
        return

    print("refitting for the December book", flush=True)
    rounds = [int(model.best_iteration_ or 200) for model in september["classifiers"]]
    event_rounds = {
        "fuel": [int(model.best_iteration_ or 200) for model in september["fuel_models"]],
        "food": [int(model.best_iteration_ or 200) for model in september["food_models"]],
    }
    X_all = customers[features].to_numpy()
    y_all = customers.Opportunity.to_numpy()
    w_all = customers.cutoff.map(SNAPSHOT_WEIGHT).to_numpy(dtype=float)
    final_classifiers = []
    for seed, n_trees in zip((7, 23), rounds):
        model = LGBMClassifier(
            n_estimators=n_trees,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(X_all, y_all, sample_weight=w_all)
        final_classifiers.append(model)
    final_fuel = []
    final_food = []
    for seed, n_trees in zip((7, 23), event_rounds["fuel"]):
        model = LGBMClassifier(
            n_estimators=n_trees,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(X_all, (y_all == FUEL_GROWTH).astype(int), sample_weight=w_all)
        final_fuel.append(model)
    for seed, n_trees in zip((7, 23), event_rounds["food"]):
        model = LGBMClassifier(
            n_estimators=n_trees,
            learning_rate=0.04,
            num_leaves=47,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            random_state=seed,
            verbosity=-1,
            n_jobs=4,
        )
        model.fit(X_all, (y_all == FOOD_GROWTH).astype(int), sample_weight=w_all)
        final_food.append(model)

    X_test = test[features].to_numpy()
    proba, classes = average_proba(final_classifiers, X_test)
    pred = apply_rule(proba, classes)
    pred = override(
        pred,
        positive_probability(final_fuel, X_test),
        positive_probability(final_food, X_test),
        chosen["threshold"],
    )
    base = pd.read_csv(BEST_FILE, dtype=str)
    out = base[["ID", "CLV_fuel", "CLV_nonfuel"]].merge(
        pd.DataFrame({"ID": test.ID.astype(str), "Opportunity": pred}),
        on="ID",
        how="left",
        validate="one_to_one",
    )
    if out.Opportunity.isna().any() or not out.ID.equals(base.ID):
        raise AssertionError("Test predictions do not cover the 0.30225 file")
    destination = HERE / "sasol_top4_growth_override_submission.csv"
    out.to_csv(destination, index=False)
    print("Wrote", destination, flush=True)
    print(out.Opportunity.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
