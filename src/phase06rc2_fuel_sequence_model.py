"""
Phase 06R-C2
Fuel-only sequence-to-sequence CLV experiment.

Research basis:
Bauer & Jannach (2021), "Improved Customer Lifetime Value Prediction
with Sequence-To-Sequence Learning and Feature-based Models."

Purpose:
Test whether ordered weekly customer purchase trajectories improve
future fuel CLV prediction beyond frozen S0.

Scope:
- Fuel only.
- 26 weekly input steps.
- Official 3-calendar-month CLV_fuel target.
- Outer temporal folds F2, F3, F4 only.
- No architecture search.
- No hyperparameter sweep.
- No Opportunity/nonfuel experiment.

Source-aligned:
- sliding historical windows;
- weekly sequences;
- two chained causal temporal convolutions;
- two GRU encoder layers;
- GRU decoder.

Fixed Sasol adaptations:
- one scalar output = official transformed 3-month CLV_fuel;
- embeddings omitted because Sasol embedding ablation was rejected;
- hidden size 32;
- no dropout;
- Adam lr 1e-3;
- batch 512;
- max 50 epochs;
- patience 5;
- latest observable training origin is used only for epoch selection,
  then the model is refit on all training origins.

Acceptance rule:
1. row-weighted F2-F4 RMSE improves vs S0;
2. at least 2/3 folds improve;
3. F4 is not worse than S0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reference.label_rules import load_transactions


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
CONFIG_PATH = ROOT / "configs" / "label_config.json"

S0_METRICS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
    / "regression_leaf_support"
    / "fold_comparison.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "sequence_fuel_c2"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_WEEKS = 26
STEP_DAYS = 7
HIDDEN_SIZE = 32
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
SEED = 20260928

FEATURE_NAMES = [
    "fuel_litres_week",
    "fuel_baskets_week",
    "nonfuel_rand_week",
    "shop_baskets_week",
    "txn_baskets_week",
    "active_days_week",
    "cum_fuel_litres",
    "cum_fuel_baskets",
    "cum_txn_baskets",
    "recency_weeks",
    "fuel_recency_weeks",
]

OUTER_FOLDS = [
    ("F2", pd.Timestamp("2025-03-01")),
    ("F3", pd.Timestamp("2025-06-01")),
    ("F4", pd.Timestamp("2025-09-01")),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype="float64")
    b = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def candidate_origins(evaluation_cutoff, data_start):
    latest = evaluation_cutoff - pd.DateOffset(months=3)
    origins = []
    current = latest
    while True:
        input_start = current - pd.Timedelta(weeks=INPUT_WEEKS)
        if input_start < data_start:
            break
        origins.append(current)
        current -= pd.Timedelta(days=STEP_DAYS)
    return sorted(origins)


def prepare_baskets(frame, included_categories, adjustment_items):
    d = frame[~frame["ItemName"].isin(adjustment_items)].copy()

    d["fuel_component"] = np.where(
        d["ItemCategoryLevel2"].eq("Fuel")
        & d["LineItemType"].eq("Sale_Fuel"),
        d["_quantity"] / 1_000_000.0,
        0.0,
    )

    d["nonfuel_component"] = np.where(
        d["ItemCategoryLevel2"].isin(included_categories)
        & ~d["ItemCategoryLevel2"].eq("Fuel"),
        d["_amount"] / 1_000_000.0,
        0.0,
    )

    baskets = (
        d.groupby(["ID", "TransactionId"], sort=False)
        .agg(
            time=("_time", "min"),
            fuel_litres=("fuel_component", "sum"),
            nonfuel_rand=("nonfuel_component", "sum"),
        )
        .reset_index()
    )

    baskets["date"] = baskets["time"].dt.normalize()
    baskets["fuel_positive"] = baskets["fuel_litres"] > 0
    baskets["shop_positive"] = baskets["nonfuel_rand"] > 0
    return baskets


def eligible_ids(frame, origin, minimum):
    counts = (
        frame[frame["_time"].lt(origin)]
        .groupby("ID")["TransactionId"]
        .nunique()
    )
    return pd.Index(
        counts[counts.ge(minimum)].index.astype(str).sort_values()
    )


def fuel_target_z(frame, ids, origin, scale, adjustment_items):
    end = origin + pd.DateOffset(months=3)

    future = frame[
        frame["_time"].ge(origin)
        & frame["_time"].lt(end)
        & frame["ID"].isin(ids)
        & ~frame["ItemName"].isin(adjustment_items)
        & frame["ItemCategoryLevel2"].eq("Fuel")
        & frame["LineItemType"].eq("Sale_Fuel")
    ]

    litres = (
        future.groupby("ID")["_quantity"]
        .sum()
        .div(1_000_000.0)
        .reindex(ids, fill_value=0.0)
        .to_numpy(dtype="float64")
    )

    if np.any(litres < 0):
        raise AssertionError(f"Negative fuel target at {origin.date()}.")

    return (np.log1p(litres) / scale).astype("float32")


def build_sequence_matrix(baskets, ids, origin):
    start = origin - pd.Timedelta(weeks=INPUT_WEEKS)

    block = baskets[
        baskets["time"].ge(start)
        & baskets["time"].lt(origin)
        & baskets["ID"].isin(ids)
    ].copy()

    seconds_per_week = 7 * 24 * 60 * 60
    block["step"] = (
        (block["time"] - start).dt.total_seconds() // seconds_per_week
    ).astype("int16")

    block = block[block["step"].between(0, INPUT_WEEKS - 1)]

    base_names = [
        "fuel_litres_week",
        "fuel_baskets_week",
        "nonfuel_rand_week",
        "shop_baskets_week",
        "txn_baskets_week",
        "active_days_week",
    ]

    weekly = (
        block.groupby(["ID", "step"], sort=False)
        .agg(
            fuel_litres_week=("fuel_litres", "sum"),
            fuel_baskets_week=("fuel_positive", "sum"),
            nonfuel_rand_week=("nonfuel_rand", "sum"),
            shop_baskets_week=("shop_positive", "sum"),
            txn_baskets_week=("TransactionId", "nunique"),
            active_days_week=("date", "nunique"),
        )
        .reset_index()
    )

    n = len(ids)
    arr = np.zeros((n, INPUT_WEEKS, len(base_names)), dtype="float32")
    id_to_row = {customer_id: i for i, customer_id in enumerate(ids)}

    if not weekly.empty:
        rows = weekly["ID"].map(id_to_row).to_numpy()
        steps = weekly["step"].to_numpy(dtype="int64")
        values = weekly[base_names].to_numpy(dtype="float32")
        arr[rows, steps, :] = values

    fuel_week = arr[:, :, 0]
    fuel_baskets = arr[:, :, 1]
    txn_baskets = arr[:, :, 4]

    cum_fuel = np.cumsum(fuel_week, axis=1)
    cum_fuel_baskets = np.cumsum(fuel_baskets, axis=1)
    cum_txn_baskets = np.cumsum(txn_baskets, axis=1)

    any_active = txn_baskets > 0
    fuel_active = fuel_baskets > 0

    recency = np.empty((n, INPUT_WEEKS), dtype="float32")
    fuel_recency = np.empty((n, INPUT_WEEKS), dtype="float32")

    for row in range(n):
        last_any = None
        last_fuel = None
        for step in range(INPUT_WEEKS):
            if any_active[row, step]:
                last_any = step
            if fuel_active[row, step]:
                last_fuel = step
            recency[row, step] = (
                step - last_any if last_any is not None else INPUT_WEEKS
            )
            fuel_recency[row, step] = (
                step - last_fuel if last_fuel is not None else INPUT_WEEKS
            )

    derived = np.stack(
        [
            cum_fuel,
            cum_fuel_baskets,
            cum_txn_baskets,
            recency,
            fuel_recency,
        ],
        axis=2,
    ).astype("float32")

    full = np.concatenate([arr, derived], axis=2)

    if full.shape[2] != len(FEATURE_NAMES):
        raise AssertionError("Unexpected sequence feature count.")

    return full


@dataclass
class ExampleBlock:
    origin: pd.Timestamp
    ids: np.ndarray
    X: np.ndarray
    y: np.ndarray


def build_example_block(
    frame,
    baskets,
    origin,
    minimum,
    scale,
    adjustment_items,
):
    ids = eligible_ids(frame, origin, minimum)
    X = build_sequence_matrix(baskets, ids, origin)
    y = fuel_target_z(
        frame,
        ids,
        origin,
        scale,
        adjustment_items,
    )
    if len(X) != len(y):
        raise AssertionError("Sequence-target alignment mismatch.")
    return ExampleBlock(
        origin=origin,
        ids=ids.to_numpy(dtype=str),
        X=X,
        y=y,
    )


class Standardizer:
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1]).astype("float64")
        self.mean = flat.mean(axis=0).astype("float32")
        std = flat.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std = std.astype("float32")
        return self

    def transform(self, X):
        return (
            (X - self.mean[None, None, :])
            / self.std[None, None, :]
        ).astype("float32")


class FuelS2S(nn.Module):
    def __init__(self, input_features, hidden_size):
        super().__init__()

        # Weekly analogues of learned monthly / quarterly smoothing.
        self.conv_month = nn.Conv1d(
            input_features,
            input_features,
            kernel_size=4,
        )
        self.conv_quarter = nn.Conv1d(
            input_features,
            input_features,
            kernel_size=13,
        )

        encoded_input = input_features * 3

        self.encoder1 = nn.GRU(
            input_size=encoded_input,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.transfer1 = nn.Linear(hidden_size, hidden_size)

        self.encoder2 = nn.GRU(
            input_size=encoded_input,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.transfer2 = nn.Linear(hidden_size, hidden_size)

        self.decoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    @staticmethod
    def causal_conv(x, conv):
        pad = conv.kernel_size[0] - 1
        x = nn.functional.pad(x, (pad, 0))
        return torch.relu(conv(x))

    def forward(self, x):
        channels = x.transpose(1, 2)

        monthly = self.causal_conv(
            channels,
            self.conv_month,
        )
        quarterly = self.causal_conv(
            monthly,
            self.conv_quarter,
        )

        combined = torch.cat(
            [channels, monthly, quarterly],
            dim=1,
        ).transpose(1, 2)

        _, h1 = self.encoder1(combined)

        h2_init = torch.tanh(
            self.transfer1(h1[-1])
        ).unsqueeze(0)

        _, h2 = self.encoder2(
            combined,
            h2_init,
        )

        decoder_init = torch.tanh(
            self.transfer2(h2[-1])
        ).unsqueeze(0)

        decoder_input = torch.zeros(
            (x.shape[0], 1, 1),
            dtype=x.dtype,
            device=x.device,
        )

        decoded, _ = self.decoder(
            decoder_input,
            decoder_init,
        )

        return self.output(
            decoded[:, -1, :]
        ).squeeze(1)


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def predict_model(model, X, device):
    model.eval()
    loader = make_loader(
        X,
        np.zeros(len(X), dtype="float32"),
        shuffle=False,
    )
    values = []

    with torch.no_grad():
        for xb, _ in loader:
            pred = model(xb.to(device))
            values.append(
                pred.detach().cpu().numpy()
            )

    return np.maximum(
        np.concatenate(values),
        0.0,
    )


def train_with_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
    device,
):
    set_seed(SEED)

    model = FuelS2S(
        input_features=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.MSELoss()

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
    )

    best_epoch = 1
    best_rmse = np.inf
    bad_epochs = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            losses.append(
                float(loss.detach().cpu())
            )

        valid_pred = predict_model(
            model,
            X_valid,
            device,
        )
        valid_rmse = rmse(
            y_valid,
            valid_pred,
        )

        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "inner_validation_rmse": valid_rmse,
            }
        )

        print(
            f"    epoch={epoch:02d} "
            f"train_mse={np.mean(losses):.6f} "
            f"inner_rmse={valid_rmse:.6f}"
        )

        if valid_rmse < best_rmse - 1e-6:
            best_rmse = valid_rmse
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= EARLY_STOP_PATIENCE:
            break

    return best_epoch, history


def refit_model(X, y, epochs, device):
    set_seed(SEED)

    model = FuelS2S(
        input_features=X.shape[2],
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.MSELoss()
    loader = make_loader(X, y, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

    return model


def load_s0_metrics():
    metrics = pd.read_csv(S0_METRICS_PATH)
    fuel = metrics[metrics["task"].eq("fuel")].copy()

    if fuel.empty:
        raise AssertionError("Fuel S0 metrics not found.")

    return {
        str(row.fold_id):
            float(row.experiment_validation_rmse)
        for row in fuel.itertuples()
    }


def main():
    set_seed(SEED)

    print("=" * 100)
    print("PHASE 06R-C2 — FUEL SEQUENCE MODEL")
    print("=" * 100)

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    minimum = int(
        config["minimum_historical_transactions"]
    )
    scale = float(
        config["normalization"]["CLV_fuel"]
    )

    included_categories = {
        category
        for category, rule
        in config["category_mapping"].items()
        if rule["included"]
    }

    adjustment_items = set(
        config["adjustment_items"]
    )

    print("Loading official Sasol transaction history...")
    frame = load_transactions(TRAIN_PATH)
    frame["ID"] = frame["ID"].astype(str)

    data_start = frame["_time"].min().normalize()

    baskets = prepare_baskets(
        frame,
        included_categories,
        adjustment_items,
    )

    s0 = load_s0_metrics()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device            : {device}")
    print(f"Input weeks       : {INPUT_WEEKS}")
    print(f"Sequential feats  : {len(FEATURE_NAMES)}")
    print(f"Hidden units      : {HIDDEN_SIZE}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Learning rate     : {LEARNING_RATE}")

    fold_rows = []
    epoch_rows = []
    prediction_frames = []

    for fold_id, outer_origin in OUTER_FOLDS:
        print()
        print("#" * 100)
        print(
            f"{fold_id} | OUTER ORIGIN "
            f"{outer_origin.date()}"
        )
        print("#" * 100)

        train_origins = candidate_origins(
            outer_origin,
            data_start,
        )

        if len(train_origins) < 2:
            raise AssertionError(
                f"{fold_id}: fewer than two training origins."
            )

        inner_origin = max(train_origins)

        earlier_origins = [
            origin
            for origin in train_origins
            if origin < inner_origin
        ]

        print(
            f"Training origins   : {len(train_origins)}"
        )
        print(
            f"Inner tune origin  : {inner_origin.date()}"
        )
        print(
            f"Outer eval origin  : {outer_origin.date()}"
        )

        cache = {}

        for origin in train_origins + [outer_origin]:
            if origin not in cache:
                print(
                    f"  building {origin.date()}..."
                )
                cache[origin] = build_example_block(
                    frame,
                    baskets,
                    origin,
                    minimum,
                    scale,
                    adjustment_items,
                )

        earlier_blocks = [
            cache[origin]
            for origin in earlier_origins
        ]
        inner_block = cache[inner_origin]
        outer_block = cache[outer_origin]

        X_early = np.concatenate(
            [block.X for block in earlier_blocks],
            axis=0,
        )
        y_early = np.concatenate(
            [block.y for block in earlier_blocks],
            axis=0,
        )

        scaler_inner = Standardizer().fit(
            X_early
        )

        X_early_s = scaler_inner.transform(
            X_early
        )
        X_inner_s = scaler_inner.transform(
            inner_block.X
        )

        print(
            f"Inner training examples : {len(X_early_s):,}"
        )
        print(
            f"Inner validation rows   : {len(X_inner_s):,}"
        )

        best_epoch, history = train_with_early_stopping(
            X_early_s,
            y_early,
            X_inner_s,
            inner_block.y,
            device,
        )

        print(
            f"Selected epoch          : {best_epoch}"
        )

        for row in history:
            epoch_rows.append(
                {
                    "fold_id": fold_id,
                    "inner_origin": inner_origin,
                    **row,
                }
            )

        all_train_blocks = [
            cache[origin]
            for origin in train_origins
        ]

        X_all = np.concatenate(
            [block.X for block in all_train_blocks],
            axis=0,
        )
        y_all = np.concatenate(
            [block.y for block in all_train_blocks],
            axis=0,
        )

        scaler_final = Standardizer().fit(
            X_all
        )

        X_all_s = scaler_final.transform(
            X_all
        )
        X_outer_s = scaler_final.transform(
            outer_block.X
        )

        print(
            f"Refit examples          : {len(X_all_s):,}"
        )

        final_model = refit_model(
            X_all_s,
            y_all,
            best_epoch,
            device,
        )

        outer_prediction = predict_model(
            final_model,
            X_outer_s,
            device,
        )

        sequence_rmse = rmse(
            outer_block.y,
            outer_prediction,
        )
        s0_rmse = float(
            s0[fold_id]
        )

        change = (
            sequence_rmse
            -
            s0_rmse
        )

        improved = (
            sequence_rmse
            <
            s0_rmse
        )

        print()
        print(
            f"{fold_id} RESULT | "
            f"S0={s0_rmse:.6f} | "
            f"S2S={sequence_rmse:.6f} | "
            f"change={change:+.6f} | "
            f"improved={improved}"
        )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "inner_origin": inner_origin,
                "outer_origin": outer_origin,
                "training_origins": len(train_origins),
                "training_examples": len(X_all_s),
                "validation_rows": len(outer_block.y),
                "selected_epoch": best_epoch,
                "s0_rmse": s0_rmse,
                "sequence_rmse": sequence_rmse,
                "rmse_change": change,
                "relative_change_pct":
                    100.0 * change / s0_rmse,
                "improved": improved,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "ID": outer_block.ids,
                    "origin": outer_origin,
                    "actual": outer_block.y,
                    "sequence_prediction":
                        outer_prediction,
                }
            )
        )

        del (
            X_early,
            y_early,
            X_early_s,
            X_inner_s,
            X_all,
            y_all,
            X_all_s,
            X_outer_s,
            final_model,
            cache,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    folds = pd.DataFrame(fold_rows)
    epochs = pd.DataFrame(epoch_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    weights = (
        folds["validation_rows"]
        /
        folds["validation_rows"].sum()
    )

    weighted_s0 = float(
        np.sum(
            folds["s0_rmse"]
            *
            weights
        )
    )

    weighted_sequence = float(
        np.sum(
            folds["sequence_rmse"]
            *
            weights
        )
    )

    folds_improved = int(
        folds["improved"].sum()
    )

    f4 = folds[
        folds["fold_id"].eq("F4")
    ].iloc[0]

    weighted_improved = (
        weighted_sequence
        <
        weighted_s0
    )

    at_least_two = (
        folds_improved >= 2
    )

    f4_not_worse = (
        float(f4["sequence_rmse"])
        <=
        float(f4["s0_rmse"])
    )

    c2_pass = (
        weighted_improved
        and
        at_least_two
        and
        f4_not_worse
    )

    summary = pd.DataFrame(
        [
            {
                "folds_improved":
                    folds_improved,
                "s0_row_weighted_rmse":
                    weighted_s0,
                "sequence_row_weighted_rmse":
                    weighted_sequence,
                "row_weighted_rmse_change":
                    weighted_sequence - weighted_s0,
                "row_weighted_relative_change_pct":
                    (
                        100.0
                        *
                        (weighted_sequence - weighted_s0)
                        /
                        weighted_s0
                    ),
                "weighted_improved":
                    weighted_improved,
                "at_least_two_folds_improved":
                    at_least_two,
                "f4_not_worse":
                    f4_not_worse,
                "c2_pass":
                    c2_pass,
            }
        ]
    )

    folds.to_csv(
        OUTPUT_DIR / "fold_comparison.csv",
        index=False,
    )
    epochs.to_csv(
        OUTPUT_DIR / "epoch_history.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / "outer_predictions.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    experiment_config = {
        "phase": "06R-C2",
        "target": "CLV_fuel",
        "input_weeks": INPUT_WEEKS,
        "weekly_features": FEATURE_NAMES,
        "architecture": {
            "causal_conv_kernel_weeks": [4, 13],
            "encoder_gru_layers": 2,
            "decoder_gru_layers": 1,
            "hidden_size": HIDDEN_SIZE,
            "dropout": 0.0,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience":
                EARLY_STOP_PATIENCE,
            "seed": SEED,
        },
        "acceptance_rule": {
            "row_weighted_rmse_improves": True,
            "minimum_folds_improved": 2,
            "F4_not_worse": True,
        },
        "hyperparameter_search": False,
        "customer_embeddings_used": False,
    }

    (
        OUTPUT_DIR / "experiment_config.json"
    ).write_text(
        json.dumps(
            experiment_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("PHASE 06R-C2 — OUTER FOLD COMPARISON")
    print("=" * 100)
    print()
    print(folds.to_string(index=False))

    print()
    print("=" * 100)
    print("C2 SUMMARY")
    print("=" * 100)
    print()
    print(summary.to_string(index=False))

    print()
    print(f"C2 PASS: {c2_pass}")
    print()
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
