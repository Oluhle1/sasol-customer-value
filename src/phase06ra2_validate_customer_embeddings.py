"""
Phase 06R-A2 integrity validation.

Purpose
-------
1. Reconcile the small event-count difference between A1 and A2.
2. Verify final competition-test embedding coverage.
3. Check whether recursive centroid refresh causes representation collapse.

This script does not fit or evaluate any predictive model.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_PATH = ROOT / "data" / "raw" / "Test.csv"

EMBEDDING_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "customer_embeddings"
    / "customer_embeddings.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "customer_embeddings"
)

ADJUSTMENT_ITEMS = {
    "TOTAL/ROUNDING DISCO",
    "PENSIONER DISCOUNT",
    "WOW_PROMO",
}

CUTOFFS = [
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
    "2025-09-01",
    "2025-12-01",
]

RNG = np.random.default_rng(42)


def event_count(frame, mask, cutoff=None):

    work = frame.loc[mask].copy()

    if cutoff is not None:
        work = work[
            work["_timestamp"].lt(
                pd.Timestamp(cutoff)
            )
        ]

    events = work.drop_duplicates(
        [
            "ID",
            "TransactionId",
            "StockItemId",
        ]
    )

    return (
        len(work),
        len(events),
    )


def cosine_rows(a, b):

    numerator = np.sum(
        a * b,
        axis=1,
    )

    denominator = (
        np.linalg.norm(
            a,
            axis=1,
        )
        *
        np.linalg.norm(
            b,
            axis=1,
        )
    )

    valid = denominator > 0

    result = np.full(
        len(a),
        np.nan,
        dtype="float64",
    )

    result[valid] = (
        numerator[valid]
        /
        denominator[valid]
    )

    return result


def representation_health(
    embeddings,
    emb_cols,
):

    rows = []

    for cutoff in CUTOFFS:

        frame = embeddings[
            embeddings["cutoff"].eq(
                cutoff
            )
        ].copy()

        X = frame[
            emb_cols
        ].to_numpy(
            dtype="float64"
        )

        if not np.isfinite(X).all():
            raise AssertionError(
                f"Non-finite vectors at {cutoff}"
            )

        dimension_std = X.std(
            axis=0,
            ddof=0,
        )

        norms = np.linalg.norm(
            X,
            axis=1,
        )

        centered = (
            X
            -
            X.mean(
                axis=0,
                keepdims=True,
            )
        )

        rank = int(
            np.linalg.matrix_rank(
                centered
            )
        )

        rounded_unique = np.unique(
            np.round(
                X,
                8,
            ),
            axis=0,
        ).shape[0]

        pair_count = min(
            5000,
            max(
                1000,
                len(X) * 2,
            ),
        )

        left = RNG.integers(
            0,
            len(X),
            size=pair_count,
        )

        right = RNG.integers(
            0,
            len(X),
            size=pair_count,
        )

        equal = left == right

        right[equal] = (
            right[equal] + 1
        ) % len(X)

        pair_cosine = cosine_rows(
            X[left],
            X[right],
        )

        rows.append(
            {
                "cutoff":
                    cutoff,

                "customers":
                    len(frame),

                "centered_matrix_rank":
                    rank,

                "dimension_std_min":
                    float(
                        dimension_std.min()
                    ),

                "dimension_std_median":
                    float(
                        np.median(
                            dimension_std
                        )
                    ),

                "dimension_std_max":
                    float(
                        dimension_std.max()
                    ),

                "mean_vector_norm":
                    float(
                        norms.mean()
                    ),

                "median_vector_norm":
                    float(
                        np.median(
                            norms
                        )
                    ),

                "unique_vectors_rounded_8dp":
                    int(
                        rounded_unique
                    ),

                "unique_vector_share":
                    float(
                        rounded_unique
                        /
                        len(frame)
                    ),

                "random_pair_cosine_mean":
                    float(
                        np.nanmean(
                            pair_cosine
                        )
                    ),

                "random_pair_cosine_median":
                    float(
                        np.nanmedian(
                            pair_cosine
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def temporal_drift(
    embeddings,
    emb_cols,
):

    rows = []

    for previous_cutoff, current_cutoff in zip(
        CUTOFFS[:-1],
        CUTOFFS[1:],
    ):

        previous = embeddings[
            embeddings["cutoff"].eq(
                previous_cutoff
            )
        ][
            ["ID"] + emb_cols
        ].copy()

        current = embeddings[
            embeddings["cutoff"].eq(
                current_cutoff
            )
        ][
            ["ID"] + emb_cols
        ].copy()

        merged = previous.merge(
            current,
            on="ID",
            how="inner",
            suffixes=(
                "_previous",
                "_current",
            ),
            validate="one_to_one",
        )

        previous_X = merged[
            [
                f"{column}_previous"
                for column
                in emb_cols
            ]
        ].to_numpy(
            dtype="float64"
        )

        current_X = merged[
            [
                f"{column}_current"
                for column
                in emb_cols
            ]
        ].to_numpy(
            dtype="float64"
        )

        delta = current_X - previous_X

        delta_norm = np.linalg.norm(
            delta,
            axis=1,
        )

        unchanged = np.all(
            previous_X == current_X,
            axis=1,
        )

        cosine = cosine_rows(
            previous_X,
            current_X,
        )

        changed_cosine = cosine[
            ~unchanged
        ]

        rows.append(
            {
                "previous_cutoff":
                    previous_cutoff,

                "current_cutoff":
                    current_cutoff,

                "common_customers":
                    len(merged),

                "unchanged_customers":
                    int(
                        unchanged.sum()
                    ),

                "unchanged_share":
                    float(
                        unchanged.mean()
                    ),

                "median_delta_norm":
                    float(
                        np.median(
                            delta_norm
                        )
                    ),

                "median_cosine_all":
                    float(
                        np.nanmedian(
                            cosine
                        )
                    ),

                "median_cosine_changed":
                    (
                        float(
                            np.nanmedian(
                                changed_cosine
                            )
                        )
                        if len(
                            changed_cosine
                        )
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():

    print(
        "Loading raw data..."
    )

    raw = pd.read_csv(
        RAW_PATH,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    raw["_timestamp"] = pd.to_datetime(
        raw["DateTimeZA"],
        format="mixed",
        errors="raise",
    )

    quantity = pd.to_numeric(
        raw["Quantity"],
        errors="raise",
    )

    amount = pd.to_numeric(
        raw["TotalExclAmount"],
        errors="raise",
    )

    non_adjustment = ~raw[
        "ItemName"
    ].isin(
        ADJUSTMENT_ITEMS
    )

    masks = {
        "positive_quantity_nonadjustment":
            quantity.gt(0)
            &
            non_adjustment,

        "positive_quantity_and_amount":
            quantity.gt(0)
            &
            amount.gt(0)
            &
            non_adjustment,

        "plus_nonempty_ID":
            quantity.gt(0)
            &
            amount.gt(0)
            &
            non_adjustment
            &
            raw["ID"].ne(""),

        "plus_nonempty_transaction":
            quantity.gt(0)
            &
            amount.gt(0)
            &
            non_adjustment
            &
            raw["ID"].ne("")
            &
            raw["TransactionId"].ne(""),

        "A2_final_mask":
            quantity.gt(0)
            &
            amount.gt(0)
            &
            non_adjustment
            &
            raw["ID"].ne("")
            &
            raw["TransactionId"].ne("")
            &
            raw["StockItemId"].ne(""),
    }

    reconciliation_rows = []

    for name, mask in masks.items():

        full_lines, full_events = event_count(
            raw,
            mask,
        )

        sep_lines, sep_events = event_count(
            raw,
            mask,
            cutoff="2024-09-01",
        )

        reconciliation_rows.append(
            {
                "mask":
                    name,

                "full_lines":
                    full_lines,

                "full_basket_product_events":
                    full_events,

                "pre_2024_09_lines":
                    sep_lines,

                "pre_2024_09_events":
                    sep_events,
            }
        )

    reconciliation = pd.DataFrame(
        reconciliation_rows
    )

    print()
    print(
        "=" * 78
    )
    print(
        "EVENT-DEFINITION RECONCILIATION"
    )
    print(
        "=" * 78
    )
    print()
    print(
        reconciliation.to_string(
            index=False
        )
    )

    print()
    print(
        "Reference from A1 feasibility audit:"
    )
    print(
        "full basket-product events : 437407"
    )
    print(
        "pre-2024-09 events         : 102605"
    )

    embeddings = pd.read_parquet(
        EMBEDDING_PATH
    )

    embeddings["ID"] = (
        embeddings["ID"]
        .astype(str)
    )

    embeddings["cutoff"] = (
        embeddings["cutoff"]
        .astype(str)
    )

    emb_cols = sorted(
        [
            column
            for column
            in embeddings.columns
            if column.startswith(
                "cust_emb_"
            )
        ]
    )

    if len(emb_cols) != 64:
        raise AssertionError(
            f"Expected 64 embedding dimensions, found {len(emb_cols)}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Expected competition test file: {TEST_PATH}"
        )

    test = pd.read_csv(
        TEST_PATH,
        dtype={
            "ID": str
        },
        keep_default_na=False,
        low_memory=False,
    )

    test_ids = set(
        test["ID"].astype(str)
    )

    final_embedding_ids = set(
        embeddings.loc[
            embeddings[
                "cutoff"
            ].eq(
                "2025-12-01"
            ),
            "ID",
        ].astype(str)
    )

    covered = len(
        test_ids
        &
        final_embedding_ids
    )

    missing_test_ids = sorted(
        test_ids
        -
        final_embedding_ids
    )

    print()
    print(
        "=" * 78
    )
    print(
        "FINAL TEST COVERAGE"
    )
    print(
        "=" * 78
    )
    print()
    print(
        f"Test customers       : {len(test_ids):,}"
    )
    print(
        f"Covered              : {covered:,}"
    )
    print(
        f"Coverage              : {covered / len(test_ids):.6f}"
    )
    print(
        f"Missing test IDs      : {len(missing_test_ids):,}"
    )

    if missing_test_ids:
        print(
            "First missing IDs:",
            missing_test_ids[:20],
        )

    health = representation_health(
        embeddings,
        emb_cols,
    )

    drift = temporal_drift(
        embeddings,
        emb_cols,
    )

    print()
    print(
        "=" * 78
    )
    print(
        "REPRESENTATION HEALTH"
    )
    print(
        "=" * 78
    )
    print()
    print(
        health.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 78
    )
    print(
        "TEMPORAL EMBEDDING DRIFT"
    )
    print(
        "=" * 78
    )
    print()
    print(
        drift.to_string(
            index=False
        )
    )

    reconciliation.to_csv(
        OUTPUT_DIR
        /
        "event_definition_reconciliation.csv",
        index=False,
    )

    health.to_csv(
        OUTPUT_DIR
        /
        "embedding_representation_health.csv",
        index=False,
    )

    drift.to_csv(
        OUTPUT_DIR
        /
        "embedding_temporal_drift.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "ID":
                missing_test_ids
        }
    ).to_csv(
        OUTPUT_DIR
        /
        "missing_test_embedding_ids.csv",
        index=False,
    )

    print()
    print(
        "A2 integrity audit complete."
    )


if __name__ == "__main__":
    main()
