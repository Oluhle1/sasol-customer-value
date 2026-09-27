"""
Phase 06R-A: Customer-embedding feasibility audit.

No embeddings and no predictive models are trained.

The script evaluates whether Sasol's historical purchase graph has the
customer-product interaction structure required by the published
SkipGram customer-embedding approach.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    ROOT
    / "data"
    / "raw"
    / "train.csv"
)

TEST_PATH = (
    ROOT
    / "data"
    / "raw"
    / "test.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "embedding_feasibility"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CUTOFFS = [
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
    "2025-09-01",
    "2025-12-01",
]

ADJUSTMENT_ITEMS = {
    "TOTAL/ROUNDING DISCO",
    "PENSIONER DISCOUNT",
    "WOW_PROMO",
}

TIME_CANDIDATES = [
    "TransactionDateTime",
    "TransactionDatetime",
    "TransactionTimestamp",
    "TransactionDate",
    "Timestamp",
    "Date",
]


def percentile_summary(values):

    values = np.asarray(
        values,
        dtype="float64",
    )

    if len(values) == 0:

        return {
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "min":
            float(
                np.min(values)
            ),

        "p25":
            float(
                np.percentile(
                    values,
                    25
                )
            ),

        "median":
            float(
                np.percentile(
                    values,
                    50
                )
            ),

        "p75":
            float(
                np.percentile(
                    values,
                    75
                )
            ),

        "p90":
            float(
                np.percentile(
                    values,
                    90
                )
            ),

        "p99":
            float(
                np.percentile(
                    values,
                    99
                )
            ),

        "max":
            float(
                np.max(values)
            ),
    }


def find_timestamp_column(columns):

    for candidate in TIME_CANDIDATES:

        if candidate in columns:

            return candidate

    raise AssertionError(
        "Could not identify transaction timestamp column. "
        f"Available columns: {list(columns)}"
    )


def main():

    print(
        "Loading Phase 06R-A raw transaction data..."
    )

    raw = pd.read_csv(
        RAW_PATH,
        dtype=str,
        keep_default_na=False,
    )

    test = pd.read_csv(
        TEST_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "ID",
        "TransactionId",
        "StockItem",
        "Quantity",
    }

    missing = (
        required
        -
        set(
            raw.columns
        )
    )

    if missing:

        raise AssertionError(
            f"Raw data missing required columns: {missing}"
        )

    timestamp_column = find_timestamp_column(
        raw.columns
    )

    print(
        f"Timestamp column: {timestamp_column}"
    )

    raw[
        "_timestamp"
    ] = pd.to_datetime(
        raw[
            timestamp_column
        ],
        errors="raise",
    )

    raw[
        "_quantity"
    ] = pd.to_numeric(
        raw[
            "Quantity"
        ],
        errors="raise",
    )

    # ============================================================
    # DEFINE A GENUINE POSITIVE PURCHASE INTERACTION
    # ============================================================

    purchase_mask = (
        raw[
            "_quantity"
        ].gt(0)
        &
        raw[
            "StockItem"
        ].ne("")
        &
        ~raw[
            "StockItem"
        ].isin(
            ADJUSTMENT_ITEMS
        )
    )

    if "LineItemType" in raw.columns:

        purchase_mask &= (
            raw[
                "LineItemType"
            ]
            .str.startswith(
                "Sale",
                na=False,
            )
        )

    purchases = raw[
        purchase_mask
    ].copy()

    # One customer-product token per basket.
    #
    # Quantity remains irrelevant to the sequence representation:
    # the research formulation is based on purchaser events rather
    # than repeating a customer token once per unit purchased.
    events = (
        purchases
        .sort_values(
            [
                "_timestamp",
                "TransactionId",
                "ID",
                "StockItem",
            ]
        )
        .drop_duplicates(
            [
                "ID",
                "TransactionId",
                "StockItem",
            ]
        )
        .copy()
    )

    print()
    print(
        f"Raw line rows               : {len(raw):,}"
    )

    print(
        f"Positive purchase lines     : {len(purchases):,}"
    )

    print(
        f"Basket-product events       : {len(events):,}"
    )

    # ============================================================
    # CORPUS SANITATION AUDIT
    # ============================================================

    sanitation = pd.DataFrame(
        [
            {
                "raw_rows":
                    len(raw),

                "positive_purchase_lines":
                    len(purchases),

                "basket_product_events":
                    len(events),

                "negative_or_zero_quantity_rows":
                    int(
                        raw[
                            "_quantity"
                        ].le(0).sum()
                    ),

                "adjustment_rows":
                    int(
                        raw[
                            "StockItem"
                        ].isin(
                            ADJUSTMENT_ITEMS
                        ).sum()
                    ),

                "timestamp_column":
                    timestamp_column,
            }
        ]
    )

    sanitation.to_csv(
        OUTPUT_DIR
        / "corpus_sanitation.csv",
        index=False,
    )

    cutoff_rows = []
    product_distribution_rows = []
    customer_distribution_rows = []
    threshold_rows = []
    context_rows = []
    product_sets = {}

    # ============================================================
    # CUTOFF-SPECIFIC AUDITS
    # ============================================================

    for cutoff_string in CUTOFFS:

        cutoff = pd.Timestamp(
            cutoff_string
        )

        history = events[
            events[
                "_timestamp"
            ].lt(
                cutoff
            )
        ].copy()

        if history.empty:

            raise AssertionError(
                f"No purchase history before {cutoff_string}"
            )

        product_stats = (
            history
            .groupby(
                "StockItem"
            )
            .agg(
                events=(
                    "ID",
                    "size",
                ),
                unique_customers=(
                    "ID",
                    "nunique",
                ),
                unique_baskets=(
                    "TransactionId",
                    "nunique",
                ),
            )
            .reset_index()
        )

        customer_stats = (
            history
            .groupby(
                "ID"
            )
            .agg(
                events=(
                    "StockItem",
                    "size",
                ),
                unique_products=(
                    "StockItem",
                    "nunique",
                ),
                unique_baskets=(
                    "TransactionId",
                    "nunique",
                ),
            )
            .reset_index()
        )

        product_sets[
            cutoff_string
        ] = set(
            product_stats[
                "StockItem"
            ]
        )

        product_purchaser_summary = (
            percentile_summary(
                product_stats[
                    "unique_customers"
                ].to_numpy()
            )
        )

        customer_product_summary = (
            percentile_summary(
                customer_stats[
                    "unique_products"
                ].to_numpy()
            )
        )

        product_distribution_rows.append(
            {
                "cutoff":
                    cutoff_string,

                **{
                    f"unique_customers_{key}":
                        value
                    for key, value
                    in product_purchaser_summary.items()
                },
            }
        )

        customer_distribution_rows.append(
            {
                "cutoff":
                    cutoff_string,

                **{
                    f"unique_products_{key}":
                        value
                    for key, value
                    in customer_product_summary.items()
                },
            }
        )

        # --------------------------------------------------------
        # PRODUCT SHARING / CUSTOMER COVERAGE
        # --------------------------------------------------------

        thresholds = [
            2,
            5,
            10,
            25,
            50,
            100,
        ]

        for threshold in thresholds:

            usable_products = set(
                product_stats.loc[
                    product_stats[
                        "unique_customers"
                    ].ge(
                        threshold
                    ),
                    "StockItem",
                ]
            )

            usable_events = history[
                history[
                    "StockItem"
                ].isin(
                    usable_products
                )
            ]

            covered_customers = (
                usable_events[
                    "ID"
                ]
                .nunique()
            )

            threshold_rows.append(
                {
                    "cutoff":
                        cutoff_string,

                    "min_unique_customers_per_product":
                        threshold,

                    "qualifying_products":
                        len(
                            usable_products
                        ),

                    "qualifying_product_share":
                        (
                            len(
                                usable_products
                            )
                            /
                            len(
                                product_stats
                            )
                        ),

                    "covered_events":
                        len(
                            usable_events
                        ),

                    "event_share":
                        (
                            len(
                                usable_events
                            )
                            /
                            len(
                                history
                            )
                        ),

                    "covered_customers":
                        covered_customers,

                    "customer_coverage":
                        (
                            covered_customers
                            /
                            history[
                                "ID"
                            ].nunique()
                        ),
                }
            )

        # --------------------------------------------------------
        # ADJACENT PURCHASER CONTEXT
        # --------------------------------------------------------

        ordered = (
            history
            .sort_values(
                [
                    "StockItem",
                    "_timestamp",
                    "TransactionId",
                    "ID",
                ]
            )
            .copy()
        )

        ordered[
            "_previous_customer"
        ] = (
            ordered
            .groupby(
                "StockItem"
            )[
                "ID"
            ]
            .shift(1)
        )

        pair_rows = ordered[
            ordered[
                "_previous_customer"
            ].notna()
        ]

        adjacent_pairs = len(
            pair_rows
        )

        cross_customer_pairs = int(
            (
                pair_rows[
                    "ID"
                ]
                !=
                pair_rows[
                    "_previous_customer"
                ]
            )
            .sum()
        )

        same_customer_pairs = (
            adjacent_pairs
            -
            cross_customer_pairs
        )

        context_rows.append(
            {
                "cutoff":
                    cutoff_string,

                "adjacent_context_pairs":
                    adjacent_pairs,

                "cross_customer_pairs":
                    cross_customer_pairs,

                "same_customer_pairs":
                    same_customer_pairs,

                "cross_customer_pair_share":
                    (
                        cross_customer_pairs
                        /
                        adjacent_pairs
                        if adjacent_pairs
                        else np.nan
                    ),
            }
        )

        # --------------------------------------------------------
        # FUEL DOMINANCE
        # --------------------------------------------------------

        if "Level2" in history.columns:

            fuel_events = int(
                history[
                    "Level2"
                ].eq(
                    "Fuel"
                ).sum()
            )

            fuel_share = (
                fuel_events
                /
                len(
                    history
                )
            )

        else:

            fuel_events = np.nan
            fuel_share = np.nan

        # --------------------------------------------------------
        # PRODUCT CONCENTRATION
        # --------------------------------------------------------

        sorted_product_events = (
            product_stats[
                "events"
            ]
            .sort_values(
                ascending=False
            )
            .to_numpy()
        )

        def top_share(n):

            return float(
                sorted_product_events[
                    :n
                ].sum()
                /
                sorted_product_events.sum()
            )

        cutoff_rows.append(
            {
                "cutoff":
                    cutoff_string,

                "purchase_events":
                    len(
                        history
                    ),

                "customers":
                    history[
                        "ID"
                    ].nunique(),

                "products":
                    history[
                        "StockItem"
                    ].nunique(),

                "baskets":
                    history[
                        "TransactionId"
                    ].nunique(),

                "products_single_customer":
                    int(
                        product_stats[
                            "unique_customers"
                        ].eq(1).sum()
                    ),

                "products_single_customer_share":
                    float(
                        product_stats[
                            "unique_customers"
                        ].eq(1).mean()
                    ),

                "top_1_product_event_share":
                    top_share(1),

                "top_10_product_event_share":
                    top_share(
                        min(
                            10,
                            len(
                                sorted_product_events
                            )
                        )
                    ),

                "top_100_product_event_share":
                    top_share(
                        min(
                            100,
                            len(
                                sorted_product_events
                            )
                        )
                    ),

                "fuel_events":
                    fuel_events,

                "fuel_event_share":
                    fuel_share,
            }
        )

    # ============================================================
    # PRODUCT UNIVERSE EVOLUTION
    # ============================================================

    persistence_rows = []

    for previous, current in zip(
        CUTOFFS[:-1],
        CUTOFFS[1:],
    ):

        previous_products = (
            product_sets[
                previous
            ]
        )

        current_products = (
            product_sets[
                current
            ]
        )

        intersection = (
            previous_products
            &
            current_products
        )

        union = (
            previous_products
            |
            current_products
        )

        newly_observed = (
            current_products
            -
            previous_products
        )

        persistence_rows.append(
            {
                "previous_cutoff":
                    previous,

                "current_cutoff":
                    current,

                "previous_products":
                    len(
                        previous_products
                    ),

                "current_products":
                    len(
                        current_products
                    ),

                "retained_products":
                    len(
                        intersection
                    ),

                "new_products":
                    len(
                        newly_observed
                    ),

                "cumulative_product_jaccard":
                    (
                        len(
                            intersection
                        )
                        /
                        len(
                            union
                        )
                    ),
            }
        )

    # ============================================================
    # FINAL COMPETITION CUSTOMER COVERAGE
    # ============================================================

    final_cutoff = pd.Timestamp(
        "2025-12-01"
    )

    final_history = events[
        events[
            "_timestamp"
        ].lt(
            final_cutoff
        )
    ].copy()

    final_product_stats = (
        final_history
        .groupby(
            "StockItem"
        )[
            "ID"
        ]
        .nunique()
    )

    test_ids = set(
        test[
            "ID"
        ].astype(str)
    )

    final_coverage_rows = []

    for threshold in [
        2,
        5,
        10,
        25,
        50,
        100,
    ]:

        usable_products = set(
            final_product_stats[
                final_product_stats.ge(
                    threshold
                )
            ].index
        )

        covered_ids = set(
            final_history.loc[
                final_history[
                    "StockItem"
                ].isin(
                    usable_products
                ),
                "ID",
            ]
        )

        covered_test = (
            test_ids
            &
            covered_ids
        )

        final_coverage_rows.append(
            {
                "min_unique_customers_per_product":
                    threshold,

                "test_customers":
                    len(
                        test_ids
                    ),

                "covered_test_customers":
                    len(
                        covered_test
                    ),

                "test_customer_coverage":
                    (
                        len(
                            covered_test
                        )
                        /
                        len(
                            test_ids
                        )
                    ),
            }
        )

    # ============================================================
    # SAVE
    # ============================================================

    cutoff_summary = pd.DataFrame(
        cutoff_rows
    )

    product_distribution = pd.DataFrame(
        product_distribution_rows
    )

    customer_distribution = pd.DataFrame(
        customer_distribution_rows
    )

    threshold_summary = pd.DataFrame(
        threshold_rows
    )

    context_summary = pd.DataFrame(
        context_rows
    )

    persistence = pd.DataFrame(
        persistence_rows
    )

    final_test_coverage = pd.DataFrame(
        final_coverage_rows
    )

    cutoff_summary.to_csv(
        OUTPUT_DIR
        / "cutoff_summary.csv",
        index=False,
    )

    product_distribution.to_csv(
        OUTPUT_DIR
        / "product_purchaser_distribution.csv",
        index=False,
    )

    customer_distribution.to_csv(
        OUTPUT_DIR
        / "customer_product_distribution.csv",
        index=False,
    )

    threshold_summary.to_csv(
        OUTPUT_DIR
        / "shared_product_coverage.csv",
        index=False,
    )

    context_summary.to_csv(
        OUTPUT_DIR
        / "adjacent_context_summary.csv",
        index=False,
    )

    persistence.to_csv(
        OUTPUT_DIR
        / "product_universe_persistence.csv",
        index=False,
    )

    final_test_coverage.to_csv(
        OUTPUT_DIR
        / "final_test_embedding_coverage.csv",
        index=False,
    )

    # ============================================================
    # CONSOLE REPORT
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 06R-A EMBEDDING FEASIBILITY AUDIT COMPLETE"
    )

    print(
        "=" * 76
    )

    print()
    print(
        "CUTOFF SUMMARY"
    )
    print()

    print(
        cutoff_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "PRODUCT PURCHASER DISTRIBUTION"
    )
    print()

    print(
        product_distribution.to_string(
            index=False
        )
    )

    print()
    print(
        "SHARED PRODUCT COVERAGE"
    )
    print()

    print(
        threshold_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "ADJACENT CONTEXT SUMMARY"
    )
    print()

    print(
        context_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "FINAL TEST EMBEDDING COVERAGE"
    )
    print()

    print(
        final_test_coverage.to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs written to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
