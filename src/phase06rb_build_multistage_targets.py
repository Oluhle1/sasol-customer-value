"""
Phase 06R-B1
Build auxiliary future outcomes for research-backed multi-stage CLV.

Research basis:
Yan & Resnick (2024), Quantitative Marketing and Economics:
    CLV(x) =
        Prob_return(x)
        * Freq_return(x)
        * AOV_return(x)

This script DOES NOT change the official Sasol labels.

It derives auxiliary components from the exact same 3-month outcome
windows and audits them against the official reference label generator.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from reference.label_rules import (
    SCALE,
    eligible_ids,
    load_transactions,
    raw_labels,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "raw"
    / "train.csv"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "label_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase06"
    / "multistage_targets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def scaled_target_to_int(series):
    """
    Official targets have at most 6 decimal places because the reference
    implementation works internally at SCALE=1,000,000.
    """

    return np.rint(
        series.to_numpy(
            dtype="float64"
        )
        * SCALE
    ).astype("int64")


def group_sum_scaled(
    frame,
    value_column,
    ids,
):
    """
    Sum an already-scaled integer transaction field by customer.
    """

    return (
        frame
        .groupby("ID")[
            value_column
        ]
        .sum()
        .reindex(
            ids,
            fill_value=0,
        )
        .astype("int64")
    )


def positive_basket_count(
    frame,
    ids,
):
    """
    Count distinct baskets containing at least one qualifying positive
    purchase line.

    This follows the official purchase definition:
        Quantity > 0
        TotalExclAmount > 0
    """

    positive = frame[
        frame["_quantity"].gt(0)
        &
        frame["_amount"].gt(0)
    ]

    return (
        positive
        .groupby("ID")[
            "TransactionId"
        ]
        .nunique()
        .reindex(
            ids,
            fill_value=0,
        )
        .astype("int64")
    )


def build_snapshot(
    transactions,
    cutoff_string,
    config,
):

    cutoff = pd.Timestamp(
        cutoff_string
    )

    outcome_end = (
        cutoff
        + pd.DateOffset(
            months=3
        )
    )

    ids = eligible_ids(
        transactions,
        cutoff,
        config[
            "minimum_historical_transactions"
        ],
    )

    # ------------------------------------------------------------
    # EXACT SAME ADJUSTMENT POLICY AS OFFICIAL LABEL GENERATOR
    # ------------------------------------------------------------

    clean = transactions[
        ~transactions[
            "ItemName"
        ].isin(
            config[
                "adjustment_items"
            ]
        )
    ].copy()

    future = clean[
        clean[
            "_time"
        ].ge(
            cutoff
        )
        &
        clean[
            "_time"
        ].lt(
            outcome_end
        )
    ].copy()

    mapping = (
        config[
            "category_mapping"
        ]
    )

    included_categories = {
        category
        for category, settings
        in mapping.items()
        if settings[
            "included"
        ]
    }

    nonfuel_categories = (
        included_categories
        -
        {"Fuel"}
    )

    # ============================================================
    # FUEL
    # ============================================================

    # Exact official regression domain:
    # Level2 = Fuel AND LineItemType = Sale_Fuel
    fuel_rows = future[
        future[
            "ItemCategoryLevel2"
        ].eq(
            "Fuel"
        )
        &
        future[
            "LineItemType"
        ].eq(
            "Sale_Fuel"
        )
    ].copy()

    fuel_total_scaled = (
        group_sum_scaled(
            fuel_rows,
            "_quantity",
            ids,
        )
    )

    fuel_baskets = (
        positive_basket_count(
            fuel_rows,
            ids,
        )
    )

    fuel_active = (
        fuel_baskets.gt(0)
    )

    # ============================================================
    # NON-FUEL
    # ============================================================

    # Exact official regression domain:
    # all included categories except Fuel.
    nonfuel_rows = future[
        future[
            "ItemCategoryLevel2"
        ].isin(
            nonfuel_categories
        )
    ].copy()

    nonfuel_total_scaled = (
        group_sum_scaled(
            nonfuel_rows,
            "_amount",
            ids,
        )
    )

    nonfuel_baskets = (
        positive_basket_count(
            nonfuel_rows,
            ids,
        )
    )

    nonfuel_active = (
        nonfuel_baskets.gt(0)
    )

    # ============================================================
    # OFFICIAL LABEL AUDIT
    # ============================================================

    official = raw_labels(
        transactions,
        cutoff_string,
        config,
    )

    official = (
        official
        .set_index(
            "ID"
        )
        .reindex(
            ids
        )
    )

    official_fuel_scaled = (
        scaled_target_to_int(
            official[
                "fuel_litres"
            ]
        )
    )

    official_nonfuel_scaled = (
        scaled_target_to_int(
            official[
                "nonfuel_rand"
            ]
        )
    )

    if not np.array_equal(
        fuel_total_scaled.to_numpy(),
        official_fuel_scaled,
    ):

        bad = np.flatnonzero(
            fuel_total_scaled.to_numpy()
            !=
            official_fuel_scaled
        )

        raise AssertionError(
            "Fuel auxiliary target does not match "
            f"official target for {len(bad)} customers."
        )

    if not np.array_equal(
        nonfuel_total_scaled.to_numpy(),
        official_nonfuel_scaled,
    ):

        bad = np.flatnonzero(
            nonfuel_total_scaled.to_numpy()
            !=
            official_nonfuel_scaled
        )

        raise AssertionError(
            "Non-fuel auxiliary target does not match "
            f"official target for {len(bad)} customers."
        )

    # Positive raw target should not exist without at least one
    # qualifying positive purchase basket.
    fuel_positive_without_purchase = (
        fuel_total_scaled.gt(0)
        &
        fuel_baskets.eq(0)
    )

    nonfuel_positive_without_purchase = (
        nonfuel_total_scaled.gt(0)
        &
        nonfuel_baskets.eq(0)
    )

    if fuel_positive_without_purchase.any():

        raise AssertionError(
            "Positive fuel target exists with zero "
            "positive fuel purchase baskets."
        )

    if nonfuel_positive_without_purchase.any():

        raise AssertionError(
            "Positive non-fuel target exists with zero "
            "positive non-fuel purchase baskets."
        )

    # ============================================================
    # AUXILIARY COMPONENTS
    # ============================================================

    fuel_litres = (
        fuel_total_scaled
        /
        SCALE
    )

    nonfuel_rand = (
        nonfuel_total_scaled
        /
        SCALE
    )

    fuel_per_basket = pd.Series(
        0.0,
        index=ids,
    )

    fuel_mask = (
        fuel_baskets.gt(0)
    )

    fuel_per_basket.loc[
        fuel_mask
    ] = (
        fuel_litres.loc[
            fuel_mask
        ]
        /
        fuel_baskets.loc[
            fuel_mask
        ]
    )

    nonfuel_per_basket = pd.Series(
        0.0,
        index=ids,
    )

    nonfuel_mask = (
        nonfuel_baskets.gt(0)
    )

    nonfuel_per_basket.loc[
        nonfuel_mask
    ] = (
        nonfuel_rand.loc[
            nonfuel_mask
        ]
        /
        nonfuel_baskets.loc[
            nonfuel_mask
        ]
    )

    result = pd.DataFrame(
        {
            "ID":
                ids,

            "cutoff":
                cutoff_string,

            "future_fuel_purchase_active":
                fuel_active
                .astype(
                    "int8"
                )
                .to_numpy(),

            "future_fuel_basket_count":
                fuel_baskets.to_numpy(),

            "future_fuel_litres":
                fuel_litres.to_numpy(),

            "future_fuel_litres_per_basket":
                fuel_per_basket.to_numpy(),

            "future_nonfuel_purchase_active":
                nonfuel_active
                .astype(
                    "int8"
                )
                .to_numpy(),

            "future_nonfuel_basket_count":
                nonfuel_baskets.to_numpy(),

            "future_nonfuel_rand":
                nonfuel_rand.to_numpy(),

            "future_nonfuel_rand_per_basket":
                nonfuel_per_basket.to_numpy(),
        }
    )

    # ============================================================
    # RECONSTRUCTION CHECK
    # ============================================================

    reconstructed_fuel = (
        result[
            "future_fuel_basket_count"
        ]
        *
        result[
            "future_fuel_litres_per_basket"
        ]
    )

    reconstructed_nonfuel = (
        result[
            "future_nonfuel_basket_count"
        ]
        *
        result[
            "future_nonfuel_rand_per_basket"
        ]
    )

    if not np.allclose(
        reconstructed_fuel,
        result[
            "future_fuel_litres"
        ],
        atol=1e-10,
    ):

        raise AssertionError(
            "Fuel decomposition reconstruction failed."
        )

    if not np.allclose(
        reconstructed_nonfuel,
        result[
            "future_nonfuel_rand"
        ],
        atol=1e-10,
    ):

        raise AssertionError(
            "Non-fuel decomposition reconstruction failed."
        )

    # ------------------------------------------------------------
    # Diagnostic edge cases
    # ------------------------------------------------------------

    fuel_active_zero_value = int(
        (
            result[
                "future_fuel_purchase_active"
            ].eq(1)
            &
            result[
                "future_fuel_litres"
            ].eq(0)
        ).sum()
    )

    nonfuel_active_zero_value = int(
        (
            result[
                "future_nonfuel_purchase_active"
            ].eq(1)
            &
            result[
                "future_nonfuel_rand"
            ].eq(0)
        ).sum()
    )

    summary = {
        "cutoff":
            cutoff_string,

        "customers":
            len(
                result
            ),

        "fuel_active_rate":
            result[
                "future_fuel_purchase_active"
            ].mean(),

        "fuel_positive_target_rate":
            result[
                "future_fuel_litres"
            ].gt(0).mean(),

        "fuel_mean_baskets_active":
            result.loc[
                result[
                    "future_fuel_purchase_active"
                ].eq(1),
                "future_fuel_basket_count",
            ].mean(),

        "fuel_mean_litres_per_basket_active":
            result.loc[
                result[
                    "future_fuel_purchase_active"
                ].eq(1),
                "future_fuel_litres_per_basket",
            ].mean(),

        "fuel_active_zero_value":
            fuel_active_zero_value,

        "nonfuel_active_rate":
            result[
                "future_nonfuel_purchase_active"
            ].mean(),

        "nonfuel_positive_target_rate":
            result[
                "future_nonfuel_rand"
            ].gt(0).mean(),

        "nonfuel_mean_baskets_active":
            result.loc[
                result[
                    "future_nonfuel_purchase_active"
                ].eq(1),
                "future_nonfuel_basket_count",
            ].mean(),

        "nonfuel_mean_rand_per_basket_active":
            result.loc[
                result[
                    "future_nonfuel_purchase_active"
                ].eq(1),
                "future_nonfuel_rand_per_basket",
            ].mean(),

        "nonfuel_active_zero_value":
            nonfuel_active_zero_value,

        "official_fuel_exact_match":
            1,

        "official_nonfuel_exact_match":
            1,

        "decomposition_reconstruction_pass":
            1,
    }

    return result, summary


def main():

    print(
        "Loading official Sasol configuration..."
    )

    config = json.loads(
        CONFIG_PATH.read_text()
    )

    print(
        "Loading transactions with official reference loader..."
    )

    transactions = (
        load_transactions(
            TRAIN_PATH
        )
    )

    all_results = []
    summaries = []

    for cutoff in config[
        "calibration_cutoffs"
    ]:

        print()
        print(
            "=" * 72
        )

        print(
            f"Building multi-stage targets: {cutoff}"
        )

        print(
            "=" * 72
        )

        result, summary = (
            build_snapshot(
                transactions,
                cutoff,
                config,
            )
        )

        output_path = (
            OUTPUT_DIR
            /
            f"multistage_targets_{cutoff}.csv"
        )

        result.to_csv(
            output_path,
            index=False,
            float_format="%.10f",
        )

        all_results.append(
            result
        )

        summaries.append(
            summary
        )

        print(
            f"Customers: {len(result):,}"
        )

        print(
            "Fuel active rate: "
            f"{summary['fuel_active_rate']:.4f}"
        )

        print(
            "Non-fuel active rate: "
            f"{summary['nonfuel_active_rate']:.4f}"
        )

        print(
            "Fuel active but zero net value: "
            f"{summary['fuel_active_zero_value']}"
        )

        print(
            "Non-fuel active but zero net value: "
            f"{summary['nonfuel_active_zero_value']}"
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    combined.to_csv(
        OUTPUT_DIR
        / "multistage_targets_all.csv",
        index=False,
        float_format="%.10f",
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "multistage_target_summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "PHASE 06R-B1 MULTI-STAGE TARGET AUDIT COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Total historical customer snapshots: "
        f"{len(combined):,}"
    )

    if len(
        combined
    ) != 22679:

        raise AssertionError(
            "Expected exactly 22,679 historical snapshots."
        )

    print()
    print(
        "Official fuel target agreement: PASS"
    )

    print(
        "Official non-fuel target agreement: PASS"
    )

    print(
        "Component reconstruction: PASS"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
