"""
Phase 03C: Momentum, previous-period comparison and regularity features.

Input:
    Phase 03B core customer feature matrix.

Adds:
- previous-quarter behaviour;
- quarter-on-quarter differences;
- relative value changes;
- latest-month acceleration;
- reactivation/dormancy states;
- purchase-gap regularity.

No category-specific modelling features and no machine-learning models
are created in this phase.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from reference.label_rules import (
    load_transactions,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT / "data" / "raw" / "train.csv"
)

LABEL_CONFIG_PATH = (
    ROOT / "configs" / "label_config.json"
)

CORE_FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase03_features"
    / "core_customer_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase03_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


QUARTER_METRICS = [
    "basket_count",
    "active_days",
    "qualifying_basket_count",
    "qualifying_active_days",
    "fuel_basket_count",
    "fuel_active_days",
    "fuel_litres",
    "nonfuel_basket_count",
    "nonfuel_active_days",
    "nonfuel_spend",
]


SHORT_TERM_METRICS = [
    "basket_count",
    "qualifying_basket_count",
    "fuel_basket_count",
    "fuel_litres",
    "nonfuel_basket_count",
    "nonfuel_spend",
]


def gap_statistics(
    baskets,
    customer_ids,
    prefix,
    include_median=True,
    include_std=True,
):
    """
    Calculate inter-basket gap statistics.

    baskets must contain:
        ID
        TransactionId
        _time

    One row per basket is enforced internally.
    """

    unique = (
        baskets[
            [
                "ID",
                "TransactionId",
                "_time",
            ]
        ]
        .drop_duplicates(
            [
                "ID",
                "TransactionId",
            ]
        )
        .sort_values(
            [
                "ID",
                "_time",
                "TransactionId",
            ]
        )
        .copy()
    )

    unique["_gap_days"] = (
        unique
        .groupby("ID")["_time"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    grouped = unique.groupby(
        "ID"
    )["_gap_days"]

    result = pd.DataFrame(
        index=pd.Index(
            customer_ids,
            name="ID",
        )
    )

    result[
        f"{prefix}_mean_days"
    ] = (
        grouped.mean()
        .reindex(customer_ids)
    )

    if include_median:

        result[
            f"{prefix}_median_days"
        ] = (
            grouped.median()
            .reindex(customer_ids)
        )

    if include_std:

        std = grouped.apply(
            lambda s:
                s.dropna().std(ddof=0)
                if s.notna().any()
                else np.nan
        )

        result[
            f"{prefix}_std_days"
        ] = std.reindex(
            customer_ids
        )

    return result


def safe_positive_ratio(
    numerator,
    denominator,
):
    """
    Divide only where denominator is strictly positive.

    Invalid denominators remain NaN.
    """

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )

    mask = (
        numerator.notna()
        & denominator.notna()
        & denominator.gt(0)
    )

    result.loc[mask] = (
        numerator.loc[mask]
        / denominator.loc[mask]
    )

    return result


def main():

    print(
        "Loading Phase 03 configuration..."
    )

    config = json.loads(
        LABEL_CONFIG_PATH.read_text()
    )

    print(
        "Loading Phase 03B core features..."
    )

    features = pd.read_csv(
        CORE_FEATURE_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    if features[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate Phase 03B snapshot IDs."
        )

    print(
        "Loading transaction history..."
    )

    transactions = load_transactions(
        TRAIN_PATH
    )

    mapping = config[
        "category_mapping"
    ]

    included_categories = sorted(
        category
        for category, settings
        in mapping.items()
        if settings["included"]
    )

    nonfuel_categories = [
        category
        for category
        in included_categories
        if category != "Fuel"
    ]

    adjustment_items = set(
        config["adjustment_items"]
    )

    output_frames = []
    audit_rows = []

    # ============================================================
    # PROCESS EACH FORECASTING ORIGIN
    # ============================================================

    for cutoff in sorted(
        features["cutoff"].unique()
    ):

        cutoff = pd.Timestamp(
            cutoff
        )

        print()
        print(
            f"Building Phase 03C features: "
            f"{cutoff.date()}"
        )

        frame = features[
            features["cutoff"].eq(
                cutoff
            )
        ].copy()

        customer_ids = (
            frame["ID"]
            .astype(str)
            .tolist()
        )

        # ========================================================
        # A. PREVIOUS QUARTER
        # ========================================================

        for metric in QUARTER_METRICS:

            frame[
                f"prev3_{metric}"
            ] = (
                frame[
                    f"m6_{metric}"
                ]
                -
                frame[
                    f"m3_{metric}"
                ]
            )

            frame[
                f"qoq_delta_{metric}"
            ] = (
                frame[
                    f"m3_{metric}"
                ]
                -
                frame[
                    f"prev3_{metric}"
                ]
            )

        # ========================================================
        # B. RELATIVE VALUE CHANGE
        # ========================================================

        for metric in [
            "fuel_litres",
            "nonfuel_spend",
        ]:

            recent = frame[
                f"m3_{metric}"
            ]

            previous = frame[
                f"prev3_{metric}"
            ]

            numerator = (
                recent - previous
            )

            frame[
                f"qoq_relative_{metric}"
            ] = safe_positive_ratio(
                numerator,
                previous,
            )

        # ========================================================
        # C. LATEST MONTH VS PRECEDING TWO MONTHS
        # ========================================================

        for metric in SHORT_TERM_METRICS:

            recent_month = frame[
                f"m1_{metric}"
            ]

            preceding_two_months = (
                frame[
                    f"m3_{metric}"
                ]
                -
                recent_month
            )

            preceding_two_month_avg = (
                preceding_two_months
                / 2.0
            )

            frame[
                f"m1_vs_prev2_monthly_{metric}"
            ] = (
                recent_month
                -
                preceding_two_month_avg
            )

        # ========================================================
        # D. REACTIVATION / DORMANCY
        # ========================================================

        state_streams = {
            "qualifying":
                "qualifying_basket_count",

            "fuel":
                "fuel_basket_count",

            "nonfuel":
                "nonfuel_basket_count",
        }

        for name, metric in state_streams.items():

            previous = frame[
                f"prev3_{metric}"
            ]

            recent = frame[
                f"m3_{metric}"
            ]

            frame[
                f"reactivated_{name}"
            ] = (
                previous.eq(0)
                & recent.gt(0)
            ).astype(
                "int8"
            )

            frame[
                f"became_dormant_{name}"
            ] = (
                previous.gt(0)
                & recent.eq(0)
            ).astype(
                "int8"
            )

        # ========================================================
        # E. PURCHASE REGULARITY
        # ========================================================

        history = transactions[
            transactions["_time"]
            < cutoff
        ].copy()

        clean = history[
            ~history["ItemName"].isin(
                adjustment_items
            )
        ]

        all_baskets = history[
            [
                "ID",
                "TransactionId",
                "_time",
            ]
        ]

        fuel_positive = clean[
            clean[
                "ItemCategoryLevel2"
            ].eq("Fuel")
            & clean[
                "LineItemType"
            ].eq("Sale_Fuel")
            & clean[
                "_quantity"
            ].gt(0)
            & clean[
                "_amount"
            ].gt(0)
        ][
            [
                "ID",
                "TransactionId",
                "_time",
            ]
        ]

        nonfuel_positive = clean[
            clean[
                "ItemCategoryLevel2"
            ].isin(
                nonfuel_categories
            )
            & clean[
                "_quantity"
            ].gt(0)
            & clean[
                "_amount"
            ].gt(0)
        ][
            [
                "ID",
                "TransactionId",
                "_time",
            ]
        ]

        six_month_start = (
            cutoff
            - pd.DateOffset(
                months=6
            )
        )

        all_baskets_m6 = (
            all_baskets[
                all_baskets["_time"]
                >= six_month_start
            ]
        )

        fuel_positive_m6 = (
            fuel_positive[
                fuel_positive["_time"]
                >= six_month_start
            ]
        )

        nonfuel_positive_m6 = (
            nonfuel_positive[
                nonfuel_positive["_time"]
                >= six_month_start
            ]
        )

        regularity = pd.DataFrame(
            index=pd.Index(
                customer_ids,
                name="ID",
            )
        )

        # General basket cadence
        general_lifetime = gap_statistics(
            all_baskets,
            customer_ids,
            prefix="lifetime_gap",
            include_median=True,
            include_std=True,
        )

        general_m6 = gap_statistics(
            all_baskets_m6,
            customer_ids,
            prefix="m6_gap",
            include_median=True,
            include_std=True,
        )

        # Fuel cadence
        fuel_lifetime = gap_statistics(
            fuel_positive,
            customer_ids,
            prefix="lifetime_fuel_gap",
            include_median=False,
            include_std=False,
        )

        fuel_m6 = gap_statistics(
            fuel_positive_m6,
            customer_ids,
            prefix="m6_fuel_gap",
            include_median=False,
            include_std=False,
        )

        # Non-fuel cadence
        nonfuel_lifetime = gap_statistics(
            nonfuel_positive,
            customer_ids,
            prefix="lifetime_nonfuel_gap",
            include_median=False,
            include_std=False,
        )

        nonfuel_m6 = gap_statistics(
            nonfuel_positive_m6,
            customer_ids,
            prefix="m6_nonfuel_gap",
            include_median=False,
            include_std=False,
        )

        for block in [
            general_lifetime,
            general_m6,
            fuel_lifetime,
            fuel_m6,
            nonfuel_lifetime,
            nonfuel_m6,
        ]:

            regularity = regularity.join(
                block,
                how="left",
            )

        regularity = (
            regularity
            .reset_index()
        )

        frame = frame.merge(
            regularity,
            on="ID",
            how="left",
            validate="one_to_one",
        )

        # ========================================================
        # F. RECENCY RELATIVE TO PERSONAL CADENCE
        # ========================================================

        frame[
            "basket_recency_to_lifetime_gap"
        ] = safe_positive_ratio(
            frame[
                "days_since_last_basket"
            ],
            frame[
                "lifetime_gap_mean_days"
            ],
        )

        frame[
            "fuel_recency_to_lifetime_gap"
        ] = safe_positive_ratio(
            frame[
                "days_since_last_fuel_purchase"
            ],
            frame[
                "lifetime_fuel_gap_mean_days"
            ],
        )

        frame[
            "nonfuel_recency_to_lifetime_gap"
        ] = safe_positive_ratio(
            frame[
                "days_since_last_nonfuel_purchase"
            ],
            frame[
                "lifetime_nonfuel_gap_mean_days"
            ],
        )

        # ========================================================
        # G. INTEGRITY CHECKS
        # ========================================================

        if frame[
            "snapshot_id"
        ].duplicated().any():

            raise AssertionError(
                "Duplicate snapshot IDs "
                "after Phase 03C."
            )

        # Previous-quarter count/day metrics cannot be negative.
        previous_count_columns = [
            c
            for c in frame.columns
            if (
                c.startswith("prev3_")
                and (
                    c.endswith("_count")
                    or c.endswith(
                        "_active_days"
                    )
                )
            )
        ]

        if (
            frame[
                previous_count_columns
            ] < 0
        ).any().any():

            raise AssertionError(
                "Negative previous-quarter "
                "count detected."
            )

        # Verify additive reconstruction:
        # m6 = prev3 + m3
        for metric in QUARTER_METRICS:

            lhs = frame[
                f"m6_{metric}"
            ].to_numpy(
                dtype="float64"
            )

            rhs = (
                frame[
                    f"prev3_{metric}"
                ]
                +
                frame[
                    f"m3_{metric}"
                ]
            ).to_numpy(
                dtype="float64"
            )

            if not np.allclose(
                lhs,
                rhs,
                rtol=1e-10,
                atol=1e-8,
                equal_nan=True,
            ):

                raise AssertionError(
                    f"Quarter reconstruction "
                    f"failed for {metric}."
                )

        # Reactivated and dormant cannot both be true.
        for stream in [
            "qualifying",
            "fuel",
            "nonfuel",
        ]:

            conflict = (
                frame[
                    f"reactivated_{stream}"
                ].eq(1)
                &
                frame[
                    f"became_dormant_{stream}"
                ].eq(1)
            )

            if conflict.any():

                raise AssertionError(
                    f"State conflict for "
                    f"{stream}."
                )

        gap_columns = [
            c
            for c in frame.columns
            if (
                "_gap_" in c
                and c.endswith("_days")
            )
        ]

        for column in gap_columns:

            negative = (
                frame[column]
                .dropna()
                .lt(0)
            )

            if negative.any():

                raise AssertionError(
                    f"Negative gap detected: "
                    f"{column}"
                )

        metadata_columns = {
            "snapshot_id",
            "ID",
            "cutoff",
            "snapshot_role",
            "is_labeled",
        }

        predictive_columns = [
            c
            for c in frame.columns
            if c not in metadata_columns
        ]

        numeric = frame[
            predictive_columns
        ].select_dtypes(
            include=[np.number]
        )

        infinite_values = int(
            np.isinf(
                numeric.to_numpy()
            ).sum()
        )

        if infinite_values != 0:

            raise AssertionError(
                "Infinite Phase 03C "
                "feature value detected."
            )

        audit_rows.append(
            {
                "cutoff":
                    cutoff.date().isoformat(),

                "rows":
                    len(frame),

                "predictive_features":
                    len(
                        predictive_columns
                    ),

                "duplicate_snapshot_ids":
                    int(
                        frame[
                            "snapshot_id"
                        ].duplicated().sum()
                    ),

                "infinite_values":
                    infinite_values,

                "reactivated_qualifying":
                    int(
                        frame[
                            "reactivated_qualifying"
                        ].sum()
                    ),

                "became_dormant_qualifying":
                    int(
                        frame[
                            "became_dormant_qualifying"
                        ].sum()
                    ),

                "reactivated_fuel":
                    int(
                        frame[
                            "reactivated_fuel"
                        ].sum()
                    ),

                "became_dormant_fuel":
                    int(
                        frame[
                            "became_dormant_fuel"
                        ].sum()
                    ),

                "reactivated_nonfuel":
                    int(
                        frame[
                            "reactivated_nonfuel"
                        ].sum()
                    ),

                "became_dormant_nonfuel":
                    int(
                        frame[
                            "became_dormant_nonfuel"
                        ].sum()
                    ),

                "missing_lifetime_gap":
                    int(
                        frame[
                            "lifetime_gap_mean_days"
                        ].isna().sum()
                    ),

                "missing_lifetime_fuel_gap":
                    int(
                        frame[
                            "lifetime_fuel_gap_mean_days"
                        ].isna().sum()
                    ),

                "missing_lifetime_nonfuel_gap":
                    int(
                        frame[
                            "lifetime_nonfuel_gap_mean_days"
                        ].isna().sum()
                    ),

                "validation_status":
                    "PASS",
            }
        )

        output_frames.append(
            frame
        )

        print(
            f"Rows: {len(frame):,}"
        )

        print(
            f"Predictive features: "
            f"{len(predictive_columns)}"
        )

    # ============================================================
    # COMBINE
    # ============================================================

    result = pd.concat(
        output_frames,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "cutoff",
            "ID",
        ]
    ).reset_index(
        drop=True
    )

    if len(result) != len(features):

        raise AssertionError(
            "Phase 03C changed total "
            "snapshot row count."
        )

    if set(
        result["snapshot_id"]
    ) != set(
        features["snapshot_id"]
    ):

        raise AssertionError(
            "Phase 03C snapshot population "
            "does not match Phase 03B."
        )

    metadata_columns = {
        "snapshot_id",
        "ID",
        "cutoff",
        "snapshot_role",
        "is_labeled",
    }

    predictive_columns = [
        c
        for c in result.columns
        if c not in metadata_columns
    ]

    EXPECTED_FEATURE_COUNT = 104

    if len(
        predictive_columns
    ) != EXPECTED_FEATURE_COUNT:

        raise AssertionError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} "
            f"predictive features but "
            f"found "
            f"{len(predictive_columns)}."
        )

    # ============================================================
    # SAVE
    # ============================================================

    result.to_csv(
        OUTPUT_DIR
        / "temporal_customer_features.csv",
        index=False,
        float_format="%.10g",
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "phase03c_feature_audit.csv",
        index=False,
    )

    # ============================================================
    # CHECKPOINT
    # ============================================================

    print()
    print(
        "=" * 72
    )

    print(
        "PHASE 03C MOMENTUM AND REGULARITY BUILD COMPLETE"
    )

    print(
        "=" * 72
    )

    print()

    print(
        audit.to_string(
            index=False
        )
    )

    print()

    print(
        f"Total snapshot rows: "
        f"{len(result):,}"
    )

    print(
        f"Predictive features: "
        f"{len(predictive_columns)}"
    )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
        / "temporal_customer_features.csv"
    )


if __name__ == "__main__":
    main()
