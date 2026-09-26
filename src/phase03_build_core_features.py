"""
Phase 03B: Build core temporal customer behavioural features.

This script creates the first predictive variables for every
Customer x Forecasting Origin observation.

Feature families:
- observed lifecycle;
- recency;
- basket frequency;
- active days;
- qualifying activity;
- fuel activity/value;
- non-fuel activity/value.

It deliberately does NOT create:
- momentum ratios;
- slopes;
- purchase regularity;
- category-specific features;
- machine-learning models.

All features use transaction information strictly before each cutoff.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from reference.label_rules import (
    load_transactions,
    SCALE,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

LABEL_CONFIG_PATH = (
    ROOT / "configs" / "label_config.json"
)

FEATURE_CONFIG_PATH = (
    ROOT / "configs" / "feature_config.json"
)

SNAPSHOT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase03_snapshots"
    / "all_snapshot_index.csv"
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


def aggregate_window(
    history,
    customer_ids,
    start_time,
    cutoff,
    prefix,
    included_categories,
    nonfuel_categories,
    adjustment_items,
):
    """
    Aggregate one historical window.

    Window definition:
        start_time <= transaction_time < cutoff

    For lifetime:
        start_time is None.
    """

    frame = history[
        history["_time"] < cutoff
    ]

    if start_time is not None:
        frame = frame[
            frame["_time"] >= start_time
        ]

    # ------------------------------------------------------------
    # ALL TRANSACTION ACTIVITY
    # ------------------------------------------------------------

    all_group = frame.groupby("ID")

    result = pd.DataFrame(
        index=pd.Index(
            customer_ids,
            name="ID",
        )
    )

    result[
        f"{prefix}_basket_count"
    ] = (
        all_group["TransactionId"]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    result[
        f"{prefix}_active_days"
    ] = (
        all_group["_date"]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    # ------------------------------------------------------------
    # REMOVE OFFICIAL ADJUSTMENT ITEMS
    # ------------------------------------------------------------

    clean = frame[
        ~frame["ItemName"].isin(
            adjustment_items
        )
    ]

    # ------------------------------------------------------------
    # QUALIFYING POSITIVE PURCHASE ACTIVITY
    # ------------------------------------------------------------

    qualifying_positive = clean[
        clean[
            "ItemCategoryLevel2"
        ].isin(
            included_categories
        )
        & clean["_quantity"].gt(0)
        & clean["_amount"].gt(0)
    ]

    q_group = qualifying_positive.groupby(
        "ID"
    )

    result[
        f"{prefix}_qualifying_basket_count"
    ] = (
        q_group["TransactionId"]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    result[
        f"{prefix}_qualifying_active_days"
    ] = (
        q_group["_date"]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    # ------------------------------------------------------------
    # FUEL
    # ------------------------------------------------------------

    fuel = clean[
        clean[
            "ItemCategoryLevel2"
        ].eq("Fuel")
        & clean["LineItemType"].eq(
            "Sale_Fuel"
        )
    ]

    fuel_positive = fuel[
        fuel["_quantity"].gt(0)
        & fuel["_amount"].gt(0)
    ]

    fuel_positive_group = (
        fuel_positive.groupby("ID")
    )

    result[
        f"{prefix}_fuel_basket_count"
    ] = (
        fuel_positive_group[
            "TransactionId"
        ]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    result[
        f"{prefix}_fuel_active_days"
    ] = (
        fuel_positive_group["_date"]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    fuel_litres = (
        fuel.groupby("ID")["_quantity"]
        .sum()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    result[
        f"{prefix}_fuel_litres"
    ] = (
        fuel_litres.astype("float64")
        / SCALE
    )

    # ------------------------------------------------------------
    # NON-FUEL
    # ------------------------------------------------------------

    nonfuel = clean[
        clean[
            "ItemCategoryLevel2"
        ].isin(
            nonfuel_categories
        )
    ]

    nonfuel_positive = nonfuel[
        nonfuel["_quantity"].gt(0)
        & nonfuel["_amount"].gt(0)
    ]

    nonfuel_positive_group = (
        nonfuel_positive.groupby("ID")
    )

    result[
        f"{prefix}_nonfuel_basket_count"
    ] = (
        nonfuel_positive_group[
            "TransactionId"
        ]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    result[
        f"{prefix}_nonfuel_active_days"
    ] = (
        nonfuel_positive_group["_date"]
        .nunique()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    nonfuel_spend = (
        nonfuel.groupby("ID")["_amount"]
        .sum()
        .reindex(
            customer_ids,
            fill_value=0,
        )
    )

    result[
        f"{prefix}_nonfuel_spend"
    ] = (
        nonfuel_spend.astype(
            "float64"
        )
        / SCALE
    )

    # Counts should remain integers.
    count_columns = [
        c
        for c in result.columns
        if (
            c.endswith("_count")
            or c.endswith(
                "_active_days"
            )
        )
    ]

    result[
        count_columns
    ] = result[
        count_columns
    ].astype(
        "int64"
    )

    return result.reset_index()


def build_recency_features(
    history,
    customer_ids,
    cutoff,
    included_categories,
    nonfuel_categories,
    adjustment_items,
):
    """
    Build cutoff-level lifecycle and recency features.

    Missing fuel/non-fuel recency is meaningful:
    the customer has never exhibited that activity
    within the released history before the cutoff.
    """

    history = history[
        history["_time"] < cutoff
    ]

    clean = history[
        ~history["ItemName"].isin(
            adjustment_items
        )
    ]

    result = pd.DataFrame(
        {
            "ID": pd.Series(
                customer_ids,
                dtype="string",
            )
        }
    ).set_index("ID")

    # ------------------------------------------------------------
    # OBSERVED LIFECYCLE
    # ------------------------------------------------------------

    first_any = (
        history.groupby("ID")["_time"]
        .min()
        .reindex(customer_ids)
    )

    last_any = (
        history.groupby("ID")["_time"]
        .max()
        .reindex(customer_ids)
    )

    result[
        "observed_tenure_days"
    ] = (
        (
            cutoff - first_any
        )
        .dt.total_seconds()
        / 86400
    )

    result[
        "days_since_last_basket"
    ] = (
        (
            cutoff - last_any
        )
        .dt.total_seconds()
        / 86400
    )

    # ------------------------------------------------------------
    # QUALIFYING PURCHASE RECENCY
    # ------------------------------------------------------------

    qualifying_positive = clean[
        clean[
            "ItemCategoryLevel2"
        ].isin(
            included_categories
        )
        & clean["_quantity"].gt(0)
        & clean["_amount"].gt(0)
    ]

    last_qualifying = (
        qualifying_positive
        .groupby("ID")["_time"]
        .max()
        .reindex(customer_ids)
    )

    result[
        "days_since_last_qualifying_purchase"
    ] = (
        (
            cutoff - last_qualifying
        )
        .dt.total_seconds()
        / 86400
    )

    # ------------------------------------------------------------
    # FUEL RECENCY
    # ------------------------------------------------------------

    fuel_positive = clean[
        clean[
            "ItemCategoryLevel2"
        ].eq("Fuel")
        & clean["LineItemType"].eq(
            "Sale_Fuel"
        )
        & clean["_quantity"].gt(0)
        & clean["_amount"].gt(0)
    ]

    last_fuel = (
        fuel_positive
        .groupby("ID")["_time"]
        .max()
        .reindex(customer_ids)
    )

    result[
        "days_since_last_fuel_purchase"
    ] = (
        (
            cutoff - last_fuel
        )
        .dt.total_seconds()
        / 86400
    )

    result[
        "has_ever_fuel"
    ] = (
        last_fuel.notna()
        .astype("int8")
    )

    # ------------------------------------------------------------
    # NON-FUEL RECENCY
    # ------------------------------------------------------------

    nonfuel_positive = clean[
        clean[
            "ItemCategoryLevel2"
        ].isin(
            nonfuel_categories
        )
        & clean["_quantity"].gt(0)
        & clean["_amount"].gt(0)
    ]

    last_nonfuel = (
        nonfuel_positive
        .groupby("ID")["_time"]
        .max()
        .reindex(customer_ids)
    )

    result[
        "days_since_last_nonfuel_purchase"
    ] = (
        (
            cutoff - last_nonfuel
        )
        .dt.total_seconds()
        / 86400
    )

    result[
        "has_ever_nonfuel"
    ] = (
        last_nonfuel.notna()
        .astype("int8")
    )

    # ------------------------------------------------------------
    # INTEGRITY
    # ------------------------------------------------------------

    if result[
        "observed_tenure_days"
    ].isna().any():

        raise AssertionError(
            "Eligible customer has no "
            "historical transaction."
        )

    if result[
        "days_since_last_basket"
    ].isna().any():

        raise AssertionError(
            "Eligible customer has no "
            "last historical basket."
        )

    return result.reset_index()


def main():

    print(
        "Loading Phase 03 configuration..."
    )

    label_config = json.loads(
        LABEL_CONFIG_PATH.read_text()
    )

    feature_config = json.loads(
        FEATURE_CONFIG_PATH.read_text()
    )

    windows = feature_config[
        "history_windows_months"
    ]

    print(
        "Loading transaction history..."
    )

    transactions = load_transactions(
        TRAIN_PATH
    )

    transactions["_date"] = (
        transactions["_time"].dt.normalize()
    )

    print(
        "Loading authoritative snapshot index..."
    )

    snapshots = pd.read_csv(
        SNAPSHOT_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    if snapshots[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Snapshot index contains "
            "duplicate snapshot IDs."
        )

    mapping = label_config[
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
        label_config[
            "adjustment_items"
        ]
    )

    feature_frames = []
    audit_rows = []

    # ============================================================
    # BUILD EACH FORECASTING ORIGIN
    # ============================================================

    for cutoff in sorted(
        snapshots["cutoff"].unique()
    ):

        cutoff = pd.Timestamp(
            cutoff
        )

        print()
        print(
            f"Building core features: "
            f"{cutoff.date()}"
        )

        snapshot = snapshots[
            snapshots["cutoff"].eq(
                cutoff
            )
        ].copy()

        customer_ids = (
            snapshot["ID"]
            .astype(str)
            .tolist()
        )

        # Strictly pre-cutoff history only.
        history = transactions[
            transactions["_time"]
            < cutoff
        ]

        base = snapshot[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "snapshot_role",
                "is_labeled",
            ]
        ].copy()

        # --------------------------------------------------------
        # LIFECYCLE + RECENCY
        # --------------------------------------------------------

        recency = build_recency_features(
            history=history,
            customer_ids=customer_ids,
            cutoff=cutoff,
            included_categories=
                included_categories,
            nonfuel_categories=
                nonfuel_categories,
            adjustment_items=
                adjustment_items,
        )

        base = base.merge(
            recency,
            on="ID",
            how="left",
            validate="one_to_one",
        )

        # --------------------------------------------------------
        # FIXED CALENDAR WINDOWS
        # --------------------------------------------------------

        for prefix, months in windows.items():

            start_time = (
                cutoff
                - pd.DateOffset(
                    months=months
                )
            )

            window_features = (
                aggregate_window(
                    history=history,
                    customer_ids=
                        customer_ids,
                    start_time=start_time,
                    cutoff=cutoff,
                    prefix=prefix,
                    included_categories=
                        included_categories,
                    nonfuel_categories=
                        nonfuel_categories,
                    adjustment_items=
                        adjustment_items,
                )
            )

            base = base.merge(
                window_features,
                on="ID",
                how="left",
                validate="one_to_one",
            )

        # --------------------------------------------------------
        # LIFETIME WINDOW
        # --------------------------------------------------------

        lifetime = aggregate_window(
            history=history,
            customer_ids=customer_ids,
            start_time=None,
            cutoff=cutoff,
            prefix="lifetime",
            included_categories=
                included_categories,
            nonfuel_categories=
                nonfuel_categories,
            adjustment_items=
                adjustment_items,
        )

        base = base.merge(
            lifetime,
            on="ID",
            how="left",
            validate="one_to_one",
        )

        # --------------------------------------------------------
        # CORE INTEGRITY CHECKS
        # --------------------------------------------------------

        if len(base) != len(snapshot):
            raise AssertionError(
                "Feature row count changed "
                "during merges."
            )

        if base[
            "snapshot_id"
        ].duplicated().any():

            raise AssertionError(
                "Duplicate snapshot IDs "
                "after feature creation."
            )

        count_cols = [
            c
            for c in base.columns
            if (
                c.endswith("_count")
                or c.endswith(
                    "_active_days"
                )
            )
        ]

        if (
            base[count_cols] < 0
        ).any().any():

            raise AssertionError(
                "Negative activity count "
                "detected."
            )

        if (
            base[
                "days_since_last_basket"
            ] < 0
        ).any():

            raise AssertionError(
                "Negative basket recency "
                "detected."
            )

        if (
            base[
                "observed_tenure_days"
            ] < 0
        ).any():

            raise AssertionError(
                "Negative observed tenure "
                "detected."
            )

        # Longer windows cannot contain
        # fewer all-basket observations
        # than shorter windows.
        for shorter, longer in [
            ("m1", "m3"),
            ("m3", "m6"),
            ("m6", "m12"),
            ("m12", "lifetime"),
        ]:

            if (
                base[
                    f"{shorter}_basket_count"
                ]
                >
                base[
                    f"{longer}_basket_count"
                ]
            ).any():

                raise AssertionError(
                    f"Window monotonicity "
                    f"failed: "
                    f"{shorter} > {longer}"
                )

        # Eligibility guarantees at least
        # three lifetime baskets.
        if (
            base[
                "lifetime_basket_count"
            ] < 3
        ).any():

            raise AssertionError(
                "Eligible snapshot contains "
                "fewer than 3 lifetime "
                "baskets."
            )

        # --------------------------------------------------------
        # AUDIT ROW
        # --------------------------------------------------------

        feature_columns = [
            c
            for c in base.columns
            if c not in {
                "snapshot_id",
                "ID",
                "cutoff",
                "snapshot_role",
                "is_labeled",
            }
        ]

        numeric = base[
            feature_columns
        ].select_dtypes(
            include=[np.number]
        )

        inf_count = int(
            np.isinf(
                numeric.to_numpy()
            ).sum()
        )

        audit_rows.append(
            {
                "cutoff":
                    cutoff.date().isoformat(),

                "snapshot_rows":
                    len(snapshot),

                "feature_rows":
                    len(base),

                "predictive_feature_count":
                    len(feature_columns),

                "duplicate_snapshot_ids":
                    int(
                        base[
                            "snapshot_id"
                        ].duplicated().sum()
                    ),

                "infinite_values":
                    inf_count,

                "missing_last_basket":
                    int(
                        base[
                            "days_since_last_basket"
                        ].isna().sum()
                    ),

                "missing_last_qualifying":
                    int(
                        base[
                            "days_since_last_qualifying_purchase"
                        ].isna().sum()
                    ),

                "never_fuel":
                    int(
                        base[
                            "has_ever_fuel"
                        ].eq(0).sum()
                    ),

                "never_nonfuel":
                    int(
                        base[
                            "has_ever_nonfuel"
                        ].eq(0).sum()
                    ),

                "negative_m1_fuel_net":
                    int(
                        base[
                            "m1_fuel_litres"
                        ].lt(0).sum()
                    ),

                "negative_m1_nonfuel_net":
                    int(
                        base[
                            "m1_nonfuel_spend"
                        ].lt(0).sum()
                    ),

                "validation_status":
                    "PASS",
            }
        )

        feature_frames.append(
            base
        )

        print(
            f"Snapshots: {len(base):,}"
        )

        print(
            f"Predictive features: "
            f"{len(feature_columns)}"
        )

    # ============================================================
    # COMBINE ALL FORECASTING ORIGINS
    # ============================================================

    features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    features = features.sort_values(
        [
            "cutoff",
            "ID",
        ]
    ).reset_index(
        drop=True
    )

    # Exact alignment with snapshot spine.
    expected_ids = set(
        snapshots[
            "snapshot_id"
        ]
    )

    actual_ids = set(
        features[
            "snapshot_id"
        ]
    )

    if actual_ids != expected_ids:
        raise AssertionError(
            "Feature matrix does not "
            "match snapshot index."
        )

    if len(features) != len(snapshots):
        raise AssertionError(
            "Feature matrix row count "
            "does not match snapshot "
            "index."
        )

    # ============================================================
    # SAVE
    # ============================================================

    feature_path = (
        OUTPUT_DIR
        / "core_customer_features.csv"
    )

    features.to_csv(
        feature_path,
        index=False,
        float_format="%.10g",
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "core_feature_audit.csv",
        index=False,
    )

    # ============================================================
    # FINAL CHECKPOINT
    # ============================================================

    metadata_columns = {
        "snapshot_id",
        "ID",
        "cutoff",
        "snapshot_role",
        "is_labeled",
    }

    predictive_columns = [
        c
        for c in features.columns
        if c not in metadata_columns
    ]

    print()
    print(
        "=" * 72
    )

    print(
        "PHASE 03B CORE FEATURE BUILD COMPLETE"
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
        f"{len(features):,}"
    )

    print(
        f"Predictive features: "
        f"{len(predictive_columns)}"
    )

    print()

    print(
        f"Output:\n{feature_path}"
    )


if __name__ == "__main__":
    main()
