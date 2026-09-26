"""
Phase 03D: Category state, maturity and reactivation features.

Adds original Level-2 category histories to the validated Phase 03C
temporal customer representation.

Original categories remain separate because Sasol determines the
winning Opportunity candidate before applying final category mapping.

No machine-learning model is trained here.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from reference.label_rules import (
    load_transactions,
    SCALE,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT / "data" / "raw" / "train.csv"
)

LABEL_CONFIG_PATH = (
    ROOT / "configs" / "label_config.json"
)

INPUT_FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase03_features"
    / "temporal_customer_features.csv"
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


def slugify_category(value):
    """
    Convert an original category name into a stable feature-safe slug.
    """

    slug = value.lower()

    slug = slug.replace(
        "&",
        " and ",
    )

    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        slug,
    )

    slug = slug.strip("_")

    if not slug:
        raise ValueError(
            f"Could not create category slug for {value!r}"
        )

    return slug


def purchase_count_matrix(
    frame,
    customer_ids,
    categories,
):
    """
    Distinct qualifying basket count by customer and original category.
    """

    if frame.empty:

        return pd.DataFrame(
            0,
            index=customer_ids,
            columns=categories,
            dtype="int64",
        )

    matrix = (
        frame
        .groupby(
            [
                "ID",
                "ItemCategoryLevel2",
            ]
        )["TransactionId"]
        .nunique()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=customer_ids,
            columns=categories,
            fill_value=0,
        )
        .astype(
            "int64"
        )
    )

    return matrix


def spend_matrix(
    frame,
    customer_ids,
    categories,
):
    """
    Signed source monetary amount by customer and original category.
    """

    if frame.empty:

        return pd.DataFrame(
            0.0,
            index=customer_ids,
            columns=categories,
            dtype="float64",
        )

    matrix = (
        frame
        .groupby(
            [
                "ID",
                "ItemCategoryLevel2",
            ]
        )["_amount"]
        .sum()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=customer_ids,
            columns=categories,
            fill_value=0,
        )
        .astype(
            "float64"
        )
        / SCALE
    )

    return matrix


def last_purchase_matrix(
    frame,
    customer_ids,
    categories,
):
    """
    Most recent qualifying positive purchase timestamp by category.
    """

    if frame.empty:

        return pd.DataFrame(
            pd.NaT,
            index=customer_ids,
            columns=categories,
        )

    matrix = (
        frame
        .groupby(
            [
                "ID",
                "ItemCategoryLevel2",
            ]
        )["_time"]
        .max()
        .unstack()
        .reindex(
            index=customer_ids,
            columns=categories,
        )
    )

    return matrix


def main():

    print(
        "Loading Phase 03 configuration..."
    )

    config = json.loads(
        LABEL_CONFIG_PATH.read_text()
    )

    print(
        "Loading Phase 03C feature matrix..."
    )

    features = pd.read_csv(
        INPUT_FEATURE_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    if features[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate Phase 03C snapshot IDs."
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

    categories = sorted(
        category
        for category, settings
        in mapping.items()
        if settings["included"]
    )

    if len(categories) != 24:

        raise AssertionError(
            f"Expected 24 included categories "
            f"but found {len(categories)}."
        )

    slugs = {
        category:
            slugify_category(category)
        for category in categories
    }

    if len(
        set(slugs.values())
    ) != len(categories):

        raise AssertionError(
            "Category slug collision detected."
        )

    category_map = pd.DataFrame(
        {
            "source_category":
                categories,
            "feature_slug":
                [
                    slugs[c]
                    for c in categories
                ],
        }
    )

    category_map.to_csv(
        OUTPUT_DIR
        / "category_feature_map.csv",
        index=False,
    )

    adjustment_items = set(
        config["adjustment_items"]
    )

    output_frames = []
    audit_rows = []

    # ============================================================
    # EACH FORECASTING ORIGIN
    # ============================================================

    for cutoff in sorted(
        features["cutoff"].unique()
    ):

        cutoff = pd.Timestamp(
            cutoff
        )

        print()
        print(
            f"Building category features: "
            f"{cutoff.date()}"
        )

        frame = features[
            features["cutoff"].eq(
                cutoff
            )
        ].copy()

        customer_ids = pd.Index(
            frame["ID"].astype(str),
            name="ID",
        )

        recent_start = (
            cutoff
            - pd.DateOffset(
                months=3
            )
        )

        previous_start = (
            cutoff
            - pd.DateOffset(
                months=6
            )
        )

        # Strictly historical data only.
        history = transactions[
            transactions["_time"]
            < cutoff
        ].copy()

        clean = history[
            ~history["ItemName"].isin(
                adjustment_items
            )
            &
            history[
                "ItemCategoryLevel2"
            ].isin(
                categories
            )
        ].copy()

        # Positive qualifying purchases determine
        # exposure, basket counts and recency.
        positive = clean[
            clean["_quantity"].gt(0)
            & clean["_amount"].gt(0)
        ].copy()

        before_m3_positive = positive[
            positive["_time"]
            < recent_start
        ]

        m3_positive = positive[
            positive["_time"].ge(
                recent_start
            )
        ]

        prev3_positive = positive[
            positive["_time"].ge(
                previous_start
            )
            &
            positive["_time"].lt(
                recent_start
            )
        ]

        # Signed spend windows retain returns.
        m3_spend_source = clean[
            clean["_time"].ge(
                recent_start
            )
        ]

        prev3_spend_source = clean[
            clean["_time"].ge(
                previous_start
            )
            &
            clean["_time"].lt(
                recent_start
            )
        ]

        # --------------------------------------------------------
        # CATEGORY MATRICES
        # --------------------------------------------------------

        lifetime_count = (
            purchase_count_matrix(
                positive,
                customer_ids,
                categories,
            )
        )

        pre_m3_count = (
            purchase_count_matrix(
                before_m3_positive,
                customer_ids,
                categories,
            )
        )

        m3_count = (
            purchase_count_matrix(
                m3_positive,
                customer_ids,
                categories,
            )
        )

        prev3_count = (
            purchase_count_matrix(
                prev3_positive,
                customer_ids,
                categories,
            )
        )

        m3_spend = spend_matrix(
            m3_spend_source,
            customer_ids,
            categories,
        )

        prev3_spend = spend_matrix(
            prev3_spend_source,
            customer_ids,
            categories,
        )

        last_purchase = (
            last_purchase_matrix(
                positive,
                customer_ids,
                categories,
            )
        )

        ever = lifetime_count.gt(0)
        seen_before_m3 = (
            pre_m3_count.gt(0)
        )

        recent_active = (
            m3_count.gt(0)
        )

        previous_active = (
            prev3_count.gt(0)
        )

        # --------------------------------------------------------
        # ADD PER-CATEGORY FEATURES
        # --------------------------------------------------------
        # --------------------------------------------------------
        # ADD PER-CATEGORY FEATURES
        # --------------------------------------------------------

        frame = frame.set_index(
            "ID",
            drop=False,
        )

        category_feature_data = {}

        for category in categories:

            slug = slugs[category]

            category_feature_data[
                f"cat__{slug}__ever_purchased"
            ] = (
                ever[category]
                .astype("int8")
            )

            category_feature_data[
                f"cat__{slug}__seen_before_m3"
            ] = (
                seen_before_m3[category]
                .astype("int8")
            )

            category_feature_data[
                f"cat__{slug}__m3_basket_count"
            ] = (
                m3_count[category]
                .astype("int64")
            )

            category_feature_data[
                f"cat__{slug}__prev3_basket_count"
            ] = (
                prev3_count[category]
                .astype("int64")
            )

            category_feature_data[
                f"cat__{slug}__m3_net_spend"
            ] = (
                m3_spend[category]
                .astype("float64")
            )

            category_feature_data[
                f"cat__{slug}__prev3_net_spend"
            ] = (
                prev3_spend[category]
                .astype("float64")
            )

            recency = (
                cutoff
                - last_purchase[category]
            )

            category_feature_data[
                f"cat__{slug}__days_since_last_purchase"
            ] = (
                recency.dt.total_seconds()
                / 86400.0
            )

        category_feature_block = pd.DataFrame(
            category_feature_data,
            index=customer_ids,
        )

        frame = pd.concat(
            [
                frame,
                category_feature_block,
            ],
            axis=1,
        )

        # --------------------------------------------------------
        # CATEGORY MATURITY SUMMARIES
        # --------------------------------------------------------

        dormant_known = (
            ever
            & ~recent_active
        )

        recent_new_to_history = (
            ~seen_before_m3
            & recent_active
        )

        recent_reactivated = (
            seen_before_m3
            & ~previous_active
            & recent_active
        )

        recent_became_dormant = (
            previous_active
            & ~recent_active
        )

        summary_feature_block = pd.DataFrame(
            {
                "lifetime_category_breadth":
                    ever.sum(axis=1).astype("int64"),

                "pre_m3_category_breadth":
                    seen_before_m3.sum(axis=1)
                    .astype("int64"),

                "m3_category_breadth":
                    recent_active.sum(axis=1)
                    .astype("int64"),

                "prev3_category_breadth":
                    previous_active.sum(axis=1)
                    .astype("int64"),

                "unseen_category_count":
                    (
                        len(categories)
                        - ever.sum(axis=1)
                    ).astype("int64"),

                "dormant_known_category_count":
                    dormant_known.sum(axis=1)
                    .astype("int64"),

                "recent_new_to_history_category_count":
                    recent_new_to_history.sum(axis=1)
                    .astype("int64"),

                "recent_reactivated_category_count":
                    recent_reactivated.sum(axis=1)
                    .astype("int64"),

                "recent_became_dormant_category_count":
                    recent_became_dormant.sum(axis=1)
                    .astype("int64"),
            },
            index=customer_ids,
        )

        frame = pd.concat(
            [
                frame,
                summary_feature_block,
            ],
            axis=1,
        )

        # One defragmented copy after bulk construction.
        frame = frame.copy()
 # --------------------------------------------------------
        # INTEGRITY CHECKS
        # --------------------------------------------------------

        # Recent positive purchase must imply ever-purchased.
        if (
            recent_active
            &
            ~ever
        ).any().any():

            raise AssertionError(
                "Recent category purchase "
                "without lifetime exposure."
            )

        # Pre-m3 exposure must imply lifetime exposure.
        if (
            seen_before_m3
            &
            ~ever
        ).any().any():

            raise AssertionError(
                "Pre-m3 category exposure "
                "without lifetime exposure."
            )

        # Recency missingness must exactly match never purchased.
        for category in categories:

            slug = slugs[
                category
            ]

            recency_missing = frame[
                f"cat__{slug}__days_since_last_purchase"
            ].isna()

            never = (
                frame[
                    f"cat__{slug}__ever_purchased"
                ]
                .eq(0)
            )

            if not recency_missing.equals(
                never
            ):

                raise AssertionError(
                    f"Category recency missingness "
                    f"mismatch: {category}"
                )

        # Breadth must equal per-category ever indicators.
        ever_feature_columns = [
            f"cat__{slugs[c]}__ever_purchased"
            for c in categories
        ]

        breadth_check = (
            frame[
                ever_feature_columns
            ].sum(axis=1)
        )

        if not (
            breadth_check
            ==
            frame[
                "lifetime_category_breadth"
            ]
        ).all():

            raise AssertionError(
                "Lifetime category breadth "
                "reconciliation failed."
            )

        if not (
            frame[
                "unseen_category_count"
            ]
            +
            frame[
                "lifetime_category_breadth"
            ]
            ==
            len(categories)
        ).all():

            raise AssertionError(
                "Unseen category count "
                "reconciliation failed."
            )

        count_columns = [
            c
            for c in frame.columns
            if (
                c.startswith("cat__")
                and c.endswith(
                    "_basket_count"
                )
            )
        ]

        if (
            frame[
                count_columns
            ] < 0
        ).any().any():

            raise AssertionError(
                "Negative category basket "
                "count detected."
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
                "Infinite category feature "
                "value detected."
            )

        recency_columns = [
            f"cat__{slugs[c]}__days_since_last_purchase"
            for c in categories
        ]

        missing_category_recencies = int(
            frame[
                recency_columns
            ]
            .isna()
            .sum()
            .sum()
        )

        unseen_slots = int(
            frame[
                "unseen_category_count"
            ]
            .sum()
        )

        if (
            missing_category_recencies
            != unseen_slots
        ):

            raise AssertionError(
                "Missing category recency "
                "count does not equal unseen "
                "category slots."
            )

        # --------------------------------------------------------
        # AUDIT
        # --------------------------------------------------------

        audit_rows.append(
            {
                "cutoff":
                    cutoff.date().isoformat(),

                "rows":
                    len(frame),

                "included_categories":
                    len(categories),

                "predictive_features":
                    len(
                        predictive_columns
                    ),

                "duplicate_snapshot_ids":
                    int(
                        frame[
                            "snapshot_id"
                        ]
                        .duplicated()
                        .sum()
                    ),

                "infinite_values":
                    infinite_values,

                "mean_lifetime_category_breadth":
                    frame[
                        "lifetime_category_breadth"
                    ].mean(),

                "median_lifetime_category_breadth":
                    frame[
                        "lifetime_category_breadth"
                    ].median(),

                "mean_unseen_category_count":
                    frame[
                        "unseen_category_count"
                    ].mean(),

                "customers_with_recent_new_category":
                    int(
                        frame[
                            "recent_new_to_history_category_count"
                        ].gt(0).sum()
                    ),

                "recent_new_category_events":
                    int(
                        frame[
                            "recent_new_to_history_category_count"
                        ].sum()
                    ),

                "customers_with_recent_category_reactivation":
                    int(
                        frame[
                            "recent_reactivated_category_count"
                        ].gt(0).sum()
                    ),

                "recent_category_reactivation_events":
                    int(
                        frame[
                            "recent_reactivated_category_count"
                        ].sum()
                    ),

                "customers_with_recent_category_dormancy":
                    int(
                        frame[
                            "recent_became_dormant_category_count"
                        ].gt(0).sum()
                    ),

                "recent_category_dormancy_events":
                    int(
                        frame[
                            "recent_became_dormant_category_count"
                        ].sum()
                    ),

                "missing_category_recencies":
                    missing_category_recencies,

                "validation_status":
                    "PASS",
            }
        )

        frame = frame.reset_index(
            drop=True
        )

        output_frames.append(
            frame
        )

        print(
            f"Rows: {len(frame):,}"
        )

        print(
            f"Included categories: "
            f"{len(categories)}"
        )

        print(
            f"Predictive features: "
            f"{len(predictive_columns)}"
        )

    # ============================================================
    # COMBINE ALL SNAPSHOTS
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
            "Phase 03D changed snapshot "
            "row count."
        )

    if set(
        result["snapshot_id"]
    ) != set(
        features["snapshot_id"]
    ):

        raise AssertionError(
            "Phase 03D snapshot population "
            "does not match Phase 03C."
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

    EXPECTED_FEATURE_COUNT = 281

    if len(
        predictive_columns
    ) != EXPECTED_FEATURE_COUNT:

        raise AssertionError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} "
            f"predictive features but found "
            f"{len(predictive_columns)}."
        )

    # ============================================================
    # SAVE
    # ============================================================

    output_path = (
        OUTPUT_DIR
        / "category_customer_features.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        float_format="%.10g",
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        OUTPUT_DIR
        / "phase03d_feature_audit.csv",
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
        "PHASE 03D CATEGORY FEATURE BUILD COMPLETE"
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
        f"Included categories: "
        f"{len(categories)}"
    )

    print(
        f"Predictive features: "
        f"{len(predictive_columns)}"
    )

    print()

    print(
        f"Output:\n{output_path}"
    )


if __name__ == "__main__":
    main()
