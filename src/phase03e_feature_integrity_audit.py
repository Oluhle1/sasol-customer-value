"""
Phase 03E: Feature integrity, leakage and temporal drift audit.

Input:
    Phase 03D category customer feature matrix.

This phase creates NO new predictive features.

Audits:
- structural integrity;
- forbidden target/future variables;
- numeric validity;
- constant and near-constant predictors;
- exact duplicate predictors;
- very high historical correlations;
- temporal monotonicity of cumulative customer states;
- independent reconstruction of sentinel features;
- missingness drift;
- historical-to-final distribution drift.

No machine-learning model is trained.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from reference.label_rules import (
    load_transactions,
    SCALE,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    ROOT
    / "data"
    / "raw"
    / "train.csv"
)

LABEL_CONFIG_PATH = (
    ROOT
    / "configs"
    / "label_config.json"
)

FEATURE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phase03_features"
    / "category_customer_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase03_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METADATA_COLUMNS = [
    "snapshot_id",
    "ID",
    "cutoff",
    "snapshot_role",
    "is_labeled",
]


FORBIDDEN_EXACT_NAMES = {
    "CLV_fuel",
    "CLV_nonfuel",
    "Opportunity",
    "fuel_litres",
    "nonfuel_rand",
    "winning_source_category",
}


FORBIDDEN_NAME_TOKENS = [
    "future",
    "outcome",
    "opportunity",
    "winning_source",
    "target_",
]


HIGH_CORRELATION_THRESHOLD = 0.995
NEAR_CONSTANT_SHARE = 0.995


CATEGORY_SUMMARY_FEATURES = {
    "lifetime_category_breadth",
    "pre_m3_category_breadth",
    "m3_category_breadth",
    "prev3_category_breadth",
    "unseen_category_count",
    "dormant_known_category_count",
    "recent_new_to_history_category_count",
    "recent_reactivated_category_count",
    "recent_became_dormant_category_count",
}


def feature_family(name):
    """
    Assign a transparent audit family to each feature.
    """

    if name.startswith("cat__"):
        return "category_state"

    if name in CATEGORY_SUMMARY_FEATURES:
        return "category_maturity"

    temporal_prefixes = (
        "prev3_",
        "qoq_",
        "m1_vs_prev2_",
        "reactivated_",
        "became_dormant_",
    )

    if name.startswith(
        temporal_prefixes
    ):
        return "momentum_regularity"

    if (
        "_gap_" in name
        or name.endswith(
            "_recency_to_lifetime_gap"
        )
    ):
        return "momentum_regularity"

    return "core_behavior"


def standardized_mean_difference(
    historical,
    final,
):
    """
    Standardized mean difference.

    Missing values are ignored when means/std are calculated.
    """

    historical = (
        historical
        .dropna()
        .astype("float64")
    )

    final = (
        final
        .dropna()
        .astype("float64")
    )

    if (
        historical.empty
        or final.empty
    ):
        return np.nan, 0

    h_mean = historical.mean()
    f_mean = final.mean()

    h_std = historical.std(
        ddof=0
    )

    f_std = final.std(
        ddof=0
    )

    pooled = np.sqrt(
        (
            h_std ** 2
            + f_std ** 2
        )
        / 2.0
    )

    if pooled > 0:
        return (
            (f_mean - h_mean)
            / pooled,
            0,
        )

    # Both populations are constant.
    if np.isclose(
        h_mean,
        f_mean,
        rtol=0,
        atol=1e-12,
    ):
        return 0.0, 0

    # Different constants cannot be expressed
    # using the ordinary SMD denominator.
    return np.nan, 1


def exact_duplicate_pairs(
    historical,
    final,
    feature_columns,
):
    """
    Detect exact duplicate feature vectors in historical data.

    Hashing creates candidate groups and Series.equals confirms
    equality, preventing hash equality from being accepted blindly.
    """

    signature_groups = {}

    for column in feature_columns:

        values = pd.util.hash_pandas_object(
            historical[column],
            index=False,
        ).values

        digest = hashlib.sha256(
            values.tobytes()
        ).hexdigest()

        signature_groups.setdefault(
            digest,
            [],
        ).append(
            column
        )

    rows = []

    for columns in signature_groups.values():

        if len(columns) < 2:
            continue

        for i in range(
            len(columns)
        ):

            for j in range(
                i + 1,
                len(columns),
            ):

                a = columns[i]
                b = columns[j]

                if not historical[
                    a
                ].equals(
                    historical[b]
                ):
                    continue

                equal_final = (
                    final[a]
                    .reset_index(drop=True)
                    .equals(
                        final[b]
                        .reset_index(
                            drop=True
                        )
                    )
                )

                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "equal_historical":
                            True,
                        "equal_final":
                            equal_final,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "feature_a",
            "feature_b",
            "equal_historical",
            "equal_final",
        ],
    )


def build_quality_summary(
    data,
    historical,
    final,
    feature_columns,
):
    """
    One-row audit summary for each predictive feature.
    """

    rows = []

    for column in feature_columns:

        full = data[column]
        hist = historical[column]
        comp = final[column]

        hist_nonmissing = (
            hist.dropna()
        )

        value_counts = (
            hist.value_counts(
                dropna=False,
                normalize=True,
            )
        )

        top_share = (
            float(
                value_counts.iloc[0]
            )
            if not value_counts.empty
            else np.nan
        )

        unique_nonmissing = int(
            hist_nonmissing.nunique(
                dropna=True
            )
        )

        rows.append(
            {
                "feature":
                    column,

                "family":
                    feature_family(
                        column
                    ),

                "dtype":
                    str(
                        data[column].dtype
                    ),

                "missing_rate_all":
                    float(
                        full.isna().mean()
                    ),

                "missing_rate_historical":
                    float(
                        hist.isna().mean()
                    ),

                "missing_rate_final":
                    float(
                        comp.isna().mean()
                    ),

                "unique_nonmissing_historical":
                    unique_nonmissing,

                "historical_min":
                    (
                        float(
                            hist_nonmissing.min()
                        )
                        if not hist_nonmissing.empty
                        else np.nan
                    ),

                "historical_max":
                    (
                        float(
                            hist_nonmissing.max()
                        )
                        if not hist_nonmissing.empty
                        else np.nan
                    ),

                "historical_mean":
                    (
                        float(
                            hist_nonmissing.mean()
                        )
                        if not hist_nonmissing.empty
                        else np.nan
                    ),

                "historical_std":
                    (
                        float(
                            hist_nonmissing.std(
                                ddof=0
                            )
                        )
                        if not hist_nonmissing.empty
                        else np.nan
                    ),

                "top_value_share_historical":
                    top_share,

                "constant_historical":
                    int(
                        unique_nonmissing <= 1
                    ),

                "near_constant_99_5":
                    int(
                        top_share
                        >= NEAR_CONSTANT_SHARE
                    )
                    if not np.isnan(
                        top_share
                    )
                    else 0,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_drift_audit(
    data,
    historical_cutoffs,
    final_cutoff,
    feature_columns,
):
    """
    Compare every historical forecasting origin with final population.
    """

    final = data[
        data["cutoff"].eq(
            final_cutoff
        )
    ]

    rows = []

    for cutoff in historical_cutoffs:

        reference = data[
            data["cutoff"].eq(
                cutoff
            )
        ]

        for column in feature_columns:

            ref = reference[column]
            comp = final[column]

            smd, constant_shift = (
                standardized_mean_difference(
                    ref,
                    comp,
                )
            )

            rows.append(
                {
                    "reference_cutoff":
                        cutoff.date().isoformat(),

                    "final_cutoff":
                        final_cutoff.date().isoformat(),

                    "feature":
                        column,

                    "family":
                        feature_family(
                            column
                        ),

                    "reference_missing_rate":
                        float(
                            ref.isna().mean()
                        ),

                    "final_missing_rate":
                        float(
                            comp.isna().mean()
                        ),

                    "missing_rate_delta_pp":
                        float(
                            (
                                comp.isna().mean()
                                -
                                ref.isna().mean()
                            )
                            * 100
                        ),

                    "reference_mean":
                        (
                            float(
                                ref.mean()
                            )
                            if ref.notna().any()
                            else np.nan
                        ),

                    "final_mean":
                        (
                            float(
                                comp.mean()
                            )
                            if comp.notna().any()
                            else np.nan
                        ),

                    "reference_median":
                        (
                            float(
                                ref.median()
                            )
                            if ref.notna().any()
                            else np.nan
                        ),

                    "final_median":
                        (
                            float(
                                comp.median()
                            )
                            if comp.notna().any()
                            else np.nan
                        ),

                    "standardized_mean_difference":
                        smd,

                    "abs_standardized_mean_difference":
                        (
                            abs(smd)
                            if not np.isnan(
                                smd
                            )
                            else np.nan
                        ),

                    "constant_distribution_shift":
                        constant_shift,
                }
            )

    drift = pd.DataFrame(
        rows
    )

    summary_rows = []

    for cutoff in historical_cutoffs:

        block = drift[
            drift[
                "reference_cutoff"
            ].eq(
                cutoff.date().isoformat()
            )
        ]

        abs_smd = (
            block[
                "abs_standardized_mean_difference"
            ]
            .dropna()
        )

        abs_missing = (
            block[
                "missing_rate_delta_pp"
            ]
            .abs()
        )

        summary_rows.append(
            {
                "reference_cutoff":
                    cutoff.date().isoformat(),

                "features_compared":
                    len(block),

                "median_abs_smd":
                    (
                        float(
                            abs_smd.median()
                        )
                        if not abs_smd.empty
                        else np.nan
                    ),

                "p90_abs_smd":
                    (
                        float(
                            abs_smd.quantile(
                                0.90
                            )
                        )
                        if not abs_smd.empty
                        else np.nan
                    ),

                "p95_abs_smd":
                    (
                        float(
                            abs_smd.quantile(
                                0.95
                            )
                        )
                        if not abs_smd.empty
                        else np.nan
                    ),

                "features_abs_smd_ge_0_5":
                    int(
                        abs_smd.ge(
                            0.5
                        ).sum()
                    ),

                "features_abs_smd_ge_1":
                    int(
                        abs_smd.ge(
                            1.0
                        ).sum()
                    ),

                "mean_abs_missing_delta_pp":
                    float(
                        abs_missing.mean()
                    ),

                "p95_abs_missing_delta_pp":
                    float(
                        abs_missing.quantile(
                            0.95
                        )
                    ),

                "constant_distribution_shifts":
                    int(
                        block[
                            "constant_distribution_shift"
                        ].sum()
                    ),
            }
        )

    return (
        drift,
        pd.DataFrame(
            summary_rows
        ),
    )


def temporal_monotonicity_audit(
    data,
    monotonic_columns,
):
    """
    Check cumulative/history features for decreases through time
    within the same customer.
    """

    ordered = data.sort_values(
        [
            "ID",
            "cutoff",
        ]
    )

    rows = []

    for column in monotonic_columns:

        previous = (
            ordered
            .groupby("ID")[
                column
            ]
            .shift(1)
        )

        current = ordered[
            column
        ]

        comparable = (
            previous.notna()
            & current.notna()
        )

        violations = (
            comparable
            & (
                current
                <
                previous
                - 1e-9
            )
        )

        rows.append(
            {
                "feature":
                    column,

                "comparisons":
                    int(
                        comparable.sum()
                    ),

                "decrease_violations":
                    int(
                        violations.sum()
                    ),

                "validation_status":
                    (
                        "PASS"
                        if not violations.any()
                        else "FAIL"
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def independent_sentinel_audit(
    features,
    transactions,
    config,
    cutoffs,
):
    """
    Independently reconstruct selected sentinel features from raw data.
    """

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
        config[
            "adjustment_items"
        ]
    )

    rows = []

    for cutoff in cutoffs:

        print(
            f"Independent sentinel audit: "
            f"{cutoff.date()}"
        )

        matrix = features[
            features["cutoff"].eq(
                cutoff
            )
        ].copy()

        ids = pd.Index(
            matrix[
                "ID"
            ].astype(str),
            name="ID",
        )

        history = transactions[
            transactions["_time"]
            < cutoff
        ].copy()

        m3_start = (
            cutoff
            - pd.DateOffset(
                months=3
            )
        )

        recent = history[
            history["_time"].ge(
                m3_start
            )
        ]

        # --------------------------------------------------------
        # Lifetime basket count
        # --------------------------------------------------------

        lifetime_baskets = (
            history
            .groupby("ID")[
                "TransactionId"
            ]
            .nunique()
            .reindex(
                ids,
                fill_value=0,
            )
        )

        # --------------------------------------------------------
        # Recent three-month basket count
        # --------------------------------------------------------

        m3_baskets = (
            recent
            .groupby("ID")[
                "TransactionId"
            ]
            .nunique()
            .reindex(
                ids,
                fill_value=0,
            )
        )

        # --------------------------------------------------------
        # Last basket recency
        # --------------------------------------------------------

        last_basket = (
            history
            .groupby("ID")[
                "_time"
            ]
            .max()
            .reindex(
                ids
            )
        )

        recency = (
            (
                cutoff
                - last_basket
            )
            .dt.total_seconds()
            / 86400.0
        )

        # --------------------------------------------------------
        # Clean source rows
        # --------------------------------------------------------

        clean_recent = recent[
            ~recent[
                "ItemName"
            ].isin(
                adjustment_items
            )
        ]

        # Fuel litres
        fuel = clean_recent[
            clean_recent[
                "ItemCategoryLevel2"
            ].eq("Fuel")
            &
            clean_recent[
                "LineItemType"
            ].eq("Sale_Fuel")
        ]

        m3_fuel_litres = (
            fuel
            .groupby("ID")[
                "_quantity"
            ]
            .sum()
            .reindex(
                ids,
                fill_value=0,
            )
            .astype("float64")
            / SCALE
        )

        # Non-fuel net spend
        nonfuel = clean_recent[
            clean_recent[
                "ItemCategoryLevel2"
            ].isin(
                nonfuel_categories
            )
        ]

        m3_nonfuel_spend = (
            nonfuel
            .groupby("ID")[
                "_amount"
            ]
            .sum()
            .reindex(
                ids,
                fill_value=0,
            )
            .astype("float64")
            / SCALE
        )

        # --------------------------------------------------------
        # Lifetime category breadth
        # --------------------------------------------------------

        clean_history = history[
            ~history[
                "ItemName"
            ].isin(
                adjustment_items
            )
            &
            history[
                "ItemCategoryLevel2"
            ].isin(
                included_categories
            )
        ]

        positive_history = clean_history[
            clean_history[
                "_quantity"
            ].gt(0)
            &
            clean_history[
                "_amount"
            ].gt(0)
        ]

        breadth = (
            positive_history
            .groupby("ID")[
                "ItemCategoryLevel2"
            ]
            .nunique()
            .reindex(
                ids,
                fill_value=0,
            )
        )

        matrix = (
            matrix
            .set_index("ID")
            .reindex(ids)
        )

        expected = {
            "lifetime_basket_count":
                lifetime_baskets,

            "m3_basket_count":
                m3_baskets,

            "days_since_last_basket":
                recency,

            "m3_fuel_litres":
                m3_fuel_litres,

            "m3_nonfuel_spend":
                m3_nonfuel_spend,

            "lifetime_category_breadth":
                breadth,
        }

        for column, reconstructed in expected.items():

            observed = matrix[
                column
            ]

            a = observed.to_numpy(
                dtype="float64"
            )

            b = reconstructed.to_numpy(
                dtype="float64"
            )

            close = np.isclose(
                a,
                b,
                rtol=1e-9,
                atol=1e-6,
                equal_nan=True,
            )

            mismatch_count = int(
                (~close).sum()
            )

            max_abs_difference = (
                float(
                    np.nanmax(
                        np.abs(
                            a - b
                        )
                    )
                )
                if len(a)
                else 0.0
            )

            rows.append(
                {
                    "cutoff":
                        cutoff.date().isoformat(),

                    "sentinel_feature":
                        column,

                    "rows_compared":
                        len(matrix),

                    "mismatches":
                        mismatch_count,

                    "max_abs_difference":
                        max_abs_difference,

                    "validation_status":
                        (
                            "PASS"
                            if mismatch_count == 0
                            else "FAIL"
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def main():

    print(
        "Loading Phase 03 configuration..."
    )

    config = json.loads(
        LABEL_CONFIG_PATH.read_text()
    )

    historical_cutoffs = [
        pd.Timestamp(
            value
        )
        for value in config[
            "calibration_cutoffs"
        ]
    ]

    final_cutoff = pd.Timestamp(
        config[
            "public_history_end_exclusive"
        ]
    )

    expected_cutoffs = (
        historical_cutoffs
        + [
            final_cutoff
        ]
    )

    print(
        "Loading Phase 03D feature matrix..."
    )

    data = pd.read_csv(
        FEATURE_PATH,
        dtype={
            "ID": str,
        },
        parse_dates=[
            "cutoff",
        ],
    )

    # ============================================================
    # STRUCTURAL INTEGRITY
    # ============================================================

    if len(data) != 28167:
        raise AssertionError(
            f"Expected 28,167 rows, "
            f"found {len(data):,}."
        )

    if data[
        "snapshot_id"
    ].duplicated().any():

        raise AssertionError(
            "Duplicate snapshot IDs detected."
        )

    actual_cutoffs = sorted(
        pd.Timestamp(
            x
        )
        for x in data[
            "cutoff"
        ].unique()
    )

    if actual_cutoffs != sorted(
        expected_cutoffs
    ):

        raise AssertionError(
            "Unexpected forecasting-origin "
            "population."
        )

    feature_columns = [
        column
        for column in data.columns
        if column not in
        METADATA_COLUMNS
    ]

    if len(feature_columns) != 281:

        raise AssertionError(
            f"Expected 281 predictive "
            f"features, found "
            f"{len(feature_columns)}."
        )

    # All predictors should be numeric.
    nonnumeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    if nonnumeric:

        raise AssertionError(
            "Non-numeric predictive "
            f"columns detected: "
            f"{nonnumeric}"
        )

    numeric_values = data[
        feature_columns
    ].to_numpy(
        dtype="float64"
    )

    infinite_values = int(
        np.isinf(
            numeric_values
        ).sum()
    )

    if infinite_values != 0:

        raise AssertionError(
            f"Detected {infinite_values} "
            f"infinite predictive values."
        )

    # ============================================================
    # ROLE INTEGRITY
    # ============================================================

    historical = data[
        data[
            "cutoff"
        ].isin(
            historical_cutoffs
        )
    ].copy()

    final = data[
        data[
            "cutoff"
        ].eq(
            final_cutoff
        )
    ].copy()

    if len(
        historical
    ) != 22679:

        raise AssertionError(
            "Historical snapshot count "
            "is not 22,679."
        )

    if len(
        final
    ) != 5488:

        raise AssertionError(
            "Competition snapshot count "
            "is not 5,488."
        )

    # ============================================================
    # FORBIDDEN / LEAKAGE-LIKE NAMES
    # ============================================================

    forbidden_hits = []

    for column in feature_columns:

        lower = column.lower()

        if column in FORBIDDEN_EXACT_NAMES:

            forbidden_hits.append(
                {
                    "feature": column,
                    "reason":
                        "forbidden exact target/"
                        "label name",
                }
            )

            continue

        matched_tokens = [
            token
            for token in FORBIDDEN_NAME_TOKENS
            if token in lower
        ]

        if matched_tokens:

            forbidden_hits.append(
                {
                    "feature": column,
                    "reason":
                        "forbidden token: "
                        + ", ".join(
                            matched_tokens
                        ),
                }
            )

    forbidden_audit = (
        pd.DataFrame(
            forbidden_hits,
            columns=[
                "feature",
                "reason",
            ],
        )
    )

    forbidden_audit.to_csv(
        OUTPUT_DIR
        / "forbidden_feature_audit.csv",
        index=False,
    )

    if not forbidden_audit.empty:

        raise AssertionError(
            "Potential target/future leakage "
            "feature names detected."
        )

    # ============================================================
    # BASIC IMPOSSIBLE-VALUE CHECKS
    # ============================================================

    # ============================================================
    # NON-NEGATIVE ABSOLUTE COUNT / DAY FEATURES
    # ============================================================

    # Some features ending in "_count" or "_active_days"
    # are intentionally signed changes rather than absolute counts.
    #
    # Examples:
    #     qoq_delta_basket_count
    #     qoq_delta_active_days
    #     m1_vs_prev2_monthly_basket_count
    #
    # Negative values are meaningful for those features because they
    # represent behavioural decline relative to another historical
    # period.
    #
    # Only absolute state/count variables should be required to be
    # non-negative.

    signed_change_prefixes = (
        "qoq_delta_",
        "m1_vs_prev2_monthly_",
    )

    count_columns = [
        column
        for column in feature_columns
        if (
            (
                column.endswith("_count")
                or column.endswith(
                    "_active_days"
                )
            )
            and not column.startswith(
                signed_change_prefixes
            )
        )
    ]

    negative_count_cells = int(
        data[
            count_columns
        ]
        .lt(0)
        .sum()
        .sum()
    )

    if negative_count_cells != 0:

        raise AssertionError(
            f"Detected "
            f"{negative_count_cells} "
            f"negative absolute count/day "
            f"values."
        )

    binary_columns = [
        column
        for column in feature_columns
        if (
            column.startswith(
                "has_ever_"
            )
            or column.startswith(
                "reactivated_"
            )
            or column.startswith(
                "became_dormant_"
            )
            or column.endswith(
                "__ever_purchased"
            )
            or column.endswith(
                "__seen_before_m3"
            )
        )
    ]

    binary_violations = 0

    for column in binary_columns:

        valid = (
            data[column]
            .dropna()
            .isin(
                [
                    0,
                    1,
                ]
            )
        )

        binary_violations += int(
            (~valid).sum()
        )

    if binary_violations != 0:

        raise AssertionError(
            "Non-binary value detected "
            "in binary-state features."
        )

    recency_gap_columns = [
        column
        for column in feature_columns
        if (
            "days_since_" in column
            or "_gap_" in column
            and column.endswith(
                "_days"
            )
        )
    ]

    negative_recency_gap = int(
        sum(
            data[column]
            .dropna()
            .lt(0)
            .sum()
            for column
            in recency_gap_columns
        )
    )

    if negative_recency_gap != 0:

        raise AssertionError(
            "Negative recency/gap value "
            "detected."
        )

    if not data[
        "lifetime_category_breadth"
    ].between(
        0,
        24,
    ).all():

        raise AssertionError(
            "Invalid lifetime category "
            "breadth."
        )

    if not data[
        "unseen_category_count"
    ].between(
        0,
        24,
    ).all():

        raise AssertionError(
            "Invalid unseen-category count."
        )

    if not (
        data[
            "lifetime_category_breadth"
        ]
        +
        data[
            "unseen_category_count"
        ]
        ==
        24
    ).all():

        raise AssertionError(
            "Category breadth reconciliation "
            "failed."
        )

    # ============================================================
    # FEATURE QUALITY
    # ============================================================

    print(
        "Building feature quality audit..."
    )

    quality = build_quality_summary(
        data,
        historical,
        final,
        feature_columns,
    )

    quality.to_csv(
        OUTPUT_DIR
        / "feature_quality_summary.csv",
        index=False,
    )

    constant_features = quality[
        quality[
            "constant_historical"
        ].eq(1)
    ].copy()

    constant_features.to_csv(
        OUTPUT_DIR
        / "constant_features.csv",
        index=False,
    )

    near_constant_features = quality[
        quality[
            "near_constant_99_5"
        ].eq(1)
    ].copy()

    near_constant_features.to_csv(
        OUTPUT_DIR
        / "near_constant_features.csv",
        index=False,
    )

    # ============================================================
    # EXACT DUPLICATE FEATURES
    # ============================================================

    print(
        "Checking exact duplicate features..."
    )

    duplicates = exact_duplicate_pairs(
        historical,
        final,
        feature_columns,
    )

    duplicates.to_csv(
        OUTPUT_DIR
        / "exact_duplicate_features.csv",
        index=False,
    )

    # ============================================================
    # VERY HIGH CORRELATION
    # ============================================================

    print(
        "Calculating high-correlation pairs..."
    )

    correlation = historical[
        feature_columns
    ].corr(
        method="pearson",
        min_periods=100,
    )

    correlation_rows = []

    columns = list(
        correlation.columns
    )

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns),
        ):

            value = correlation.iloc[
                i,
                j,
            ]

            if pd.isna(
                value
            ):
                continue

            if abs(
                value
            ) >= HIGH_CORRELATION_THRESHOLD:

                correlation_rows.append(
                    {
                        "feature_a":
                            columns[i],

                        "feature_b":
                            columns[j],

                        "correlation":
                            float(
                                value
                            ),

                        "abs_correlation":
                            float(
                                abs(
                                    value
                                )
                            ),
                    }
                )

    high_correlation = pd.DataFrame(
        correlation_rows,
        columns=[
            "feature_a",
            "feature_b",
            "correlation",
            "abs_correlation",
        ],
    )

    if not high_correlation.empty:

        high_correlation = (
            high_correlation
            .sort_values(
                "abs_correlation",
                ascending=False,
            )
        )

    high_correlation.to_csv(
        OUTPUT_DIR
        / "high_correlation_pairs.csv",
        index=False,
    )

    # ============================================================
    # TEMPORAL MONOTONICITY
    # ============================================================

    print(
        "Checking temporal monotonicity..."
    )

    category_ever_columns = [
        column
        for column in feature_columns
        if column.endswith(
            "__ever_purchased"
        )
    ]

    monotonic_columns = [
        "lifetime_basket_count",
        "observed_tenure_days",
        "has_ever_fuel",
        "has_ever_nonfuel",
        "lifetime_category_breadth",
    ] + category_ever_columns

    monotonicity = (
        temporal_monotonicity_audit(
            data,
            monotonic_columns,
        )
    )

    monotonicity.to_csv(
        OUTPUT_DIR
        / "temporal_monotonicity_audit.csv",
        index=False,
    )

    monotonic_violations = int(
        monotonicity[
            "decrease_violations"
        ].sum()
    )

    if monotonic_violations != 0:

        raise AssertionError(
            f"Detected "
            f"{monotonic_violations} "
            f"temporal cumulative-state "
            f"violations."
        )

    # ============================================================
    # INDEPENDENT SENTINEL RECONSTRUCTION
    # ============================================================

    print(
        "Loading raw transactions for "
        "independent sentinel audit..."
    )

    transactions = load_transactions(
        TRAIN_PATH
    )

    sentinel = independent_sentinel_audit(
        features=data,
        transactions=transactions,
        config=config,
        cutoffs=expected_cutoffs,
    )

    sentinel.to_csv(
        OUTPUT_DIR
        / "sentinel_reconstruction_audit.csv",
        index=False,
    )

    sentinel_mismatches = int(
        sentinel[
            "mismatches"
        ].sum()
    )

    if sentinel_mismatches != 0:

        raise AssertionError(
            f"Independent sentinel audit "
            f"found {sentinel_mismatches} "
            f"mismatches."
        )

    # ============================================================
    # DISTRIBUTION DRIFT
    # ============================================================

    print(
        "Building historical-to-final "
        "drift audit..."
    )

    drift, drift_summary = (
        build_drift_audit(
            data=data,
            historical_cutoffs=
                historical_cutoffs,
            final_cutoff=
                final_cutoff,
            feature_columns=
                feature_columns,
        )
    )

    drift.to_csv(
        OUTPUT_DIR
        / "feature_drift_by_cutoff.csv",
        index=False,
    )

    drift_summary.to_csv(
        OUTPUT_DIR
        / "cutoff_drift_summary.csv",
        index=False,
    )

    latest_reference = (
        historical_cutoffs[-1]
        .date()
        .isoformat()
    )

    latest_to_final = drift[
        drift[
            "reference_cutoff"
        ].eq(
            latest_reference
        )
    ].copy()

    latest_to_final = (
        latest_to_final
        .sort_values(
            "abs_standardized_mean_difference",
            ascending=False,
            na_position="last",
        )
    )

    latest_to_final.to_csv(
        OUTPUT_DIR
        / "latest_to_final_feature_drift.csv",
        index=False,
    )

    # ============================================================
    # FEATURE AUDIT MANIFEST
    # ============================================================

    duplicate_feature_set = set()

    if not duplicates.empty:

        duplicate_feature_set.update(
            duplicates[
                "feature_a"
            ].tolist()
        )

        duplicate_feature_set.update(
            duplicates[
                "feature_b"
            ].tolist()
        )

    high_corr_feature_set = set()

    if not high_correlation.empty:

        high_corr_feature_set.update(
            high_correlation[
                "feature_a"
            ].tolist()
        )

        high_corr_feature_set.update(
            high_correlation[
                "feature_b"
            ].tolist()
        )

    quality_lookup = (
        quality.set_index(
            "feature"
        )
    )

    manifest_rows = []

    for position, column in enumerate(
        feature_columns,
        start=1,
    ):

        manifest_rows.append(
            {
                "feature_position":
                    position,

                "feature":
                    column,

                "family":
                    feature_family(
                        column
                    ),

                "historical_missing_rate":
                    quality_lookup.loc[
                        column,
                        "missing_rate_historical",
                    ],

                "final_missing_rate":
                    quality_lookup.loc[
                        column,
                        "missing_rate_final",
                    ],

                "constant_historical":
                    quality_lookup.loc[
                        column,
                        "constant_historical",
                    ],

                "near_constant_99_5":
                    quality_lookup.loc[
                        column,
                        "near_constant_99_5",
                    ],

                "in_exact_duplicate_pair":
                    int(
                        column
                        in duplicate_feature_set
                    ),

                "in_high_correlation_pair":
                    int(
                        column
                        in high_corr_feature_set
                    ),

                "candidate_predictor":
                    1,
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        OUTPUT_DIR
        / "feature_audit_manifest.csv",
        index=False,
    )

    # ============================================================
    # OVERALL AUDIT SUMMARY
    # ============================================================

    latest_drift_row = (
        drift_summary[
            drift_summary[
                "reference_cutoff"
            ].eq(
                latest_reference
            )
        ]
        .iloc[0]
    )

    summary = pd.DataFrame(
        [
            {
                "total_rows":
                    len(data),

                "historical_rows":
                    len(historical),

                "competition_rows":
                    len(final),

                "predictive_features":
                    len(
                        feature_columns
                    ),

                "forbidden_feature_hits":
                    len(
                        forbidden_audit
                    ),

                "infinite_values":
                    infinite_values,

                "negative_count_cells":
                    negative_count_cells,

                "binary_violations":
                    binary_violations,

                "negative_recency_gap_cells":
                    negative_recency_gap,

                "constant_historical_features":
                    len(
                        constant_features
                    ),

                "near_constant_features":
                    len(
                        near_constant_features
                    ),

                "exact_duplicate_pairs":
                    len(
                        duplicates
                    ),

                "high_correlation_pairs":
                    len(
                        high_correlation
                    ),

                "temporal_monotonicity_violations":
                    monotonic_violations,

                "sentinel_mismatches":
                    sentinel_mismatches,

                "latest_to_final_median_abs_smd":
                    latest_drift_row[
                        "median_abs_smd"
                    ],

                "latest_to_final_p90_abs_smd":
                    latest_drift_row[
                        "p90_abs_smd"
                    ],

                "latest_to_final_features_abs_smd_ge_0_5":
                    latest_drift_row[
                        "features_abs_smd_ge_0_5"
                    ],

                "latest_to_final_mean_abs_missing_delta_pp":
                    latest_drift_row[
                        "mean_abs_missing_delta_pp"
                    ],

                "validation_status":
                    "PASS",
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "phase03e_audit_summary.csv",
        index=False,
    )

    # ============================================================
    # CONSOLE CHECKPOINT
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 03E FEATURE INTEGRITY AUDIT COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "CUTOFF DRIFT SUMMARY"
    )

    print()

    print(
        drift_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "SENTINEL RECONSTRUCTION"
    )

    print()

    print(
        sentinel.to_string(
            index=False
        )
    )

    print()

    print(
        "Audit outputs written to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
