"""
Phase 04C: Targeted error analysis for the fixed Phase 04B baseline.

No model is retrained.

The script analyses:
- regression error by zero/positive outcome;
- regression error by target magnitude;
- regression bias;
- positive-outcome ranking diagnostics;
- full Opportunity confusion;
- macro Opportunity confusion;
- predicted class frequency;
- true-class probability and rank;
- top-k Opportunity recovery;
- adoption probability behaviour;
- feature-importance concentration.

Saved Phase 04B LightGBM models are reused only to recover validation
class probabilities that were not stored in the original OOF file.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

PHASE04_DIR = (
    ROOT
    / "data"
    / "processed"
    / "phase04"
)

BASELINE_DIR = (
    PHASE04_DIR
    / "baseline"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "phase04_baseline"
)

OUTPUT_DIR = (
    PHASE04_DIR
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUPERVISED_PATH = (
    PHASE04_DIR
    / "supervised_historical.csv"
)

OOF_PATH = (
    BASELINE_DIR
    / "oof_predictions.csv"
)

FEATURE_MANIFEST_PATH = (
    PHASE04_DIR
    / "model_feature_manifest.csv"
)

CLASS_MAP_PATH = (
    PHASE04_DIR
    / "opportunity_class_map.csv"
)

FOLD_MEMBERSHIP_PATH = (
    PHASE04_DIR
    / "temporal_fold_membership.csv"
)

IMPORTANCE_PATH = (
    BASELINE_DIR
    / "feature_importance_by_fold.csv"
)


def rmse(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype="float64",
    )

    y_pred = np.asarray(
        y_pred,
        dtype="float64",
    )

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def mae(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype="float64",
    )

    y_pred = np.asarray(
        y_pred,
        dtype="float64",
    )

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def opportunity_macro(label):

    if label == "Inactivity":
        return "Inactivity"

    if label == "Stable":
        return "Stable"

    if label.startswith(
        "Existing-category growth:"
    ):
        return "Growth"

    if label.startswith(
        "New category adoption:"
    ):
        return "Adoption"

    raise ValueError(
        f"Unknown Opportunity label: "
        f"{label}"
    )


def regression_group_summary(
    frame,
    target,
    prediction,
    task,
):

    rows = []

    masks = {
        "all":
            np.ones(
                len(frame),
                dtype=bool,
            ),

        "true_zero":
            frame[target].eq(0),

        "true_positive":
            frame[target].gt(0),
    }

    for group, mask in masks.items():

        block = frame.loc[
            mask
        ]

        if block.empty:
            continue

        residual = (
            block[prediction]
            -
            block[target]
        )

        rows.append(
            {
                "task":
                    task,

                "group":
                    group,

                "rows":
                    len(block),

                "target_mean":
                    float(
                        block[
                            target
                        ].mean()
                    ),

                "prediction_mean":
                    float(
                        block[
                            prediction
                        ].mean()
                    ),

                "rmse":
                    rmse(
                        block[target],
                        block[prediction],
                    ),

                "mae":
                    mae(
                        block[target],
                        block[prediction],
                    ),

                "mean_residual_pred_minus_true":
                    float(
                        residual.mean()
                    ),

                "median_residual_pred_minus_true":
                    float(
                        residual.median()
                    ),
            }
        )

    return rows


def magnitude_error_summary(
    frame,
    target,
    prediction,
    task,
):

    positive = frame[
        frame[target].gt(0)
    ].copy()

    if positive.empty:
        return []

    # Rank-based bins avoid assuming equal target intervals.
    positive[
        "target_magnitude_bin"
    ] = pd.qcut(
        positive[
            target
        ],
        q=5,
        duplicates="drop",
        labels=False,
    )

    rows = []

    for bin_id, block in positive.groupby(
        "target_magnitude_bin"
    ):

        residual = (
            block[prediction]
            -
            block[target]
        )

        rows.append(
            {
                "task":
                    task,

                "target_magnitude_bin":
                    int(bin_id) + 1,

                "rows":
                    len(block),

                "target_min":
                    float(
                        block[
                            target
                        ].min()
                    ),

                "target_median":
                    float(
                        block[
                            target
                        ].median()
                    ),

                "target_max":
                    float(
                        block[
                            target
                        ].max()
                    ),

                "prediction_mean":
                    float(
                        block[
                            prediction
                        ].mean()
                    ),

                "rmse":
                    rmse(
                        block[target],
                        block[prediction],
                    ),

                "mae":
                    mae(
                        block[target],
                        block[prediction],
                    ),

                "mean_residual_pred_minus_true":
                    float(
                        residual.mean()
                    ),
            }
        )

    return rows


def ranking_summary(
    frame,
    target,
    prediction,
    task,
):

    truth = (
        frame[
            target
        ].gt(0)
        .astype(int)
    )

    if truth.nunique() < 2:

        auc = np.nan

    else:

        auc = roc_auc_score(
            truth,
            frame[
                prediction
            ],
        )

    zero_scores = frame.loc[
        frame[target].eq(0),
        prediction,
    ]

    positive_scores = frame.loc[
        frame[target].gt(0),
        prediction,
    ]

    return {
        "task":
            task,

        "rows":
            len(frame),

        "positive_prevalence":
            float(
                truth.mean()
            ),

        "positive_ranking_roc_auc":
            float(auc)
            if not np.isnan(auc)
            else np.nan,

        "mean_prediction_true_zero":
            float(
                zero_scores.mean()
            ),

        "mean_prediction_true_positive":
            float(
                positive_scores.mean()
            ),

        "median_prediction_true_zero":
            float(
                zero_scores.median()
            ),

        "median_prediction_true_positive":
            float(
                positive_scores.median()
            ),
    }


def main():

    print(
        "Loading Phase 04 baseline outputs..."
    )

    supervised = pd.read_csv(
        SUPERVISED_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    oof = pd.read_csv(
        OOF_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    manifest = pd.read_csv(
        FEATURE_MANIFEST_PATH
    )

    class_map = pd.read_csv(
        CLASS_MAP_PATH
    )

    membership = pd.read_csv(
        FOLD_MEMBERSHIP_PATH,
        dtype={"ID": str},
        parse_dates=["cutoff"],
    )

    importance = pd.read_csv(
        IMPORTANCE_PATH
    )

    selected_features = (
        manifest.loc[
            manifest[
                "selected_for_baseline"
            ].eq(1),
            "feature",
        ]
        .tolist()
    )

    if len(selected_features) != 276:
        raise AssertionError(
            "Expected 276 selected features."
        )

    class_map = class_map.sort_values(
        "class_id"
    )

    class_ids = (
        class_map[
            "class_id"
        ]
        .astype(int)
        .tolist()
    )

    class_names = (
        class_map[
            "Opportunity"
        ]
        .astype(str)
        .tolist()
    )

    class_name_lookup = dict(
        zip(
            class_ids,
            class_names,
        )
    )

    # ============================================================
    # REGRESSION ERROR DECOMPOSITION
    # ============================================================

    print(
        "Analysing regression errors..."
    )

    regression_group_rows = []
    magnitude_rows = []
    ranking_rows = []

    for fold_id in sorted(
        oof[
            "fold_id"
        ].unique()
    ):

        fold = oof[
            oof[
                "fold_id"
            ].eq(
                fold_id
            )
        ].copy()

        for (
            task,
            target,
            prediction,
        ) in [
            (
                "fuel",
                "CLV_fuel",
                "pred_CLV_fuel",
            ),
            (
                "nonfuel",
                "CLV_nonfuel",
                "pred_CLV_nonfuel",
            ),
        ]:

            group_rows = (
                regression_group_summary(
                    fold,
                    target,
                    prediction,
                    task,
                )
            )

            for row in group_rows:
                row[
                    "fold_id"
                ] = fold_id

            regression_group_rows.extend(
                group_rows
            )

            fold_magnitude_rows = (
                magnitude_error_summary(
                    fold,
                    target,
                    prediction,
                    task,
                )
            )

            for row in fold_magnitude_rows:
                row[
                    "fold_id"
                ] = fold_id

            magnitude_rows.extend(
                fold_magnitude_rows
            )

            ranking = ranking_summary(
                fold,
                target,
                prediction,
                task,
            )

            ranking[
                "fold_id"
            ] = fold_id

            ranking_rows.append(
                ranking
            )

    regression_groups = pd.DataFrame(
        regression_group_rows
    )

    magnitude_errors = pd.DataFrame(
        magnitude_rows
    )

    ranking = pd.DataFrame(
        ranking_rows
    )

    # ============================================================
    # OPPORTUNITY PROBABILITY RECOVERY
    # ============================================================

    print(
        "Recovering Opportunity validation probabilities..."
    )

    probability_frames = []

    for fold_id in sorted(
        oof[
            "fold_id"
        ].unique()
    ):

        valid_ids = set(
            membership.loc[
                (
                    membership[
                        "fold_id"
                    ].eq(
                        fold_id
                    )
                    &
                    membership[
                        "role"
                    ].eq(
                        "validation"
                    )
                ),
                "snapshot_id",
            ]
        )

        valid = supervised[
            supervised[
                "snapshot_id"
            ].isin(
                valid_ids
            )
        ].copy()

        valid = valid.sort_values(
            "snapshot_id"
        ).reset_index(
            drop=True
        )

        model_path = (
            MODEL_DIR
            / f"{fold_id}_opportunity.txt"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                model_path
            )

        model = lgb.Booster(
            model_file=str(
                model_path
            )
        )

        probabilities = model.predict(
            valid[
                selected_features
            ]
        )

        if probabilities.shape != (
            len(valid),
            len(class_ids),
        ):

            raise AssertionError(
                f"{fold_id} unexpected "
                f"probability shape "
                f"{probabilities.shape}."
            )

        probability_frame = valid[
            [
                "snapshot_id",
                "ID",
                "cutoff",
                "Opportunity",
                "Opportunity_class_id",
            ]
        ].copy()

        probability_frame[
            "fold_id"
        ] = fold_id

        predicted_id = np.argmax(
            probabilities,
            axis=1,
        )

        probability_frame[
            "pred_Opportunity_class_id"
        ] = predicted_id

        probability_frame[
            "pred_Opportunity"
        ] = [
            class_name_lookup[
                int(value)
            ]
            for value in predicted_id
        ]

        for class_id in class_ids:

            probability_frame[
                f"prob_class_{class_id}"
            ] = probabilities[
                :,
                class_id
            ]

        true_ids = probability_frame[
            "Opportunity_class_id"
        ].to_numpy(
            dtype=int
        )

        true_probability = probabilities[
            np.arange(
                len(probability_frame)
            ),
            true_ids,
        ]

        probability_frame[
            "true_class_probability"
        ] = true_probability

        # rank 1 = highest model probability.
        order = np.argsort(
            -probabilities,
            axis=1,
        )

        ranks = np.empty_like(
            order
        )

        ranks[
            np.arange(
                len(order)
            )[:, None],
            order,
        ] = (
            np.arange(
                len(class_ids)
            )[None, :]
            + 1
        )

        probability_frame[
            "true_class_rank"
        ] = ranks[
            np.arange(
                len(probability_frame)
            ),
            true_ids,
        ]

        probability_frame[
            "top2_correct"
        ] = (
            probability_frame[
                "true_class_rank"
            ].le(2)
            .astype(int)
        )

        probability_frame[
            "top3_correct"
        ] = (
            probability_frame[
                "true_class_rank"
            ].le(3)
            .astype(int)
        )

        probability_frame[
            "top5_correct"
        ] = (
            probability_frame[
                "true_class_rank"
            ].le(5)
            .astype(int)
        )

        probability_frame[
            "true_macro"
        ] = (
            probability_frame[
                "Opportunity"
            ].map(
                opportunity_macro
            )
        )

        probability_frame[
            "pred_macro"
        ] = (
            probability_frame[
                "pred_Opportunity"
            ].map(
                opportunity_macro
            )
        )

        probability_frames.append(
            probability_frame
        )

    probabilities = pd.concat(
        probability_frames,
        ignore_index=True,
    )

    # Confirm recovered argmax equals saved OOF predictions.
    check = (
        probabilities[
            [
                "snapshot_id",
                "pred_Opportunity",
            ]
        ]
        .merge(
            oof[
                [
                    "snapshot_id",
                    "pred_Opportunity",
                ]
            ],
            on="snapshot_id",
            suffixes=(
                "_recovered",
                "_oof",
            ),
            validate="one_to_one",
        )
    )

    prediction_disagreements = int(
        (
            check[
                "pred_Opportunity_recovered"
            ]
            !=
            check[
                "pred_Opportunity_oof"
            ]
        ).sum()
    )

    if prediction_disagreements != 0:

        raise AssertionError(
            f"Recovered Opportunity "
            f"predictions disagree with OOF "
            f"for {prediction_disagreements} rows."
        )

    # ============================================================
    # CLASS DISTRIBUTION / CONFUSION
    # ============================================================

    print(
        "Building Opportunity confusion diagnostics..."
    )

    predicted_distribution_rows = []
    class_probability_rows = []
    macro_metric_rows = []
    confusion_rows = []
    macro_confusion_rows = []

    macro_order = [
        "Adoption",
        "Growth",
        "Inactivity",
        "Stable",
    ]

    for fold_id in sorted(
        probabilities[
            "fold_id"
        ].unique()
    ):

        block = probabilities[
            probabilities[
                "fold_id"
            ].eq(
                fold_id
            )
        ].copy()

        # Predicted versus true class frequency.
        for class_id, class_name in zip(
            class_ids,
            class_names,
        ):

            true_count = int(
                block[
                    "Opportunity_class_id"
                ].eq(
                    class_id
                ).sum()
            )

            predicted_count = int(
                block[
                    "pred_Opportunity_class_id"
                ].eq(
                    class_id
                ).sum()
            )

            predicted_distribution_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "class_id":
                        class_id,

                    "Opportunity":
                        class_name,

                    "true_count":
                        true_count,

                    "predicted_count":
                        predicted_count,

                    "true_share":
                        true_count
                        / len(block),

                    "predicted_share":
                        predicted_count
                        / len(block),
                }
            )

        # Full confusion matrix.
        cm = confusion_matrix(
            block[
                "Opportunity_class_id"
            ],
            block[
                "pred_Opportunity_class_id"
            ],
            labels=class_ids,
        )

        for i, true_name in enumerate(
            class_names
        ):

            for j, pred_name in enumerate(
                class_names
            ):

                count = int(
                    cm[i, j]
                )

                if count == 0:
                    continue

                confusion_rows.append(
                    {
                        "fold_id":
                            fold_id,

                        "true_class":
                            true_name,

                        "predicted_class":
                            pred_name,

                        "count":
                            count,

                        "share_of_true_class":
                            count
                            /
                            cm[i, :].sum(),
                    }
                )

        # Per-true-class probability/rank diagnostics.
        for class_id, class_name in zip(
            class_ids,
            class_names,
        ):

            class_block = block[
                block[
                    "Opportunity_class_id"
                ].eq(
                    class_id
                )
            ]

            if class_block.empty:
                continue

            class_probability_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "class_id":
                        class_id,

                    "Opportunity":
                        class_name,

                    "macro":
                        opportunity_macro(
                            class_name
                        ),

                    "support":
                        len(
                            class_block
                        ),

                    "mean_true_class_probability":
                        float(
                            class_block[
                                "true_class_probability"
                            ].mean()
                        ),

                    "median_true_class_probability":
                        float(
                            class_block[
                                "true_class_probability"
                            ].median()
                        ),

                    "mean_true_class_rank":
                        float(
                            class_block[
                                "true_class_rank"
                            ].mean()
                        ),

                    "median_true_class_rank":
                        float(
                            class_block[
                                "true_class_rank"
                            ].median()
                        ),

                    "top1_recall":
                        float(
                            class_block[
                                "true_class_rank"
                            ].le(1).mean()
                        ),

                    "top2_recall":
                        float(
                            class_block[
                                "top2_correct"
                            ].mean()
                        ),

                    "top3_recall":
                        float(
                            class_block[
                                "top3_correct"
                            ].mean()
                        ),

                    "top5_recall":
                        float(
                            class_block[
                                "top5_correct"
                            ].mean()
                        ),
                }
            )

        # Macro confusion.
        macro_cm = confusion_matrix(
            block[
                "true_macro"
            ],
            block[
                "pred_macro"
            ],
            labels=macro_order,
        )

        for i, true_macro in enumerate(
            macro_order
        ):

            for j, pred_macro in enumerate(
                macro_order
            ):

                count = int(
                    macro_cm[
                        i,
                        j,
                    ]
                )

                if count == 0:
                    continue

                macro_confusion_rows.append(
                    {
                        "fold_id":
                            fold_id,

                        "true_macro":
                            true_macro,

                        "predicted_macro":
                            pred_macro,

                        "count":
                            count,

                        "share_of_true_macro":
                            count
                            /
                            macro_cm[
                                i,
                                :
                            ].sum(),
                    }
                )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                block[
                    "true_macro"
                ],
                block[
                    "pred_macro"
                ],
                labels=macro_order,
                zero_division=0,
            )
        )

        for index, macro in enumerate(
            macro_order
        ):

            macro_metric_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "macro":
                        macro,

                    "support":
                        int(
                            support[index]
                        ),

                    "precision":
                        float(
                            precision[index]
                        ),

                    "recall":
                        float(
                            recall[index]
                        ),

                    "f1":
                        float(
                            f1[index]
                        ),
                }
            )

    predicted_distribution = pd.DataFrame(
        predicted_distribution_rows
    )

    class_probability = pd.DataFrame(
        class_probability_rows
    )

    confusion = pd.DataFrame(
        confusion_rows
    )

    macro_confusion = pd.DataFrame(
        macro_confusion_rows
    )

    macro_metrics = pd.DataFrame(
        macro_metric_rows
    )

    # ============================================================
    # FEATURE IMPORTANCE CONCENTRATION
    # ============================================================

    print(
        "Summarising feature importance concentration..."
    )

    importance_rows = []

    for (
        fold_id,
        task
    ), block in importance.groupby(
        [
            "fold_id",
            "task",
        ]
    ):

        ordered = block.sort_values(
            "gain",
            ascending=False,
        )

        total_gain = ordered[
            "gain"
        ].sum()

        for top_n in [
            5,
            10,
            20,
            50,
        ]:

            gain = ordered.head(
                top_n
            )[
                "gain"
            ].sum()

            importance_rows.append(
                {
                    "fold_id":
                        fold_id,

                    "task":
                        task,

                    "top_n":
                        top_n,

                    "gain_share":
                        (
                            float(
                                gain
                                / total_gain
                            )
                            if total_gain > 0
                            else np.nan
                        ),
                }
            )

    importance_concentration = (
        pd.DataFrame(
            importance_rows
        )
    )

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    regression_groups.to_csv(
        OUTPUT_DIR
        / "regression_error_by_zero_status.csv",
        index=False,
    )

    magnitude_errors.to_csv(
        OUTPUT_DIR
        / "regression_error_by_positive_magnitude.csv",
        index=False,
    )

    ranking.to_csv(
        OUTPUT_DIR
        / "regression_positive_ranking.csv",
        index=False,
    )

    probabilities.to_csv(
        OUTPUT_DIR
        / "opportunity_probability_diagnostics.csv",
        index=False,
        float_format="%.10g",
    )

    predicted_distribution.to_csv(
        OUTPUT_DIR
        / "opportunity_predicted_distribution.csv",
        index=False,
    )

    class_probability.to_csv(
        OUTPUT_DIR
        / "opportunity_true_class_probability.csv",
        index=False,
    )

    confusion.to_csv(
        OUTPUT_DIR
        / "opportunity_confusion_long.csv",
        index=False,
    )

    macro_confusion.to_csv(
        OUTPUT_DIR
        / "opportunity_macro_confusion.csv",
        index=False,
    )

    macro_metrics.to_csv(
        OUTPUT_DIR
        / "opportunity_macro_metrics.csv",
        index=False,
    )

    importance_concentration.to_csv(
        OUTPUT_DIR
        / "feature_importance_concentration.csv",
        index=False,
    )

    # ============================================================
    # CONSOLE SUMMARY
    # ============================================================

    print()
    print(
        "=" * 76
    )

    print(
        "PHASE 04C BASELINE ERROR ANALYSIS COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "F4 REGRESSION ERROR BY ZERO STATUS"
    )

    print()

    print(
        regression_groups[
            regression_groups[
                "fold_id"
            ].eq(
                "F4"
            )
        ][
            [
                "task",
                "group",
                "rows",
                "target_mean",
                "prediction_mean",
                "rmse",
                "mae",
                "mean_residual_pred_minus_true",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "F4 POSITIVE-OUTCOME RANKING"
    )

    print()

    print(
        ranking[
            ranking[
                "fold_id"
            ].eq(
                "F4"
            )
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "F4 OPPORTUNITY MACRO METRICS"
    )

    print()

    print(
        macro_metrics[
            macro_metrics[
                "fold_id"
            ].eq(
                "F4"
            )
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "F4 OPPORTUNITY PREDICTED DISTRIBUTION"
    )

    print()

    print(
        predicted_distribution[
            predicted_distribution[
                "fold_id"
            ].eq(
                "F4"
            )
        ][
            [
                "Opportunity",
                "true_count",
                "predicted_count",
            ]
        ].sort_values(
            "true_count",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "F4 TRUE-CLASS PROBABILITY / RANK"
    )

    print()

    print(
        class_probability[
            class_probability[
                "fold_id"
            ].eq(
                "F4"
            )
        ][
            [
                "Opportunity",
                "macro",
                "support",
                "mean_true_class_probability",
                "median_true_class_rank",
                "top1_recall",
                "top3_recall",
                "top5_recall",
            ]
        ].sort_values(
            "support",
            ascending=False,
        ).to_string(
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
