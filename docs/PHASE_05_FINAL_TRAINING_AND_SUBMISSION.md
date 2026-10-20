# Phase 05 - Final Training, Prediction and Submission

## 1. Purpose

Phase 05 converts the frozen Phase 04 modelling specification into final
competition predictions.

No further model selection or tuning is permitted during this phase.

The final models are trained using all available labelled historical
customer-origin observations and applied to the unlabelled competition
population at the final forecasting origin.

---

## 2. Frozen Training Population

Historical labelled forecasting origins:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01
    2025-09-01

Total historical observations:

    22,679

Competition forecasting origin:

    2025-12-01

Competition observations:

    5,488

---

## 3. Frozen Feature Representation

Canonical Phase 03 predictors:

    281

Excluded predictors:

    5

Final modelling predictors:

    276

The feature set is read from the frozen Phase 04 model feature manifest.

No feature is added, removed or regenerated during final model fitting.

---

## 4. Final Fuel Model

Algorithm:

    LightGBM regression

Target:

    CLV_fuel

Frozen parameters:

    num_leaves = 15
    min_data_in_leaf = 100
    num_boost_round = 168

Prediction post-processing:

    maximum(prediction, 0)

The model is trained on all 22,679 historical snapshots.

---

## 5. Final Non-Fuel Model

Algorithm:

    LightGBM regression

Target:

    CLV_nonfuel

Frozen parameters:

    num_leaves = 15
    min_data_in_leaf = 50
    num_boost_round = 145

Prediction post-processing:

    maximum(prediction, 0)

The accepted model remains direct regression.

The Phase 04E two-part architecture is not used.

---

## 6. Final Opportunity Model

Algorithm:

    LightGBM multiclass

Classes:

    17

Frozen structural parameters:

    num_leaves = 31
    min_data_in_leaf = 50
    num_boost_round = 232

Training weights:

    (N / (K * n_c)) ^ 0.50

Weights are calculated using the final historical training population
only and normalized to mean one.

Prediction rule:

    argmax class probability

---

## 7. Final Training Principles

Phase 05 final model construction follows these rules:

- no competition labels are available or used;
- no validation split is created;
- no early stopping is performed;
- no hyperparameters are changed;
- no additional feature selection is performed;
- no ensembling is introduced;
- no leaderboard feedback is used to alter the frozen specification.

Final training is therefore a direct implementation of the Phase 04M
specification.


---

## 8. Phase 05A Final Training Result

The frozen Phase 04M specification was trained on all:

    22,679

historical customer-origin observations.

Final competition population:

    5,488

Three final models were trained:

    Fuel:
        168 boosting rounds
        num_leaves = 15
        min_data_in_leaf = 100

    Non-fuel:
        145 boosting rounds
        num_leaves = 15
        min_data_in_leaf = 50

    Opportunity:
        232 boosting rounds
        num_leaves = 31
        min_data_in_leaf = 50
        class-weight exponent = 0.50

No validation split or early stopping was used during final training.

---

## 9. Final Prediction Diagnostics

Fuel transformed-CLV predictions:

    minimum     0.038532
    mean        1.466796
    median      1.629893
    maximum     3.144977

Raw negative Fuel predictions:

    0

Non-fuel transformed-CLV predictions:

    minimum     0.021964
    mean        0.711787
    median      0.456709
    maximum     3.084665

Raw negative non-fuel predictions:

    0

Opportunity predictions contained:

    14 of the 17 valid output classes.

The three highest-frequency predictions were:

    Inactivity                     1,884
    Stable                         1,568
    Existing-category growth: Fuel 1,209

No manual adjustment was applied to the final prediction distribution.

---

## 10. Phase 05B Submission Integrity Audit

The final submission was reconstructed using the official
SampleSubmission.csv as the structural authority.

All integrity checks passed:

    official columns                 PASS
    official row count               PASS
    exact ID-set match               PASS
    official ID ordering             PASS
    missing-value check              PASS
    Fuel finite/non-negative         PASS
    Non-fuel finite/non-negative     PASS
    Opportunity-label validity       PASS
    serialized-file recheck          PASS

Final submission rows:

    5,488

Unique customer IDs:

    5,488

Predicted Opportunity classes:

    14

The presence of fewer than 17 predicted classes is valid because every
prediction belongs to the official class set. No artificial class
predictions are introduced.

---

## 11. Final Submission File

The audited competition file is:

    submissions/sasol_phase05_frozen_lightgbm_submission.csv

This file represents the first submission produced from the frozen
temporal-validation modelling framework.

Leaderboard feedback must be recorded separately and must not silently
change the frozen modelling pipeline.


---

## 12. First Public Leaderboard Result

The first submission produced by the frozen Phase 05 modelling pipeline
was submitted without any leaderboard-driven adjustment.

Public leaderboard result:

    Public score: 0.289392053
    Public rank: 23

This score is designated:

    S0

and serves as the external benchmark for subsequent controlled
experiments.

At the time of observation, visible leaderboard reference scores
included:

    Rank 7:  0.307567003
    Rank 1:  0.310346593

Therefore the observed gap between S0 and the rank-7 score was:

    0.018174950

and the observed gap between S0 and the leading score was:

    0.020954540

The leaderboard result will not be used as a hyperparameter optimization
signal.

Subsequent changes must first be supported by historical temporal
validation and documented as explicit modelling hypotheses.

