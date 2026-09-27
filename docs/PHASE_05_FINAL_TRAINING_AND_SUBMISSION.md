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

