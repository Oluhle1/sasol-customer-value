# Phase 04 - LightGBM Modelling and Temporal Validation

**Project:** Sasol Customer Value Recruitment Challenge
**Phase:** 04
**Status:** In Progress
**Started:** 26 September 2026

---

## 1. Purpose

Phase 04 evaluates whether the temporal customer representation developed
and audited in Phase 03 can predict the three official future customer
outcomes.

The fixed modelling family is LightGBM.

The project will not compare unrelated model families or conduct a model
zoo.

Improvement will occur through:

- temporal validation;
- error analysis;
- feature-group analysis;
- and controlled LightGBM tuning.

---

## 2. Prediction Tasks

Three predictive tasks are modelled separately.

### Fuel customer value

Target:

    CLV_fuel

Evaluation:

    RMSE

The official transformed target supplied by the Phase 02 label
reconstruction is used directly.

### Non-fuel customer value

Target:

    CLV_nonfuel

Evaluation:

    RMSE

The official transformed target supplied by the Phase 02 label
reconstruction is used directly.

### Customer Opportunity

Target:

    Opportunity

Evaluation:

    weighted F1

The problem contains 17 official Opportunity classes.

---

## 3. Modelling Unit

The modelling observation remains:

    Customer x Forecasting Origin

The same customer may occur at multiple forecasting origins.

This is intentional.

Each observation represents the customer's information state at a
different point in time.

---

## 4. Feature Policy

The canonical Phase 03 representation contains:

    281 candidate predictors.

The frozen baseline modelling policy excludes five deterministic or
training-ambiguous variables:

    cat__fuel__days_since_last_purchase
    cat__fuel__ever_purchased
    cat__fuel__prev3_basket_count
    cat__fuel__m3_basket_count
    unseen_category_count

The baseline modelling representation therefore contains:

    276 predictors.

The canonical 281-feature research matrix remains unchanged.

---

## 5. Supervised Historical Population

Historical modelling snapshots occur at:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01
    2025-09-01

Expected labelled observations:

    22,679

Final competition observations at:

    2025-12-01

Expected unlabelled observations:

    5,488

---

## 6. Temporal Validation Strategy

Validation uses expanding-window rolling forecasting origins.

Random cross-validation is not used.

### Fold F1

Train:

    2024-09-01

Validate:

    2024-12-01

Expected:

    Train = 3,608
    Validation = 4,223

### Fold F2

Train:

    2024-09-01
    2024-12-01

Validate:

    2025-03-01

Expected:

    Train = 7,831
    Validation = 4,649

### Fold F3

Train:

    2024-09-01
    2024-12-01
    2025-03-01

Validate:

    2025-06-01

Expected:

    Train = 12,480
    Validation = 4,961

### Fold F4

Train:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01

Validate:

    2025-09-01

Expected:

    Train = 17,441
    Validation = 5,238

The final production model will later be trained using all five
historical origins before generating predictions for:

    2025-12-01.

---

## 7. Why Repeated Customers Across Folds Are Allowed

A customer may appear in training at an earlier forecasting origin and
in validation at a later forecasting origin.

This is not treated as leakage because each observation contains only
information available before its own cutoff.

For example:

    customer at 2025-06-01 -> training state

and:

    same customer at 2025-09-01 -> validation state.

Transactions occurring between the two dates legitimately become known
history by the later forecasting origin.

The validation design therefore represents sequential operational
forecasting rather than independent-person prediction.

---

## 8. Target-Window Boundary Rule

Every historical target represents the three calendar months beginning
at its forecasting origin.

For validation origin t, all training target windows must finish on or
before t.

Therefore:

    maximum training target end <= validation cutoff

must hold for every validation fold.

This prevents the model from being trained using outcomes that would
not yet have been observable at validation time.

---

## 9. Phase 04A Objective

Phase 04A will:

- assemble the 276-feature supervised historical dataset;
- independently reconcile Phase 03 features with Phase 02 labels;
- create the 5,488-row competition feature matrix;
- freeze the 17-class Opportunity encoding;
- generate exact temporal fold membership;
- validate target-window chronology;
- and produce modelling-ready audit files.

No LightGBM model is trained during Phase 04A.

---

## 10. Phase 04A Exit Criteria

Phase 04A passes only if:

    historical rows = 22,679
    competition rows = 5,488
    selected predictors = 276
    Opportunity classes = 17

and:

- feature-label joins are one-to-one;
- no historical target is missing;
- no unexpected customer-cutoff row appears;
- all temporal folds reproduce their expected populations;
- training target windows precede validation origins;
- every validation snapshot contains all required targets;
- and selected predictor values remain free of infinity.

---

## 11. Current Status

Phase 04A supervised dataset assembly is being implemented.

No predictive model has yet been trained.

---

## 12. Phase 04A Results - Supervised Dataset Assembly

Phase 04A successfully assembled the modelling-ready historical and
competition populations.

The canonical Phase 03 representation contained:

    281 predictors

The frozen feature policy excluded:

    5 predictors

leaving:

    276 baseline predictors.

Observed populations were:

    Historical feature rows     22,679
    Historical target rows      22,679
    Joined supervised rows      22,679
    Competition rows             5,488

Feature-label reconciliation found:

    features without labels        0
    labels without features        0

All 17 Opportunity classes were present.

---

## 13. Feature-Label Join Integrity

Every forecasting origin reconciled exactly between the Phase 03
feature matrix and Phase 02 transformed labels.

| Cutoff | Feature rows | Label rows | Joined rows |
|---|---:|---:|---:|
| 2024-09-01 | 3,608 | 3,608 | 3,608 |
| 2024-12-01 | 4,223 | 4,223 | 4,223 |
| 2025-03-01 | 4,649 | 4,649 | 4,649 |
| 2025-06-01 | 4,961 | 4,961 | 4,961 |
| 2025-09-01 | 5,238 | 5,238 | 5,238 |

At every cutoff:

    features_without_labels = 0
    labels_without_features = 0

The supervised join therefore preserves the intended:

    Customer x Forecasting Origin

target alignment.

---

## 14. Temporal Validation Population

The frozen expanding-window validation scheme produced:

| Fold | Train rows | Validation rows | Validation origin |
|---|---:|---:|---|
| F1 | 3,608 | 4,223 | 2024-12-01 |
| F2 | 7,831 | 4,649 | 2025-03-01 |
| F3 | 12,480 | 4,961 | 2025-06-01 |
| F4 | 17,441 | 5,238 | 2025-09-01 |

Every training and validation population contained all 17 official
Opportunity classes.

---

## 15. Target-Window Chronology

For every validation fold:

    latest training target end
        <=
    validation forecasting origin

Observed:

    F1 latest target end = 2024-12-01
    F2 latest target end = 2025-03-01
    F3 latest target end = 2025-06-01
    F4 latest target end = 2025-09-01

All temporal boundary checks passed.

The training process therefore does not use outcomes that would still
be in the future at the corresponding validation origin.

---

## 16. Historical Target Drift

Fuel customer value changes substantially through time.

Transformed mean CLV_fuel declines from approximately:

    1.898

at 2024-09-01 to:

    1.496

at 2025-09-01.

Fuel zero prevalence increases from:

    11.89%

to:

    29.25%.

Non-fuel customer value remains strongly zero-inflated.

Non-fuel zero prevalence rises from:

    57.65%

to approximately:

    65.06%.

The transformed median CLV_nonfuel remains:

    0

at every historical forecasting origin.

These results reinforce the need to report predictive performance by
forecasting origin rather than relying exclusively on one pooled score.

---

## 17. Phase 04A Decision

Phase 04A is complete because:

- all 22,679 historical feature snapshots were matched one-to-one with
  targets;
- all 5,488 competition snapshots were preserved;
- the frozen modelling representation contains 276 predictors;
- all 17 Opportunity classes occur in every historical origin;
- all four expanding-window validation folds reproduce their expected
  populations;
- and every training target window ends on or before its corresponding
  validation forecasting origin.

No predictive model has yet been trained.

The project may now proceed to:

    Phase 04B - First LightGBM Temporal Baseline

