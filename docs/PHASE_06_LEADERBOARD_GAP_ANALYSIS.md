# Phase 06 - Leaderboard Gap Analysis

## 1. External Baseline

The first frozen Phase 05 submission produced:

    Public score: 0.289392053
    Public rank: 23

This submission is designated:

    S0

Visible leaderboard references at the time included:

    Rank 7: 0.307567003
    Rank 1: 0.310346593

Observed gap from S0 to rank 7:

    0.018174950

Observed gap from S0 to rank 1:

    0.020954540

Leaderboard performance is treated as an external benchmark only.

It is not used to select hyperparameters or retrospectively modify
historical validation criteria.

---

## 2. Phase 06 Strategy

The gap to the leading group is sufficiently large that Phase 06 will
focus on modelling assumptions rather than fine-grained LightGBM
parameter search.

The principal unresolved hypotheses are:

1. temporal recency weighting;
2. ablation of the drift-sensitive observed_tenure_days predictor;
3. Opportunity architecture if classification remains the major
   unresolved weakness.

Phase 06A investigates temporal recency weighting first because temporal
distribution change was observed across all three competition targets.

---

## 3. Phase 06A - Temporal Recency Weighting

Historical target distributions changed substantially across forecasting
origins.

Examples observed during Phase 02 and Phase 04 include:

    Fuel zero-rate:
        approximately 11.9% -> 29.2%

    Inactivity prevalence:
        approximately 11.1% -> 27.4%

    Adoption prevalence:
        approximately 14.3% -> 7.0%

    Fuel transformed target mean:
        approximately 1.90 -> 1.50

The Phase 05 final models nevertheless assign equal training importance
to all historical forecasting origins.

Phase 06A tests whether giving greater weight to more recent customer
snapshots improves forward temporal generalization.

---

## 4. Pre-Registered Recency Policy

Exactly one temporal-weighting policy will be tested.

For a training origin lagged q quarters behind the most recent training
origin:

    temporal_weight = 2 ^ (-q / 2)

This corresponds to a two-quarter, approximately six-month half-life.

Raw relative weights are therefore approximately:

    newest origin        1.000
    one quarter older    0.707
    two quarters older   0.500
    three quarters older 0.354
    four quarters older  0.250

Weights are normalized to mean one within each training fold.

No alternative half-life will be tested as part of Phase 06A.

---

## 5. Model Specifications Remain Frozen

Fuel:

    LightGBM regression
    num_leaves = 15
    min_data_in_leaf = 100

Non-fuel:

    LightGBM regression
    num_leaves = 15
    min_data_in_leaf = 50

Opportunity:

    LightGBM multiclass
    num_leaves = 31
    min_data_in_leaf = 50
    class-weight exponent = 0.50

For Opportunity, temporal weights multiply the accepted class weights.

The combined weights are then normalized to mean one.

No feature or structural parameter changes are made.

---

## 6. Temporal Validation Design

Phase 06A evaluates folds F2, F3 and F4.

F1 is excluded from the acceptance decision because it contains only one
training origin, making relative recency weighting meaningless.

Evaluation remains fully forward-looking:

    F2:
        earlier origins -> 2025-03 validation

    F3:
        earlier origins -> 2025-06 validation

    F4:
        earlier origins -> 2025-09 validation

Validation observations remain unweighted.

---

## 7. Pre-Registered Acceptance Rule

Each target is evaluated independently.

For Fuel and non-fuel, the recency-weighted model is accepted only if:

1. row-weighted F2-F4 RMSE improves;
2. F4 RMSE does not worsen;
3. at least two of three folds improve.

For Opportunity, the recency-weighted model is accepted only if:

1. row-weighted F2-F4 weighted F1 improves;
2. F4 weighted F1 does not worsen;
3. at least two of three folds improve.

A target may accept recency weighting even if another target rejects it.

No combined local competition score is constructed because the official
leaderboard aggregation should not be approximated without an explicit
verified formula.

