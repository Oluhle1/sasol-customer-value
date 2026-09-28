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


---

## 8. Phase 06A Results - Temporal Recency Weighting

Phase 06A evaluated the pre-registered two-quarter temporal half-life
across validation folds F2-F4.

The intervention changed only the training observation weights.

All structural model parameters and predictors remained frozen.

### Fuel

Results:

    F2:
        accepted RMSE     0.631574
        recency RMSE      0.630047
        change           -0.001527

    F3:
        accepted RMSE     0.595067
        recency RMSE      0.595561
        change           +0.000493

    F4:
        accepted RMSE     0.600904
        recency RMSE      0.600903
        change           -0.000001

Two of three folds improved.

Row-weighted F2-F4 RMSE improved from:

    0.608557

to:

    0.608243.

Fuel therefore satisfies the pre-registered Phase 06A acceptance rule.

Temporal recency weighting is accepted for Fuel.

---

### Non-Fuel

Results:

    F2:
        accepted RMSE     0.782150
        recency RMSE      0.782179
        change           +0.000029

    F3:
        accepted RMSE     0.739133
        recency RMSE      0.737508
        change           -0.001624

    F4:
        accepted RMSE     0.742953
        recency RMSE      0.742726
        change           -0.000227

Two of three folds improved.

Row-weighted F2-F4 RMSE improved from:

    0.753949

to:

    0.753336.

Non-fuel therefore satisfies the pre-registered Phase 06A acceptance
rule.

Temporal recency weighting is accepted for non-fuel.

---

### Opportunity

Results:

    F2:
        accepted weighted F1     0.475097
        recency weighted F1      0.474661
        change                  -0.000437

    F3:
        accepted weighted F1     0.496634
        recency weighted F1      0.495572
        change                  -0.001063

    F4:
        accepted weighted F1     0.504765
        recency weighted F1      0.510985
        change                  +0.006220

Only one of three folds improved.

Although the row-weighted F2-F4 metric increased from:

    0.492759

to:

    0.494462,

the experiment fails the pre-registered requirement that at least two
of three folds improve.

Temporal recency weighting is therefore rejected for Opportunity.

The strong F4 improvement is retained as evidence of possible
latest-regime sensitivity but does not override the pre-registered
selection rule.

---

## 9. Phase 06A Decision

Accepted:

    Fuel:
        temporal half-life = 2 quarters

    Non-fuel:
        temporal half-life = 2 quarters

Rejected:

    Opportunity:
        temporal weighting

The accepted Opportunity model therefore remains the Phase 04D
square-root class-weighted multiclass LightGBM without temporal
recency weighting.

The Phase 06A regression improvements are statistically consistent
enough to retain but small in absolute magnitude.

They are not interpreted as sufficient to explain the current public
leaderboard gap.


---

## 10. Phase 06B - Drift-Sensitive Feature Ablation

Phase 03E identified observed_tenure_days as the strongest distribution
shift between the latest historical feature origin and the final
competition origin.

Its latest-to-final standardized mean difference was approximately:

    +0.53

It was the only predictor with absolute SMD greater than or equal to:

    0.50

Phase 06B therefore tests whether dependence on this feature reduces
forward temporal generalization.

Exactly one predictor is removed:

    observed_tenure_days

The modelling feature count changes from:

    276

to:

    275.

No additional feature is added or modified.

---

## 11. Phase 06B Control Models

The ablation is compared against the currently accepted specification
for each target.

Fuel control:

    LightGBM regression
    num_leaves = 15
    min_data_in_leaf = 100
    Phase 06A temporal recency weighting enabled
    half-life = 2 quarters

Non-fuel control:

    LightGBM regression
    num_leaves = 15
    min_data_in_leaf = 50
    Phase 06A temporal recency weighting enabled
    half-life = 2 quarters

Opportunity control:

    LightGBM multiclass
    num_leaves = 31
    min_data_in_leaf = 50
    class-weight exponent = 0.50
    temporal recency weighting disabled

Thus the only Phase 06B intervention is removal of:

    observed_tenure_days

---

## 12. Phase 06B Acceptance Rule

Validation folds:

    F2
    F3
    F4

For Fuel and non-fuel, the ablation is accepted only if:

1. row-weighted F2-F4 RMSE improves;
2. F4 RMSE does not deteriorate;
3. at least two of three folds improve.

For Opportunity, the ablation is accepted only if:

1. row-weighted F2-F4 weighted F1 improves;
2. F4 weighted F1 does not deteriorate;
3. at least two of three folds improve.

These rules are defined before Phase 06B results are observed.


---

## 13. Phase 06B Results - Drift-Sensitive Feature Ablation

Phase 06B tested removal of:

    observed_tenure_days

This predictor had previously been identified as the strongest
latest-to-final distribution shift, with absolute standardized mean
difference of approximately:

    0.53

The feature count changed from:

    276

to:

    275

for the experimental models.

No other predictor or model parameter was changed.

---

### Fuel

Results:

    F2:
        control RMSE       0.630047
        no-tenure RMSE     0.629614
        change            -0.000433

    F3:
        control RMSE       0.595561
        no-tenure RMSE     0.596055
        change            +0.000495

    F4:
        control RMSE       0.600903
        no-tenure RMSE     0.601321
        change            +0.000417

Only one of three folds improved.

Row-weighted F2-F4 RMSE worsened from:

    0.608243

to:

    0.608420.

The Fuel ablation is rejected.

---

### Non-Fuel

Results:

    F2:
        control RMSE       0.782179
        no-tenure RMSE     0.784078
        change            +0.001899

    F3:
        control RMSE       0.737508
        no-tenure RMSE     0.739711
        change            +0.002202

    F4:
        control RMSE       0.742726
        no-tenure RMSE     0.743380
        change            +0.000655

No validation fold improved.

Row-weighted F2-F4 RMSE worsened from:

    0.753336

to:

    0.754897.

The non-fuel ablation is rejected.

---

### Opportunity

Results:

    F2:
        control weighted F1       0.475097
        no-tenure weighted F1     0.477869
        change                   +0.002772

    F3:
        control weighted F1       0.496634
        no-tenure weighted F1     0.495417
        change                   -0.001217

    F4:
        control weighted F1       0.504765
        no-tenure weighted F1     0.496956
        change                   -0.007809

Only one of three folds improved.

Row-weighted F2-F4 weighted F1 fell from:

    0.492759

to:

    0.490466.

The deterioration was especially pronounced at the latest validation
origin.

The Opportunity ablation is rejected.

---

## 14. Phase 06B Decision

The observed_tenure_days predictor is retained for all three targets.

Although its marginal feature distribution shifts between the latest
historical and competition origins, temporal ablation demonstrates that
removing it reduces predictive performance.

Therefore distribution shift alone is not treated as sufficient
evidence for feature removal.

Current accepted state remains:

    Fuel:
        276 predictors
        Phase 06A recency weighting enabled

    Non-fuel:
        276 predictors
        Phase 06A recency weighting enabled

    Opportunity:
        276 predictors
        alpha 0.50 class weighting
        no temporal recency weighting


---

## 15. Phase 06C - Rule-Aware Hierarchical Opportunity Architecture

The accepted Opportunity model treats the target as a single 17-class
multiclass problem.

However, the official target construction has an explicit behavioural
hierarchy:

    Inactivity
    Stable
    Existing-category Growth
    New-category Adoption

Growth then contains four output categories.

Adoption contains eleven output categories.

Phase 04 error analysis also showed that several Growth and Adoption
classes contain ranking signal without frequently becoming the
highest-probability class in the direct 17-class model.

Phase 06C therefore tests a hierarchical probability decomposition.

---

## 16. Hierarchical Architecture

Three LightGBM multiclass models are trained.

Model 1 - Macro Behaviour:

    Growth
    Adoption
    Inactivity
    Stable

Training population:

    all training observations

Model 2 - Growth Subtype:

    Existing-category growth: Beverages
    Existing-category growth: Food & beverage
    Existing-category growth: Fuel
    Existing-category growth: Other

Training population:

    true Growth observations only

Model 3 - Adoption Subtype:

    eleven official New category adoption labels

Training population:

    true Adoption observations only

---

## 17. Probability Composition

Hard routing is not used.

Final Growth probabilities are:

    P(Growth class j)
        =
    P(Growth)
        *
    P(class j | Growth)

Final Adoption probabilities are:

    P(Adoption class k)
        =
    P(Adoption)
        *
    P(class k | Adoption)

Inactivity and Stable probabilities are taken directly from the macro
classifier.

The resulting 17-class probability vector must sum to one.

Final prediction remains:

    argmax final class probability.

---

## 18. Phase 06C Fixed Modelling Policy

The experiment remains within the LightGBM modelling framework.

All three stages use:

    num_leaves = 31
    min_data_in_leaf = 50

The previously accepted class-balancing principle is retained:

    inverse-frequency exponent = 0.50

and is calculated independently within each classification stage.

Features:

    all 276 accepted predictors

Temporal recency weighting:

    disabled

This is consistent with the accepted Opportunity control model.

No structural hyperparameter search is performed.

---

## 19. Phase 06C Acceptance Rule

Validation folds:

    F2
    F3
    F4

The hierarchical Opportunity model is accepted only if:

1. row-weighted F2-F4 Weighted F1 improves;
2. F4 Weighted F1 does not deteriorate;
3. at least two of three validation folds improve.

Macro-level and subtype-level metrics are diagnostic only.

The final 17-class Weighted F1 remains the model-selection criterion.


---

## 20. Phase 06C Results - Hierarchical Opportunity Architecture

The hierarchical Opportunity architecture was evaluated across F2-F4.

Results:

| Fold | Direct control F1 | Hierarchical F1 | Change |
|---|---:|---:|---:|
| F2 | 0.475097 | 0.473146 | -0.001951 |
| F3 | 0.496634 | 0.472702 | -0.023932 |
| F4 | 0.504765 | 0.469738 | -0.035027 |

No temporal fold improved.

Row-weighted F2-F4 weighted F1 fell from:

    0.492759

to:

    0.471795.

The experiment therefore fails every pre-registered acceptance
criterion and is rejected.

---

## 21. Structural Failure Analysis

The failure reveals an important problem with the probability
composition used in this experiment.

Stable and Inactivity are singleton macro classes and therefore retain
their complete macro probability:

    P(Stable)
    P(Inactivity)

Growth probability is distributed among four child classes:

    P(Growth class j)
        =
    P(Growth) * P(class j | Growth)

Adoption probability is distributed among eleven child classes:

    P(Adoption class k)
        =
    P(Adoption) * P(class k | Adoption)

The final argmax therefore compares full singleton macro probabilities
against fractions of the Growth and Adoption macro probabilities.

This creates a structural granularity disadvantage for classes that
belong to multi-child branches.

The effect is especially severe for Adoption because its macro
probability is distributed across eleven child labels.

At F4 the hierarchical model predicted:

    6 of 17 final classes

and predicted:

    0 Adoption observations.

This occurred even though the conditional Adoption subtype classifier
achieved weighted F1 of approximately:

    0.254692

on true Adoption observations.

Therefore the experiment does not establish that hierarchical
classification is inherently unsuitable.

It establishes that the specific probability-composition and final
argmax formulation used in Phase 06C is unsuitable for this target
structure.

---

## 22. Phase 06C Decision

The hierarchical Opportunity architecture is rejected.

The accepted Opportunity control remains:

    Direct 17-class LightGBM
    276 predictors
    num_leaves = 31
    min_data_in_leaf = 50
    class-weight exponent = 0.50

No revised hierarchical architecture will be attempted until the
methodological basis for further Opportunity modelling has been
reviewed.


---

## 23. Phase 06C Public Leaderboard Diagnostic

The Phase 06C hierarchical Opportunity architecture was submitted as a
controlled external diagnostic.

The submission preserved the Phase 05 Fuel and non-fuel predictions
exactly and replaced only the Opportunity predictions.

Control submission:

    S0
    Public score = 0.289392053

Phase 06C diagnostic submission:

    S1
    Public score = 0.283652312

Absolute public-score change:

    -0.005739741

Relative change from S0:

    approximately -1.98%

Because Fuel and non-fuel predictions were unchanged between S0 and S1,
the public-score deterioration is attributable to the Opportunity model
replacement.

This external result agrees directionally with historical temporal
validation.

Phase 06C local F4 weighted F1:

    Direct control      0.504765
    Hierarchical        0.469738

Phase 06C row-weighted F2-F4 weighted F1:

    Direct control      0.492759
    Hierarchical        0.471795

The hierarchical architecture therefore performs worse both locally and
on the public leaderboard.

This strengthens confidence in the temporal validation framework as a
model-selection mechanism.

The Phase 06C architecture remains rejected.


---

# Phase 06R-G0 — Exact Grok Architecture Temporal Replication

Public benchmark:

    0.301425001

Purpose:

Determine whether the complete architecture that achieved the stronger
public leaderboard score also generalises under nested temporal
validation.

F1 was excluded because G0 requires an earlier labelled period for
internal recipe selection.

Nested outer folds:

    F2: tune Dec-2024 -> evaluate Mar-2025
    F3: tune Mar-2025 -> evaluate Jun-2025
    F4: tune Jun-2025 -> evaluate Sep-2025

## Results

Fuel:

    S0 row-weighted RMSE     0.608626
    G0 row-weighted RMSE     0.611020
    folds won by G0          1 / 3

    F4:
        S0                  0.600904
        G0                  0.605988

Decision:

    REJECT

Nonfuel:

    S0 row-weighted RMSE     0.754297
    G0 row-weighted RMSE     0.757748
    folds won by G0          1 / 3

    F4:
        S0                  0.742953
        G0                  0.750256

Decision:

    REJECT

Opportunity:

    S0 row-weighted F1       0.485527
    G0 row-weighted F1       0.485776
    folds won by G0          2 / 3

    F4:
        S0                  0.504765
        G0                  0.499061

The aggregate difference was only:

    +0.000249 weighted F1

and the latest temporal origin deteriorated materially.

## Interpretation

The 0.301425001 public leaderboard result is retained as valid
same-challenge empirical evidence.

However, the full architecture did not demonstrate robust superiority
under nested temporal validation.

Fuel and nonfuel both deteriorated overall.

Opportunity was effectively tied overall and deteriorated at the latest
historical forecasting origin.

The selected internal recipes also varied across temporal origins.

## Decision

    PROMOTE G0 DIRECTION: NO

The G0 architecture will not replace S0.

Individual G0 components will not be searched independently merely
because the public score was higher.

S0 remains the primary validated architecture.

Phase 06R-G0 is closed.

Next research hypothesis:

    Phase 06R-C
    Sequence-aware CLV representation.

