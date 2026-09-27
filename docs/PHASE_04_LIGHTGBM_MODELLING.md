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


---

## 18. Phase 04B Results - First LightGBM Temporal Baseline

The first fixed-methodology LightGBM baseline was successfully trained
across all four expanding-window temporal validation folds.

Three models were trained per fold:

    transformed CLV_fuel regression
    transformed CLV_nonfuel regression
    17-class Opportunity classification

This produced:

    4 folds x 3 tasks
    = 12 fitted LightGBM models.

No hyperparameter search, class reweighting, ensembling or additional
feature engineering was performed.

---

## 19. Temporal Baseline Metrics

Observed validation results were:

| Fold | Validation cutoff | Fuel RMSE | Non-fuel RMSE | Opportunity weighted F1 |
|---|---|---:|---:|---:|
| F1 | 2024-12-01 | 0.624804 | 0.823599 | 0.446296 |
| F2 | 2025-03-01 | 0.633544 | 0.783620 | 0.474763 |
| F3 | 2025-06-01 | 0.596600 | 0.740761 | 0.490870 |
| F4 | 2025-09-01 | 0.602348 | 0.743258 | 0.503033 |

Pooled out-of-fold performance was:

    Fuel RMSE                    0.613621
    Non-fuel RMSE               0.770952
    Opportunity weighted F1     0.482080

These values define the untuned temporal baseline against which later
controlled improvements will be evaluated.

---

## 20. Opportunity Temporal Pattern

Opportunity weighted F1 increases through the successive validation
origins:

    0.446296
    0.474763
    0.490870
    0.503033

The absolute difference between the earliest and latest validation
origins is approximately:

    +0.0567 weighted F1

or approximately:

    +12.7% relative to the F1 score.

This improvement is consistent with a combination of increased training
population, more mature historical customer representations and reduced
distributional distance between later historical snapshots and the
competition environment.

The relative contribution of these mechanisms has not yet been
established and will be investigated during error analysis.

---

## 21. Non-Fuel Temporal Pattern

Non-fuel RMSE improves substantially from:

    0.823599

at F1 to:

    0.743258

at F4.

The reduction is approximately:

    9.8%.

The strongest observed non-fuel result is:

    F3 RMSE = 0.740761.

This improvement occurs despite the target remaining strongly
zero-inflated across all forecasting origins.

The later regression models therefore appear to extract useful signal
from the temporal customer representation.

---

## 22. Fuel Temporal Pattern

Fuel validation performance is comparatively stable.

Observed RMSE values were:

    F1    0.624804
    F2    0.633544
    F3    0.596600
    F4    0.602348

The strongest observed fold was:

    F3 RMSE = 0.596600.

Unlike Opportunity and non-fuel, fuel does not exhibit monotonic
improvement across validation origins.

Error decomposition is required before making feature or parameter
changes.

---

## 23. Regression Prediction Floor

Very few raw LightGBM regression predictions were below zero.

Observed raw negative prediction counts were:

    Fuel:
        F1    0
        F2    0
        F3    1
        F4    2

    Non-fuel:
        all folds    0

The fixed zero-floor post-processing therefore has negligible effect on
the current baseline.

---

## 24. Baseline Best Iterations

Observed early-stopping iterations were:

| Fold | Fuel | Non-fuel | Opportunity |
|---|---:|---:|---:|
| F1 | 102 | 62 | 57 |
| F2 | 160 | 87 | 99 |
| F3 | 133 | 126 | 46 |
| F4 | 145 | 128 | 147 |

Regression best-iteration values are relatively stable as training data
expands.

Opportunity best iteration is more variable across folds.

This variation is recorded for later error analysis and controlled
parameter selection rather than immediately triggering hyperparameter
search.

---

## 25. Phase 04B Decision

Phase 04B is considered complete.

The first LightGBM temporal baseline has established reproducible
performance across all four rolling forecasting origins.

The baseline will not yet be tuned.

The next stage is:

    Phase 04C - Baseline Error Analysis

Phase 04C will determine where the models fail before any modelling
changes are made.


---

## 26. Phase 04C - Baseline Error Analysis

Phase 04C does not tune or replace the baseline models.

Its purpose is to identify where the fixed LightGBM baseline fails before
any modelling intervention is introduced.

The analysis focuses on three distinct questions.

### Fuel regression

Determine whether residual error is concentrated among:

    future-zero customers
    future-positive customers
    high-value customers
    dormant or reactivated customers
    abrupt behavioural changes

### Non-fuel regression

Because future non-fuel value is strongly zero-inflated, distinguish:

    zero-versus-positive discrimination

from:

    positive-value magnitude estimation.

This is a diagnostic decomposition only.

No two-stage model is introduced during Phase 04C.

### Opportunity classification

Determine:

    predicted class distribution
    full 17-class confusion structure
    macro behaviour confusion
    adoption-versus-growth behaviour
    probability assigned to true rare classes
    true-class rank
    top-k recovery

This analysis is particularly important because the initial F4 baseline
has zero recall for all New-category adoption classes.

---

## 27. Initial F4 Opportunity Finding

At the latest labelled forecasting origin the three highest-support
classes dominate the validation population:

    Inactivity                       1,436
    Existing-category growth: Fuel  1,435
    Stable                           1,404

Together these classes account for approximately 81.6% of the F4
population.

Observed F4 F1 scores were approximately:

    Inactivity                       0.748
    Stable                           0.577
    Existing-category growth: Fuel  0.461

All eleven New-category adoption classes recorded zero recall.

The baseline weighted F1 of approximately 0.503 therefore reflects
substantial success on the dominant behavioural states but not yet
successful prediction across the complete 17-class label space.

---

## 28. Initial Feature-Importance Finding

Opportunity importance is led by:

    purchase recency
    recent Fuel momentum
    Fuel monetary state
    category-specific recency
    purchase cadence
    observed history depth

Fuel regression is dominated by recent Fuel quantity and recency.

Non-fuel regression is dominated by recent and medium-term non-fuel
active days, suggesting that purchase occurrence and regularity are
central to the target.

Phase 04C will quantify these failure modes before any parameter or
method change is accepted.


---

## 29. Phase 04C Results - Regression Error Decomposition

The F4 regression models exhibit systematic regression toward intermediate
values.

### Fuel

For future-zero customers:

    Rows                1,532
    True mean           0.000
    Prediction mean     0.563
    RMSE                0.743

For future-positive customers:

    Rows                3,706
    True mean           2.114
    Prediction mean     1.866
    RMSE                0.533

The model therefore overpredicts customers with zero future Fuel value
while underpredicting positive customers.

Despite this, the continuous Fuel prediction strongly ranks future
positive activity:

    ROC AUC = 0.921

This indicates that substantial occurrence information exists in the
Phase 03 representation.

---

## 30. Non-Fuel Error Decomposition

The zero-versus-positive shrinkage pattern is stronger for non-fuel.

For future-zero customers:

    Rows                3,408
    True mean           0.000
    Prediction mean     0.412
    RMSE                0.551

For future-positive customers:

    Rows                1,830
    True mean           2.018
    Prediction mean     1.220
    RMSE                1.008

The model therefore substantially underestimates future-positive
non-fuel customer value.

However, positive-outcome ranking remains informative:

    ROC AUC = 0.841

This motivates a later zero-inflation hypothesis based on decomposing
expected value into occurrence probability and conditional positive
magnitude.

No such model is introduced during Phase 04C.

---

## 31. Opportunity Macro Behaviour

F4 macro-level performance was:

| Macro state | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Adoption | 368 | 0.250 | 0.005 | 0.011 |
| Growth | 2,030 | 0.591 | 0.458 | 0.516 |
| Inactivity | 1,436 | 0.716 | 0.783 | 0.748 |
| Stable | 1,404 | 0.482 | 0.717 | 0.577 |

The most severe baseline failure is Adoption recall.

Only approximately 0.5% of actual adoption cases are recovered at
top-1.

---

## 32. Opportunity Prediction Distribution

The classifier strongly favours the dominant Opportunity classes.

Observed F4 counts included:

    True Stable               1,404
    Predicted Stable          2,088

    True Inactivity           1,436
    Predicted Inactivity      1,569

    True Fuel growth          1,435
    Predicted Fuel growth     1,309

The F4 validation population contained:

    368 true Adoption observations.

Across all eleven adoption classes, the classifier produced only:

    8 Adoption predictions.

Only two of these were correct.

The resulting Adoption macro precision was therefore approximately:

    0.25

while recall remained approximately:

    0.005.

This identifies an extremely conservative rare-class decision pattern.

---

## 33. Opportunity Probability and Rank Diagnostics

Top-k diagnostics show that some rare classes contain substantially more
model signal than top-1 recall suggests.

Examples include:

    New category adoption: Beverages
        Top-1 recall    0.0%
        Top-3 recall   20.6%
        Top-5 recall   76.2%

    New category adoption: Food & beverage
        Top-5 recall   52.0%

    New category adoption: Lubricants
        Top-5 recall   44.4%

Similarly:

    Existing-category growth: Beverages
        Top-1 recall     2.1%
        Top-5 recall    96.3%

These cases indicate that the correct class frequently receives
meaningful model probability but fails to defeat higher-prior dominant
classes under the ordinary argmax decision.

Other rare adoption classes display much weaker evidence.

Examples include Snacking, Ice-cream, Groceries, Breads & rolls and
Braai & Ice, where top-5 recovery is zero or nearly zero.

Phase 04C therefore identifies both:

    probability suppression caused by class imbalance

and:

    genuinely weak rare-class discrimination.

These mechanisms should not be treated as equivalent.

---

## 34. Phase 04C Decision

The first controlled modelling intervention will target Opportunity
class imbalance.

The baseline classifier used no class weighting.

Phase 04D will retain:

    the same 276 predictors
    the same LightGBM algorithm
    the same temporal folds
    the same tree hyperparameters
    the same early-stopping metric
    the same argmax decision rule

and change only:

    training observation weights.

A fixed square-root inverse-frequency weighting rule will be evaluated.

This moderated weighting is preferred over full inverse-frequency
weighting for the first experiment because the official metric is
weighted F1 and the high-support classes remain important.

Regression methodology will remain unchanged during this experiment.

The separate zero-inflation regression hypothesis is retained for a
later controlled experiment.


---

## 35. Phase 04D Results - Controlled Opportunity Class Weighting

Phase 04D changed exactly one modelling assumption relative to the
Phase 04B Opportunity baseline.

Training observations were assigned square-root inverse-frequency class
weights calculated independently from each fold's training population.

All other components remained fixed:

    276 predictors
    LightGBM multiclass model
    four temporal validation folds
    tree parameters
    random seed
    weighted-F1 early stopping
    argmax prediction rule

No regression model was changed.

---

## 36. Temporal Weighted-F1 Comparison

Observed results were:

| Fold | Baseline weighted F1 | Weighted-model F1 | Change |
|---|---:|---:|---:|
| F1 | 0.446296 | 0.450449 | +0.004154 |
| F2 | 0.474763 | 0.475097 | +0.000334 |
| F3 | 0.490870 | 0.496634 | +0.005764 |
| F4 | 0.503033 | 0.504765 | +0.001732 |

The intervention improved weighted F1 at every temporal validation
origin.

The row-weighted temporal result improved from:

    0.480414

to:

    0.483390

for an absolute increase of:

    +0.002976.

The latest-origin F4 score improved from:

    0.503033

to:

    0.504765.

---

## 37. Minority Growth-Class Effect

The strongest F4 gains occurred among smaller Existing-category growth
classes.

Examples include:

    Existing-category growth: Other
        Baseline F1     approximately 0.262
        Weighted F1     approximately 0.318

    Existing-category growth: Beverages
        Baseline F1     approximately 0.038
        Weighted F1     approximately 0.143

    Existing-category growth: Food & beverage
        Baseline F1     approximately 0.068
        Weighted F1     approximately 0.142

The weighting intervention therefore allows the multiclass model to use
more of the category-specific signal developed in Phase 03D.

---

## 38. Dominant-Class Trade-Off

Improvement in minority classes is accompanied by some deterioration in
large classes.

At F4:

    Inactivity
        F1 remains approximately 0.748.

    Existing-category growth: Fuel
        F1 decreases from approximately 0.461
        to approximately 0.448.

    Stable
        F1 decreases from approximately 0.577
        to approximately 0.563.

The official weighted-F1 metric nevertheless improves, indicating that
the minority-class gains outweigh these losses under the competition
evaluation structure.

---

## 39. Adoption Remains Unresolved

Moderate class weighting does not solve the New-category adoption
problem.

At F4, only Beverages and Lubricants adoption achieve non-zero recall:

    New category adoption: Beverages
        recall approximately 0.016

    New category adoption: Lubricants
        recall approximately 0.019

Most remaining adoption classes continue to record zero recall.

The model therefore remains weak at distinguishing individual rare
adoption outcomes even after majority-class suppression is reduced.

This confirms the Phase 04C finding that some adoption classes contain
usable but suppressed signal while others exhibit genuinely weak
class-specific discrimination.

---

## 40. Prediction-Space Breadth

The weighted F4 model predicts:

    13 of the 17 official classes.

This is broader than the highly concentrated baseline decision pattern
and is consistent with improved macro-level class coverage.

The F4 weighted-model macro F1 is:

    0.142461.

The corresponding baseline macro F1 is approximately:

    0.127.

Class weighting therefore produces a larger relative improvement in
label-space coverage than in the official weighted-F1 metric.

Weighted F1 remains the primary acceptance criterion.

---

## 41. Phase 04D Decision

The square-root inverse-frequency weighting experiment is accepted.

The reasons are:

- weighted F1 improved at all four temporal validation origins;
- row-weighted temporal F1 improved;
- latest-origin F4 performance improved;
- minority growth-class discrimination improved materially;
- prediction coverage expanded across the label space;
- and no change to features, algorithm family or validation methodology
  was required.

The gain is modest and does not resolve the rare Adoption classes.

Therefore stronger class weighting is not introduced immediately.

The accepted Opportunity training policy is now:

    LightGBM multiclass
    +
    square-root inverse-frequency training weights.

Further changes must continue to be evaluated as isolated temporal
experiments.


---

## 42. Regression Magnitude-Bin Diagnostics

Phase 04C positive-target magnitude analysis was reviewed before
changing the regression formulation.

Fuel positive-target RMSE remains approximately between 0.48 and 0.60
across the five positive-value quintiles.

The highest-value Fuel quintile does not exhibit a uniquely severe error
increase.

This indicates that the remaining Fuel error is not primarily an
extreme-target-tail problem.

Non-fuel displays stronger systematic shrinkage.

Across positive-target quintiles, mean predictions remain substantially
below typical observed target values.

Approximate positive non-fuel quintile results were:

| Quintile | Target median | Mean prediction | RMSE |
|---|---:|---:|---:|
| 1 | 1.446 | 0.807 | 0.801 |
| 2 | 1.743 | 0.871 | 1.049 |
| 3 | 1.981 | 1.062 | 1.116 |
| 4 | 2.246 | 1.387 | 1.076 |
| 5 | 2.625 | 1.974 | 0.966 |

The error is therefore not confined to the highest-value positive
customers.

Together with the Phase 04C zero-status analysis, the dominant pattern
is:

    zero customers      -> systematic overprediction
    positive customers  -> systematic underprediction

This supports testing a zero-inflation-aware expected-value formulation.

---

## 43. Feature-Importance Concentration

F4 gain concentration was:

### Fuel

    Top 5 features       86.2%
    Top 10 features      90.7%
    Top 20 features      95.0%
    Top 50 features      98.6%

### Non-fuel

    Top 5 features       63.8%
    Top 10 features      73.7%
    Top 20 features      83.0%
    Top 50 features      94.6%

### Opportunity

    Top 5 features       20.9%
    Top 10 features      34.3%
    Top 20 features      52.5%
    Top 50 features      79.1%

Fuel therefore relies on a highly concentrated set of recent quantity
and recency predictors.

Non-fuel uses a broader behavioural representation.

These findings do not currently support increasing general LightGBM
capacity as the first regression intervention.

---

## 44. Phase 04E Hypothesis

Phase 04E will test a two-part expected-value model for non-fuel only.

The official transformed target satisfies:

    CLV_nonfuel >= 0

and can therefore be decomposed as:

    E[CLV_nonfuel | X]
        =
    P(CLV_nonfuel > 0 | X)
        *
    E[CLV_nonfuel | CLV_nonfuel > 0, X]

The first component will be a LightGBM binary classifier.

The second component will be a LightGBM regression model trained only
on positive non-fuel targets.

The final prediction will be the product of occurrence probability and
conditional positive magnitude.

The calculation remains entirely in the official transformed target
space.

Fuel modelling remains unchanged during this experiment so that the
effect of the new formulation can be evaluated independently.


---

## 45. Phase 04E Results - Two-Part Non-Fuel Expected Value

Phase 04E evaluated a two-part formulation for transformed non-fuel
customer value.

The model decomposed expected value as:

    P(CLV_nonfuel > 0 | X)

multiplied by:

    E[CLV_nonfuel | CLV_nonfuel > 0, X].

The same 276 predictors, temporal folds, LightGBM family and common tree
configuration were retained.

---

## 46. Temporal RMSE Comparison

Observed results were:

| Fold | Direct baseline RMSE | Two-part RMSE | Change |
|---|---:|---:|---:|
| F1 | 0.823599 | 0.818754 | -0.004845 |
| F2 | 0.783620 | 0.781686 | -0.001934 |
| F3 | 0.740761 | 0.738296 | -0.002465 |
| F4 | 0.743258 | 0.744649 | +0.001391 |

Three of the four temporal folds improved.

Pooled temporal RMSE changed from:

    0.770952

to:

    0.769082

for an absolute reduction of:

    0.001870.

The row-weighted mean fold RMSE changed from:

    0.770238

to:

    0.768435.

However, the latest and most competition-proximate validation origin
did not improve.

At F4:

    Direct baseline RMSE    0.743258
    Two-part RMSE           0.744649

The deterioration was:

    +0.001391 RMSE.

---

## 47. Occurrence Discrimination

The non-fuel occurrence classifier achieved ROC AUC values of:

    F1    0.799015
    F2    0.809955
    F3    0.841902
    F4    0.840508

Future non-fuel occurrence therefore contains substantial predictable
signal.

Discrimination improves as historical training data expands and remains
strong at the latest validation origin.

---

## 48. Conditional Positive Magnitude

The positive-only magnitude model performed substantially better than
the final combined prediction when evaluated among actual positive
customers.

Observed positive-magnitude RMSE was:

    F1    0.360995
    F2    0.344928
    F3    0.366504
    F4    0.349306

This indicates that, conditional on future non-fuel value being
positive, magnitude estimation is comparatively strong.

The principal difficulty in the two-part formulation therefore appears
to involve the occurrence-probability component and its interaction with
the conditional magnitude prediction.

---

## 49. F4 Zero and Positive Decomposition

At F4, the two-part formulation did not improve either zero-target or
positive-target RMSE.

True-zero customers:

    Direct RMSE      0.551164
    Two-part RMSE    0.553873

True-positive customers:

    Direct RMSE      1.007718
    Two-part RMSE    1.007890

The latest-origin deterioration is therefore not caused by an obvious
trade-off between improved zero predictions and worsened positive
predictions.

Both groups deteriorate slightly.

---

## 50. Phase 04E Decision

The two-part non-fuel formulation is retained as a promising research
candidate but is not promoted to the current accepted production
specification.

Reasons:

- three of four temporal folds improve;
- pooled RMSE improves;
- occurrence discrimination is strong;
- conditional positive-magnitude prediction is strong;
- but the nearest competition-like F4 origin deteriorates slightly;
- and the overall gain remains small.

The accepted non-fuel specification therefore remains:

    direct LightGBM regression

pending further diagnosis.

The next analysis will investigate probability calibration of the
two-part occurrence component.

No additional model architecture will be introduced until that
diagnostic is complete.


---

## 51. Phase 04F Results - Non-Fuel Occurrence Calibration

Phase 04F audited the probability calibration of the Phase 04E
non-fuel occurrence classifier.

No model was trained.

Calibration improved substantially as forecasting origins advanced.

| Fold | Validation prevalence | Mean probability | Bias | ECE |
|---|---:|---:|---:|---:|
| F1 | 0.3758 | 0.4477 | +0.0719 | 0.0733 |
| F2 | 0.3500 | 0.3823 | +0.0324 | 0.0345 |
| F3 | 0.3447 | 0.3491 | +0.0044 | 0.0212 |
| F4 | 0.3494 | 0.3435 | -0.0058 | 0.0132 |

The nearest competition-like validation origin therefore exhibits only
minor global calibration bias.

---

## 52. F4 Reliability

The F4 reliability table shows close correspondence between predicted
probability and observed occurrence across most of the probability
distribution.

Examples include:

    Decile 7
        Mean predicted probability    0.366
        Observed positive rate        0.371

    Decile 8
        Mean predicted probability    0.531
        Observed positive rate        0.534

    Decile 10
        Mean predicted probability    0.925
        Observed positive rate        0.939

The largest local discrepancy occurs around the middle of the
probability distribution and is not large enough to explain the
Phase 04E F4 RMSE result.

F4 occurrence diagnostics were:

    ROC AUC             0.8405
    Brier score         0.1469
    Binary log loss     0.4537
    ECE                 0.0132

---

## 53. Temporal Prevalence Drift

Early-fold probability overprediction coincides with declining non-fuel
positive prevalence.

For F1:

    Training prevalence      42.35%
    Validation prevalence    37.58%
    Mean prediction          44.77%

For F2:

    Training prevalence      39.78%
    Validation prevalence    35.00%
    Mean prediction          38.23%

By F3 and F4, predicted prevalence closely tracks the later validation
environment.

The calibration behaviour therefore provides further evidence of
temporal population evolution rather than a persistent probability
calibration defect.

---

## 54. Pooled Calibration Interpretation

Across all temporal validation observations:

    Actual positive prevalence       0.3542
    Mean predicted probability       0.3775
    Probability bias                 +0.0234
    Expected calibration error       0.0271

This pooled result combines different historical regimes and is not used
to derive a global probability correction.

The latest forecasting-origin behaviour is considered more relevant to
the final competition environment.

---

## 55. Phase 04F Decision

Probability calibration is not identified as the principal limitation
of the Phase 04E two-part model at the latest validation origin.

At F4:

- occurrence discrimination is strong;
- occurrence probability calibration is good;
- conditional positive-magnitude prediction is strong;
- but the final two-part expected-value RMSE remains slightly worse than
  direct regression.

No probability recalibration method is therefore introduced.

The two-part architecture remains a documented research candidate but is
not promoted to the accepted model.

The current accepted regression specifications remain:

    Fuel:
        direct LightGBM regression

    Non-fuel:
        direct LightGBM regression

The accepted Opportunity specification remains:

    LightGBM multiclass
    with square-root inverse-frequency training weights.


---

## 56. Phase 04G Results - Regression Generalization Gap

Phase 04G applied the saved Phase 04B regression models to their original
training and validation populations.

No model was retrained.

The purpose was to quantify whether the fixed baseline appeared strongly
overfit, balanced, or potentially capacity-limited before tuning.

---

## 57. Fuel Generalization

Observed Fuel results were:

| Fold | Train RMSE | Validation RMSE | Gap | Validation / Train |
|---|---:|---:|---:|---:|
| F1 | 0.501639 | 0.624804 | 0.123166 | 1.246 |
| F2 | 0.507726 | 0.633544 | 0.125819 | 1.248 |
| F3 | 0.550343 | 0.596600 | 0.046257 | 1.084 |
| F4 | 0.554273 | 0.602348 | 0.048075 | 1.087 |

The early historical folds show substantial train-validation separation.

The gap becomes much smaller at the later forecasting origins but
remains material relative to non-fuel.

Fuel therefore does not provide strong evidence that a larger tree
capacity should be the first tuning intervention.

The accepted Fuel baseline remains unchanged.

---

## 58. Non-Fuel Generalization

Observed non-fuel results were:

| Fold | Train RMSE | Validation RMSE | Gap | Validation / Train |
|---|---:|---:|---:|---:|
| F1 | 0.749938 | 0.823599 | 0.073661 | 1.098 |
| F2 | 0.739866 | 0.783620 | 0.043754 | 1.059 |
| F3 | 0.727648 | 0.740761 | 0.013113 | 1.018 |
| F4 | 0.724418 | 0.743258 | 0.018841 | 1.026 |

At the latest two forecasting origins, training and validation errors
are close.

In particular, F4 validation RMSE is only approximately 2.6% higher
than training RMSE.

A small generalization gap does not by itself prove underfitting because
irreducible target variation may also limit performance.

However, together with the broad non-fuel feature-importance structure
and the absence of strong latest-period overfitting, it supports a
controlled model-capacity experiment.

---

## 59. Phase 04G Decision

Fuel and non-fuel will no longer be assumed to require identical
LightGBM tuning.

Fuel remains frozen at the current baseline configuration.

The next controlled experiment will modify non-fuel only.

The experiment will test whether increasing tree interaction capacity
improves temporal RMSE while holding all other modelling assumptions
fixed.

The first parameter changed will be:

    num_leaves

Baseline:

    31

Experiment:

    63

All other parameters, predictors, temporal folds and target definitions
remain unchanged.


---

## 60. Phase 04H Results - Non-Fuel Capacity Experiment

Phase 04H tested whether increasing LightGBM interaction capacity could
improve direct non-fuel regression.

Exactly one parameter changed:

    num_leaves

Baseline:

    31

Experiment:

    63

All other predictors, model parameters, temporal folds and evaluation
rules remained unchanged.

---

## 61. Temporal RMSE Results

Observed performance was:

| Fold | Baseline RMSE | 63-leaf RMSE | Change |
|---|---:|---:|---:|
| F1 | 0.823599 | 0.826437 | +0.002837 |
| F2 | 0.783620 | 0.785251 | +0.001631 |
| F3 | 0.740761 | 0.742440 | +0.001679 |
| F4 | 0.743258 | 0.745510 | +0.002252 |

No temporal fold improved.

The row-weighted temporal RMSE changed from:

    0.770238

to:

    0.772319

for a deterioration of:

    +0.002081 RMSE.

The pooled experimental RMSE was:

    0.773043

which is also worse than the pooled Phase 04B baseline.

---

## 62. Generalization Effect

The larger model reduced training error while increasing validation
error.

At F4:

    Baseline train RMSE          0.724418
    Baseline validation RMSE     0.743258

With 63 leaves:

    Experimental train RMSE      0.691419
    Experimental validation RMSE 0.745510

Increasing model capacity therefore improved in-sample fit but worsened
future-period generalization.

The F4 validation-to-training RMSE ratio increased from approximately:

    1.026

under the baseline to:

    1.078

under the larger model.

This provides direct evidence that additional leaf complexity encourages
overfitting rather than resolving a capacity limitation.

---

## 63. Phase 04H Decision

The 63-leaf non-fuel model is rejected.

The accepted non-fuel specification remains:

    direct LightGBM regression
    num_leaves = 31

No Phase 04H model will be promoted to final competition training.

The experiment also revises the interpretation of the Phase 04G small
generalization gap.

The baseline model should not be regarded as clearly underfit merely
because training and validation errors are close.

When additional tree capacity was supplied, training error decreased
but temporal validation deteriorated.

The next controlled experiment will therefore test the opposite
direction:

    lower model complexity.


---

## 64. Phase 04I Results - Lower Non-Fuel Tree Complexity

Phase 04I tested the opposite complexity direction following the failure
of the 63-leaf experiment.

Exactly one parameter changed relative to the original direct non-fuel
baseline:

    num_leaves

Baseline:

    31

Experiment:

    15

All other features, temporal folds, LightGBM parameters, target
definitions and evaluation rules remained unchanged.

---

## 65. Temporal RMSE Results

Observed results were:

| Fold | 31-leaf RMSE | 15-leaf RMSE | Change |
|---|---:|---:|---:|
| F1 | 0.823599 | 0.817758 | -0.005841 |
| F2 | 0.783620 | 0.782150 | -0.001470 |
| F3 | 0.740761 | 0.739133 | -0.001629 |
| F4 | 0.743258 | 0.742953 | -0.000305 |

All four temporal folds improved.

The row-weighted temporal RMSE changed from:

    0.770238

to:

    0.768079

for an absolute improvement of:

    -0.002159 RMSE.

The pooled 15-leaf RMSE was:

    0.768720.

The latest F4 RMSE improved from:

    0.743258

to:

    0.742953.

---

## 66. Complexity Direction

The combined capacity experiments now show:

    63 leaves:
        worse than 31 in all four folds

    31 leaves:
        original baseline

    15 leaves:
        better than 31 in all four folds

The non-fuel task therefore benefits from stronger structural
regularization through simpler trees.

Additional leaf-count searching is not justified.

The project will not continue testing nearby values merely to optimize
the validation score.

---

## 67. Comparison With Two-Part Non-Fuel Model

The accepted 15-leaf direct regression also outperforms the Phase 04E
two-part formulation.

Row-weighted temporal RMSE:

    Two-part model      0.768435
    15-leaf direct      0.768079

Latest F4 RMSE:

    Two-part model      0.744649
    15-leaf direct      0.742953

The simpler direct regression therefore provides both lower complexity
and stronger temporal validation performance.

---

## 68. Train-Validation Interpretation

At later folds, 15-leaf training RMSE may exceed validation RMSE.

For example, F4 produced:

    Train RMSE          0.748529
    Validation RMSE     0.742953

This is not interpreted using the assumptions of random IID
cross-validation.

The expanding training population combines multiple older forecasting
regimes, whereas each validation set represents one later temporal
origin.

Differences in target difficulty and population composition therefore
contribute to the train-validation relationship.

Temporal validation performance remains the primary selection
criterion.

---

## 69. Phase 04I Decision

The 15-leaf direct non-fuel LightGBM model is accepted.

The current non-fuel specification is now:

    Direct LightGBM regression
    276 predictors
    num_leaves = 15

The num_leaves parameter is frozen.

The 31-leaf baseline and 63-leaf experiment are superseded for the
non-fuel task.


---

## 70. Phase 04J Results - Lower Fuel Tree Complexity

Phase 04J tested whether the Fuel regression model also benefited from
stronger structural regularization.

Exactly one parameter changed:

    num_leaves

Baseline:

    31

Experiment:

    15

All predictors, target definitions, temporal folds, regularization
parameters and evaluation procedures remained fixed.

---

## 71. Temporal Fuel RMSE

Observed performance was:

| Fold | 31-leaf RMSE | 15-leaf RMSE | Change |
|---|---:|---:|---:|
| F1 | 0.624804 | 0.622905 | -0.001899 |
| F2 | 0.633544 | 0.632140 | -0.001404 |
| F3 | 0.596600 | 0.595613 | -0.000987 |
| F4 | 0.602348 | 0.601137 | -0.001211 |

All four temporal validation folds improved.

The row-weighted temporal RMSE improved from:

    0.613430

to:

    0.612078

for an absolute reduction of:

    -0.001352.

The pooled 15-leaf Fuel RMSE was:

    0.612264.

The latest-origin F4 RMSE improved from:

    0.602348

to:

    0.601137.

---

## 72. Fuel Generalization Effect

At F4 the original 31-leaf model produced:

    Train RMSE          0.554273
    Validation RMSE     0.602348
    Validation / Train  1.086735

The 15-leaf model produced:

    Train RMSE          0.575423
    Validation RMSE     0.601137
    Validation / Train  1.044688

Lower tree complexity deliberately increases training error while
improving future-period validation error.

The train-validation separation therefore narrows substantially.

This is consistent with beneficial structural regularization rather
than additional capacity being required.

---

## 73. Cross-Regression Complexity Finding

Both customer-value regression tasks independently favour lower tree
complexity.

Fuel:

    31 -> 15 leaves
    improved all 4 temporal folds.

Non-fuel:

    63 leaves
    worsened all 4 temporal folds.

    31 -> 15 leaves
    improved all 4 temporal folds.

The evidence therefore supports comparatively conservative tree
complexity for the regression tasks.

The shared value of 15 leaves is accepted because it was independently
validated for each target rather than imposed as a common assumption.

---

## 74. Phase 04J Decision

The 15-leaf Fuel model is accepted.

The current regression specifications are:

    Fuel:
        Direct LightGBM regression
        276 predictors
        num_leaves = 15

    Non-fuel:
        Direct LightGBM regression
        276 predictors
        num_leaves = 15

The Fuel num_leaves parameter is now frozen.

Further nearby leaf-count values will not be tested without a new
evidence-based reason.


---

## 75. Phase 04K - Minimum Leaf Support Experiment

The preceding regression experiments established that both Fuel and
non-fuel customer-value models benefit from lower tree complexity.

The accepted regression configuration currently uses:

    num_leaves = 15
    min_data_in_leaf = 50

Phase 04K tests whether additional structural regularization improves
temporal generalization by changing exactly one parameter:

    min_data_in_leaf

Baseline:

    50

Experiment:

    100

The accepted value of:

    num_leaves = 15

remains fixed.

All predictors, target definitions, temporal folds, learning rate,
feature sampling, bagging, L2 regularization and random seeds remain
unchanged.

Fuel and non-fuel are evaluated independently.

---

## 76. Phase 04K Acceptance Rule

The experimental min_data_in_leaf value will be accepted separately for
each regression target only if:

1. row-weighted temporal RMSE improves;
2. the latest F4 RMSE does not deteriorate;
3. at least three of four temporal validation folds improve.

This rule is defined before observing Phase 04K results.

The purpose is to avoid selecting a parameter based only on one
favourable validation period.


---

## 77. Phase 04K Results - Minimum Leaf Support

Phase 04K tested stronger terminal-leaf regularization while preserving
the accepted 15-leaf regression structure.

Exactly one parameter changed:

    min_data_in_leaf

Accepted value before experiment:

    50

Experimental value:

    100

Fuel and non-fuel were evaluated independently using the previously
defined acceptance rule.

---

## 78. Fuel Results

Fuel performance was:

| Fold | Min-leaf 50 RMSE | Min-leaf 100 RMSE | Change |
|---|---:|---:|---:|
| F1 | 0.622905 | 0.621366 | -0.001540 |
| F2 | 0.632140 | 0.631574 | -0.000566 |
| F3 | 0.595613 | 0.595067 | -0.000546 |
| F4 | 0.601137 | 0.600904 | -0.000233 |

All four temporal folds improved.

Row-weighted temporal RMSE improved from:

    0.612078

to:

    0.611393.

All three pre-registered acceptance criteria passed.

Fuel therefore adopts:

    num_leaves = 15
    min_data_in_leaf = 100

---

## 79. Non-Fuel Results

Non-fuel performance was:

| Fold | Min-leaf 50 RMSE | Min-leaf 100 RMSE | Change |
|---|---:|---:|---:|
| F1 | 0.817758 | 0.819457 | +0.001699 |
| F2 | 0.782150 | 0.780753 | -0.001396 |
| F3 | 0.739133 | 0.738884 | -0.000248 |
| F4 | 0.742953 | 0.742954 | approximately 0.000001 |

Only two of four folds improved.

The row-weighted temporal difference was negligible:

    0.768079 -> 0.768050.

Although aggregate RMSE improved marginally, the experiment failed the
pre-registered requirement that at least three temporal folds improve.

The F4 result was also microscopically worse.

Non-fuel therefore retains:

    num_leaves = 15
    min_data_in_leaf = 50

---

## 80. Phase 04K Decision

The stronger minimum-leaf regularization is accepted for Fuel and
rejected for non-fuel.

Current accepted regression specifications are:

    Fuel:
        Direct LightGBM regression
        num_leaves = 15
        min_data_in_leaf = 100

    Non-fuel:
        Direct LightGBM regression
        num_leaves = 15
        min_data_in_leaf = 50

The decision follows the pre-registered acceptance rule rather than
post-hoc selection from individual fold results.


---

## 81. Phase 04L - Opportunity Weight-Intensity Experiment

Phase 04D established that moderate Opportunity class weighting improves
the official weighted-F1 metric across all four temporal validation
origins.

The accepted Opportunity weighting rule is:

    class weight proportional to
    (N / (K * n_c)) ^ 0.50

where:

    N   = number of training observations
    K   = number of Opportunity classes
    n_c = training frequency of class c.

Phase 04L tests whether stronger, but still moderated, class balancing
provides additional benefit.

The experimental weighting rule is:

    class weight proportional to
    (N / (K * n_c)) ^ 0.75

The experiment does not use full inverse-frequency weighting.

All other modelling assumptions remain fixed:

    276 predictors
    LightGBM multiclass objective
    num_leaves = 31
    min_data_in_leaf = 50
    same learning rate
    same bagging and feature fractions
    same L2 regularization
    same four expanding-window temporal folds
    same random seed
    same weighted-F1 early stopping
    same argmax decision rule

Weights are calculated from each fold's training population only.

---

## 82. Phase 04L Acceptance Rule

The 0.75 weighting exponent will replace the accepted 0.50 exponent only
if all of the following conditions hold:

1. row-weighted temporal weighted F1 improves;
2. F4 weighted F1 does not deteriorate;
3. at least three of four temporal folds improve.

Macro F1, rare-class recall and prediction-space breadth are secondary
diagnostics.

They do not override deterioration in the official weighted-F1 metric.

The acceptance criteria are defined before observing Phase 04L results.


---

## 83. Phase 04L Results - Opportunity Weight Intensity

Phase 04L tested whether stronger Opportunity class balancing improved
the accepted square-root weighting policy.

Accepted exponent:

    0.50

Experimental exponent:

    0.75

All other modelling assumptions remained fixed.

---

## 84. Temporal Weighted-F1 Results

Observed results were:

| Fold | Alpha 0.50 F1 | Alpha 0.75 F1 | Change |
|---|---:|---:|---:|
| F1 | 0.450449 | 0.439856 | -0.010593 |
| F2 | 0.475097 | 0.463447 | -0.011650 |
| F3 | 0.496634 | 0.486262 | -0.010372 |
| F4 | 0.504765 | 0.501600 | -0.003164 |

No temporal fold improved.

Row-weighted temporal weighted F1 fell from:

    0.483390

to:

    0.474637

for a deterioration of:

    -0.008753.

All pre-registered acceptance criteria failed.

---

## 85. Rare-Class Effect

Stronger weighting increased prediction activity for some Adoption
classes.

At F4, Adoption macro recall reached approximately:

    0.0353.

The experimental model recovered small numbers of:

    Adoption: Beverages
    Adoption: Lubricants
    Adoption: Other
    Adoption: Food & beverage

However, this additional rare-class recovery was accompanied by weaker
performance on high-support classes.

F4 F1 values included approximately:

    Inactivity       0.742
    Fuel growth      0.447
    Stable           0.556

Because these classes account for a large proportion of the evaluation
population, the official weighted-F1 metric deteriorated.

---

## 86. Optimization Behaviour

The experimental best iterations were:

    F1      5
    F2      7
    F3      8
    F4    360

This pattern is substantially less stable than the accepted 0.50
weighting experiment.

The early folds indicate that stronger weighting rapidly moves the model
away from the validation weighted-F1 optimum.

The latest fold tolerates substantially longer training but still fails
to outperform the accepted weighting strength.

---

## 87. Phase 04L Decision

The 0.75 weighting exponent is rejected.

The accepted Opportunity specification remains:

    LightGBM multiclass
    276 predictors
    num_leaves = 31
    min_data_in_leaf = 50
    class-weight exponent = 0.50
    argmax decision rule

The class-weight exponent is now frozen.

No nearby weighting exponents will be tested without a new substantive
modelling hypothesis.


---

## 88. Phase 04M - Final Model Specification Freeze

Phase 04 experimentation is now closed.

All three competition targets have an accepted modelling specification
selected through expanding-window temporal validation.

No further structural or class-weight parameter search will be
performed before final training.

---

## 89. Accepted Fuel Specification

Model:

    LightGBM regression

Predictors:

    276 frozen Phase 03 predictors

Accepted structural parameters:

    num_leaves = 15
    min_data_in_leaf = 100

All other common LightGBM parameters remain those defined by the
Phase 04 baseline configuration.

Latest temporal validation result:

    F4 RMSE = 0.600904

The specification reflects successful controlled experiments in:

    lower tree complexity
    stronger terminal-leaf support.

---

## 90. Accepted Non-Fuel Specification

Model:

    LightGBM regression

Predictors:

    276 frozen Phase 03 predictors

Accepted structural parameters:

    num_leaves = 15
    min_data_in_leaf = 50

Latest temporal validation result:

    F4 RMSE = 0.742953

The two-part occurrence-times-magnitude architecture remains a
documented research candidate but was not promoted because the latest
F4 temporal origin deteriorated slightly.

Direct regression remains the accepted model.

---

## 91. Accepted Opportunity Specification

Model:

    LightGBM multiclass

Predictors:

    276 frozen Phase 03 predictors

Structural parameters:

    num_leaves = 31
    min_data_in_leaf = 50

Training class weighting:

    weight proportional to
    (N / (K * n_c)) ^ 0.50

Prediction rule:

    maximum class probability

Latest temporal validation result:

    F4 weighted F1 = 0.504765

A stronger weighting exponent of:

    0.75

was rejected because weighted F1 deteriorated at every temporal
validation origin.

---

## 92. Final Boosting-Round Rule

Final models cannot use competition labels for early stopping.

The number of boosting rounds must therefore be determined entirely
from historical temporal validation.

The frozen rule is:

    use the accepted model's F4 best iteration.

F4 is selected because it is the latest labelled forecasting origin and
the closest available temporal analogue to the final 2025-12-01
competition forecasting origin.

Accepted final boosting rounds are:

    Fuel           168
    Non-fuel       145
    Opportunity    232

No extrapolation beyond the F4 best iteration is performed.

This rule is fixed before final competition training.

---

## 93. Final Historical Training Population

Final models will train on all five labelled customer-origin snapshots:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01
    2025-09-01

Total supervised observations:

    22,679

Final predictions will be generated for:

    2025-12-01

Competition observations:

    5,488

No competition-period labels are available or used.

---

## 94. Phase 04M Decision

Phase 04 modelling experimentation is frozen.

Final model construction must reproduce exactly the accepted
specifications recorded above.

Any later model change must be treated as a new explicitly documented
experiment rather than silently altering the final pipeline.

