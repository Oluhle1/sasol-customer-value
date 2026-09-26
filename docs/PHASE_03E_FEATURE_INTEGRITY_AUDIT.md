# Phase 03E - Feature Integrity, Leakage and Temporal Drift Audit

**Project:** Sasol Customer Value Recruitment Challenge
**Phase:** 03E
**Status:** In Progress
**Started:** 26 September 2026

---

## 1. Purpose

Phase 03E is the final feature-engineering quality gate before predictive
modelling.

No additional predictive variables are created.

The objective is to audit the complete Phase 03D customer
representation for:

- structural integrity;
- target or future leakage;
- impossible feature values;
- temporal inconsistency;
- redundant or constant predictors;
- meaningful missingness;
- historical-to-final distribution drift;
- and independent reproducibility of selected sentinel features.

The Phase 03D input contains:

    28,167 Customer x Forecasting Origin observations

and:

    281 predictive features.

---

## 2. Audit Principle

A feature matrix is not considered modelling-ready merely because the
feature-generation scripts execute successfully.

The matrix must independently satisfy logical, temporal and statistical
checks before being supplied to LightGBM.

Phase 03E therefore attempts to falsify the correctness of the current
representation rather than assuming that previous successful runs prove
correctness.

---

## 3. Structural Integrity

The audit must verify:

    total rows = 28,167
    predictive features = 281

Customer-cutoff snapshot identifiers must remain unique.

The matrix must contain exactly the five historical calibration
forecasting origins plus the final competition forecasting origin.

Historical and competition row roles must remain distinct.

---

## 4. Leakage Audit

Predictive features must not contain:

    future outcomes
    CLV targets
    Opportunity labels
    winning source category
    target-window information
    snapshot role
    labelled/unlabelled status

The following columns are experimental metadata and must remain outside
the predictive feature manifest:

    snapshot_id
    ID
    cutoff
    snapshot_role
    is_labeled

The cutoff is required for feature construction and temporal validation,
but is not automatically treated as a customer behavioural predictor.

Any later decision to use calendar information derived from the cutoff
must be separately justified.

---

## 5. Temporal Consistency

Certain features represent cumulative observed customer history.

Once observed, these states cannot logically reverse at a later cutoff.

Examples include:

    lifetime_basket_count
    observed_tenure_days
    has_ever_fuel
    has_ever_nonfuel
    lifetime_category_breadth
    category ever_purchased indicators

For customers appearing at multiple forecasting origins, these features
must therefore be non-decreasing through time.

A decrease would indicate a feature-generation or snapshot-alignment
error.

---

## 6. Independent Sentinel Reconstruction

Selected features will be independently recalculated from raw
transactions using only data strictly before each cutoff.

The sentinel variables are:

    lifetime_basket_count
    m3_basket_count
    days_since_last_basket
    m3_fuel_litres
    m3_nonfuel_spend
    lifetime_category_breadth

The independent reconstruction is intentionally simpler than the main
feature pipeline.

Its purpose is to provide an external checksum against accidental
misalignment or temporal leakage.

---

## 7. Feature Quality Audit

Every predictive variable will be examined for:

    data type
    missingness
    number of unique historical values
    historical minimum and maximum
    historical mean and standard deviation
    constant behaviour
    near-constant behaviour
    exact duplication
    extremely high linear correlation

Constant or highly correlated variables are diagnostics rather than
automatic deletion rules.

Rare category states may legitimately produce highly sparse variables.

Any feature removal must therefore be justified rather than performed
blindly.

---

## 8. Missingness

Missing values can encode legitimate customer state.

Examples include:

    no historical non-fuel purchase
    insufficient repeated transactions for a purchase gap
    historically unseen category

Phase 03E therefore audits missingness patterns but does not
automatically impute them.

Missingness itself may drift through time as customer histories mature.

---

## 9. Distribution Drift

The final competition population occurs at:

    2025-12-01

Historical forecasting origins occur at:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01
    2025-09-01

For each feature, historical distributions will be compared with the
final competition snapshot.

Primary diagnostics are:

    standardized mean difference
    missing-rate difference

The purpose is not to declare a feature invalid when drift exists.

Instead, the analysis determines how representative each historical
forecasting origin is of the final prediction environment.

---

## 10. Drift Interpretation

For numeric feature x, standardized mean difference is:

    SMD =
        (mean_final - mean_historical)
        /
        sqrt((sd_final^2 + sd_historical^2) / 2)

The absolute value measures distributional separation in standard
deviation units.

This is used as a screening statistic rather than a formal hypothesis
test.

Large values indicate that the final competition population differs
substantially from the historical reference population for that
feature.

---

## 11. Redundancy

Exact duplicates and near-perfect correlations are reported.

They are not automatically removed during Phase 03E.

Some deterministic relationships were deliberately created for
interpretability.

For example:

    unseen_category_count

is mathematically related to:

    lifetime_category_breadth

because the total number of included categories is fixed.

Such relationships should be identified before modelling so the final
model feature manifest is deliberate rather than accidental.

---

## 12. Phase 03E Exit Criteria

Phase 03E can pass when:

- all 28,167 snapshots are preserved;
- all 281 candidate predictors are present;
- no predictive target/future metadata is detected;
- snapshot identifiers remain unique;
- cumulative feature states are temporally consistent;
- independent sentinel reconstructions reconcile;
- no infinite predictive values are present;
- impossible count/state values are absent;
- feature quality and redundancy are documented;
- and distribution drift has been quantified.

Only then may Phase 04 predictive modelling begin.

---

## 13. Current Status

Phase 03E audit code is being implemented.

No Phase 03E conclusion has yet been accepted.

No machine-learning model has been trained.

---

## 14. Phase 03E Results

The Phase 03E integrity audit completed successfully.

The audited representation contained:

    total snapshots            28,167
    historical snapshots       22,679
    competition snapshots       5,488
    candidate predictors          281

All hard integrity gates passed.

Observed:

    forbidden target/future feature hits       0
    infinite feature values                    0
    invalid absolute count values              0
    invalid binary values                      0
    negative recency/gap values                0
    temporal cumulative-state violations       0
    independent sentinel mismatches            0

Phase 03 therefore contains no detected structural or temporal integrity
failure that prevents modelling.

---

## 15. Independent Sentinel Reconstruction

Six sentinel features were independently reconstructed from raw
transactions at each of the six forecasting origins.

The tested features were:

    lifetime_basket_count
    m3_basket_count
    days_since_last_basket
    m3_fuel_litres
    m3_nonfuel_spend
    lifetime_category_breadth

This produced:

    6 features x 6 forecasting origins
    = 36 independent reconciliation tests.

All 36 tests returned zero mismatches.

Recency reconstructions differed only by approximately:

    5e-08 days

in maximum absolute numerical difference.

This is attributable to floating-point and CSV representation precision
and is far below the audit tolerance.

The sentinel results provide independent evidence that the temporal
feature pipeline correctly applies pre-cutoff transaction history.

---

## 16. Temporal Cumulative-State Integrity

No temporal monotonicity violations were detected.

For customers observed at multiple forecasting origins, cumulative
states such as:

    lifetime_basket_count
    observed_tenure_days
    has_ever_fuel
    has_ever_nonfuel
    lifetime_category_breadth
    per-category ever_purchased

never decreased through time.

Observed:

    temporal monotonicity violations = 0

This supports the validity of the Customer x Forecasting Origin
architecture.

---

## 17. Historical-to-Final Distribution Drift

Feature distributions become progressively more similar to the final
competition population as forecasting origins move forward through time.

| Historical cutoff | Median abs SMD | P90 abs SMD | Features >= 0.5 | Features >= 1.0 |
|---|---:|---:|---:|---:|
| 2024-09-01 | 0.093 | 0.735 | 48 | 21 |
| 2024-12-01 | 0.067 | 0.564 | 34 | 11 |
| 2025-03-01 | 0.043 | 0.416 | 25 | 2 |
| 2025-06-01 | 0.034 | 0.253 | 6 | 1 |
| 2025-09-01 | 0.020 | 0.107 | 1 | 0 |

The latest labelled forecasting origin is therefore substantially closer
to the competition environment than the earliest historical origin.

This independently reinforces the decision to use temporal rather than
random validation.

Earlier snapshots remain useful training observations, but they are not
assumed to be exchangeable with later snapshots.

---

## 18. Missingness Drift

From the latest labelled forecasting origin to the final competition
origin:

    mean absolute missing-rate difference
        = 0.166 percentage points

and:

    95th percentile absolute missing-rate difference
        = 1.547 percentage points.

Meaningful missing-value states are therefore relatively stable between
the nearest labelled snapshot and the competition forecasting origin.

---

## 19. Near-Constant Features

The audit identified:

    40 near-constant predictors

using a historical dominant-value threshold of 99.5%.

All were category-state features associated primarily with rare
categories or rare category behaviours.

Examples include:

    Frozen food
    Promotional items
    Miscellaneous
    Krispy Kreme
    Gas
    Automotive products

These variables are retained.

Sparsity is not treated as sufficient evidence for removal because rare
category histories may contain information relevant to rare Opportunity
classes.

No feature is removed merely because it is near constant.

---

## 20. Highly Correlated Predictors

The audit identified:

    23 feature pairs

with absolute historical Pearson correlation of at least 0.995.

Many relationships have clear behavioural explanations.

Examples include:

    basket activity versus qualifying basket activity
    active days versus qualifying active days
    fuel litres versus fuel monetary spend

These variables represent conceptually distinct quantities despite high
empirical correlation and are retained.

Tree-based LightGBM models do not require removal of predictors solely
because of linear multicollinearity.

---

## 21. Deterministic Category-Breadth Redundancy

The relationship:

    unseen_category_count
        =
    24 - lifetime_category_breadth

is deterministic because the included category universe contains 24
categories.

The two variables therefore contain identical information in opposite
directions.

The canonical Phase 03 matrix retains both for audit transparency.

For baseline modelling:

    lifetime_category_breadth

is retained and:

    unseen_category_count

is excluded.

---

## 22. Fuel Feature Duplication

Three Fuel category-state features were exact duplicates of existing
core behavioural features in both historical and competition data:

    cat__fuel__days_since_last_purchase
    cat__fuel__ever_purchased
    cat__fuel__prev3_basket_count

Their retained equivalents are:

    days_since_last_fuel_purchase
    has_ever_fuel
    prev3_fuel_basket_count

The duplicate category versions are excluded from the baseline modelling
manifest.

---

## 23. Fuel Recent-Basket Definition Divergence

The pair:

    m3_fuel_basket_count
    cat__fuel__m3_basket_count

was exactly identical throughout the historical modelling population but
not in the final competition population.

The two variables originate from slightly different semantic
definitions.

The core Fuel activity feature requires the Fuel Sale_Fuel transaction
definition, while the generic category-state representation identifies
qualifying positive activity within the Fuel source category.

Because the two predictors are identical throughout training, a model
cannot learn separate effects for them.

Allowing two training-identical but inference-divergent predictors into
the same baseline model could make feature selection dependent on
arbitrary tree split choice.

The baseline representation therefore retains:

    m3_fuel_basket_count

and excludes:

    cat__fuel__m3_basket_count.

Category-level Fuel monetary state remains represented separately.

---

## 24. Observed Tenure Drift Watch

The only predictor with absolute standardized mean difference of at
least 0.5 between the latest labelled snapshot and final competition
snapshot was:

    observed_tenure_days

Observed values were approximately:

    September 2025 mean     442.43 days
    December 2025 mean      516.12 days

with:

    SMD = 0.531.

This shift is consistent with increasing observation depth as the
forecasting origin advances.

The feature is therefore retained for the baseline rather than
automatically removed.

It is flagged for particular attention during temporal validation and
feature-importance analysis.

---

## 25. Baseline Feature Retention Decision

The canonical Phase 03 feature matrix remains unchanged at:

    281 candidate predictors.

Five deterministic or training-ambiguous variables are excluded from
the initial modelling manifest:

    cat__fuel__days_since_last_purchase
    cat__fuel__ever_purchased
    cat__fuel__prev3_basket_count
    cat__fuel__m3_basket_count
    unseen_category_count

The Phase 04 baseline therefore receives:

    276 predictive features.

Near-constant, sparse and highly correlated variables are otherwise
retained when they represent distinct behavioural concepts.

---

## 26. Phase 03E Decision

Phase 03E is considered complete.

The complete Phase 03 pipeline has:

- reconstructed a leakage-safe Customer x Forecasting Origin structure;
- generated core customer behavioural state;
- represented multi-period momentum and purchase regularity;
- preserved original category-level state;
- quantified customer maturity and category transitions;
- independently reconstructed sentinel predictors from raw data;
- verified cumulative temporal consistency;
- quantified missingness and distribution drift;
- and frozen a documented initial model feature policy.

No predictive model has yet been trained.

Phase 03 is now complete.

The project may proceed to:

    Phase 04 - LightGBM Baseline Modelling and Temporal Validation

