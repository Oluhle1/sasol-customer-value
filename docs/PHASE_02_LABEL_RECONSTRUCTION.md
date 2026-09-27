# Phase 02 — Label Reconstruction and Target Forensics

**Project:** Sasol Customer Value Recruitment Challenge  
**Phase:** 02  
**Status:** In Progress  
**Started:** 25 September 2026

---

## 1. Purpose

Phase 02 investigates the outcomes that the machine-learning system
will eventually attempt to predict.

Before feature engineering or model development, the official Sasol
label-generation rules are reproduced across historical forecasting
origins.

The objectives are to:

- reproduce the official historical target snapshots;
- verify target-generation integrity;
- understand how the eligible population changes through time;
- quantify fuel and non-fuel target sparsity;
- investigate the distributions of future fuel volume and non-fuel spend;
- quantify Opportunity class imbalance;
- investigate temporal changes in target behaviour;
- establish the supervised-learning population that will be used later.

No predictive feature engineering or machine-learning modelling is
performed during this phase.

---

## 2. Authoritative Label Definition

The project does not independently redefine the competition targets.

Historical outcomes are generated using the official Sasol reference
implementation stored at:

    src/reference/label_rules.py

with configuration stored at:

    configs/label_config.json

Our Phase 02 code acts only as a reproducible wrapper around this
reference implementation.

---

## 3. Historical Forecasting Origins

The following official historical cutoffs are used:

    2024-09-01
    2024-12-01
    2025-03-01
    2025-06-01
    2025-09-01

At each cutoff:

    historical information = transactions strictly before cutoff

and:

    outcomes = following three calendar months

Eligibility is recalculated independently at every cutoff.

---

## 4. Reconstruction Outputs

For every cutoff, two datasets are generated locally.

### Raw outcomes

    labels_raw_<cutoff>.csv

Contains:

    ID
    fuel_litres
    nonfuel_rand
    Opportunity
    winning_source_category
    winning_increment_rand

### Competition-transformed outcomes

    labels_transformed_<cutoff>.csv

Contains:

    ID
    CLV_fuel
    CLV_nonfuel
    Opportunity

Derived datasets are stored under:

    data/processed/phase02_labels/

and are excluded from Git.

---

## 5. Validation Principle

The reconstruction must succeed before any target analysis begins.

The expected eligible-customer counts from an independent execution
of the supplied Sasol reference implementation are:

| Cutoff | Expected eligible customers |
|---|---:|
| 2024-09-01 | 3,608 |
| 2024-12-01 | 4,223 |
| 2025-03-01 | 4,649 |
| 2025-06-01 | 4,961 |
| 2025-09-01 | 5,238 |

A discrepancy means the pipeline must be investigated before proceeding.

---

## 6. Current Status

Phase 02A — label reconstruction is being implemented.

Target forensics have not yet begun.

Feature engineering has not begun.

Model training has not begun.

## 7. Label Reconstruction Results

The official Sasol label generator was successfully executed across all five historical forecasting origins.

The resulting eligible-customer populations were:

| Cutoff | Eligible customers |
|---|---:|
| 2024-09-01 | 3,608 |
| 2024-12-01 | 4,223 |
| 2025-03-01 | 4,649 |
| 2025-06-01 | 4,961 |
| 2025-09-01 | 5,238 |

These values exactly matched an independent execution of the official reference implementation.

Phase 02A label reconstruction is therefore considered validated.

All five historical snapshots contain all 17 authoritative Opportunity classes.

---

## 8. Regression Target Sparsity

Fuel and non-fuel outcomes display materially different levels of sparsity.

| Cutoff | Fuel zero rate | Non-fuel zero rate |
|---|---:|---:|
| 2024-09-01 | 11.89% | 57.65% |
| 2024-12-01 | 17.69% | 62.42% |
| 2025-03-01 | 23.79% | 65.00% |
| 2025-06-01 | 27.29% | 65.53% |
| 2025-09-01 | 29.25% | 65.06% |

Future non-fuel expenditure is therefore strongly zero-inflated.

The median future non-fuel spend is R0 in every historical snapshot.

Fuel also becomes increasingly sparse through time.

The median future fuel volume decreases from approximately 144.7 litres at the September 2024 forecasting origin to approximately 76.6 litres at the September 2025 forecasting origin.

This indicates meaningful temporal change in the target population.

---

## 9. Opportunity Macro-Class Behaviour

The 17 authoritative Opportunity labels can be grouped descriptively into four macro families for analysis only:

- Existing-category growth
- New-category adoption
- Inactivity
- Stable

These analytical groupings do not replace the official 17-class prediction target.

Their prevalence was:

| Cutoff | Growth | Adoption | Inactivity | Stable |
|---|---:|---:|---:|---:|
| 2024-09-01 | 37.33% | 14.27% | 11.06% | 37.33% |
| 2024-12-01 | 36.94% | 10.42% | 16.32% | 36.32% |
| 2025-03-01 | 38.50% | 7.81% | 22.18% | 31.51% |
| 2025-06-01 | 36.85% | 7.86% | 25.80% | 29.49% |
| 2025-09-01 | 38.76% | 7.03% | 27.42% | 26.80% |

Existing-category growth remains comparatively stable at approximately 37–39%.

Inactivity increases materially through time.

Stable behaviour declines.

New-category adoption also declines substantially.

---

## 10. Interpretation of Adoption Drift

The decline in adoption prevalence must not automatically be interpreted as a change in customer preferences.

Under the official label rules, adoption requires that the customer has never previously made a qualifying purchase in the candidate category.

Adoption checks the complete observed history before the forecasting cutoff.

As customer histories become longer, customers naturally accumulate previously observed categories.

The set of categories still eligible for future "new-category adoption" therefore tends to shrink over time.

Part of the decline in adoption prevalence is consequently a structural property of the target definition.

This creates an important modelling requirement:

customer category-history depth and historical category exposure must eventually be represented accurately in the predictive feature set.

---

## 11. Opportunity Class Imbalance

Although all 17 classes occur at every historical forecasting origin, their frequencies differ substantially.

At the latest historical cutoff, 2025-09-01, the dominant classes are:

| Opportunity class | Count | Share |
|---|---:|---:|
| Inactivity | 1,436 | 27.42% |
| Existing-category growth: Fuel | 1,435 | 27.40% |
| Stable | 1,404 | 26.80% |
| Existing-category growth: Other | 278 | 5.31% |
| Existing-category growth: Beverages | 188 | 3.59% |
| Existing-category growth: Food & beverage | 129 | 2.46% |

The remaining adoption classes individually account for relatively small proportions of customers.

The rarest class in the latest snapshot contains only 14 observations.

The ratio between the largest and smallest Opportunity classes is therefore approximately:

    1436 / 14 = 102.6

This confirms severe class imbalance within the full 17-class prediction problem.

Because the competition evaluates Opportunity using Weighted F1 rather than Macro F1, class support will affect each class's contribution to the final classification metric.

---

## 12. Relationship Between Opportunity and CLV Targets

The three prediction tasks are statistically related.

Every customer labelled Inactivity in all examined historical snapshots has:

    fuel_litres = 0

and:

    nonfuel_rand = 0

during the corresponding future outcome window.

However, zero fuel or zero non-fuel alone does not imply Inactivity.

For the 2025-09-01 snapshot:

    Inactivity customers = 1,436
    Zero-fuel customers = 1,532
    Zero-nonfuel customers = 3,408

Therefore some customers remain active exclusively through non-fuel behaviour, while many others remain active through fuel purchases without qualifying non-fuel spend.

The targets should consequently be treated as related outcomes rather than independent descriptions of customer behaviour.

The selected modelling framework remains three separate LightGBM models, but later evaluation should include cross-target consistency checks.

---

## 13. Temporal Persistence of Inactivity

Customer Opportunity states were compared between consecutive quarterly forecasting origins.

Among previously inactive customers, approximately:

    68.7%
    75.3%
    77.7%
    76.0%

remained inactive in the following quarterly snapshot.

Inactivity therefore exhibits substantial temporal persistence.

This suggests that recency, recent activity density and length of inactivity may eventually be important predictive information.

---

## 14. Reactivation and Growth Label Mechanics

An important consequence of the official growth definition was identified.

A customer may have purchased a category historically but recorded no purchases in that category during the immediately preceding three-month comparison period.

If the customer then returns during the future outcome period, the previous-quarter comparison spend is zero.

A sufficiently large returning purchase can therefore satisfy the Existing-category growth conditions.

Consequently, customers moving from Inactivity back to activity frequently transition to Existing-category growth rather than Stable.

This is a structural consequence of the label definition and not necessarily evidence of conventional commercial growth.

Future feature engineering must preserve:

- all-history category exposure;
- recent-quarter category activity;
- category recency;
- and the distinction between reactivation and continuous growth.

---

## 15. Phase 02A Conclusion

The historical labels have been reconstructed successfully and pass the required integrity checks.

The target analysis demonstrates substantial temporal structure:

- the eligible population expands over time;
- future inactivity increases;
- future fuel activity becomes more sparse;
- non-fuel CLV is highly zero-inflated;
- adoption opportunities decline as customer histories mature;
- Opportunity classes are strongly imbalanced;
- and several target states are mechanically related through the official label-generation rules.

These findings reinforce the decision to use temporal validation and customer-state representations rather than random customer splitting.

Phase 02A is complete.

The next stage is Phase 02B — Detailed Target Forensics.
