# Sasol Customer Value Challenge - Project Log

This document records the development history of the project.

---

## 2026-09-25 - Project Initiation

### Objective

Begin development of a reproducible solution for the Sasol Customer Value Recruitment Challenge.

### Project Philosophy

The challenge will be treated as an applied data science research project.

The goal is not only to achieve competitive predictive performance but also to produce a solution that can be:

- reproduced,
- reviewed,
- explained,
- defended during technical interviews,
- and maintained by another data scientist.

### Selected Modelling Framework

Temporal Customer-Snapshot Gradient Boosting Framework.

The project will use customer behaviour observed before a forecasting cutoff to predict behaviour during the following three months.

The selected modelling family is LightGBM.

Three predictive tasks will be implemented:

1. CLV fuel regression
2. CLV non-fuel regression
3. Opportunity multiclass classification

### Initial Data Findings

Training data:

- 467,880 transaction-line records
- 365,638 unique baskets
- 6,204 customers
- 390 locations
- 6,849 stock items
- Historical period: 19 March 2024 to 30 November 2025

Final test population:

- 5,488 customers

The 5,488 test customers exactly correspond to customers satisfying the competition eligibility requirement of at least three historical transactions.

### Important Structural Decision

The fundamental modelling observation will be:

    Customer x Forecasting Cutoff

rather than simply:

    Customer

This allows historical forecasting snapshots to be used for realistic temporal model development and validation.

### Current Status

Repository structure created locally.

No machine-learning model has been trained.

No feature engineering has begun.

### Next Step

Phase 1 documentation and reproducible transaction-data inspection.


---

## 2026-09-25 - Phase 01 Complete: Transaction Data Forensics

Phase 01 has been completed and documented in:

docs/PHASE_01_DATA_FORENSICS.md

Key conclusions:

- Training data contains 467,880 line-item records and 365,638 baskets.
- Historical data contains 6,204 customers.
- The 5,488 final test customers exactly match customers satisfying the minimum-three-transaction eligibility rule at the final cutoff.
- Historical coverage runs from 19 March 2024 through 30 November 2025.
- The final forecasting origin is 1 December 2025.
- The modelling unit is defined as Customer x Forecasting Origin.
- Historical eligibility will be recalculated independently at every cutoff.
- Raw competition data will remain unchanged.
- Temporal validation will be used instead of random customer splitting.
- Feature engineering and modelling have deliberately not started yet.

Next phase:

Phase 02 - Label Reconstruction and Target Forensics.



## 2026-10-10 — Monthly pipeline and current best

Recovered from the Windows working folder and documented on 2026-10-13. The October baseline advanced from 0.304344557 to 0.309164264 through monthly Opportunity training, category-dynamics features, two-part nonfuel and small monthly fuel models. A group-specialist upload deteriorated publicly and was excluded from later best submissions. Subsequent local diversity and compact-model tests were rejected.

See [the submission journal](submission_journal_2026-10-10/README.md) for exact files and score lineage, [methods](submission_journal_2026-10-10/METHODS_AND_DECISIONS.md) for recipes and rationale, and [results](docs/OCT10_EXPERIMENT_RESULTS.md) for measured validation and failures. The September frozen specification remains a historical baseline, not the current October recipe.
