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

