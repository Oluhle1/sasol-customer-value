# Phase 06R-A - Purchase Representation Feasibility Audit

## Purpose

This phase evaluates whether the Sasol transactional data can faithfully
support a published customer-embedding approach before any representation
model is trained.

No predictive model is trained in this phase.

## Research basis

Bauer and Jannach (2021) reproduce a customer embedding method using
retail purchase data.

For each product:

    purchasers are ordered by purchase time

The embedding formulation treats:

    customers as Word2Vec tokens
    products as sequences/documents

An adapted SkipGram model with negative sampling learns dense
customer representations.

The resulting customer embedding dimensions are then used as additional
features in a gradient boosting model.

Yan and Resnick independently apply a closely related construction to
product-level retail purchase data in their multi-brand CLV system.

Therefore the Sasol feasibility audit focuses specifically on whether
its item-level transaction data provide sufficient product-customer
interaction density for this representation.

## Important limitation from prior research

Bauer and Jannach report that embeddings provide less benefit on their
smaller UKRetail dataset than on their substantially larger dataset.

Their full UKRetail data contain approximately:

    4,373 customers
    541,909 purchase records

This is close to the scale of the Sasol challenge.

Dataset size alone is therefore not sufficient evidence that customer
embeddings will be effective.

The relevant quantity is the usable product-customer interaction
structure.

## Sasol corpus interpretation

The candidate representation corpus contains genuine positive purchase
interactions only.

Known discount/accounting adjustment rows are excluded because they are
not purchased products.

Negative-quantity return events are not treated as product purchases in
the embedding corpus.

This preprocessing decision is a challenge-specific semantic alignment
with the purchase-event formulation in the cited studies. It is not
claimed to be a literature-derived hyperparameter.

The predictive target rules remain unchanged.

## Audit questions

For every historical forecasting cutoff and the final competition
origin, measure:

- positive purchase interaction count;
- unique customers;
- unique products;
- distinct purchasers per product;
- purchase events per product;
- unique products per customer;
- concentration of activity among highly popular products;
- coverage of customers by products shared across multiple customers;
- cross-customer adjacent purchaser relationships;
- Fuel versus non-Fuel corpus composition;
- product-universe growth through time;
- final competition-customer coverage.

No embedding dimension, context-window size, negative-sampling rate or
other Word2Vec hyperparameter is selected during this audit.

A representation experiment will only be designed after these results
are compared with the empirical literature.

