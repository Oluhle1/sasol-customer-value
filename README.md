# Sasol Customer Value Recruitment Challenge

## Project Purpose

This repository documents the development of a reproducible machine-learning solution for the Sasol Customer Value Recruitment Challenge.

The project is being approached as an applied data science research project rather than only as a leaderboard optimisation exercise.

## Challenge Objectives

The solution must predict three customer outcomes for the following three-month period:

1. Fuel customer lifetime value (CLV_fuel)
2. Non-fuel customer lifetime value (CLV_nonfuel)
3. Customer Opportunity behaviour

## Modelling Strategy

The project will use a temporal customer-snapshot modelling framework.

Historical transaction data will be converted into customer-level behavioural representations at specific forecasting cutoffs.

The selected modelling family is LightGBM.

Separate models will be developed for:

- Fuel CLV regression
- Non-fuel CLV regression
- Opportunity multiclass classification

## Core Research Principles

- No future-data leakage
- Time-based validation
- Reproducible experiments
- Documented feature hypotheses
- Documented modelling decisions
- Error analysis before model changes
- No blind model switching
- Competition data will not be committed to this repository

## Repository Structure

    configs/       Configuration files
    data/          Local competition and processed data
    docs/          Research and technical documentation
    experiments/   Experiment tracking
    models/        Generated model artefacts
    notebooks/     Exploratory and analytical notebooks
    outputs/       Analysis outputs
    src/           Reusable Python source code
    submissions/   Generated competition submissions

## Status

Current phase: Project setup and documentation.

Next phase: Phase 1 - Transaction Data Forensics.

