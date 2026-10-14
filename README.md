# Sasol Customer Value Recruitment Challenge

## Current result

Latest recorded best public score: **0.309164264**, reported by the user after the October 10, 2026 Zindi submission. The starting October baseline was **0.304344557**, a gain of **0.004819707**. Private score is unknown.

The best file is [sasol_oct10_small_fuel_values.csv](submission_journal_2026-10-10/submissions/sasol_oct10_small_fuel_values.csv), with 5,488 customers. SHA-256: `11a4da31d86ab4738c2366b9c06f8439124991fc1e138bef685dbc38e61eb79a`.

| Component | Accepted recipe |
|---|---|
| Opportunity | Two-seed monthly LightGBM, category dynamics, 305 features and frozen class decision multipliers |
| Nonfuel CLV | 50% direct regression + 50% positive-outcome hurdle, on the official transformed scale |
| Fuel CLV | Small monthly LightGBM models, 40% direct + 35% hurdle + 25% prior-quarter residual |

Final training uses 13 monthly origins and 59,300 correlated customer snapshots. Nested temporal round selection excludes outcome windows crossing validation origins. Later failed experiments did not replace this baseline.

## October 14 experiment round

All six June/September baseline component metrics reproduced exactly. The calendar-exposure variant and temporal OOF fuel-rule correction both failed their preregistered gates; no replacement submission was exported. Public baseline remains **0.309164264**.

Read the [October 14 results](docs/OCT14_EXPERIMENT_RESULTS.md), [source and protocol](experiments/next_round/README.md), and [aggregate evidence](docs/oct14_evidence/). December uses untuned class decisions and fixed rounds because earlier calibration/inner validation is unavailable; its prior-year history is missing. It is a reused seasonal diagnostic.

## Results and decision record

- [Complete submission journal](submission_journal_2026-10-10/README.md): all five October uploads, exact CSVs, hashes and provenance.
- [Methods and decision rationale](submission_journal_2026-10-10/METHODS_AND_DECISIONS.md): labels, features, formulas, parameters, seeds, temporal validation, acceptance gates and exceptions.
- [Experiment results](docs/OCT10_EXPERIMENT_RESULTS.md): successful and failed tests and component validation metrics.
- [Reproduction instructions](submission_journal_2026-10-10/REPRODUCIBILITY.md) and [canonical source](experiments/oct10/README.md).
- [Rejected experiments](submission_journal_2026-10-10/FAILED_EXPERIMENTS.md) and [resume notes](submission_journal_2026-10-10/NEXT_ROUND.md).

The September Phase 04–06 documents describe an earlier frozen 276-feature baseline. The October experiments are a subsequent separate pipeline adapted from the newer local model source; the older frozen configuration is preserved as historical evidence.

## Repository layout

`configs/` contains label and model policies; `src/` contains the original phase scripts and reference labels; `experiments/oct10/` contains the portable October pipeline; `docs/` contains technical reports; `submission_journal_2026-10-10/` preserves October prediction artifacts, evidence and source snapshots. Raw competition data, generated caches and trained models remain local.

Public scores are user-reported results, not newly retrieved authenticated receipts. Repeated historical validation and overlapping label horizons limit claims about generalization. Recorded rationale is documented from source and evidence; unrecorded reasoning is not invented.

