# October 14 experiment results

Confirmed public baseline remains **0.309164264**. New local metrics are not a measured public score. User-reported top10 reference is **0.317136582**; target gap **0.007972318**.

## Audit findings

The three large classes comprise 81.6% of September rows. Fuel growth has 1,435 examples and F1 0.4803745; its weighted F1 deficit contribution is 0.142356, larger than Stable 0.114368 and Inactivity 0.067620. Only 1 of 1,436 eligible fuel-growth events lost to another category in September. Threshold/purchase uncertainty is better supported than category competition as a fuel-error explanation.

Global price_index and price_change are constant within origins. Their original raw SMD values are numerically meaningless; use population_comparison_corrected.csv, which sets constant-population SMD to missing and reports absolute mean changes. Other comparisons show changing tenure and price gap; matched_population_changes.csv separates customer evolution from population composition.

December 2024 is a reused matched-season diagnostic. It had only one eligible training origin and no nested early-stop fold; 40 boosting rounds were fixed before results. Its lower metrics must not be interpreted as an unbiased seasonal effect because training history also differs.

## Prior-year coverage

All December 2024 prior-year fuel and shop features are zero because that history was not released; December 2025 has coverage. Zero padding must not be interpreted as observed customer inactivity. The matched-season backtest also differs in training-history length. See prior_year_coverage.csv.

## Baseline reproduction

| origin | target | fresh_metric | archived_metric | metric_difference | rows |
| --- | --- | --- | --- | --- | --- |
| 2024-12-01 | Opportunity | 0.43289703 | — | — | 4223 |
| 2024-12-01 | CLV_fuel | 0.63274327 | — | — | 4223 |
| 2024-12-01 | CLV_nonfuel | 0.82293876 | — | — | 4223 |
| 2025-06-01 | Opportunity | 0.50952715 | 0.50952715 | 0.00000000 | 4961 |
| 2025-06-01 | CLV_fuel | 0.59151178 | 0.59151178 | 0.00000000 | 4961 |
| 2025-06-01 | CLV_nonfuel | 0.73410006 | 0.73410006 | 0.00000000 | 4961 |
| 2025-09-01 | Opportunity | 0.51655683 | 0.51655683 | 0.00000000 | 5238 |
| 2025-09-01 | CLV_fuel | 0.59678626 | 0.59678626 | 0.00000000 | 5238 |
| 2025-09-01 | CLV_nonfuel | 0.74235956 | 0.74235956 | 0.00000000 | 5238 |

Full precision and alignment fields: `baseline_reproduction_corrected.csv`.

## Calendar exposure

| origin | target | baseline_metric | candidate_metric | gain | rows |
| --- | --- | --- | --- | --- | --- |
| 2024-12-01 | Opportunity | 0.43289703 | 0.43903003 | 0.00613300 | 4223 |
| 2025-06-01 | Opportunity | 0.50952715 | 0.51007395 | 0.00054680 | 4961 |
| 2025-09-01 | Opportunity | 0.51655683 | 0.51534482 | -0.00121201 | 5238 |
| 2024-12-01 | CLV_fuel | 0.63274327 | 0.63334663 | -0.00060335 | 4223 |
| 2025-06-01 | CLV_fuel | 0.59151178 | 0.59114437 | 0.00036741 | 4961 |
| 2025-09-01 | CLV_fuel | 0.59678626 | 0.59660691 | 0.00017935 | 5238 |
| 2024-12-01 | CLV_nonfuel | 0.82293876 | 0.82285970 | 0.00007907 | 4223 |
| 2025-06-01 | CLV_nonfuel | 0.73410006 | 0.73424106 | -0.00014100 | 4961 |
| 2025-09-01 | CLV_nonfuel | 0.74235956 | 0.74248077 | -0.00012121 | 5238 |

Full precision and alignment fields: `calendar_comparison_corrected.csv`.

## Fuel-rule OOF correction

| origin | target | baseline_metric | candidate_metric | matching_control_metric | gain | rows |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-06-01 | Opportunity | 0.50952715 | 0.49300813 | 0.49471734 | -0.01651902 | 4961 |
| 2025-09-01 | Opportunity | 0.51655683 | 0.49782582 | 0.49643319 | -0.01873100 | 5238 |

Full precision and alignment fields: `rule_comparison.csv`.

## Calendar decision

```json
{
  "accepted_targets": [],
  "rows": [
    {
      "variant": "calendar_exposure_v1",
      "origin": "2024-12-01",
      "target": "Opportunity",
      "baseline_metric": 0.4328970319911569,
      "candidate_metric": 0.4390300321838049,
      "gain": 0.006133000192647953,
      "rows": 4223,
      "missing_rows": 0,
      "gain_vs_archived": null
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2025-06-01",
      "target": "Opportunity",
      "baseline_metric": 0.5095271507692096,
      "candidate_metric": 0.5100739511193448,
      "gain": 0.0005468003501352,
      "rows": 4961,
      "missing_rows": 0,
      "gain_vs_archived": 0.0005468003501352
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2025-09-01",
      "target": "Opportunity",
      "baseline_metric": 0.5165568275346171,
      "candidate_metric": 0.515344815174794,
      "gain": -0.001212012359823,
      "rows": 5238,
      "missing_rows": 0,
      "gain_vs_archived": -0.001212012359823
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2024-12-01",
      "target": "CLV_fuel",
      "baseline_metric": 0.6327432738799041,
      "candidate_metric": 0.6333466256342675,
      "gain": -0.0006033517543633,
      "rows": 4223,
      "missing_rows": 0,
      "gain_vs_archived": null
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2025-06-01",
      "target": "CLV_fuel",
      "baseline_metric": 0.5915117844522825,
      "candidate_metric": 0.5911443705823219,
      "gain": 0.0003674138699605,
      "rows": 4961,
      "missing_rows": 0,
      "gain_vs_archived": 0.0003674138699605
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2025-09-01",
      "target": "CLV_fuel",
      "baseline_metric": 0.5967862625611445,
      "candidate_metric": 0.5966069115503811,
      "gain": 0.0001793510107633,
      "rows": 5238,
      "missing_rows": 0,
      "gain_vs_archived": 0.0001793510107633
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2024-12-01",
      "target": "CLV_nonfuel",
      "baseline_metric": 0.822938763236278,
      "candidate_metric": 0.8228596977275678,
      "gain": 7.906550871017437e-05,
      "rows": 4223,
      "missing_rows": 0,
      "gain_vs_archived": null
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2025-06-01",
      "target": "CLV_nonfuel",
      "baseline_metric": 0.7341000618559954,
      "candidate_metric": 0.7342410575853909,
      "gain": -0.0001409957293955,
      "rows": 4961,
      "missing_rows": 0,
      "gain_vs_archived": -0.0001409957293955
    },
    {
      "variant": "calendar_exposure_v1",
      "origin": "2025-09-01",
      "target": "CLV_nonfuel",
      "baseline_metric": 0.7423595601227382,
      "candidate_metric": 0.7424807692625404,
      "gain": -0.0001212091398021,
      "rows": 5238,
      "missing_rows": 0,
      "gain_vs_archived": -0.0001212091398021
    }
  ],
  "gate": "Unchanged predeclared calendar protocol; no automatic promotion for a December-only win.",
  "public_score": null,
  "measurement_amendment": "December uniform multipliers declared before candidate fitting; raw evidence preserved.",
  "raw_decision_accepted_targets": []
}

```

## Rule decision

```json
{
  "accepted": false,
  "comparisons": [
    {
      "variant": "fuel_rule_correction_v1",
      "origin": "2025-06-01",
      "target": "Opportunity",
      "baseline_metric": 0.5095271507692096,
      "candidate_metric": 0.493008126121514,
      "gain": -0.016519024647695613,
      "rows": 4961,
      "missing_rows": 0,
      "gain_vs_matching_logistic_control": -0.0017092091135259002,
      "matching_control_metric": 0.4947173352350399,
      "unobserved_training_classes": [],
      "gain_vs_archived": -0.016519024647695613
    },
    {
      "variant": "fuel_rule_correction_v1",
      "origin": "2025-09-01",
      "target": "Opportunity",
      "baseline_metric": 0.5165568275346171,
      "candidate_metric": 0.49782582475470455,
      "gain": -0.01873100277991252,
      "rows": 5238,
      "missing_rows": 0,
      "gain_vs_matching_logistic_control": 0.0013926354197049129,
      "matching_control_metric": 0.49643318933499964,
      "unobserved_training_classes": [],
      "gain_vs_archived": -0.01873100277991252
    }
  ],
  "public_score": null,
  "export": "Not exported if screen fails. An eligible candidate requires separate final OOF refit and export audit."
}

```

## December measurement correction

{
  "December_baseline_untuned_F1": 0.4328970319911569,
  "source": "b838f9b6679e05b7_2024-12-01_Opportunity.json",
  "amendment": {
    "recorded_utc": "2026-10-13T23:32:07.381382+00:00",
    "stage": "Before calendar proposal/fitting",
    "reason": "Existing Opportunity decision multipliers were tuned on March2025 labels, which are unavailable at December2024.",
    "amendment": "December Opportunity metrics use uniform multipliers (untuned argmax), for baseline and candidate. No model predictions or data change. June/September retain frozen multipliers.",
    "initial_december_metric_status": "Superseded diagnostic using later-tuned multipliers; must not be presented as a causal December backtest.",
    "acceptance": "Same numerical gates and no-December-deterioration guard; apply using corrected December Opportunity scoring. Correction declared before candidate fitting, not a retrospective gate relaxation.",
    "implementation": "Result finalization will preserve raw stage files and write corrected baseline/calendar comparison and decision files."
  },
  "no_model_refit": true,
  "December_calendar_untuned_F1": 0.4390300321838049,
  "December_calendar_gain": 0.006133000192647953,
  "calendar_source": "d64bcc6797d6201b_2024-12-01_Opportunity.json",
  "accepted_targets_corrected": [],
  "raw_acceptance_changed": false
}

## Statistical and reproducibility limits

Per-customer paired rows and customer-cluster bootstrap intervals are archived alongside comparisons. Intervals condition on reused historical dates, repeated customers and overlapping horizons; they do not remove repeated-selection bias or measure future seasonal uncertainty. Exact library versions and source/config/panel fingerprints accompany caches. Inputs and canonical labels matched the recorded hashes. Original archives and best submission were preserved.

## Cohort diagnostics

See calendar_cohort_comparison.csv for recency, short-history, past-threshold-proxy and true-class comparisons. Cohorts overlap and are diagnostic rather than new model-selection tests.

## Final integrity

[]

No public-score improvement is established by this round. Rejected variants are retained with their original gates and evidence. No candidate is exported unless it meets its preregistered rule.
