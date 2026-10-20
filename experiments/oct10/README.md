# October 10 temporal experiments

These experiments are separate from the frozen Phase 04–06 pipeline.
`feature_pipeline.py` adapts the user's newer
`Downloads/sasol-fresh/model_clv.py` into portable helpers. It reads the
workspace's raw competition files, official reference generator and label
configuration. It does not depend on Downloads or a Windows Python environment.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sasol-mpl python3 -u experiments/oct10/run_experiment.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sasol-mpl python3 -u experiments/oct10/run_values.py
```

Python package versions for this run are in `experiments/oct10/requirements.txt`. Single-thread
model fitting limits contention in the current environment. Models use fixed
seeds. Outputs and caches stay under `outputs/oct10`, which is ignored by Git.
The cache is a locally generated pickle; load only your own trusted cache.

## Opportunity experiment

The comparison is a two-seed LightGBM control, a CatBoost classifier, and a
50/50 probability blend. All use the same 197 reconstructed historical features
and the same snapshot weights. The control uses the documented 0.302250514
submission multipliers; this is a reproduced recipe on a different feature
representation, not a reproduction of the user's current 0.304344557 submission.

March 2025 selects candidate class multipliers. Those multipliers remain fixed
for June and September. Each outer fold selects boosting rounds on the latest
earlier training quarter, then refits on all earlier quarters. No outer labels
are used for early stopping. All reconstructed historical labels are checked
against the project's canonical transformed label CSVs.

Accept a candidate only when both June and September weighted F1 improve over
the larger of the control and the confirmed baseline's recorded scores
(June 0.5007932, September 0.5130711), with a mean improvement of at least 0.003.
The confirmed baseline is Downloads/submission.csv, public score 0.304344557;
its bytes match Downloads/sasol-fresh/submission.csv. A snapshot copy and
SHA-256 audit are under outputs/oct10. The baseline's historical metrics come
from its existing score_report.json and are not yet independently reproduced.
March is a tuning fold;
its score is not evidence of generalization. June and September have already
been studied elsewhere in the project, so they are historical checks, not a
previously untouched test set. Public or private leaderboard improvement is
not established by these local results.

`comparison.csv`, per-customer predictions, probability arrays and
`decision.json` retain the results. An accepted candidate also writes
`candidate_opportunity.csv` and the fitted CatBoost model. No submission is
written automatically by the experiment.

To apply an accepted candidate to the exact submission whose score is known:

```bash
python3 experiments/oct10/build_submission.py --baseline /path/to/best_submission.csv
```

The builder verifies official IDs, row order, finite nonnegative regression
predictions, authoritative labels, and preservation of both regression columns.
It writes a provenance audit with the baseline SHA-256 and changed-label count.

## Monthly training origins

`run_monthly.py` adds October/November 2024 and January/February/April/May/July/
August 2025 snapshots, for 13 monthly origins and 59,300 final training rows.
These are correlated repeated observations, not independent new customers.
Feature and label rules stay identical. For each outer validation origin T,
training outcomes must end at or before T. The inner validation origin is T-3
months and its training outcomes must end at or before that origin. Adjacent
monthly training labels may overlap each other, but never a validation origin.
March fixes decision multipliers; June and September assess them. Acceptance
uses the same confirmed-baseline gate. Run with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sasol-mpl python3 -u experiments/oct10/run_monthly.py
python3 experiments/oct10/build_submission.py --candidate monthly --baseline outputs/oct10/baseline_0304344557.csv --output submissions/sasol_oct10_monthly.csv
```

## Value experiment

The regression control uses the canonical feature set and frozen structural
LightGBM parameters, but chooses rounds on the preceding quarter for a nested
comparison. The candidate uses the richer features and a fixed hurdle blend:
30% hurdle for fuel and 50% for non-fuel. Targets remain in their official
transformed scale. Accept each target separately only if RMSE improves in both
June and September with a mean reduction of at least 0.001.

The comparison is against the canonical regression recipe, not the current
0.304344557 submission's unknown regression recipe. Accepted value predictions
are exported separately and are not automatically applied to the Opportunity
submission.

No estimated composite leaderboard score is used for model selection. The
normalization denominators assumed by previous scripts have not been verified
against an authoritative scoring implementation.

## Monthly value regressors

`run_monthly_values.py` tests two-seed, 15-leaf LightGBM regressors on the same
purged monthly origins. Boosting rounds are selected on the preceding quarter.
Each target must improve the confirmed baseline's recorded RMSE on both outer
folds, with a mean reduction of at least 0.001. These forecasts are separate
from the Opportunity experiment and never alter its regression columns.

## User-requested public leaderboard experiment

The monthly Opportunity model scores 0.5093886782 on June and 0.5131400293
on September, against the confirmed baseline's recorded 0.5007931647 and
0.5130711212. It fails our conservative acceptance gate because its mean gain
against the larger available control or recorded baseline is below 0.003.
The user explicitly requested a CSV to test whether these local gains translate
to the public leaderboard. This export keeps that failed gate recorded and
preserves both CLV columns from the confirmed 0.304344557 baseline.

```bash
python3 experiments/oct10/export_monthly_experiment.py
python3 experiments/oct10/build_submission.py --candidate monthly --experimental --baseline outputs/oct10/baseline_0304344557.csv --output submissions/sasol_oct10_monthly_experiment.csv
```

The submission audit records the experimental override, validation gains,
baseline hash, number of changed labels, and preservation of both CLV columns.
This CSV has no measured public leaderboard score until the user submits it.

### Confirmed public result

The user reported a public leaderboard score of **0.305007187** for
`submissions/sasol_oct10_monthly_experiment.csv`, up from **0.304344557**
(absolute gain **0.000662630**). Only Opportunity changed. This establishes
the monthly model as the best user-reported public baseline so far; it does
not establish a private leaderboard gain. The CSV is snapshotted at
`outputs/oct10/baseline_0305007187.csv` with its hash and score provenance in
`outputs/oct10/confirmed_monthly_baseline.json`. Future candidates should be
compared against the monthly model's local validation scores as well as the
original baseline, and preserve its submission until a better result is known.

## Next submission: category purchase dynamics

`run_decision_refinement.py` tried a small regularized decision-weight search
using March only. It raised March F1 but reduced June by 0.00749 and September
by 0.01002 against the monthly baseline, so it was rejected.

`run_category_dynamics.py` instead adds 108 past-only features: per-category
last-purchase recency, previous-quarter spend, quarter-on-quarter log-spend
changes, qualifying purchases in the most recent quarter, and customer/fuel/
shop visit counts over 0–14, 14–28, and 28–56 days. It keeps the monthly model's
class weights, training-origin weights, two random seeds, and purged temporal
round selection. No customer ID is used as a model input.

The predefined candidates are the enriched classifier and a 50/50 probability
average with the confirmed monthly classifier. Export requires positive F1
gains in both June and September and a mean gain of at least 0.001 over the
**0.305007187** baseline. These historical periods have been examined in prior
experiments; they are validation periods, not an untouched test set. Public
and private leaderboard gains remain unknown until scored.

```bash
python3 -u experiments/oct10/run_category_dynamics.py
python3 experiments/oct10/export_category_submission.py
```

The export preserves both CLV columns and writes
`submissions/sasol_oct10_category_dynamics.csv` plus a validation audit.

The user reported **0.307534678** for this submission: **+0.002527491** over
the monthly submission, and **+0.003190121** over the original baseline.
It is the new best user-reported public baseline. Its immutable snapshot is
`outputs/oct10/baseline_0307534678.csv`; score provenance, hash, and local
validation scores are in `outputs/oct10/confirmed_category_baseline.json`.
Future candidates should compare against this model's June/September F1
(0.5095271508 / 0.5165568275), not only the earlier monthly baseline.

## Group specialist experiment

`run_group_specialist.py` trains a three-class classifier for fuel growth,
inactivity, and all remaining outcomes. These frequent classes drive much of
the weighted F1 loss. It uses the same 305 features, monthly training origins,
snapshot weights, two seeds, and purged inner validation as the best model.
The specialist has 31 leaves, minimum 50 observations per leaf and L2=2.

The specialist's three-group probabilities are averaged 50/50 with the best
model's summed group probabilities. The original model's relative probabilities
within the remaining 15 labels are preserved. Class decision weights stay
unchanged; no blend-weight or decision-weight search uses June or September.
Export requires improvement on both historical periods and mean F1 gain
at least 0.001 over the confirmed **0.307534678** category baseline.

```bash
python3 -u experiments/oct10/run_group_specialist.py
python3 experiments/oct10/export_category_submission.py --candidate group
```

Output: `submissions/sasol_oct10_group_specialist.csv`, with the same CLV
predictions as the best confirmed submission and an adjacent validation audit.

Observed F1 gains were **+0.000143689** on June and **+0.001561888** on
September. Their mean **0.000852788** falls below the pre-set 0.001 threshold.
Given the ongoing user-authorized leaderboard experiments, export the
positive-in-both-periods candidate as an experiment without changing the
failed acceptance result:

```bash
python3 -u experiments/oct10/run_group_specialist.py --experimental
python3 experiments/oct10/export_category_submission.py --candidate group --experimental
```

The export and audit explicitly record the experimental override and failed
mean-gain threshold. Both CLV columns remain identical to the best submission.

The user reported **0.306894638** for the group specialist, a drop of
**0.000640040** from the category baseline. Reject this submission; the best
confirmed public model remains **0.307534678**. Local improvements alone
did not establish a public improvement. Score provenance is saved in
`outputs/oct10/group_specialist_public_result.json`.

## CLV experiments with six prediction origins

`run_category_values.py` tests direct monthly regressors with the 305 enriched
features. Both targets were rejected: gains against the recorded baseline in
June did not carry into September. This experiment does not alter any CSV.

`run_two_part_values.py` predicts each official transformed CLV target as
P(target > 0) times E(target | target > 0), then averages that forecast 50/50
with the direct regressor. Two seeds, 15-leaf models, monthly snapshot weights,
and purged inner round selection are fixed before evaluation. Fuel and nonfuel
are selected independently; Opportunity stays at the best category model.

Each target must first beat the better of direct-control RMSE and the best
baseline's recorded RMSE in June and September, with mean gain >=0.001.
Passing targets are then evaluated at April, May, July and August origins.
Across all six origins, they must beat the like-for-like direct control at
least five times, improve RMSE by at least 0.001 on average, and never lose
more than 0.002. Adjacent three-month outcomes overlap; these are six
historical origins rather than six independent holdouts. The extra-month
control is a reproducible direct regressor, not the unavailable original
submission's exact per-customer validation predictions.

```bash
python3 -u experiments/oct10/run_category_values.py
python3 -u experiments/oct10/run_two_part_values.py
python3 experiments/oct10/export_value_submission.py --candidate two-part
```

The export independently checks the gate and baseline hash, applies only
accepted value targets, preserves all Opportunity labels and any unselected
CLV column, validates IDs and transformed-value ranges, and writes an audit.
No experimental bypass is supported for this six-origin gate.

The user reported **0.308109087** for the two-part nonfuel submission,
**+0.000574409** over the category-dynamics best. It is now our best confirmed
public submission, snapshotted at `outputs/oct10/baseline_0308109087.csv`.
`outputs/oct10/confirmed_two_part_baseline.json` records its hash, score
provenance, unchanged Opportunity metrics and six nonfuel validation metrics.
Fuel remains the original baseline prediction; the rejected group specialist
is not included. Future experiments should preserve these accepted components
unless their replacements are validated.

## Fuel checks and station-context Opportunity

`run_residual_fuel.py` tried adding a fixed 25% change-from-last-quarter fuel
forecast. It improved the two known origins, but failed the pre-set 0.001
mean-gain screen. `run_monthly_fuel_recipe.py` applied the original 47-leaf
fuel recipe and fixed 40/35/25 direct/hurdle/residual weights to enriched
monthly data. It failed the screen against smaller models and the recorded
baseline. Neither changes the accepted fuel predictions.

`run_station_context.py` adds 20 features to the category-dynamics classifier:
seven smoothed station peer outcome proportions from the previous quarter,
known peer count, and recent station activity levels/changes for all, fuel,
and shop transactions. Each customer's own previous outcome is excluded
from its station and global smoothing counts. Only previous-quarter outcomes
that ended at the feature cutoff are used. Primary station and activity
windows use transactions strictly before the cutoff. `check_station_priors.py`
verifies self-exclusion, peer sensitivity, singleton/unknown-prior handling
and finite probabilities.

June and September must both improve over the best Opportunity model, with
mean F1 gain >=0.001. Passing candidates also forecast April/May using models
fitted at March, and July/August using models fitted at June. Both enriched
and control models use the same frozen fitting origin; forecast-date features
use all history available at that date. These extra checks assess stability
across dates without refitting on their outcomes. They are not additional
fully retrained models or independent holdouts. At least five of the six dates
must improve, with mean gain >=0.001 and no loss worse than 0.002.

```bash
python3 experiments/oct10/check_station_priors.py
python3 -u experiments/oct10/run_station_context.py
python3 experiments/oct10/export_station_submission.py
```

The export applies only Opportunity changes to the **0.308109087** baseline,
preserves the accepted fuel and two-part nonfuel values, rechecks the full
validation gate and writes the audit alongside the CSV. No bypass is supported.

The `--variant recent` experiment uses the existing 305 category features and
multiplies the original monthly snapshot weights by a fixed six-month
exponential half-life relative to the latest eligible training origin.
Decision weights, seeds and model settings are unchanged. It uses the same
two-quarter screen and four frozen-origin forecast checks, with separate
model/probability caches and reports:

```bash
python3 -u experiments/oct10/run_station_context.py --variant recent
python3 experiments/oct10/export_station_submission.py --candidate recent
```

The six-month half-life is fixed before evaluating this variant; no range of
decay strengths is searched using the validation outcomes.

Station context was rejected: June/September F1 0.509483 / 0.507189.
Recency weighting was rejected after June F1 0.509442 failed the screen;
the September fit was stopped to avoid further work on an ineligible model.

The `--variant early` experiment adds June/July/August 2024 customer snapshots
using `run_early_history.py`. Eligibility remains at least three historical
transactions. It adds 9,167 rows, for 68,467 total rows and 16 monthly origins.
The same category feature builder creates their 305 features. Existing
feature tables are checked unchanged with exact dataframe comparisons.
Original snapshot weights, model settings and decision weights remain fixed.
It uses the same full validation rule and exports against 0.308109087:

```bash
python3 -u experiments/oct10/run_station_context.py --variant early
python3 experiments/oct10/export_station_submission.py --candidate early
```

The early-history variant was rejected: June/September F1 was
0.510498 / 0.515695; September was below the accepted category model.

## Recent fuel cadence

`run_fuel_cadence.py` follows the control-model error audit saved in
`fuel_recency_error_audit.csv`. Eleven past-only features describe the last
three positive fuel transactions' gaps and fill sizes, last-fill ratios,
recent versus lifetime cadence, overdue status and a recent-cycle fuel
volume estimate. Candidate and control use the same fixed low-capacity
40/35/25 direct/hurdle/residual blend, two seeds per component and purged
monthly training. Only the candidate adds these features.

The existing six-origin gate is unchanged: beat both the stronger control
and recorded fuel baseline on June/September with mean gain >=0.001; then
improve at least five of six origins with mean gain >=0.001 and no loss >0.002.
The accepted Opportunity and nonfuel predictions remain unchanged.

```bash
python3 -u experiments/oct10/run_fuel_cadence.py
python3 experiments/oct10/export_value_submission.py --candidate fuel-cadence
```

The cadence features failed the September comparison and were rejected.

## Smaller monthly fuel recipe

The existing 305-feature small monthly control is evaluated against a
reconstruction of the original quarterly fuel recipe, and independently
against its recorded June/September RMSE. The original cached feature pickle
requires unavailable pyarrow; exact reproduction is not claimed. The fixed
40/35/25 direct/hurdle/residual blend uses 15-leaf models and two seeds per
component. All six dates improve against the reconstructed original, and
both recorded quarter metrics improve. The full validation gate passes.
Only fuel changes from the confirmed 0.308109087 submission.

```bash
python3 -u experiments/oct10/run_small_fuel_recipe.py
python3 experiments/oct10/export_value_submission.py --candidate small-fuel
```

## Diversity tests against 0.309164264

`run_shop_residual.py` tests a fixed 25% spending-change component in nonfuel,
anchored to `shop_z_q1`. It failed June and was rejected.
`run_cat_shop.py` tests a fixed 25% CatBoost blend in nonfuel, with depth 6,
two seeds, 305 features and the same purged monthly origins. Both periods
improved but mean gain 0.000385 failed the 0.001 screen.
`run_cat_fuel.py` reuses this CatBoost fitting procedure for fuel against the
publicly confirmed small monthly recipe. The original six-origin gate is
unchanged. Accepted candidates preserve the other targets from 0.309164264.

```bash
python3 -u experiments/oct10/run_shop_residual.py
python3 -u experiments/oct10/run_cat_shop.py
python3 -u experiments/oct10/run_cat_fuel.py
python3 experiments/oct10/export_value_submission.py --candidate cat-fuel
```

The fuel CatBoost blend failed both quarter dates and was rejected.

## Compact Opportunity classifier

The `compact` variant retains the 305-feature category panel, monthly origins,
sample weights, two seeds and confirmed class decision weights. Only model
capacity changes: 31 leaves, minimum group size 60 and L2 penalty 2.4. Settings
are fixed before evaluation. The existing six-date classifier gate applies,
including four forecasts using frozen March/June models with matched controls.
If accepted, both CLV columns are preserved from 0.309164264.

```bash
python3 -u experiments/oct10/run_station_context.py --variant compact
python3 experiments/oct10/export_station_submission.py --candidate compact
```

Compact Opportunity failed June F1 (0.508626 versus 0.509527) and was
stopped before completing September. No candidate from this four-test batch
passed, and no new CSV was exported. All rejected decision files are retained.
