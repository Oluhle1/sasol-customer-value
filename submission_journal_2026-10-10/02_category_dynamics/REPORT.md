# 02 category dynamics

Public score: **0.307534678**. Comparison baseline: **0.305007187**. Public change: **0.002527491**.

Score provenance: user-reported Zindi result; private score unavailable.

Changed targets: Opportunity. Decision: retained as best at this stage.

## Experiment evidence and rationale

# Sasol model experiments — 10 October 2026

## Public leaderboard results reported by the user

| Submission | Public score | Change |
|---|---:|---:|
| Original `Downloads/submission.csv` | 0.304344557 | — |
| `submissions/sasol_oct10_monthly_experiment.csv` | **0.305007187** | **+0.000662630** |
| `submissions/sasol_oct10_category_dynamics.csv` | **0.307534678** | **+0.002527491** |
| `submissions/sasol_oct10_group_specialist.csv` | 0.306894638 | −0.000640040 vs best; rejected |
| `submissions/sasol_oct10_two_part_values.csv` | **0.308109087** | **+0.000574409** vs previous best |

The monthly submission changed 671 Opportunity predictions across 5,488
customers and preserved both CLV columns. Its public gain is confirmed by the
user's score report; its private leaderboard score is unknown.

## Local Opportunity validation

| Model | June weighted F1 | September weighted F1 |
|---|---:|---:|
| Original baseline's saved report | 0.500793 | 0.513071 |
| Monthly classifier, confirmed public improvement | **0.509389** | **0.513140** |
| Monthly classifier with four value forecast inputs | 0.508617 | 0.510449 |
| Refined monthly decision weights, tuned on March | 0.501902 | 0.503116 |
| Monthly classifier with category dynamics | **0.509527** | **0.516557** |
| 50/50 category dynamics / monthly blend | 0.506369 | 0.513661 |
| Category dynamics plus three-group specialist | **0.509671** | **0.518119** |

The original baseline's historical metrics came from its saved report. The
monthly and subsequent experiments were evaluated locally. Classifiers use
historical transactions from the competition dataset, with outcomes ending
at or before each prediction cutoff. Outcomes crossing an inner validation
cutoff are excluded from its training set.

The confirmed monthly model trains on 13 monthly origins rather than five
quarterly origins and averages two LightGBM models. Final training contains
59,300 repeated customer snapshots; these are correlated observations, not
59,300 independent customers. March set the class decision weights.

Further decision-weight tuning improved March but harmed both later periods,
so that change was rejected. The next experiment adds category purchase
recency, spending momentum and recent visit counts while retaining the
confirmed monthly decision weights. It tests an enriched model and a fixed
50/50 blend against the monthly baseline on both later periods.

The category dynamics classifier improved both periods against the confirmed
monthly model, with F1 gains of 0.0001385 and 0.0034168. It was selected over
the blend, which reduced June F1. Its 108 extra features bring the model to
305 inputs. The final candidate retains the monthly model's class decision
weights and preserves both CLV columns from the 0.305007187 submission.
The exported file is `submissions/sasol_oct10_category_dynamics.csv`, with
5,488 rows and 396 Opportunity changes from that baseline. Final models use
73 and 74 boosting rounds, selected using earlier historical outcomes.
The user reported a public score of **0.307534678**, bringing the total gain
from the original submission to **0.003190121**. It is now the best confirmed
public baseline. Its snapshot and score provenance are saved in
`outputs/oct10/baseline_0307534678.csv` and
`outputs/oct10/confirmed_category_baseline.json`.
The adjacent audit JSON records its hash,
baseline hash, validation gains, and CLV preservation checks.

The next specialist models fuel growth, inactivity, and all remaining outcomes
using the same 305 past-only features. Its three-group probabilities are
averaged 50/50 with the best model's summed probabilities. Relative
probabilities within the remaining 15 labels and all decision weights are
preserved. Its June/September gains are 0.0001437 / 0.0015619, for a mean
gain of 0.0008528. Both periods improve, but this fails the pre-set mean-gain
threshold of 0.001. The next CSV is an explicitly marked experiment; the
failed threshold remains recorded, rather than being relaxed retrospectively.

The user reported a top-10 public score threshold of **0.316766793**,
**0.009232115** above the current best **0.307534678**. This threshold is
user-reported rather than independently verified live. The user reported a
specialist public score of **0.306894638**, a decrease of **0.000640040**
from the category model. It is rejected; the best baseline remains
**0.307534678**. Its local gains did not transfer to the public evaluation.
Score provenance is recorded in `outputs/oct10/group_specialist_public_result.json`.
The file `submissions/sasol_oct10_group_specialist.csv` contains all 5,488
customers, with 115 Opportunity changes from the 0.307534678 baseline and
unchanged CLV columns. Its audit explicitly marks the experimental export
and failed mean-gain gate. Final specialist fits use 135 and 140 rounds,
selected on past-only validation.

## CLV experiments after the specialist rejection

The direct monthly regressors with category dynamics failed the screen on
both targets. Fuel RMSE was 0.592414 / 0.597498 on June/September; nonfuel
was 0.735850 / 0.743531. Neither target consistently beat the best recorded
baseline and the matching feature control, so neither was exported.

The next candidate uses a two-part model for the official transformed CLV:
probability of a positive target times its conditional expected value. A
fixed 50/50 blend combines this with the direct model; both use monthly
origins, two seeds, 305 past-only features and purged inner round selection.
Fuel failed the initial screen. Nonfuel passed: June/September RMSE was
0.734100 / 0.742360 against the best recorded 0.737693 / 0.742717.

Passing targets are checked on April, May, July and August as well. The
extra-month control is the direct model with the same features and temporal
training procedure; it is not the original submission's unavailable exact
validation predictions. Export requires gains on at least five of the six
origins, a mean RMSE gain >=0.001, and no origin losing more than 0.002.
Adjacent three-month outcome windows overlap, so these observations are
correlated rather than six independent tests.

The nonfuel two-part blend passed all six origins against its direct control:

| Origin | Direct RMSE | Two-part blend RMSE |
|---|---:|---:|
| April | 0.770720 | **0.768003** |
| May | 0.757135 | **0.756439** |
| June | 0.735850 | **0.734100** |
| July | 0.737651 | **0.736945** |
| August | 0.737644 | **0.736055** |
| September | 0.743531 | **0.742360** |

The mean RMSE improvement is approximately 0.0014384. Fuel was rejected.
The final nonfuel candidate preserves fuel and Opportunity from the best
0.307534678 submission. It passes the complete pre-set rule; no threshold
override is used. The user reported a public score of **0.308109087**, an
improvement of **0.000574409** over the previous best and **0.003764530**
over the original submission. This is now the best confirmed public baseline.
The remaining gap to the user-reported top-10 score is **0.008657706**.
Its private leaderboard score remains unknown.
The validated file is `submissions/sasol_oct10_two_part_values.csv`, with
5,488 rows. Only CLV_nonfuel changes; fuel and all Opportunity labels are
preserved from the best baseline. Its adjacent audit records six-origin
evidence, the unchanged baseline hash and the new submission hash.
The best CSV is snapshotted at `outputs/oct10/baseline_0308109087.csv`, with
score provenance, SHA256, and local component metrics in
`outputs/oct10/confirmed_two_part_baseline.json`.

## Further tests against 0.308109087

The fuel residual blend improved June/September RMSE against the recorded
baseline but failed the pre-set screen against the stronger control. The
original-capacity monthly fuel recipe then lost against the smaller-model
control on both periods and lost against the recorded baseline on September.
Neither fuel experiment was exported.

The station-context Opportunity model passed the self-exclusion leakage
checks but failed validation: June F1 0.509483 and September 0.507189,
against 0.509527 / 0.516557. It was rejected. Six-month recency weighting
produced June F1 0.509442, also below baseline; that run was stopped before
finishing September because it could no longer pass the required screen.
Rejected experiments do not change the best submission.

The next data-expansion experiment adds June, July and August 2024 origins,
with 2,745, 3,083 and 3,339 eligible customers respectively. Final training
therefore has 68,467 rows from 16 monthly origins. Labels use the original
eligibility and category rules, and all three new outcome windows finish
before the validation dates. Feature tables for all 14 previously generated
prediction dates were verified unchanged with exact dataframe comparisons.
The model retains 305 features and the confirmed class decision weights.
It must pass the same two-period screen and four additional frozen-origin
forecast checks before export.
It improved June F1 to 0.510498 but reduced September to 0.515695, so it
failed the screen and was not exported.

The next fuel check follows an error audit of the reproducible small-model
control, not the unavailable original model's individual validation vectors.
Its largest fuel RMSE occurs among customers with last purchases 30–90 days
before the September cutoff (0.830980 within that group). Eleven additional
features describe recent purchase gaps, fill quantities, cadence changes and
a recent-cycle volume estimate. They use only positive fuel purchases before
each feature cutoff. The candidate and control share a fixed low-capacity
40/35/25 direct/hurdle/residual recipe. The full two-period screen and six-origin
rule apply without a bypass. Fuel is the only target eligible for replacement.

The cadence features were rejected: June RMSE improved against the matching
small-model control (0.591138 versus 0.591512), but September worsened
(0.596859 versus 0.596786).

The smaller monthly fuel control was then evaluated as a replacement for the
deployed original quarterly fuel recipe. It uses the existing 305 features,
15-leaf models and a fixed 40/35/25 direct/hurdle/residual blend. The original
197-feature, 47-leaf quarterly recipe was reconstructed for comparison.
Its original Windows feature cache requires unavailable pyarrow, so this is
a recipe reconstruction, not exact reproduction of its validation vectors.
The candidate must also beat the original recorded June and September RMSE.
This comparison does not reverse the rejection of the cadence features or
the larger monthly models; those are different experiments.

| Cutoff | Reconstructed original fuel RMSE | Candidate fuel RMSE |
| --- | ---: | ---: |
| April | 0.624952 | 0.618481 |
| May | 0.615192 | 0.609128 |
| June | 0.594561 | 0.591512 |
| July | 0.599083 | 0.594968 |
| August | 0.608108 | 0.603314 |
| September | 0.597112 | 0.596786 |

All six dates improve, passing the unchanged six-origin gate. June and
September also improve against the recorded original RMSE of 0.594327 and
0.597050. Only fuel is eligible for replacement; Opportunity and nonfuel
remain from the confirmed 0.308109087 submission. The user reported a public
score of **0.309164264**, an improvement of **0.001055177** over the previous
best and **0.004819707** over the original submission. This is the new best
confirmed baseline, with a gap of **0.007602529** to the user-reported
top-10 threshold of 0.316766793. Its private score remains unknown.
The submission is `submissions/sasol_oct10_small_fuel_values.csv`; its
immutable snapshot is `outputs/oct10/baseline_0309164264.csv` and score,
hash and component validation provenance are recorded in
`outputs/oct10/confirmed_small_fuel_baseline.json`.

## Tests against 0.309164264

The nonfuel spending-change model adds a 25% residual component anchored to
the previous quarter's shop CLV. June RMSE worsened from 0.734100 to 0.734229;
September improved from 0.742360 to 0.742062. It failed the two-period gate.

A fixed 25% CatBoost direct-regression blend improved June/September nonfuel
RMSE to 0.733669 / 0.742020, but its mean gain of 0.000385 was below the
unchanged 0.001 screen. It was rejected. CatBoost uses two fixed seeds,
depth 6, inner purged time validation and the existing 305 features. These
tests preserve the best confirmed submission; no threshold bypass is used.
The same fixed diversity blend is next tested on fuel against the confirmed
small monthly 40/35/25 recipe, rather than the weaker original model.
Fuel CatBoost also failed: June/September RMSE was 0.592017 / 0.596847,
worse than the current 0.591512 / 0.596786 on both dates.

The next Opportunity variant holds the data, 305 features and decision
weights fixed while changing the classifier to 31 leaves, minimum group
size 60 and L2 penalty 2.4. Its two-period screen and four additional
frozen-origin forecast checks use the existing validation gates. Both CLV
columns are preserved from 0.309164264 if it passes.
The compact classifier failed June: F1 0.508626 versus baseline 0.509527.
It was stopped before completing September because the two-period screen
could no longer pass. No new submission was exported from this four-test
batch. The SHA256 of the confirmed 0.309164264 snapshot was rechecked and
is unchanged. Per-class baseline error metrics are saved in
`outputs/oct10/best_opportunity_class_audit.csv`; September fuel-growth F1
is 0.480375, with 1,435 examples, making it a substantial remaining source
of classification error.

Historical validation periods have been examined across experiments. They
help reject inconsistent candidates, but do not provide an unbiased estimate
of public or private leaderboard improvement. No top-10 rank is guaranteed.

Reproduction commands and methodology are in
[`experiments/oct10/README.md`](../../experiments/oct10/README.md). Exact CSV
hashes and export checks accompany each submission in its audit JSON.


## Exact export audit

```json
{
  "baseline_public_score": 0.305007187,
  "baseline_sha256": "e24c931018b953b1d9a2131ae3df76dd21e3e535573894e99dddfc450f20f35e",
  "output": "/mnt/c/Users/oluhl/sasol-customer-value/submissions/sasol_oct10_category_dynamics.csv",
  "output_sha256": "a70340046ff7971c80bfee9213a641ccda37523472202166f818ee3be910ea44",
  "rows": 5488,
  "changed_opportunity_labels": 396,
  "chosen_model": "category",
  "features": 305,
  "june_september_f1_gains": [
    0.00013847258659627215,
    0.003416798236364982
  ],
  "regression_columns_preserved": true,
  "public_score": null
}
```

## Source and method

The principal scripts are archived in this folder’s `python/` directory. Run the canonical scripts in `experiments/oct10/` from the repository root. Shared formulas, parameters, seeds, temporal splits and selection gates are documented in [METHODS_AND_DECISIONS.md](../METHODS_AND_DECISIONS.md). This report preserves recorded rationale and measured evidence; it does not claim to reconstruct unrecorded private reasoning.
