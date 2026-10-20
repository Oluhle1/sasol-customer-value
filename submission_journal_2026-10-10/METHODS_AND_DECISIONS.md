# Shared methods and decision rationale

## 1. The prediction problem

The raw data has 467,880 transaction line items from 6,204 customers. A line
item is not necessarily a separate shopping visit: multiple rows can belong
to one transaction. The final submission has 5,488 IDs from the sample file.
History ends before 1 December 2025; the prediction window is December 2025
through February 2026. Local counts are properties of the workspace data,
not fresh measurements from the public website.

There are three separate outputs. Fuel predicts total litres over the next
three months. Nonfuel predicts spending in rands over the next three months.
Opportunity selects one of the 17 configured behavior/category labels.
The official competition page describes transformed regression RMSE and
weighted F1, with component weights 0.3, 0.3 and 0.4.
[Competition evaluation](https://zindi.world/competitions/sasol-customer-value-recruitment-challenge).

The submitted regression quantities are **already transformed**:

```text
z_fuel    = log(1 + litres) / 2.352825386185113
z_nonfuel = log(1 + rands)  / 2.4200904988927476
```

These constants come from `configs/label_config.json`. They are fixed; they
are not recomputed on the final test set. Predicting and minimizing RMSE on
this scale aligns the regression loss with the supplied target definition.
There is no final second log transform. For inspection only, raw units can
be recovered as `exp(scale * z) - 1`.

The old source contains guessed leaderboard normalization denominators and
reconstructed composite scores. They are preserved as provenance, but were
not used as authoritative public-score estimates for the new selection gates.
Component validation and your actual public scores drove decisions.

## 2. Label and feature provenance

The reference label generator and configuration determine eligibility,
adjustments, included categories, adoption/growth rules and label priorities.
They are copied under `source_snapshot/src/reference/` and
`source_snapshot/configs/`. Customers require at least three historical
transactions. Growth settings include an absolute increase of 20 rands,
a relative increase of 25% and the configured purchase condition; inspect
the generator for the complete hierarchy rather than recreating it from
this abbreviated description.

The portable `feature_pipeline.py` adapts your newer original
`Downloads/sasol-fresh/model_clv.py`. It reads project data instead of relying
on the old Windows environment. Its 197 features summarize customer history,
monthly/quarterly volumes and spending, category behavior, recency, purchase
gaps, visits, transaction timing, station context and prior observed outcomes.
Some inherited forecasting features include cadence, exponential smoothing
and BG/NBD-style signals. Their presence comes from your source; it is not
evidence that those models independently improved today's score.

Historical labels at five quarter origins were compared with the project's
canonical transformed label CSVs. Opportunity labels matched and regression
values matched within numerical tolerance. This checks that the reconstructed
training problem is consistent with the local authoritative tables.

Customer ID is an alignment key, not a predictor. Features at cutoff T use
transactions strictly before T. Previous-quarter outcomes used as features
must have ended by T. Negative adjustments and category exclusions follow
the supplied rules; different feature families sometimes require positive
qualifying purchases, as explicitly coded.

## 3. Why monthly snapshots were introduced

Five quarter origins provide relatively few views of changing behavior.
Monthly origins from September 2024 through September 2025 supply 13 dates
and 59,300 rows for the final training set. A customer can occur repeatedly.
This adds temporal examples, not 59,300 independent customers.

The hypothesis was that more examples of transitions into growth, inactivity
and adoption would help generalization. The tradeoff is correlation between
rows and overlapping three-month labels. That makes time-aware validation
essential and makes random row splitting unsuitable.

Original quarter weights were 0.70, 0.85, 1.00, 1.15 and 1.30. Monthly weights
interpolate the same trend:

```text
w(origin) = 0.70 + 0.05 * months_since_September_2024
```

This gives later snapshots slightly greater influence. It is a retained
design choice, not an optimized weighting result. A separate six-month
half-life experiment failed and is not part of the best submission.

## 4. Outer and inner temporal validation

For a forecast origin T, only training origins whose three-month outcomes
finish by T can be used. The latest eligible training origin is T minus
three months. The inner early-stopping validation uses that latest origin;
its own training outcomes must finish before that inner origin.

For June 2025, the latest training origin is March 2025. Round selection
trains on origins through December 2024 and validates on March. The model
then refits on all eligible origins through March and predicts June. For
September, the corresponding inner origin is June and the inner training
limit is March. Three-month label boundaries are implemented with calendar
month offsets, not an approximate 90-day row gap.

This prevents training labels from crossing a validation origin. Adjacent
training outcomes may overlap each other. Repeated customers remain possible
because the task is forecasting existing customers; a customer-disjoint
split would answer a different question.

March 2025 was used for initial Opportunity decision multipliers. Those
multipliers were then frozen for June and September. These periods have
been examined repeatedly, so they are historical checks rather than untouched
test sets. April, May, July and August checks add stability evidence, but
their overlapping horizons do not make them independent holdouts.

## 5. Opportunity model and decisions

The accepted Opportunity models average two LightGBM classifiers, seeds 7
and 23. Baseline settings: 47 leaves, learning rate 0.04, minimum child
samples 35, column fraction 0.70, row fraction 0.85 with bagging frequency 1,
L2 penalty 1.2 and maximum 1,200 rounds. Inner early stopping waits 60 rounds;
the refit uses at least 40 rounds. Later two-thread fits request deterministic
column-wise training. Early monthly evaluation and export thread counts
differed; the archived arrays and CSV hashes preserve the realized output.

Each class probability is multiplied by its frozen decision multiplier,
then the largest weighted value supplies the categorical label:

```text
predicted_label = classes[argmax(probability * multiplier)]
```

The multipliers are: beverages growth 1.3, food/beverage growth 2.0, fuel
growth 1.0, other growth 1.3, inactivity 1.0, each adoption label 2.5 and
Stable 0.85. This is a label decision rule, not a change to the regression
targets or a probability submitted in place of a label. The coordinate
search was limited to March; no June/September label search was used for
these accepted weights.

Weighted F1 is the mean of class F1 values weighted by each class's actual
support. It is not the same as accuracy or macro F1. Its nonlinearity means
small probability changes can move labels without reliably improving the
public metric. The failed specialist and decision-refinement experiments
demonstrate this risk.

The 108 added category-dynamics features capture when and how purchasing is
changing: 24 included categories times four summaries, plus 12 recent-visit
summaries. Recency distinguishes an old category purchase from a recent
one; prior-quarter spend and spend change indicate momentum; qualifying
counts separate repeated purchasing from a single basket. Recent windows
capture short-term activity that quarter totals can hide. These are
plausible signals; the experiment and public score provide the evidence.

## 6. Direct, two-part and residual CLV models

Direct regression estimates expected transformed CLV from the features.
The accepted small models have 15 leaves, learning rate 0.04, feature and
row fractions 0.90, bagging frequency 1 and L2 penalty 1.2. Direct fuel has
minimum child samples 100; direct nonfuel has 50. Maximum rounds are 1,000,
with 60-round inner stopping and a 40-round minimum refit. Fuel seeds are
7/21; nonfuel seeds are 11/29.

A two-part model separates a zero outcome from the amount conditional on
purchasing:

```text
hurdle(x) = P(z > 0 | x) * E[z | z > 0, x]
```

Both submodels use 15 leaves and minimum child samples 50. Probability and
conditional transformed value are multiplied for each seed, then those
products are averaged. This conditional-expectation decomposition is in
the official transformed scale. It does not estimate expected raw rands
by exponentiating the expected log value; those are different quantities.

The accepted nonfuel prediction is `0.50 * direct + 0.50 * hurdle`. The
rationale is that many customers make no future nonfuel purchase, while
positive spending has a different distribution. Direct and two-part fits
may have complementary errors; equal weighting avoids searching many
weights on reused validation periods. The formula is a hypothesis tested
against a direct control, not an assumption that a hurdle must always win.

Residual regression models `z_future - z_previous_quarter`, then adds the
previous-quarter anchor back and clips the result at zero. The hypothesis
is that learning the change can be easier than relearning a customer's
entire spending level. It can fail when zero/inactive outcomes or abrupt
behavior changes make persistence misleading.

The accepted fuel forecast is:

```text
fuel = 0.40 * direct + 0.35 * hurdle + 0.25 * residual
```

Each component averages two fuel seeds. The weights were retained from
your original fuel recipe. The replacement combines monthly training,
305 features and smaller models; it is not an isolated test of tree size.
Larger monthly models and additional cadence features failed their own
comparisons and are not included in the best fuel prediction.

## 7. Acceptance gates and the two early exceptions

Initial Opportunity tests required both quarter dates to improve against
the stronger available control/recorded baseline, with mean F1 gain at
least 0.003. The monthly experiment failed this conservative gate but was
exported after you explicitly asked for a public test. It improved publicly.

Category dynamics and specialist tests used both positive quarter gains
and mean F1 gain at least 0.001. The specialist missed that mean threshold,
was exported as explicitly experimental and fell publicly. This was a poor
decision in hindsight; later gates did not gain an override because of it.

Later regression screens require both June and September to beat the better
of the recorded baseline and matching control, with mean RMSE reduction
at least 0.001. Across six origins, at least five must improve, the mean
gain must be at least 0.001, and no loss can exceed 0.002. Export independently
rechecks the evidence and hash. These numerical margins are engineering
filters, not statistical significance tests.

Later Opportunity tests use the analogous F1 gate. Their extra four dates
are forecast with frozen March/June models and matching frozen controls,
whereas regression's extra dates refit at their eligible origin. Reports
must not describe these as identical validation procedures.

## 8. Submission integrity and what a public gain establishes

Each export aligns to official IDs, checks unique rows and configured labels,
requires finite nonnegative CLVs, preserves unchanged columns and hashes
the source baseline. Float output uses 17 significant digits. Re-reading
the file verifies preserved numerical columns within tight tolerance.

Changing one target at a time makes an observed public gain attributable to
that submitted component, subject to the leaderboard's scoring behavior.
It does not identify which feature caused the gain or guarantee private
generalization. The current best is the accepted category Opportunity model,
two-part nonfuel model and smaller monthly fuel model; the failed specialist
is absent.
