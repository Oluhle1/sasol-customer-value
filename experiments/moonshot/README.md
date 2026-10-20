# Last-hours model experiments

User explicitly requested a higher-risk batch after three feature additions
failed. This replaces the one-block-at-a-time scope for this separate batch.
No accepted baseline or earlier gate/decision is modified.

## Fixed experiments

1. `memory17`: GPU CatBoost MultiClass, 794 numeric features plus categorical
   customer ID, depth8, 1200-iteration cap, LR .05, L2 6, seed7. Learn all
   17 outcomes directly, including persistent customer effects across eligible
   historical origins. IDs do not grant access to future labels.
2. `balanced17`: same inputs, GPU depth10, 1500 cap, LR .035, L2 10,
   SqrtBalanced class weights. A deliberate larger minority-class tradeoff.
3. `rank17`: CPU LightGBM LambdaRank, 17 candidate outcome rows per customer,
   one positive relevance label. Uses core history and prior-label context,
   candidate category-state descriptors, candidate ID and known eligibility.
   Up to800 iterations, 63 leaves, LR .04, seed7. It changes the learning
   objective and representation, not the official 17-class targets.

Each family has two predeclared output views: standalone raw argmax, and
an equal mixture with the original baseline decision distribution. The
baseline decision distribution is product*vR normalized; it is not claimed
to be calibrated probabilities. New models get no fitted multipliers or
post-result temperature/blend search. There are six fixed views, not AutoML.

All five previously constructed feature blocks are used together in the
CatBoost arms, under the new user authorization. This combination has not
been validated by the failed one-block tests. Ranking descriptors summarize
original-category states as predictive inputs; they do not recompute labels
or pool source categories during official target generation.

## Gates and exports

The original promotion gates remain: screen March/April/May/July/August
mean >=+.004 and each >=-.002; only passing views evaluate June/September,
requiring mean >=+.004 and each >=+.0015. No tuning on evaluation labels.
Candidate models use the same outer eligible origins <=T-3 months, inner
training <=T-6, inner validation T-3 and full eligible-history refit.

A separate exploratory rule is predeclared before fitting: screen mean >=0,
worst >=-.01, and July AND August strictly positive. Such a view may produce
a December probe even if it fails promotion. Strict screen failures never
fit June/September. A strict screen pass later failing evaluation can still
produce an exploratory probe if the independent screen probe rule holds.
This intentionally accepts greater risk; it is not a validated improvement.

`validated/validated_<view>.csv` and
`exploratory/exploratory_<view>.csv` are distinct. Every export preserves
5488 official IDs/order and exact original CLV strings. No public score is
estimated, no upload is automated, and the current best is not overwritten.

## Runtime and resuming

Use free Colab T4 GPU. Upload `sasol_moonshot_bundle.zip` to MyDrive/sasol
and run `run_colab_cell.py` as one normal Python cell. No other ZIP needed.
The CPU ranker and GPU classifiers run sequentially, with a hard 10-minute
subprocess timeout per origin and a 90-minute wall-clock budget per invocation.
On an arm error, record it and continue; no parameter/device fallback.
Restarting with unchanged inputs reuses completed fits and starts a new
invocation budget. A killed incomplete fit is not cached as complete.

All fitted boosters (.cbm/.txt), probabilities, IDs and metadata are saved
under predictions/<family>/ on Drive. GPU CatBoost is not bit reproducible;
saved models and hashes preserve the actual fitted artifacts. Cache identity
covers frozen input/code manifests and pinned package versions. Output JSON:
`moonshot_decision.json`. Keep prediction/model artifacts for reproducibility.

## Preparation and validation

`proposal.json` is preregistered under outputs/moonshot before any real
candidate fits. `build_bundle.py` packages frozen arrays and cached baseline
probabilities; it fits nothing. `verify.py` exercises classification with
categorical IDs and listwise ranking using synthetic CPU data only. Local
GPU execution is not tested because this machine has no usable GPU.

Rebuild:

```bash
python3 -B experiments/moonshot/build_bundle.py
```

The bundle checker verifies manifest hashes, all five blocks, IDs, original
baseline scores and temporal splits. The best source CSV is manifest-bound.
No label generator, normalization constant, frozen model pipeline or old
prediction cache is changed.

Implementation references:
https://catboost.ai/docs/en/features/training-on-gpu
https://catboost.ai/docs/en/references/training-parameters/
https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRanker.html

## Reported run and separately registered CPU variants

Colab GPU quota prevented both GPU classifiers from running. The CPU ranker
ran successfully and failed: standalone mean -0.0195408/min -0.0270191;
half blend mean -0.00360494/min -0.0102984. Neither met the promotion or
exploratory rule; no June/September, December fits or exports. Five fits
total154.2s. Evidence: outputs/moonshot/rank17_reported_screen.json.

Separate smaller-tree CPU variants are preregistered in
experiments/moonshot_cpu/README.md and outputs/moonshot_cpu/proposal.json.
They have a different input/code identity and Drive folder. They do not
change the original GPU proposal or count as completed GPU tests.
