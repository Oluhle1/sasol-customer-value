# Final solution

This is the model to keep, and the method to copy onto a similar problem.

A similar problem has three properties. Each entity is predicted from its own history at a cutoff. At least one target is a nonnegative amount with many exact zeros. One target is a single label chosen from many classes, a few of which cover most of the rows and most of the score.

## Outcome

| Item | Value |
|---|---|
| Best public submission | `submissions/sasol_sprint16_r4full_vR.csv` |
| Public score / rank | 0.313217559 / 32 |
| SHA-256 of that file | `67478c9c861376834f54976a828bcd753a81c3103a3c15398fdff8a4195a593e` |
| Rows | 5,488 official IDs, sample order |
| What it changes | Opportunity labels only. Both value columns are the 10 October fuel file |
| Privately scored file | `submissions/sasol_phase05_frozen_lightgbm_submission.csv` |
| Private score / rank | 0.302880883 / 116 |
| Intended private pair | `sasol_sprint16_r4full_vR.csv` and `sasol_sprint16_blend_vR.csv` (public 0.309967901). Neither was selected |

S0 is the frozen quarterly LightGBM from Phase 05 (276 features, one model per target, argmax). It remains the file the platform scored. The recipe below is the later model, and it is the one that held the best public score through every subsequent round.

## Score

The leaderboard number is higher when it is better. It mixes three terms with weights 0.3, 0.3, and 0.4:

- fuel RMSE on the transformed target
- non-fuel RMSE on the transformed target
- weighted F1 of the Opportunity label

The transform is fixed for every participant:

```text
z = ln(1 + y) / s
```

`s` is the population standard deviation (ddof = 0) of `ln(1 + y)` on the five historical quarterly snapshots in `configs/label_config.json`. Do not recompute it from a new window.

| Target | `s` |
|---|---|
| CLV_fuel (litres) | 2.352825386185113 |
| CLV_nonfuel (pre-tax rand) | 2.4200904988927476 |

A fit to public leaderboard rows during sprint 16 is consistent with both RMSE terms entering as `1 - rmse / 0.78`. That fit was a diagnostic. Acceptance used the component metrics on later time folds. A reconstructed leaderboard number was never the selection rule.

On this contest the class term was the one with room left. By 16 October the value columns were mid-pack, and almost all of the gap from 0.31322 to the public top-10 cut (0.31671) was Opportunity weighted F1.

## Recipe

Three models, one row per customer per cutoff. Features use transactions strictly before the cutoff. Labels use the next three calendar months. The implementation of the labels is `src/reference/label_rules.py`. Read [docs/LABEL_RULES.md](docs/LABEL_RULES.md) before changing anything about eligibility, returns, or which category wins.

### 1. Features

`experiments/oct10/feature_pipeline.py` builds the 305-column panel. The families that stayed in the final file:

- recency, tenure, gap mean, gap variation, and recency divided by the typical gap
- lifetime fuel litres and shop rand, plus "never bought" flags
- fuel regularity: gap variation and the share of fill-ups on the modal weekday
- last four quarters and last six months of fuel litres and shop rand
- the same window one year earlier, and the change from the previous quarter to the latest quarter, on the `z` scale
- per category: days since the last qualifying purchase, previous-quarter log spend, the change into the latest quarter, and log purchase count (`experiments/oct10/run_category_dynamics.py`)

Later blocks did not earn a place: weekly fuel paths, price and discount, pretrained item embeddings, basket attachment, shop cadence, and a second category-state layer. See `experiments/sprint17/RESULTS.md` and `experiments/feature_rebuild/RESULTS.md`.

### 2. Value columns

Both columns are copies of `submissions/sasol_oct10_small_fuel_values.csv` (public 0.309164264 when it was the Opportunity of its day). They were frozen before the final Opportunity work.

Train on `z`. Clip predictions at 0. Use small trees (15 leaves), two seeds, and choose the number of rounds on the previous quarter only. Refit on every earlier origin before predicting the outer origin.

Fuel is three forecasts blended with frozen weights:

```text
CLV_fuel = 0.40 * direct + 0.35 * hurdle + 0.25 * residual
```

- direct: regression of `z`
- hurdle: probability of a positive total, times a regression fit on the positive rows (Cragg hurdle in `lgb_hurdle_predict`)
- residual: regression of the change from last quarter, added back to last quarter's `z`

Non-fuel is:

```text
CLV_nonfuel = 0.50 * direct + 0.50 * hurdle
```

Code: `experiments/oct10/run_small_fuel_recipe.py`, `run_two_part_values.py`, `run_residual_fuel.py`. Sprint 16 rechecked the fuel weights on a simplex. The shipped 40/35/25 blend reproduced exactly, and no nearby weight improved both evaluation folds (`experiments/sprint16/tune_clv_ensembles.py`).

### 3. Opportunity

Seventeen official class names, in this order. The order is the order of the multiplier vector.

```text
Existing-category growth: Beverages
Existing-category growth: Food & beverage
Existing-category growth: Fuel
Existing-category growth: Other
Inactivity
New category adoption: Beverages
New category adoption: Braai & Ice
New category adoption: Breads & rolls
New category adoption: Confectionary
New category adoption: Food & beverage
New category adoption: Groceries
New category adoption: Ice-cream
New category adoption: Lubricants
New category adoption: Other
New category adoption: Snacking
New category adoption: Tobacco
Stable
```

**Base model.** Monthly-origin LightGBM on the 305 features, with the category-dynamics block. Average seeds 7 and 23. This is `category_dynamics` in `experiments/oct10/run_category_dynamics.py`.

**Four-class head.** A second LightGBM collapses the label to Inactivity, Stable, Fuel growth, and Other. `experiments/top10_round/run_rank4.py`.

**Full recombination.** The head sets the probability of the three large classes and the total mass of the other fourteen. The 17-class model only splits that leftover mass. This is the `full` mode. The `within` mode, which only reshuffles mass the 17-class model already put on the three large classes, lost the evaluation folds and was not shipped.

```python
def rank4_full(p17, p4, classes):
    """p4 columns: Inactivity, Stable, Fuel growth, Other."""
    big = {
        classes.index("Inactivity"),
        classes.index("Stable"),
        classes.index("Existing-category growth: Fuel"),
    }
    other = [i for i in range(len(classes)) if i not in big]
    cond = p17[:, other]
    cond = cond / np.maximum(cond.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(p17)
    out[:, classes.index("Inactivity")] = p4[:, 0]
    out[:, classes.index("Stable")] = p4[:, 1]
    out[:, classes.index("Existing-category growth: Fuel")] = p4[:, 2]
    out[:, other] = p4[:, 3:4] * cond
    return out / out.sum(axis=1, keepdims=True)
```

The shipped function is `rank4_combine(..., mode="full")` in `experiments/sprint16/evaluate_saved.py`.

**Decode.** Multiply each class probability by a constant, then take argmax. The constants below are `vR`. They were fit by coordinate ascent on the r4full probabilities, on four origins that are not the evaluation folds: 2025-04-01, 2025-05-01, 2025-07-01, 2025-08-01. June and September were scored after the vector was frozen. Source: `outputs/sprint16/pool_r4full_vector.json`.

| Class group | Multiplier |
|---|---|
| Existing-category growth: Beverages | 2.0 |
| Existing-category growth: Food & beverage | 2.0 |
| Existing-category growth: Fuel | 1.0 |
| Existing-category growth: Other | 1.3 |
| Inactivity | 0.8 |
| Each of the 11 New-category adoption labels | 3.0 |
| Stable | 0.85 |

```python
vR = [2.0, 2.0, 1.0, 1.3, 0.8] + [3.0] * 11 + [0.85]
label = classes[np.argmax(r4full * vR, axis=1)]
```

Export: `experiments/sprint16/build_vR_candidates.py`. The value columns are copied from the baseline file. Only Opportunity changes. Against the previous best public file this changed 291 labels.

### 4. Why this decode, and why it stopped here

The same vector on a different probability blend did not win. On the public board, `vR` moved r4full by about +0.00381, the plain category-dynamics model by about +0.00157, and the 0.4/0.4/0.2 blend by about −0.00041. Tune the constants on the probability product you intend to ship.

Two nearby probes lost on the public board: Stable at 0.90 instead of 0.85 (`r4full_vR2`, 0.31135), and a 0.6/0.4 mix with the early-history model (`rank4early_vR`, 0.30978). Local gains under about 0.001 did not survive.

Sprint 16 then closed the remaining levers on this family: a longer tuning window, probability temperatures, more seeds, a CatBoost member, an adoption split, neural nets, and value-weight search. Sprint 17 closed a different model family (weekly paths, cascades, stacking, XGBoost, TabPFN, FT-Transformer, item embeddings, a GRU over item sequences). None of them beat r4full + vR on the pre-registered screen. The errors that remain are mostly confident swaps among Inactivity, Stable, and Fuel growth. A second-best label is often correct, and the margin is too wide for a threshold to find those rows.

## Method to reuse

Use this sequence on the next problem of the same shape.

1. **Freeze the label contract** before fitting. One module owns eligibility, the horizon, returns, and ties. Every later number is void if that module changes.
2. **One row per entity per cutoff.** Features at cutoff T use events strictly before T. Training labels for a validation origin must end at or before that origin. Adjacent monthly rows may overlap each other. They may not overlap the origin you are scoring.
3. **Hold out at least two later origins.** Use an earlier origin to choose rounds, blend weights, and class multipliers. Score the held-out origins once. Do not retune on them.
4. **Split the targets.** Fit amounts and the class label separately. A gain on one column is exported by copying the other columns from the current best file.
5. **Treat exact zeros as their own event.** A direct regression of `log1p` shrinks zeros and large values toward each other. A hurdle (probability of a positive outcome, times a size model on positive rows) fixes that. Blend the two, and a residual-from-last-period model if the amount is persistent. Freeze the blend weights on the tuning origins.
6. **Give the common classes their own head** when three or four labels dominate the weighted score. Let that head set their probabilities. Let the full model split only the remaining mass, in proportion to its own probabilities on those rare classes. Then apply class multipliers, and fit those multipliers on that exact product.
7. **Require the gain on every held-out origin** before you keep a change, and require a mean gain large enough to matter. A rule that wins the average by losing one origin will usually lose on the next unseen window.
8. **Change one component, then stop when the surface is sharp.** If a 0.05 move in a multiplier reverses a public result, more grid search will not find a new model. The next candidate has to be a new feature family or a new learner, and it has to clear the same screen.
9. **If the contest keeps a private selection, check the selected files after every score update.** The selection is a setting on the platform. It does not follow the best file on disk. This portfolio's private rank is the result of leaving that setting on S0.

## What a new project can delete

Copy `label_rules` (or your own label module), the feature builder, the hurdle and residual trainers, `rank4_combine`, and the multiplier fit. The phase scripts under `src/phase04*` and `src/phase06*` are the path that produced S0 and the rejected quarterly variants. The folders `top4_papers/`, `top4_shipped/`, and `top4_submission/` are closed side experiments from 28–29 September. Their exports scored below the file they were trying to replace.

## Local data

`data/raw/` holds the released train, test, and sample files. `outputs/` holds panels, probability arrays, and decision JSON. Both are gitignored. `submission_journal_2026-10-10/input_data_manifest.json` lists the input hashes. Do not delete `outputs/`. Later rounds read those caches and do not refit an accepted baseline.
