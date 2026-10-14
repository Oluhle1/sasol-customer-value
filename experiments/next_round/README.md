# October 14 experiment round

Execution brief: `C:\Users\oluhl\Downloads\octber14th.md`. Confirmed baseline is **0.309164264**, SHA256 `11a4da31d86ab4738c2366b9c06f8439124991fc1e138bef685dbc38e61eb79a`. User-reported top10 reference **0.317136582** implies target gain **0.007972318**; this is not a fresh leaderboard measurement.

Run from the existing Windows checkout through WSL:

```bash
/home/oluhle/.venvs/sasol-oct14/bin/python -B -u experiments/next_round/run_round.py --phase all
```

Use `--phase audit`, `calendar`, or `rules` to resume an individual stage. The environment is an isolated venv exposing already-installed system packages whose exact versions match `experiments/oct10/requirements.txt`; no package downloads were required. Actual Python version, packages and hardware are saved in `outputs/next_round/environment.json`. All models run sequentially on one thread. Each invocation has a one-hour wall-clock budget checked before new fits.

## Phase 1

Verify raw and canonical-label hashes, realized submission hashes, and the completed published archive. The Windows journal predates the GitHub completion: its verifier/manifest/reports are missing locally. Run the published verifier from `/home/oluhle/sasol-github-update/` and compare every available original local artifact with its published copy. No historic archive is overwritten.

Reuse only the existing trusted 305-feature category panel. Fit fresh baseline components on matching December, June and September customers; compare with realized archived June/September predictions. December2024 was previously used in Phase04 F1 and label calibration, so it is a reused seasonal diagnostic. Only September2024 training outcomes end before December; all components use 40 rounds fixed before December labels, with no inner early stopping. Report the limitation explicitly.

Audit population means, missing/zero-history proxies, confusion, adjusted decision-score margins, per-class weighted F1 contributions, recency cohorts and exact fuel-growth-event thresholds. Normalized weighted class scores are decision diagnostics, not calibrated probabilities.

## Phase 2

One calendar variant adds month sine/cosine, day-weighted forecast-window sine/cosine, exact calendar-window days, weekend/December fractions, the future/past-quarter day ratio and customer fuel/shop exposure interactions. All quantities are known at origin; no holidays or external datasets are used. Other features, rows, seeds, structural model capacity and decision multipliers stay fixed. The identical purged round-selection procedure is retained. December's fixed-round rule is unchanged.

The new protocol preserves numerical gates and records its December guard in advance: June and September must each beat both the fresh reconstruction and historical archive, mean gain >=0.001. Passing targets need positive gain on at least five of six April–September dates, mean gain >=0.001, worst loss <=0.002, and no December deterioration. For extra dates, both candidate and comparator are refitted on their eligible prior origins. This differs from the October10 classifier extra-date protocol using frozen March/June fits and is explicitly a new protocol, not identical historical validation.

A pooled seasonal variant is deferred because there is only one released December outcome season. A December-only win does not trigger promotion. New proposals are saved before fitting.

## Phase 3

If the audit supports fuel-threshold uncertainty, construct two unconditional binary signals: qualifying future fuel-purchase probability and exact joint fuel-growth-event probability. The latter includes historic fuel eligibility, future purchases and spend thresholds. They overlap and are never multiplied as independent events. The generator stays unchanged. The binary event can lose to another category under official rules; the downstream model retains all 17 labels.

Upstream models use the existing 305 features, two seeds7/23, 15 leaves, min child50 and 40 rounds fixed in advance. Downstream training uses exclusively temporally out-of-fold baseline class probabilities and event probabilities. Each training origin's Platt calibration uses only outcomes ending before that origin; insufficient earlier OOF history uses identity calibration. Evaluation calibration uses all eligible earlier OOF records. No same-row auxiliary outcomes calibrate its own downstream input.

The regularized multinomial logistic correction has C=.05 and max_iter500, input standardization, log baseline probabilities, known fuel growth threshold, historical fuel eligibility and two calibrated event logits. A matched control omits the two new logits. All original decision multipliers remain frozen. Acceptance must beat both baseline and this matched control under the predeclared two-date and six-date gates. No early-December OOF records support an honest December correction fit, so that limitation is reported rather than constructing in-sample upstream inputs. Binary calibration, Brier and AUC reports are diagnostic; no quantile or calibration variant search is performed.

## Evidence and integrity

Evidence is written only under ignored `outputs/next_round/`. New prediction caches include source, rule-source, label config, helper, trusted panel and package fingerprints, with stored IDs checked on reuse. Initial Git status, exact inputs and baseline hash are retained. Every comparison reports identical evaluation rows and zero missing IDs or fails explicitly.

Paired cluster bootstrap samples customer IDs and retains each sampled customer's full set of date rows, with the same sample for baseline and candidate. Intervals describe customer variation conditional on reused dates; they do not capture seasonal uncertainty, independent future performance or repeated-selection bias. Gate decisions use per-origin metrics, without unverified leaderboard score normalization.

Official source checked October14: https://zindi.world/competitions/sasol-customer-value-recruitment-challenge . It confirms transformed RMSE and weighted F1 with weights .3/.3/.4, but the page does not expose RMSE normalization denominators. Dataset rules limit modelling inputs to challenge data. No external holidays are added.

Eligible exports preserve baseline components not selected for replacement and receive a hash/ID/value audit. No candidate has a measured new public score until manually submitted and reported. Failed experiments never overwrite baseline CSVs or historical decisions.

## Recorded setup corrections

All six June/September component metrics reproduced exactly. The first calendar construction failed before any candidate fit because pandas returned an Index; converting month angles to a NumPy array resolved it. The initial cache format stored object IDs; new caches use Unicode IDs. `migrate_baseline_cache.py` proved baseline model-producing functions AST-identical after normalizing only ID serialization, then checked IDs and prediction arrays exactly unchanged before reusing the nine completed baseline fits. Migration metadata and the initial source/audit are retained.

The initial December Opportunity diagnostic incorrectly applied decision multipliers selected on March2025 outcomes. A measurement amendment was recorded **before any calendar candidate fit**: use uniform, untuned argmax for both December models. The corrected runner applies this directly, and `finalize_december.py` recomputes the reported December metric and gate from saved raw probabilities. June/September retain their legitimate frozen multipliers. Original stage evidence is preserved under `outputs/next_round/audit_stage_v1/`; it must not be cited as the corrected December result.

After the round, produce the corrected report and integrity checks:

```bash
/home/oluhle/.venvs/sasol-oct14/bin/python -B experiments/next_round/summarize.py
```

`RESULTS.md` and files ending `_corrected.csv` / `_corrected.json` are the final measurement authority. The raw population SMD table includes undefined standardized values for constant origin-level prices; use the corrected table with absolute differences instead.

The joint fuel-growth probability is zeroed when known historical Fuel eligibility is false, before and after calibration. Both the correction and matched control zero the final Fuel-growth class in this same known-ineligible cohort and renormalize other classes. This implements a known eligibility constraint; it does not multiply overlapping probabilities or assume independence. The constraint was finalized before the rule proposal and all rule evaluation results.

Optional chart regeneration uses Matplotlib 3.10.8 (separate from the six pinned model dependencies):

```bash
python3 experiments/next_round/plot_audit.py
```

The original prediction archive, released input data and October10 source snapshots remain unchanged. The next-round code and evidence are additive.
