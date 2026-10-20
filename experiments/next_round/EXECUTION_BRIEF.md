# Local Codex handoff: next Sasol experiments

Use this document as an execution brief. Inspect the local repository, implement the experiments below, run them where the required data are available, and keep an evidence trail. Continue through setup issues and diagnosed failures without waiting for me to request each next step. Ask only when missing data, credentials, competition rules, or a material decision prevents useful progress.

## Objective and current baseline

- Repository: `Oluhle1/sasol-customer-value`.
- Best recorded public score: **0.309164264**.
- User-reported current top-10 threshold: **0.317136582**.
- Required public-score gain: **0.007972318**. Treat this as a target, not a promised outcome or independently verified live rank.
- Best archived CSV: `submission_journal_2026-10-10/submissions/sasol_oct10_small_fuel_values.csv`.
- Expected SHA-256: `11a4da31d86ab4738c2366b9c06f8439124991fc1e138bef685dbc38e61eb79a`.
- Accepted components: category-dynamics monthly LightGBM Opportunity; 50% direct + 50% hurdle nonfuel; small monthly fuel with 40% direct + 35% hurdle + 25% residual.

The immediate goal is a reproducible, informative next experiment round. Do not equate a local improvement with a measured leaderboard improvement.

## Read before implementation

Read applicable `AGENTS.md` files, then:

1. `README.md`.
2. `submission_journal_2026-10-10/REPRODUCIBILITY.md`.
3. `submission_journal_2026-10-10/METHODS_AND_DECISIONS.md`.
4. `submission_journal_2026-10-10/FAILED_EXPERIMENTS.md` and `NEXT_ROUND.md`.
5. `submission_journal_2026-10-10/evidence/best_opportunity_class_audit.csv`.
6. `experiments/oct10/README.md`, the canonical scripts, `src/reference/label_rules.py`, and `configs/label_config.json`.

Read the machine-readable decisions relevant to each proposed experiment. Do not repeat failed specialist, cadence, recency-weighting, station-context, compact-model or generic blend experiments under a new name. Identify exactly how a new hypothesis differs.

## Preserve existing work and establish reproducibility

- Discover the existing local checkout and use it. Do not create a worktree unless requested.
- Record Git status and preserve all existing changes. Do not reset, clean, overwrite baseline files, or replace historical decisions.
- This brief authorizes new experiment code and documentation. Prefer a new `experiments/next_round/` directory, thin wrappers around canonical helpers, and outputs under ignored `outputs/next_round/`. Keep official label logic and configuration unchanged. Do not alter archived source snapshots or realized prediction files.
- Verify the best CSV hash and run `submission_journal_2026-10-10/tools/verify_package.py`. If verification fails, diagnose whether local changes or corruption explain it before using the affected artifact.
- Locate authorized raw inputs and trusted generated caches. Compare inputs with `submission_journal_2026-10-10/input_data_manifest.json`; never invent or silently substitute missing competition data.
- Use the pinned `experiments/oct10/requirements.txt` in an isolated Python environment. Retain the existing local platform if it works. The cloud setup successfully used Python 3.12.14; the separate historical `requirements-py313.txt` is not the October environment.
- Start with one thread per model and a modest run budget. Check available CPU, memory and disk before expanding. Record library versions, seeds and actual threads.
- Reproduce the accepted baseline on matching validation rows. Archived final CSV verification does not establish reproduction of historical validation vectors. Label reconstructions explicitly and report any metric discrepancy.
- Give new caches distinct names with source/config/input fingerprints. Reuse only matching, trusted caches. Do not let an old cache silently mask a feature change.

## Measurement contract

Verify the official evaluation formula and competition rules from an authoritative source if access permits. The repository's RMSE leaderboard normalization denominators are explicitly unverified. Do not use them to claim how much a local RMSE change will improve the public score. If the exact formula remains unavailable, report fuel RMSE, nonfuel RMSE and weighted Opportunity F1 separately.

For a forecast origin T:

- Every feature must use transactions strictly before T.
- Every training outcome window must end at or before T.
- Early stopping, calibration, blend weights and decision rules must be learned on earlier eligible origins, with their own outcome-window purging.
- Construct stacked training inputs from temporal out-of-fold predictions. Never train the downstream model on upstream in-sample predictions.
- Keep ID as an alignment key; do not introduce arbitrary customer-ID encoding or random row validation.
- Align baseline and candidate predictions on identical customers. Report missing or ineligible rows explicitly.

June and September, and the additional April/May/July/August windows, have already been examined. They are reused historical checks, not untouched holdouts. Audit the December 2024 origin's prior use before calling it fresh. Overlapping horizons and repeated customers are dependent.

Report paired metric deltas, per-origin results and customer-cluster bootstrap intervals where practical: sample customer IDs and retain each sampled customer's complete set of origin rows, using the same sample for both models. Such intervals describe customer variation conditional on these dates; they do not capture future seasonal uncertainty or correct repeated model selection.

Before each candidate runs, save a proposal with hypothesis, features/targets, comparator, seeds, tuning dates, evaluation dates, budget and numerical acceptance gate. Preserve existing historical gates and rejection records. A different gate must be declared as a new protocol before seeing its results. Never relax a gate after failure.

## Phase 1: December forecast and error audit

Do this before expensive new models.

The final prediction origin is **2025-12-01**, covering December 2025 through February 2026. Main historical checks use June and September. The hypothesis is that calendar effects or population drift explain some of the plateau; this has not been established.

1. Compare final-test customer features with historical-origin features: tenure, recent activity, fuel/shop mix, category breadth, prices, prior-quarter spend, growth thresholds, and missing or short history.
2. Produce a matched **December 2024 to February 2025** backtest if the data and eligible earlier training origins support it. Build only pre-December features and use training labels ending by December. If nested early stopping is infeasible with the available history, use settings fixed in advance and state the limitation.
3. Reconstruct baseline confusion matrices, probability margins and per-class precision/recall/F1. Inspect fuel growth versus Stable and Inactivity, errors near growth thresholds, recency cohorts, and competing category events.
4. Report weighted error contributions, not only the worst class F1. September fuel-growth F1 is about 0.4804 over 1,435 customers; the three large classes represent about 81% of September labels. Confirm these from the actual validation rows.
5. Audit current seasonality features. The pipeline already includes previous-year volumes/spend and an elapsed origin-month feature. Do not claim seasonality is absent. Tree splits on elapsed month cannot extrapolate a trend beyond observed values.

Deliver `audit.md`, population comparisons, confusion matrices and a decision on which hypothesis to test first. Do not use the audit to search dozens of variants.

## Phase 2: Small, explicit calendar experiment

Hold model capacity, seeds, decision multipliers, rows and other features fixed. Predeclare at most two variants:

- Calendar features: cyclic month encoding, forecast-window composition and known calendar exposure, including correct calendar-month window lengths. Use public holiday features only if rules permit and their provenance is recorded.
- Pooled seasonal adjustments: shrunk population/category effects estimated exclusively from history available at each cutoff, with explicit fallback for insufficient prior-year customer history.

Use the December-matched backtest alongside the existing origins. One observed holiday season is weak evidence; avoid detailed customer-specific seasonal fitting. No future realized prices, promotions, station availability or transactions may enter historical features.

Record which change helps which cohort. A December-only win with losses elsewhere is exploratory evidence, not automatic acceptance.

## Phase 3: Rule-aware Opportunity probabilities

Proceed if the audit supports threshold or category-competition errors. This must differ materially from the rejected three-group specialist.

The official generator determines growth/adoption eligibility, minimum purchases, thresholds, category winners, tie handling and Inactivity precedence. Read the implementation rather than reproducing it from an abbreviated description. Keep the official generator unchanged.

Start with a minimal fuel-focused candidate:

1. Estimate probabilities of future qualifying purchases and crossing the exact fuel growth-spend threshold. If these are trained on all customers, define clearly whether they are unconditional; do not multiply overlapping probabilities as though independent.
2. Predict spending-increment uncertainty or a small set of conditional quantiles where useful. Evaluate probability calibration and threshold discrimination on earlier temporal folds.
3. Add only these temporal out-of-fold signals to a regularized Opportunity model or a small correction model alongside baseline class probabilities and known thresholds. Use all configured labels, preserving competition from nonfuel growth and adoption.