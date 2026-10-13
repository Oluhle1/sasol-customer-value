# Sasol submission journal — 10 October 2026

This folder records the five CSV submissions reported in today's conversation,
the code that produced their changed components, the validation evidence,
the public results, failed experiments and a practical plan for the next round.
The original starting CSV is archived separately.

The best confirmed public score is **0.309164264**. The starting score was
**0.304344557**, a total improvement of **0.004819707**. The gap to your
reported top-10 threshold of **0.316766793** is **0.007602529**. The threshold
is your reported reference, not a fresh leaderboard measurement. None of
these submissions has a known private score.

## Submission history

| Record | Public score | Change from its comparison baseline | What changed | Decision |
| --- | ---: | ---: | --- | --- |
| Original baseline | 0.304344557 | — | Starting submission | Preserve as provenance |
| [01 Monthly Opportunity](01_monthly_opportunity/REPORT.md) | 0.305007187 | +0.000662630 | Opportunity only | Public improvement; experimental export |
| [02 Category dynamics](02_category_dynamics/REPORT.md) | 0.307534678 | +0.002527491 | Opportunity only | Keep |
| [03 Group specialist](03_group_specialist_rejected/REPORT.md) | 0.306894638 | −0.000640040 | Opportunity only | Reject; never carried into later best files |
| [04 Two-part nonfuel](04_two_part_nonfuel/REPORT.md) | 0.308109087 | +0.000574409 | CLV_nonfuel only | Keep |
| [05 Small monthly fuel](05_small_monthly_fuel_best/REPORT.md) | **0.309164264** | +0.001055177 | CLV_fuel only | Current best |

The fourth submission builds on the category-dynamics best, not the rejected
group specialist. Each gain is measured against the best file it actually
replaced, not necessarily the immediately preceding upload.

## Where to start

1. Read [METHODS_AND_DECISIONS.md](METHODS_AND_DECISIONS.md) for the shared
   data preparation, temporal splits, formulas, settings and rationale.
2. Read the five individual reports for what changed, why, measured results,
   exact Python entry points and limitations.
3. Read [FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md) before repeating a test.
4. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for execution and environment
   workarounds, and [NEXT_ROUND.md](NEXT_ROUND.md) for the proposed next tests.
5. Read [research/REFERENCES.md](research/REFERENCES.md) for papers, books and
   official documentation, with links and the role of each source.

## Files supplied

`submissions/` holds exact copies of the five uploaded CSVs and their original
export audits. `original_baseline/` holds the starting CSV, the available
original model source and its saved score report. `evidence/` holds recorded
scores, decisions, comparisons and training metadata. `submission_ledger.json`
links each reported public score to its file hash and actual baseline.

`source_snapshot/` preserves the current experiment scripts, label configuration
and reference label generator. Each submission's `python/` folder also contains
its principal scripts and common helpers for easy reading. Those per-submission
copies are source references, not standalone runnable projects: use the
canonical layout described in REPRODUCIBILITY.md.

The snapshot is taken at documentation time. Some shared scripts gained optional
arguments later in the session; their original defaults remain available.
This is not a claim that historical source versions were separately committed
at the exact time of every submission.

Raw transaction data and large feature/model caches are not duplicated here.
The original project contains them. `input_data_manifest.json` records available
input paths and hashes. All files in the package are indexed by
`package_manifest.json`; `tools/verify_package.py` checks their hashes, submitted
IDs and preserved components without retraining.

## Evidence and reference honesty

The reported public scores came from you after manual Zindi uploads. We did
not retrieve authenticated submission receipts or private scores. The original
export audits may still say `public_score: null`; the ledger and confirmed
baseline metadata hold subsequent user-reported results. Original audits are
preserved rather than silently rewritten.

Today's modeling used your existing code, released challenge data, label rules
and measured experiments. There is no recorded evidence that a particular
paper or book was consulted before each model was created. The bibliography
was checked while preparing this journal and is explicitly labeled as
explanatory background or proposed further reading. It is not retroactively
presented as the source of the observed leaderboard gains.

The reports provide decision rationale and evidence, including mistakes and
limitations. They do not turn an experimental hypothesis into a proven causal
explanation: some successful changes altered several related ingredients at
once.
