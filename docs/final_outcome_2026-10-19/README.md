# Final outcome — Sasol Customer Value Recruitment Challenge (19 Oct 2026)

The competition closed on 18 Oct 2026 and the final private leaderboard was
revealed on 19 Oct 2026. This is the official record of the outcome for the
Oluhle portfolio. All figures were fetched from the Zindi public API on
2026-10-19 (raw dump: `outputs/sprint17/final_private_leaderboard_raw_2026-10-19.json`)
and are not user-reported. Scores remain provisional until Zindi's code review
completes (the page stated 2026-11-08).

## Headline

| item | value |
|---|---|
| Final public rank / score | 32 / 0.313217559 |
| Final private rank / score | **116 / 0.302880883** |
| Private-scoring file | `submissions/sasol_phase05_frozen_lightgbm_submission.csv` (S0); public 0.289392053 → private 0.302880883 (+0.01348883) |
| Intended pair | `sasol_sprint16_r4full_vR.csv` (0.313217559) + `sasol_sprint16_blend_vR.csv` (0.309967901) — never privately scored |
| Final private top-10 cut | 0.312723243 |
| Winner | DrMarcoLotz, 0.315560685 |

The two private-scoring slots were left on S0-era submissions (best private
submission uploaded 2026-09-27T01:20:57Z) and were never updated after the
sprint 16 upgrades. The better of the two scored files is identified as S0 by
its exact public score 0.289392053; the second scored file is unidentified and
scored at most 0.302880883.

## Public → private transfer (156 same-submission pairs)

Delta = private score minus the public score of the same submission that was
privately scored.

| stat | value |
|---|---|
| mean | +0.001097 |
| median | +0.001196 |
| sd (ddof=1) | 0.006160 |
| min / max | -0.024760 / +0.018848 |
| share positive | 62.8% |

The transfer is strongly score-level dependent: at the top of this
leaderboard public scores regress. Around the intended best file's public
score (0.313217559):

| window | n | median private | median private rank | share above top-10 cut |
|---|---|---|---|---|
| ±0.0005 | 5 | 0.310785 | 31 | 0% |
| ±0.001 | 9 | 0.309721 | 39 | 11.1% |
| ±0.002 | 19 | 0.309721 | 39 | 10.5% |
| ±0.003 | 30 | 0.309966 | 36 | 16.7% |

S0's own transfer was +0.01348883, the 98.1st percentile of the 156 pairs —
yet its entry public level (0.289392053) still left the portfolio 0.009842
below the top-10 cut.

## Counterfactual if the intended pair had been selected

**Estimate only — not a recorded score.** The intended files never received a
private score because the platform did not score them.

- Conditioned neighborhood estimate: ~0.3097 (rank ~39).
- Local linear fit over ±0.003 of target: ~0.3103 (rank ~34).
- Naive median-transfer projection: ~0.3144 (rank ~5) — optimistic; it
  ignores the observed regression at the top and is an upper bound only.
- Share of comparable rows that beat the top-10 cut: 10.5–16.7%.

If one of the intended files had (contrary to the user's account) been the
unidentified second scored file, its private score would have to be
≤ 0.302880883: for r4full_vR that is a transfer of -0.010337 or worse
(5/156 = 3.2% of pairs), for blend_vR -0.007087 or worse (15/156 = 9.6%).

## Takeaways

1. Selection logistics are part of the pipeline. The private-scoring pair is
   set on the platform and is easy to leave stale; the final-state checklist
   must include verifying it after every score update, not just computing the
   best file.
2. Private transfer is score-level dependent. The global mean (+0.0011) is
   not a usable projection near the top of the leaderboard; band-conditioned
   transfer (this doc's window table) is the honest estimator.
3. The modelling work closed at its ceiling: `sasol_sprint16_r4full_vR.csv`
   remained the best public file (0.313217559) through sprint 17, whose E1–E14
   closed the remaining levers. The miss was operational, not modelling.

## Artifacts

- `outputs/sprint17/final_private_leaderboard_2026-10-19.json` — curated
  evidence (committed copy in this folder).
- `outputs/sprint17/final_private_leaderboard_raw_2026-10-19.json` — raw API
  dump, 242 participation rows.
- `outputs/sprint17/final_private_leaderboard_ranked_2026-10-19.csv` — the 156
  ranked rows with public, private and selection scores.
- `outputs/sprint17/final_leaderboard_analysis.py` — reproducible analysis
  script that rebuilds the curated JSON from the raw dump.
- `experiments/sprint17/RESULTS.md` — final outcome section appended.
