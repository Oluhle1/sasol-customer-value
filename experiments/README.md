# Experiments

Read [SOLUTION.md](../SOLUTION.md) for the model that was kept. This page is the map of the rounds.

Each round wrote its hypothesis and numerical gate before fitting. June 2025 and September 2025 were evaluation folds. March 2025, and later the April/May/July/August pool, were for tuning. A gate was not relaxed after the results.

| Folder | Status | What it decided |
|---|---|---|
| `oct10/` | Kept, in part | 305-feature panel. Category-dynamics Opportunity. Fuel blend 40/35/25. Non-fuel blend 50/50. Public 0.309164264 |
| `top10_round/` | Kept, in part | Four-class head. The `full` recombination is in the final file. Rank-corrector, five-class head, and lapsed-fuel regressor were rejected |
| `sprint16/` | Kept | Decode vector `vR` on the r4full product. Best public file, 0.313217559. E7–E11 rejected |
| `next_round/` | Rejected | Bridge between the October file and the four-class head. See `README.md` |
| `sprint17/` | Rejected | Weekly fuel, cascades, stacking, XGBoost, TabPFN, FT-Transformer, embeddings, GRU. E1–E14 all lost their screens |
| `feature_rebuild/` | Rejected | Shop cadence, basket attachment, category state |
| `moonshot/`, `moonshot_cpu/` | Rejected or untested | All-feature and customer-ID GPU models. The rank-17 screen lost. GPU arms were not fit (Colab quota). CPU variants were prepared and not fit on real customers |

Curated write-ups:

- `docs/OCT10_EXPERIMENT_RESULTS.md`
- `docs/OCT14_EXPERIMENT_RESULTS.md`
- `docs/PHASE_06_LEADERBOARD_GAP_ANALYSIS.md`
- `experiments/sprint16/RESULTS.md`
- `experiments/sprint17/RESULTS.md`
- `docs/final_outcome_2026-10-19/README.md`

`submission_journal_2026-10-10/` is the 10 October lab book. Its "current best" of 0.309164264 was superseded by `top10_round` and then by `sprint16`.

The original quarterly pipeline that produced S0 lives in `src/phase04*` and `src/phase05*`, with the frozen spec in `configs/final_model_spec.json`. New work does not modify those files.
