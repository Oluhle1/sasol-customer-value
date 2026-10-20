# Sasol Customer Value

Temporal LightGBM solution for a three-target customer forecast: fuel value, non-fuel value, and one behaviour class. The same pattern applies to any problem where each entity is scored on a future window, some targets are zero-inflated amounts, and one target is a rare-class label that dominates the score.

The competition closed on 18 October 2026. Private scores were revealed on 19 October 2026 and stay provisional until Zindi's code review (stated 2026-11-08).

| | |
|---|---|
| Best public file | `submissions/sasol_sprint16_r4full_vR.csv` — **0.313217559**, public rank 32 |
| File that was privately scored | `submissions/sasol_phase05_frozen_lightgbm_submission.csv` (S0) — **0.302880883**, private rank 116 |
| Private top 10 | 0.312723243 |
| Winner | DrMarcoLotz, 0.315560685 |

The two private-scoring slots were left on the September frozen model. The October file above was never privately scored. Full account: [docs/final_outcome_2026-10-19/README.md](docs/final_outcome_2026-10-19/README.md).

## Start here

1. [SOLUTION.md](SOLUTION.md) — the recipe, the decision rule, and how to reuse it.
2. [experiments/README.md](experiments/README.md) — which round produced which piece, and which rounds were rejected.
3. [docs/LABEL_RULES.md](docs/LABEL_RULES.md) — the label contract. Do not reimplement it from memory.

`PROJECT_LOG.md` and the phase notes under `docs/PHASE_*.md` are the research diary. They stop being the source of truth once a later round replaces a component.

## What is in this checkout

| Path | Role |
|---|---|
| `SOLUTION.md` | Final recipe and the reusable method |
| `src/reference/label_rules.py`, `configs/` | Frozen labels, feature policy, and the original quarterly spec |
| `src/phase0*` | Original quarterly pipeline that produced S0. Leave it frozen |
| `experiments/oct10/` | 305-feature panel, value blends, category-dynamics classifier |
| `experiments/top10_round/` | Four-class ranker (`run_rank4.py`) |
| `experiments/sprint16/` | Decode vector `vR` and the best public file |
| `experiments/sprint17/`, `feature_rebuild/`, `moonshot/` | Later families. All rejected |
| `data/`, `models/`, `outputs/`, `submissions/` | Local only. A fresh clone has none of them |

## Environment

WSL, system Python 3.12. The packages used to fit the solution are pinned in [experiments/oct10/requirements.txt](experiments/oct10/requirements.txt): numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, lightgbm 4.6.0, catboost 1.2.10.

Run from the repository root, one process at a time:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sasol-mpl python3 -u experiments/oct10/run_experiment.py
```

Do not use `.venv313/`. That tree is a Windows environment and is not the fit that produced these scores. `requirements-py313.txt` records that Windows environment only.
