# Reproduction and artifact verification

## Verify the archived predictions

Run from the repository root:

```bash
python3 submission_journal_2026-10-10/tools/verify_package.py
```

This standard-library check verifies package hashes, each CSV and comparison-baseline hash, 5,488 unique IDs, row order, finite nonnegative values, configured labels and preservation of untouched targets. It requires no raw data, model training or external network access.

## Reproduce training

Use the canonical `experiments/oct10/` layout and `src/reference/label_rules.py`, with the repository label configuration. Install the pinned experiment requirements in an isolated environment. Raw competition files and generated feature/model caches remain local and are not included in this documentation archive. See the exact input paths and hashes in `input_data_manifest.json`. Paths under `/mnt/c/Users/oluhl/` in historical audits describe the original run and must be adapted for a different machine.

Read `experiments/oct10/README.md` for all entry points and `METHODS_AND_DECISIONS.md` for formulas, parameters, seeds and purged inner/outer splits. Successful components were generated with:

```bash
python3 -u experiments/oct10/run_monthly.py
python3 experiments/oct10/export_monthly_experiment.py
python3 -u experiments/oct10/run_category_dynamics.py
python3 experiments/oct10/export_category_submission.py
python3 -u experiments/oct10/run_category_values.py
python3 -u experiments/oct10/run_two_part_values.py
python3 experiments/oct10/export_value_submission.py --candidate two-part
python3 -u experiments/oct10/run_small_fuel_recipe.py
python3 experiments/oct10/export_value_submission.py --candidate small-fuel
```

These commands need the locally generated caches and baseline snapshots specified by their scripts. Archived CSVs are authoritative for realized predictions. Frozen seeds help reproducibility, but changed library versions, thread counts and later shared-script options mean a fresh training run is not claimed to reproduce historical bytes. The original quarterly fuel comparator was reconstructed because its Windows feature pickle required unavailable pyarrow.

The October source snapshots were captured after the experiments, rather than committed at each submission. Original export audits remain intact. Score metadata records subsequent user-reported outcomes; no authenticated receipts or private scores were recovered.
