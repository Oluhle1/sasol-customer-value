# Layered customer features

Construction-only pipeline for the released Sasol data. The accepted model,
original label generator and 305-feature cache remain the comparison.
The construction command produces no model fits, label changes or submissions.
Separate Colab experiment runners are documented below.

## Run

From the repository root, using the installed system packages:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B -u experiments/feature_rebuild/build_layers.py
python3 -B experiments/feature_rebuild/package_features.py
```

The default output is `outputs/feature_rebuild/`. Use `--phase lines`,
`--phase baskets`, or `--phase features` to stop after / build through that
layer. Later phases load prior verified layers or construct missing ones.
Existing layer caches require an exact source/code identity and checksum.
After a code/input change, select a new directory using `--out`; never
silently reuse a mismatched artifact. The raw transaction CSV is about 72 MB;
typed lines occupy about 284 MB in pandas. The initial full build took 157
seconds with 912 MiB peak resident memory on this machine.

Hand-checked semantic fixtures:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B experiments/feature_rebuild/verify_layers.py
python3 -B experiments/feature_rebuild/verify_artifacts.py
```

## Layer 1: original typed lines

`layers/lines.pkl` preserves all 16 raw columns and adds the authoritative
loader's timestamp and integer-scaled quantities/amounts. It marks adjustments,
in-scope categories, qualifying purchases, fuel volume and nonfuel amounts.
No price-modifier subtraction, returns clipping, inferred labels or fitting.
Repeated source category strings are compressed as pandas categoricals;
customer and receipt IDs remain strings. Literal `NA` and `Unknown` remain
real original categories. Local pandas pickle files are not the Colab format.

## Layer 2: receipts and original-category receipts

`layers/baskets.pkl`: one row per `(ID, TransactionId)`, including every
receipt for eligibility. Holds source total once, signed fuel/nonfuel sums,
purchase flags, category/product breadth, site and time. Excluded-category
or adjustment-only receipts count for eligibility but not qualifying activity.

`layers/category_baskets.pkl`: one row per customer, receipt and original
included Level2 category. Keeps signed spend and a separate qualifying
purchase flag. Original categories are never pooled into `Other` here.

The build checks that each receipt has one timestamp, site and repeated total;
it refuses inconsistent receipts rather than aggregating across a cutoff.
Exact integer sums reconcile with the typed lines. Basket totals are not
expected to equal line sums because of source tax/scope semantics.

## Layer 3: customer-cutoff blocks

At all 14 baseline cutoffs, generate separate finite float32 NPZ arrays with
Unicode IDs, ordered feature names and a source/code identity. Each builder
filters all three input layers strictly before the cutoff before aggregating.
No future outcomes or label-derived feature selection are used.

| Block | New columns | Meaning |
|---|---:|---|
| `shop_cadence` | 21 | Shop-only gaps, unavailable-gap flags, recency and corrected shop-cadence proxy |
| `basket_attachment` | 57 | Same-receipt and same-day fuel/shop association, ticket sizes and quarter changes |
| `category_state` | 366 | 15 fields for each of 24 original categories plus six maturity counts |
| `coverage_and_ratios` | 28 | Data coverage, first-observed exposure, log gap ratios and empty-quarter transitions |
| `fine_category_and_site` | 17 | Level3 breadth/concentration and personal site changes |

The 489 columns are separate candidate blocks, not an accepted 794-feature
model. Adding only shop cadence to the 305 baseline would give 326 inputs.
Some blocks intentionally expose states already indirectly recoverable from
the baseline. Numerical gains and predictive gains are distinct questions.
Coverage is split into dataset availability versus customer first-observed
exposure; first observed activity is not true enrollment or lifetime tenure.

`feature_catalog.json` documents every column, unit and missing fill. NPZ
files live under `features/<block>/<cutoff>.npz`. `load_blocks.py` verifies
checksums, identity and exact requested ID order, then returns NumPy arrays
for explicitly selected blocks. It never silently reorders customers.

```python
from load_blocks import load_blocks
extra, extra_names = load_blocks(
    'outputs/feature_rebuild', '2025-03-01', baseline_ids, ['shop_cadence']
)
# Downstream experiment only, after its fitting proposal is registered:
# X = np.column_stack([baseline_X, extra])
```

## Evidence and portable artifacts

`construction_proposal.json` records definitions and future evaluation gates.
`construction_manifest.json` records source and builder hashes.
`layer*_report.json`, `layer_checksums.json`, `feature_build_report.json`
and `decision.json` record alignment, reconciliation and construction status.
The build deletes all future lines and rebuilds receipts at March, September
and December to require exact equality for all five feature blocks. Original
input, baseline cache and best-submission hashes must remain unchanged.

`feature_blocks_colab.zip` packages only portable arrays, metadata, this
README and the NumPy consumer; it excludes raw/pandas layer files. Uploading
or fitting is a separate step. Combining blocks, model parameters and tuning
on June/September is not authorized by this construction pass.

For the next model experiment, preregister one block and hold architecture,
training rows, seeds, blend and vR fixed. Screen mean gain >=0.004 and each
origin >=-0.002; only after passing, evaluate June/September, each >=0.0015
and mean >=0.004. CLV requires its own regression proposal and comparator.

## Run the first model experiment in Colab

The feature-only archive does not train models. Use the self-contained
`outputs/feature_rebuild/colab_shop_cadence_bundle.zip` and companion
`Sasol_Shop_Cadence.ipynb`. Upload the ZIP to `MyDrive/sasol/`; upload the
notebook to Colab and run its cells in order in a fresh CPU runtime. The ZIP
includes all original training inputs and cached baseline predictions plus
only the 21 shop features. No other ZIP is required. Dependencies are pinned.
The final cell displays `shop_cadence_decision.json`. Completed origins are
saved under `shop_cadence_bundle/shop_cadence_predictions/` on Drive.
To resume, mount Drive and rerun the fit cell with the same pinned packages.

Rebuild the experiment bundle locally without fitting:

```bash
python3 -B experiments/feature_rebuild/build_shop_colab_bundle.py
```

Frozen design: `PROPOSAL_SHOP_CADENCE.md`. Test gates and cache/export logic
on a separately extracted temporary bundle, using synthetic rows only:

```bash
python3 -B experiments/feature_rebuild/verify_shop_runner.py --bundle-dir /tmp/extracted_shop_bundle --report /tmp/shop_runner_verification.json
```

This is a 326-feature LightGBM test of the new representation, separate from
the earlier Transformer experiment. No GPU is required. Baseline arrays keep
their original float64 precision; the candidate block is cast from its fixed
float32 artifact. No accepted model is refitted during bundle preparation.

## E2 basket attachment — rejected

E1 shop cadence failed its screen (mean -0.001864, minimum -0.004653).
E2 tests only the 57 basket-attachment features plus the original 305;
it does not carry E1 features forward. Proposal: `PROPOSAL_BASKET_ATTACHMENT.md`.

Upload `outputs/feature_rebuild/colab_basket_attachment_bundle.zip` to
`MyDrive/sasol/`, then copy `run_basket_colab_cell.py` into one normal Python
cell in a fresh Colab notebook. CPU runtime is sufficient. It mounts Drive,
extracts the separate bundle, installs pinned packages, validates inputs,
streams training and prints `basket_attachment_decision.json`. No bash magic
is used. Re-running reuses compatible completed prediction caches on Drive.
The companion `Sasol_Basket_Attachment.ipynb` is also available.

Rebuild the E2 bundle locally without fitting:

```bash
python3 -B experiments/feature_rebuild/build_basket_colab_bundle.py
```

E2 retained the original E1 model recipe and gates. The user-reported Colab
screen failed (mean -0.001044, minimum -0.004190); no export.

## E3 category state — current Colab experiment

Test only 366 category-state columns plus the original 305, giving 671 inputs.
E1 shop cadence and E2 basket attachment are excluded. E3 retains the existing
LightGBM recipe and all acceptance gates. Proposal: `PROPOSAL_CATEGORY_STATE.md`.

Upload `outputs/feature_rebuild/colab_category_state_bundle.zip` to
`MyDrive/sasol/`. Paste `run_category_colab_cell.py` into one normal Python
cell in a fresh Colab notebook (CPU runtime is sufficient). The launcher
installs pinned dependencies, checks inputs, streams fitting and prints
`category_state_decision.json`. Completed origins are cached in the separate
`category_state_bundle/category_state_predictions/` Drive folder. A companion
`Sasol_Category_State.ipynb` is also available. No bash magic is needed.

Rebuild the E3 bundle locally, without fitting:

```bash
python3 -B experiments/feature_rebuild/build_category_colab_bundle.py
```

The known historical growth requirements in this block are predictive inputs,
not reconstructed future labels; the official generator/config stay unchanged.
No real-customer E3 model was fitted during preparation.
