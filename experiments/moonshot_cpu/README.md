# CPU customer-memory variants

The original LambdaRank family failed: standalone screen mean -0.0195408,
minimum -0.0270191; half blend mean -0.00360494, minimum -0.0102984.
Both failed promotion and the separate exploratory rule. No exports.
The original GPU CatBoost families remain untested because Colab's GPU quota
was exhausted. These are separately preregistered CPU designs, not silent
fallbacks under the original proposal. No original result is overwritten.

Keep the original sasol_moonshot_bundle.zip on Drive and upload the small
sasol_moonshot_cpu_patch.zip beside it under MyDrive/sasol. Paste
run_colab_cell.py into one normal Python cell, using a CPU runtime.
The launcher checks the original input manifest, extracts into the separate
moonshot_cpu_memory_bundle folder and applies only the frozen code/proposal
patch. Model, prediction and result caches remain separate from the ranker.

## Two CPU families and four views

All794 numerical inputs plus categorical customer ID, unchanged labels and
purged temporal fitting/full refit. Unweighted memory model: depth6, cap500,
LR .06, L2 8. SqrtBalanced variant: depth7, cap650, LR .05, L2 10. Both use
CPU, seed7, one thread, border32, feature subsampling .5, Bernoulli row
sampling .8 and CTR complexity1. No additional parameter/device fallback.
Each family has standalone and the original fixed half-baseline blend.

Original promotion gates and exploratory-export rules are unchanged.
90-minute invocation budget, 10-minute hard subprocess timeout per origin;
errors are recorded without switching algorithms. Completed fits can resume.
Only Opportunity can change in CSVs; official IDs and original CLV strings
are preserved. Exploratory CSVs are never accepted improvements.

The patch manifest covers all unchanged original input files plus the new
runner/proposal. Source/output folders: experiments/moonshot_cpu and
outputs/moonshot_cpu. Real-customer CPU candidate fits have not run locally;
only synthetic execution and original input/baseline checks are performed.
