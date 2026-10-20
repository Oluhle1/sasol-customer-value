# September gap: 0.5118 is not in the code on disk

0.5118 is claimed by the 28 September 2026 handoff as the September weighted F1 of a two-step Opportunity rule: every existing-category growth times 1.3 and every adoption times 2.5, then Fuel growth times 0.9 and Food and beverage growth times 3.5, with Beverages and Other left at 1, seeds 7 and 23 averaged, then argmax. That rule is not in the code on disk, so a replay cannot be scored from this script without changing it. The script was not changed. No submission was written.

## What is on disk

`C:\Users\oluhl\Downloads\Y2YrnikRKzLjsyEg-grok-workspace\ml\run_pipeline.py` and `public\run_pipeline.py` are the same file (48909 bytes, SHA-256 prefix 915c9799b152f9f9, dated 28 Sep 2026 01:26). `public\submission.csv` is there (303370 bytes) and is byte-identical to `C:\Users\oluhl\Downloads\submission.csv`. `Downloads\submission (1).csv` is a different file (297295 bytes). `pipeline_v4.log` is not in the grok workspace.

`rescale()` in `ml/run_pipeline.py` (about line 625) multiplies every label that starts with "Existing-category growth" by one growth factor and every label that starts with "New category adoption" by one adoption factor. The September-holdout grid (about lines 649-650) is growth in {1.0, 1.2, 1.45, 1.7} and adoption in {1.0, 2.0, 4.0, 7.0}. There is no per-class Fuel or Food factor. The strings 0.5118, a Fuel factor of 0.9, and a Food factor of 3.5 are not in the file. The classifier fitted for that search is one model, seed 7 (about line 596), not the average of seeds 7 and 23.

The saved briefing next to that script, `src\data\briefing.json`, records this pipeline's September selection fold as weighted F1 0.5029, fuel RMSE 0.6056, shop RMSE 0.7491, opportunity approach "multiclass". No September probability file from `run_pipeline.py` is saved in the workspace, so the handoff rule cannot be applied to that fold without editing and rerunning the script.

## Where 0.5118 is written in this repo

It is a hardcoded gate, not a computed score:

- `top4_shipped/train_growth_override.py` line 22: `SHIPPED_SEPTEMBER_F1 = 0.5118`
- `top4_shipped/README.md` (the gate and the result paragraph)
- `top4_shipped/train_adoption_multiplier.py` line 25
- `top4_shipped/outputs/decision.json` and `adoption_multiplier_decision.json` copy that constant into `published_september_f1`
- `top4_papers/apply_category_recency.py` line 23 and `top4_papers/README.md`

## The 0.5083 figure is a different pipeline

`top4_shipped/train_growth_override.py` is not the same code as `ml/run_pipeline.py`. It execs the older feature builders, then adds history features the pipeline function does not emit, trains two classifiers (seeds 7 and 23) with different tree settings, and applies snapshot sample weights 0.70, 0.85, 1.00, 1.15, 1.30. Its control path, `apply_rule` (lines 217-221), is a single argmax after collapsed multipliers (Fuel 1.17, Beverages 1.3, Food and beverage 4.55, Other 1.3, adoption 2.5). That is not the handoff's two-step rule.

The Stable-to-growth override is a separate function (`override`, lines 233-240). It is applied on September only when June selects a threshold (`main`, lines 352-362). `top4_shipped/outputs/decision.json` has `"threshold": null`, so the override was not applied. `september_shipped_f1` and `september_override_f1` are both 0.5082986958829383. The adoption-multiplier control on the same rule, with no override at all, repeated that number (`september_control_weighted_f1` in `adoption_multiplier_decision.json`).

No September label file from the run that claimed 0.5118 was found, so the 0.0035 gap was not measured as a row disagreement. It is a pipeline mismatch: 0.5118 is not produced by the script on disk.

Later tests of this shipped script should use the same-run control 0.508299, not 0.5118. 0.5118 is not reproducible without changing `ml/run_pipeline.py`.
