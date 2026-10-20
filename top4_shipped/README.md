# Shipped 0.30225 rule, then one June-gated growth override

This folder is separate from the frozen Phase 04–06 pipeline and from `top4_submission/`.

The control is the flat multiclass rule in the file that scored 0.302250514.

Two unweighted LightGBM classifiers, seeds 7 and 23, are averaged. Alpha is 0. The label is the argmax after these multipliers:

| Label | Multiplier |
|---|---|
| Inactivity, Stable | 1 |
| Growth: Fuel | 1.17 |
| Growth: Beverages | 1.3 |
| Growth: Food & beverage | 4.55 |
| Growth: Other | 1.3 |
| Every adoption label | 2.5 |

Hierarchy, the Markov factor, and the growth-demotion rule stay off.

## The one change being tested

A pair of models predicts the official Fuel growth event and the official Food & beverage growth event. On a row the shipped rule called Stable, the event model may replace that label with Fuel growth or Food & beverage growth when its probability clears a threshold.

The threshold is chosen only on the June 2025 fold. September 2025 is scored once with that threshold already frozen.

## When a submission file is written

All three have to hold:

1. June weighted F1 is higher than the shipped rule.
2. September weighted F1 is higher than 0.5118, the shipped rule’s published September score.
3. The average of the June and September gains against the shipped rule in this run is at least 0.01.

Otherwise the 0.302250514 file stays the one to select. Fuel and shop values are copied from that file when a submission is written.

## Result

The rebuilt rule scored weighted F1 0.4994 on June and 0.5083 on September. The published September score of the shipped file is 0.5118.

No Stable-to-growth threshold improved June. September was left at 0.5083. The gate failed, and no submission file was written.
