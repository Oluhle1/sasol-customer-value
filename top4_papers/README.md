# Research applied to the 0.30225 model

The control stays the flat multiclass rule that scored 0.302250514. Multipliers stay frozen. A new idea is kept only if weighted F1 rises on both June 2025 and September 2025, and the September score is above the published 0.5118.

## What this run applies

Anderson, Kumar, Tomkins and Vassilvitskii, "The Dynamics of Repeat Consumption", WWW 2014.

Their result is that, given items a person has already consumed, the strongest predictor of which item they consume again is how recently they consumed it. Frequency and popularity matter less. A time-since-last-consumption effect, close to a power law with an exponential cutoff, fits the repeat sequences.

The Sasol growth rule is that claim in operational form. Existing-category growth requires another qualifying purchase in a category the customer has already bought, with spend at least R20 and 25% above the previous quarter. The shipped feature table has one recency number for fuel and one for the whole shop. It does not have recency inside each category. This run adds, for every included category, the days since the last qualifying purchase and the qualifying-basket count in the last quarter. The trees can split on those values. The September multipliers are not searched again.

## Result

| Feature set | June weighted F1 | September weighted F1 |
|---|---:|---:|
| Shipped features | 0.4994 | 0.5083 |
| Plus category recency | 0.5016 | 0.5084 |

June rose by 0.0021. September rose by 0.0001 and stayed below the published 0.5118. The gate failed. No submission file was written.

## What is not run again

These papers describe the field, and their mechanisms have already been measured here.

- Schmittlein, Morrison and Colombo (1987), Pareto/NBD, and Fader, Hardie and Lee (2005), BG/NBD. A model that forecasts category purchase and spend and then applies the official rule scored about 0.41 weighted F1. The handoff switches that path off.
- Vanderveld, Pandey, Han and Parekh (2016), KDD, and Cragg (1971). Separate models for the chance of a purchase and the amount if a purchase happens. The 0.30225 file already uses that hurdle, at 30% for fuel and 50% for shop.
- Wang, Liu and Miao, "A Deep Probabilistic Model for Customer Lifetime Value Prediction" (arXiv:1912.07753). A zero-inflated lognormal replaces the two-stage hurdle with one likelihood. The leaderboard scores squared error on z = ln(1+y)/s, and the shipped model already trains on z and blends the hurdle. Re-fitting that mixture is not a new test.
- Chamberlain, Cardoso, Liu, Pagliari and Deisenroth (2017), KDD. Customer embeddings from purchase histories. The warm-start embedding ablation in this repository failed its acceptance rule.
- Bauer and Jannach (2021), sequence-to-sequence lifetime value. Weekly histories are long enough for the model to be feasible. A sequence network has not beaten the shipped rule on both later quarters, so it is not the submission.
- Balanced class weights, the hierarchical family model, and another September multiplier grid. All three have been measured. Balanced weights fell to about 0.47. Hierarchy did not beat the flat rule. The Food & beverage multiplier raised September F1 by 0.008 and the public score by 0.0008.
