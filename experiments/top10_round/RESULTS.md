# 14 October top-10 round

Best public file is `submissions/sasol_top10_rank4.csv` at **0.309411451**, reported on 14 October 2026. That is 0.000247187 above `submissions/sasol_oct10_small_fuel_values.csv` at 0.309164264. The rank4 file was kept. No newer submission was written.

User-reported score to beat: **0.316927909**. Gap from the current file: **0.007516458**.

The local reconstruction `0.3 * (1 - fuel_rmse / 0.6905) + 0.3 * (1 - shop_rmse / 0.9296) + 0.4 * weighted_f1` matches that public score within about 0.001 when the accepted June and September metrics are inserted. Those denominators are the project's persistence scales, not a figure published on the Zindi page. Under that reconstruction the gap is about 0.019 weighted F1, or 0.018 fuel RMSE, or 0.024 non-fuel RMSE.

## What was tested

The rank corrector could only replace the category-dynamics label with its own second label. It was fit on March–May out-of-fold scores. It lost on all four later months. September weighted F1 fell by 0.0156. Decision: `outputs/top10_round/rank_corrector_decision.json`.

A recency-specific blend of the three saved fuel components, with weights fit only on April and May, was worse on June, July, August and September.

Other saved classifiers (monthly, CatBoost, early-history, station) contain the true label for only about one tenth of the rows where the main head ranks that label second. Voting with them does not raise weighted F1.

Year-ago fuel and last-quarter fuel, blended with the accepted fuel forecast using April–May weights, received zero weight. Pure non-fuel hurdle instead of the 50/50 blend gained well under 0.001 RMSE and lost July.

Diesel versus petrol history is absent from the 305 features and has partial correlation about -0.14 with lapsed September fuel CLV after fill size and last-quarter volume. Retraining the accepted 40/35/25 fuel recipe on that history lost April and May, so the six-origin rule can no longer pass. June and July gains were 0.000089 and 0.000024. The run was stopped. Decision: `outputs/top10_round/fuel_grade/decision.json`.

## Where the error still is

On September, the true Opportunity class is the head's second class for 1,206 of 5,238 customers. Correcting every such row would raise weighted F1 by 0.227. The median margin on a wrong row is 0.19, so these are not near-tie cases. The saved models do not know which of those rows to flip.

Fuel RMSE is 0.405 for customers who bought fuel in the last week and 0.831 for customers whose last fuel purchase was 30–90 days earlier. In that lapsed group the accepted forecast correlates 0.43 with the outcome. Year-ago volume correlates 0.16 and is a worse forecast than the model.

## Four-class ranker

A 47-leaf four-class model reallocated Inactivity, Stable, Fuel growth and the remaining mass. The 17-class model still chose the label inside that remaining mass. The full rule gained weighted F1 on five of six origins. September rose from 0.516557 to 0.520405. June rose by 0.000103 and August fell by 0.000297. The mean of June through September was 0.001121, below the pre-registered 0.004, so the rule was not accepted. The within-three reallocation lost June.

The September gain came from Food & beverage growth, Other growth and a few adoption labels. Fuel-growth F1 moved from 0.480375 to 0.480637. Stable F1 fell by 0.001059.

`submissions/sasol_top10_rank4.csv` keeps both CLV columns and changes 312 Opportunity labels. It is an experimental export of the failed-gate full rule. The public board later scored it at 0.309411451, so it is the file to keep.

A regressor trained only on customers with fuel recency above 30 days raised fuel RMSE on all six origins, by 0.00376 on June and 0.00096 on September. It was not exported.

## Extensions of the four-class rule

Two extensions were scored against the four-class full rule. The gate required June and September each to improve, a June–September mean gain of at least 0.001, five of six origins improving, and no loss worse than 0.002. Decision: `outputs/top10_round/rank5/decision.json`.

The five-class mass model separates other growth from adoption. Gains versus the four-class rule were April −0.007000, May −0.004970, June +0.000870, July +0.003714, August +0.002312, September −0.003973. September Food & beverage growth predictions fell from 75 to 55 and Stable predictions rose from 1682 to 1754. September weighted F1 fell back to 0.516432, below the original 17-class model.

The 14-class remainder model keeps the four-class masses and replaces the split inside them. Gains were April +0.002211, May +0.000419, June −0.001662, July +0.000229, August +0.002266, September −0.002528. September Food & beverage growth F1 rose by 0.0136 and Other growth F1 fell by 0.0242.

A mix of the original 17-class probabilities and the four-class full probabilities was chosen on April and May only, from strengths 0.25, 0.50, 0.75 and 1.00. Mean weighted F1 rose at every step, and full strength won: April–May means were 0.495310, 0.497414, 0.499485 and 0.501527. No blended file was written. Decision: `outputs/top10_round/rank4_blend/decision.json`.
