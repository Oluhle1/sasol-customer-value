# December forecast and error audit

Baseline: 0.309164264; exact CSV and input hashes verified. December 2024 was already used in Phase 04 F1 and label calibration. Only September 2024 is eligible for its training; nested early stopping is infeasible, so all components use 40 rounds fixed before results. December is a reused matched-season diagnostic, not an untouched holdout.

## Population shift

historical_origin,feature,historical_mean,final_mean,smd,historical_zero_fraction,final_zero_fraction,historical_rows,final_rows
2025-09-01,price_change,0.00907127429805632,-0.021830985915492925,-1.2596357894267676e+16,0.0,0.0,5238,5488
2025-09-01,price_index,21.59827213822894,21.126760563380284,-187692599025318.06,0.0,0.0,5238,5488
2025-09-01,price_gap,0.002742249495739582,0.015150771628816635,0.5677293717058707,0.26384116074837727,0.28589650145772594,5238,5488
2025-09-01,age_days,442.4269191038821,516.1177174012787,0.5308042121509563,0.0,0.0,5238,5488
2025-09-01,recency_days,71.57432103304467,84.68490532100762,0.10427696902748622,0.0,0.0,5238,5488
2025-09-01,n_tx,58.478235967926686,66.44806851311954,0.09110226641718726,0.0,0.0,5238,5488
2025-09-01,log_n_tx,3.4559905533112465,3.5573062262510775,0.08726584924588283,0.0,0.0,5238,5488
2025-09-01,fuel_recency,87.12015987740718,101.7647411831222,0.08701880807437948,0.0,0.0,5238,5488
2025-09-01,dyn_fuel_recency,87.12015987740718,101.7647411831222,0.08701880807437948,0.0,0.0,5238,5488
2025-09-01,n_cat_ever,3.665139366170294,3.876275510204082,0.07738849848864089,0.0,0.0,5238,5488
2025-09-01,fuel_growth_gap,-0.39675956355001774,-0.42242947052498964,-0.07366169398284103,0.013936617029400534,0.015123906705539359,5238,5488
2025-09-01,dyn_fuel_q2,5.9494340955861675,5.679014973591225,-0.07315778711693474,0.2527682321496755,0.2857142857142857,5238,5488

## Large-class errors

[
  {
    "origin": "2025-06-01",
    "rows": 4961,
    "majority_three_share": 0.8171739568635356,
    "fuel_growth_support": 1311,
    "median_adjusted_probability_margin": 0.2133978364246364,
    "errors_low_margin_005": 461
  },
  {
    "origin": "2025-09-01",
    "rows": 5238,
    "majority_three_share": 0.8161512027491409,
    "fuel_growth_support": 1435,
    "median_adjusted_probability_margin": 0.24833696166154356,
    "errors_low_margin_005": 390
  }
]

## Threshold and category competition

origin,cohort,n,error_rate
2025-06-01,within_100_rand_of_threshold,1057,0.16367076631977295
2025-06-01,fuel_growth_event_eligible,1315,0.5133079847908745
2025-06-01,fuel_event_lost_category_competition,4,1.0
2025-09-01,within_100_rand_of_threshold,1249,0.16092874299439552
2025-09-01,fuel_growth_event_eligible,1436,0.5355153203342619
2025-09-01,fuel_event_lost_category_competition,1,1.0

## Reproduction

origin,target,fresh_metric,archived_metric,metric_difference,max_prediction_difference,same_validation_ids,rows,December_fixed_rounds
2024-12-01,Opportunity,0.4286594364369827,,,,True,4223,40.0
2024-12-01,CLV_fuel,0.6327432738799041,,,,True,4223,40.0
2024-12-01,CLV_nonfuel,0.822938763236278,,,,True,4223,40.0
2025-06-01,Opportunity,0.5095271507692096,0.5095271507692096,0.0,4.440892098500626e-16,True,4961,
2025-06-01,CLV_fuel,0.5915117844522825,0.5915117844522825,0.0,0.0,True,4961,
2025-06-01,CLV_nonfuel,0.7341000618559954,0.7341000618559954,0.0,0.0,True,4961,
2025-09-01,Opportunity,0.5165568275346171,0.5165568275346171,0.0,4.440892098500626e-16,True,5238,
2025-09-01,CLV_fuel,0.5967862625611445,0.5967862625611445,0.0,0.0,True,5238,
2025-09-01,CLV_nonfuel,0.7423595601227382,0.7423595601227382,0.0,0.0,True,5238,

## First decision

Test one explicit calendar-exposure variant first. Existing seasonality features include prior-year outcomes and elapsed origin_month; seasonality is not absent. Tree thresholds on elapsed month cannot extrapolate a trend. Defer pooled seasonal fitting because only one historical December season is observed. Then test fuel-rule event probabilities via temporally out-of-fold inputs, preserving all class competition; this differs from the rejected three-group specialist.

Metrics remain separate: official component weights are known but leaderboard RMSE normalization constants are unavailable. No external holidays, realized future prices or promotions enter features. See population_comparison.csv, confusion CSVs, class_error_contributions.csv, recency_cohorts.csv, and baseline_reproduction.csv.
