\## 7. Target Correlation Structure



Future fuel volume and future non-fuel expenditure are positively related, but they are not interchangeable customer-value measures.



Observed correlations were:



| Cutoff | Raw Pearson | Raw Spearman | Transformed Pearson | Transformed Spearman |

|---|---:|---:|---:|---:|

| 2024-09-01 | 0.2212 | 0.4142 | 0.3889 | 0.4142 |

| 2024-12-01 | 0.2271 | 0.4408 | 0.4084 | 0.4408 |

| 2025-03-01 | 0.2246 | 0.4810 | 0.4479 | 0.4810 |

| 2025-06-01 | 0.2106 | 0.4971 | 0.4731 | 0.4971 |

| 2025-09-01 | 0.2663 | 0.5170 | 0.4895 | 0.5170 |



The raw Pearson relationship is relatively modest.



The considerably larger Spearman correlations indicate that customers with higher future fuel value also tend to rank higher in future non-fuel value, even though the direct numerical relationship between litres and rand spend is not strongly linear.



The official logarithmic transformations increase Pearson correlation substantially.



This is consistent with the transformation compressing extreme values and reducing the influence of highly valuable customers.



Spearman correlation remains unchanged because the transformation is monotonic and therefore preserves rank ordering.



Fuel and non-fuel CLV should consequently remain separate regression targets.



\---



\## 8. Exact Relationship Between Inactivity and Zero CLV



A deterministic relationship was identified between the Opportunity target and the two regression targets.



The historical snapshots produced the following results:



| Cutoff | Both targets zero | Inactivity |

|---|---:|---:|

| 2024-09-01 | 399 | 399 |

| 2024-12-01 | 689 | 689 |

| 2025-03-01 | 1,031 | 1,031 |

| 2025-06-01 | 1,280 | 1,280 |

| 2025-09-01 | 1,436 | 1,436 |



Across all five snapshots:



&#x20;   inactivity\_with\_nonzero\_target = 0



and:



&#x20;   both\_zero\_but\_not\_inactivity = 0



Therefore, within the official historical labels:



&#x20;   CLV\_fuel = 0

&#x20;   AND

&#x20;   CLV\_nonfuel = 0



is exactly equivalent to:



&#x20;   Opportunity = Inactivity



This does not imply that either regression target independently determines inactivity.



For example, at the latest historical cutoff:



&#x20;   Both targets zero       = 1,436

&#x20;   Fuel zero only          =    96

&#x20;   Non-fuel zero only      = 1,972

&#x20;   Neither target zero     = 1,734



A customer may therefore remain active exclusively through fuel or exclusively through non-fuel activity.



The three prediction tasks are related but not redundant.



This deterministic relationship should later be used as a diagnostic consistency check across model outputs rather than immediately forcing one model's prediction into another model.



\---



\## 9. Macro Opportunity Drift



The broad Opportunity families exhibit different temporal behaviour.



| Cutoff | Growth | Adoption | Inactivity | Stable |

|---|---:|---:|---:|---:|

| 2024-09-01 | 37.33% | 14.27% | 11.06% | 37.33% |

| 2024-12-01 | 36.94% | 10.42% | 16.32% | 36.32% |

| 2025-03-01 | 38.50% | 7.81% | 22.18% | 31.51% |

| 2025-06-01 | 36.85% | 7.86% | 25.80% | 29.49% |

| 2025-09-01 | 38.76% | 7.03% | 27.42% | 26.80% |



Existing-category growth remains relatively stable at approximately 37–39%.



The remaining customer states change considerably.



Between the earliest and latest forecasting origins:



&#x20;   Inactivity:

&#x20;   11.06% -> 27.42%



&#x20;   Stable:

&#x20;   37.33% -> 26.80%



&#x20;   Adoption:

&#x20;   14.27% -> 7.03%



&#x20;   Growth:

&#x20;   37.33% -> 38.76%



This represents approximately:



&#x20;   +16.36 percentage points for Inactivity



&#x20;   -10.53 percentage points for Stable



&#x20;   -7.24 percentage points for Adoption



&#x20;   +1.43 percentage points for Growth



The target-generating process is therefore not temporally stationary.



\---



\## 10. Implications of Temporal Drift



The historical forecasting origins should not be treated as exchangeable samples from an identical population.



The 2025-09-01 forecasting origin is behaviourally more similar in time to the final 2025-12-01 competition cutoff than the earliest 2024 snapshots.



This does not mean earlier snapshots should be discarded.



They provide valuable additional training observations and expose the model to a broader set of customer states.



However, later validation design should explicitly measure performance on recent forecasting origins rather than relying only on an average across all historical periods.



A model performing well on early snapshots but poorly on the latest period would provide weak evidence of generalisation to the final competition window.



\---



\## 11. Full Opportunity-Class Drift



The largest individual class change is Inactivity:



&#x20;   11.06% -> 27.42%



or:



&#x20;   +16.36 percentage points



The largest decline is Stable:



&#x20;   37.33% -> 26.80%



or:



&#x20;   -10.53 percentage points



Among growth categories, overall prevalence is comparatively stable.



The Fuel growth class changes only modestly:



&#x20;   28.16% -> 27.40%



while several smaller growth categories increase slightly.



The decline in adoption is distributed across several categories rather than being caused by one isolated class.



For example:



&#x20;   New category adoption: Beverages

&#x20;   2.91% -> 1.20%



&#x20;   New category adoption: Food \& beverage

&#x20;   2.25% -> 0.95%



&#x20;   New category adoption: Snacking

&#x20;   1.27% -> 0.48%



&#x20;   New category adoption: Tobacco

&#x20;   1.14% -> 0.38%



This pattern is consistent with customers accumulating category history as their observed tenure increases.



Because adoption requires that a category has never previously been purchased, an increasingly mature customer population naturally has fewer categories remaining that can qualify as genuinely new adoption opportunities.



\---



\## 12. Opportunity Class Imbalance



The classifier will face substantial class imbalance.



At the most recent historical forecasting origin, three classes dominate:



&#x20;   Inactivity



&#x20;   Existing-category growth: Fuel



&#x20;   Stable



Together they represent more than 80% of eligible customers.



The remaining fourteen classes divide the comparatively small remaining population.



This imbalance is structurally important because the competition evaluates Opportunity using Weighted F1.



Therefore:



\- dominant classes will strongly influence the overall classification metric;

\- rare categories cannot be ignored completely;

\- and class-specific precision and recall must still be examined during later error analysis.



Macro F1 may still be useful as a diagnostic statistic, but it will not be treated as the competition optimisation objective.



\---



\## 13. Inactivity as a Persistent Customer State



Sequential historical snapshots demonstrate substantial persistence of inactivity.



Among customers classified as Inactivity at one forecasting origin, approximately:



&#x20;   68.7%

&#x20;   75.3%

&#x20;   77.7%

&#x20;   76.0%



remain inactive at the following quarterly forecasting origin.



This implies that customer inactivity is not usually a one-quarter random event.



Historical recency and duration of inactivity are therefore likely to contain useful predictive information.



This conclusion is carried forward as a Phase 03 modelling hypothesis rather than being converted into a feature during Phase 02.



\---



\## 14. Reactivation and the Growth Definition



An important structural behaviour follows from the official growth rules.



A customer can have historical experience with a category while recording no purchases in that category during the most recent three-month comparison window.



If that customer returns during the following outcome period, previous-quarter category spend is zero.



Provided the returned spend satisfies the absolute growth threshold and positive-purchase requirement, the category may therefore qualify as Existing-category growth.



Consequently, transitions from:



&#x20;   Inactivity -> Growth



or:



&#x20;   Dormant category -> Growth



may represent reactivation rather than continuous expansion of an already-active purchasing pattern.



This distinction is important for later customer-state representation.



The future modelling system should be capable of distinguishing:



&#x20;   continuous activity,



&#x20;   declining activity,



&#x20;   dormant historical activity,



&#x20;   reactivation,



&#x20;   and genuinely new category adoption.



\---



\## 15. Implications for Phase 03



Phase 02B establishes several requirements for the future customer representation.



The following concepts must be expressible from information available before each forecasting cutoff:



\### Customer activity state



The representation must distinguish highly recent, intermittently active and dormant customers.



\### Historical category exposure



The representation must record whether each relevant category has ever previously been purchased.



This is necessary because adoption depends on complete historical category exposure.



\### Recent category state



Historical exposure alone is insufficient.



The representation must separately describe behaviour during the most recent comparison period because growth depends on recent-quarter category activity.



\### Reactivation state



A historically known category with no recent purchases must be distinguishable from both:



\- continuously active categories; and

\- completely unseen categories.



\### Fuel and non-fuel behaviour



Fuel and non-fuel histories should remain separately represented because their future values are related but not equivalent.



\### Temporal momentum



The observed drift and label mechanics suggest that static lifetime totals alone will be insufficient.



Recent activity must eventually be compared with longer-term customer behaviour.



These are requirements for Phase 03 feature design, not completed feature definitions.



\---



\## 16. Phase 02B Conclusion



Detailed target forensics confirms that the Sasol challenge is a temporally changing, multi-output customer-state prediction problem.



The principal findings are:



1\. Fuel and non-fuel future value are positively related but remain distinct regression targets.



2\. The official log transformations reduce raw-value distortion and produce a more linear relationship between the two value measures.



3\. Inactivity is exactly associated with simultaneous zero fuel and zero non-fuel future value in all historical snapshots.



4\. Inactivity becomes substantially more common through time.



5\. Stable behaviour and new-category adoption decline as the customer population matures.



6\. Existing-category growth remains relatively stable at the macro level.



7\. The full 17-class Opportunity target is strongly imbalanced.



8\. Inactivity shows substantial quarter-to-quarter persistence.



9\. Reactivation can legitimately appear as Existing-category growth because of the official comparison-window rules.



10\. The most recent historical forecasting periods must receive particular attention during later validation because the target distribution is changing over time.



Phase 02B therefore supports the original temporal customer-snapshot methodology and provides the behavioural requirements needed to begin feature design.



\---



\## 17. Phase 02 Exit Decision



Phase 02 is considered complete when:



\- Phase 02A historical labels have been reproduced successfully;

\- Phase 02B target-forensics integrity checks pass;

\- temporal target behaviour is documented;

\- cross-target relationships are documented;

\- Opportunity imbalance is documented;

\- and the resulting requirements for customer-state representation are recorded.



No machine-learning model has been trained during Phase 02.



No final predictive feature matrix has been created during Phase 02.



The project may now proceed to:



\# Phase 03 — Temporal Customer Representation and Feature Engineering

