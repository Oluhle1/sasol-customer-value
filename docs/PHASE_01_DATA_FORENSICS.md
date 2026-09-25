\# Phase 01 — Transaction Data Forensics



\*\*Project:\*\* Sasol Customer Value Recruitment Challenge  

\*\*Phase:\*\* 01  

\*\*Status:\*\* Complete  

\*\*Date completed:\*\* 25 September 2026  

\*\*Next phase:\*\* Phase 02 — Label Reconstruction and Target Forensics



\---



\## 1. Purpose



The purpose of Phase 01 was to understand the structure, quality, scope and temporal characteristics of the Sasol transaction data before performing feature engineering or predictive modelling.



The phase was intentionally completed before model development.



The main principle was:



> Do not model a dataset until the unit of observation, temporal structure, business meaning and data-generating process are understood.



This phase therefore focused on answering the following questions:



\- What does one row represent?

\- What constitutes a transaction?

\- What constitutes a customer?

\- What period of customer history is available?

\- How does the final test population relate to the training population?

\- Are there missing values, duplicates or structural inconsistencies?

\- What is the balance between fuel and non-fuel behaviour?

\- How heterogeneous are customers?

\- Which implementation details in the official rules could lead to incorrect labels or leakage?

\- What should the fundamental modelling unit be?



No predictive model was trained during this phase.



\---



\## 2. Source Files



The following official challenge files were used.



\### Transaction data



`train.csv`



Historical Sasol loyalty transaction data at line-item level.



`test.csv`



Final prediction population containing customer IDs.



`SampleSubmission.csv`



Required final submission structure.



\### Official documentation



`DATA\_DICTIONARY.md`



Defines the supplied transaction fields and their meaning.



`LABEL\_RULES.md`



Defines eligibility, outcome windows, CLV construction and Opportunity-label generation.



`label\_config.json`



Contains authoritative thresholds, category mappings, Opportunity class names, normalization constants and historical calibration cutoffs.



`label\_rules.py`



Official reproducible implementation of Sasol's target-generation rules.



These files are treated as the authoritative specification for the competition.



\---



\## 3. Raw Dataset Structure



The full training dataset contains:



| Property | Observed value |

|---|---:|

| Transaction-line records | 467,880 |

| Unique transactions/baskets | 365,638 |

| Unique customers | 6,204 |

| Final test customers | 5,488 |

| Locations | 390 |

| Stock items | 6,849 |

| Level-2 categories | 27 |

| First transaction date | 19 March 2024 |

| Last transaction date | 30 November 2025 |



The raw training file therefore represents approximately twenty months of historical loyalty activity immediately preceding the final forecast date.



\---



\## 4. Fundamental Data Hierarchy



The raw data is not at customer level.



The observed hierarchy is:



```text

Customer

&#x20;   |

&#x20;   +-- Transaction / Basket

&#x20;           |

&#x20;           +-- Line item

&#x20;           +-- Line item

&#x20;           +-- Line item

```



One `TransactionId` may appear on multiple rows because a single transaction can contain multiple products.



The official data dictionary confirms that `TransactionId` is the basket identifier and that multiple rows may share the same transaction.



Therefore:



```text

row count != transaction count

```



and:



```text

line-item frequency != visit frequency

```



This distinction is fundamental.



For example, a transaction containing fuel, coffee and confectionery represents one basket containing three line items, not three customer visits.



\---



\## 5. Basket Structure



The transaction data contains:



\*\*365,638 unique baskets\*\*



from:



\*\*467,880 line items\*\*



The distribution of lines per basket is highly concentrated.



| Statistic | Lines per basket |

|---|---:|

| Median | 1 |

| 90th percentile | 2 |

| 99th percentile | 6 |

| Maximum | 103 |



Most transactions therefore contain very few items, but some baskets contain substantially more.



This has two modelling implications.



First, basket-level features must count unique `TransactionId` values rather than transaction rows.



Second, large multi-item baskets must not artificially inflate behavioural frequency measures.



\---



\## 6. Important Monetary Field Behaviour



`TransactionTotalInclAmount` contains the whole basket total and is repeated across the line items belonging to the transaction.



It must therefore never be summed directly across line-item rows.



Doing so would multiply the economic value of transactions according to the number of items in the basket.



The official label rules instead use the signed line-level `TotalExclAmount` field where applicable.



`TotalModifierAmount` must also not be subtracted from `TotalExclAmount` again because the supplied amount already reflects the source transaction amount required by the competition rules.



This distinction will be preserved throughout all subsequent phases.



\---



\## 7. Data Integrity Assessment



The source dataset was tested for major structural problems.



The following results were observed:



| Integrity test | Result |

|---|---:|

| Exact duplicate rows | 0 |

| Duplicate `TransactionId + LineItemSequence` combinations | 0 |

| Baskets linked to multiple customers | 0 |

| Baskets linked to multiple locations | 0 |

| Baskets with inconsistent timestamps | 0 |

| Numeric parsing failures | 0 |

| Blank fields detected in supplied columns | 0 |



The repeated basket totals were also internally consistent within transactions.



The source data is therefore structurally clean enough that extensive generic cleaning is not justified.



Data preparation will focus primarily on preserving Sasol's business rules rather than applying arbitrary data-cleaning transformations.



\---



\## 8. Customer Eligibility Verification



The official challenge rules state that a customer is eligible at a forecasting cutoff only when the customer has at least:



```text

3 distinct historical TransactionId values

```



strictly before the cutoff.



At the final prediction cutoff, the transaction history contains:



\*\*6,204 customers\*\*



Applying the official minimum-three-transaction rule produces:



\*\*5,488 eligible customers\*\*



The supplied `test.csv` also contains:



\*\*5,488 customers\*\*



The customer sets match exactly.



Therefore:



```text

final eligible historical population

=

test population

```



This provides an important independent validation that the official eligibility logic has been interpreted correctly.



The remaining:



```text

6,204 - 5,488 = 716

```



customers have insufficient historical transaction depth for the final prediction task.



\---



\## 9. Consequence for Historical Training Snapshots



Eligibility must be recalculated independently at every historical cutoff.



A customer who is eligible on 1 December 2025 may not have been eligible on 1 September 2024.



Therefore the final test population must not simply be reused across all historical experiments.



Doing that would introduce information from the future about which customers eventually become sufficiently active.



For a historical cutoff \\(t\\):



\\\[

Eligible\_i(t)

=

I(TransactionCount\_i(<t) \\ge 3)

\\]



Only customers satisfying this condition at that particular point in time may form part of that historical training snapshot.



\---



\## 10. Temporal Coverage



The observed transaction history runs from:



```text

19 March 2024

```



through:



```text

30 November 2025

```



The final production-style forecasting cutoff is:



```text

1 December 2025

```



The unseen outcome period is therefore:



```text

1 December 2025

to

28 February 2026

```



or, formally:



\\\[

\[2025\\text{-}12\\text{-}01,\\;2026\\text{-}03\\text{-}01)

\\]



The problem is therefore inherently temporal.



Random train/test splitting would not accurately represent the final task.



\---



\## 11. Official Historical Forecasting Origins



The configuration file supplies five historical quarterly cutoffs:



```text

1 September 2024

1 December 2024

1 March 2025

1 June 2025

1 September 2025

```



These will be used as historical forecasting origins.



The resulting research design is:



```text

History before cutoff

&#x20;       |

&#x20;       v

Customer snapshot at t

&#x20;       |

&#x20;       v

Predict next three months

```



For example:



```text

History before 1 Sep 2024

&#x20;       ->

Sep-Nov 2024 outcomes



History before 1 Dec 2024

&#x20;       ->

Dec 2024-Feb 2025 outcomes



History before 1 Mar 2025

&#x20;       ->

Mar-May 2025 outcomes



History before 1 Jun 2025

&#x20;       ->

Jun-Aug 2025 outcomes



History before 1 Sep 2025

&#x20;       ->

Sep-Nov 2025 outcomes

```



The final competition task then becomes:



```text

History before 1 Dec 2025

&#x20;       ->

Dec 2025-Feb 2026 unknown outcomes

```



This provides a natural framework for realistic backtesting.



\---



\## 12. Fundamental Modelling Observation



The project will not treat the machine-learning observation simply as:



\\\[

Customer

\\]



Instead, the fundamental observation will be:



\\\[

\\boxed{Customer \\times Forecasting\\ Origin}

\\]



A conceptual training table will therefore resemble:



```text

ID          cutoff        features...        targets...

Customer A  2024-09-01    ...                ...

Customer A  2024-12-01    ...                ...

Customer A  2025-03-01    ...                ...

Customer B  2024-12-01    ...                ...

Customer B  2025-03-01    ...                ...

```



provided each customer satisfied the eligibility requirement at the relevant cutoff.



This design allows customer state to change over time and creates multiple legitimate historical forecasting observations from the transaction history.



\---



\## 13. Transaction Types



The dataset contains three observed line-item types:



| Line-item type | Rows |

|---|---:|

| `Sale\_Fuel` | 342,760 |

| `Sale` | 123,760 |

| `Sale\_Voucher` | 1,360 |



Fuel therefore represents the majority of recorded transaction lines.



However, line-item counts must not be confused with basket counts or customer-level behaviour.



\---



\## 14. Basket-Level Fuel and Non-Fuel Composition



Baskets were classified according to whether they contained fuel, qualifying non-fuel merchandise or both.



The resulting composition was approximately:



| Basket type | Share |

|---|---:|

| Fuel only | 88.08% |

| Non-fuel only | 8.85% |

| Fuel + non-fuel | 3.08% |



The customer journey is therefore heavily fuel dominated.



Only a relatively small proportion of baskets combine fuel and convenience-store purchases.



This creates a potentially meaningful behavioural dimension for future analysis:



\\\[

Fuel \\rightarrow NonFuel

\\]



interaction.



At this phase, this is treated only as a feature hypothesis.



No claim is made that such behaviour predicts the Opportunity target until it is tested under temporal validation.



\---



\## 15. Customer Transaction Depth



Customer activity is highly heterogeneous.



Among the final eligible customer population:



| Statistic | Historical baskets |

|---|---:|

| Median | 36 |

| 90th percentile | 153 |

| 99th percentile | approximately 468 |

| Maximum | 1,302 |



Therefore different customers provide substantially different amounts of historical evidence.



A customer with three historical baskets cannot be treated as behaviourally equivalent to a customer with hundreds of observed baskets.



Future features should therefore preserve concepts such as:



```text

customer tenure

number of historical baskets

active days

transaction density

observation depth

```



rather than relying on total value alone.



\---



\## 16. Customer Recency



At the final forecasting cutoff, the distribution of customer recency is highly skewed.



Approximate observed values include:



| Statistic | Days since most recent transaction |

|---|---:|

| Median | 14.2 days |

| 75th percentile | approximately 100 days |

| 90th percentile | approximately 315 days |



Some customers therefore transact very recently while others remain eligible despite not having transacted for many months.



This indicates substantial variation in customer lifecycle state.



Recency is consequently expected to be particularly relevant for modelling future inactivity.



This remains a hypothesis to be tested rather than an assumed model relationship.



\---



\## 17. Recent-Quarter Activity



During the most recent three-month comparison window before the final cutoff:



```text

September-November 2025

```



approximately:



\*\*4,052 of 5,488\*\*



eligible customers recorded qualifying in-scope positive activity.



Therefore approximately:



\*\*1,436 customers\*\*



had no qualifying activity during the latest three-month comparison period.



This represents approximately:



\\\[

26.2\\%

\\]



of the final prediction population.



A lack of recent activity does not automatically imply future Inactivity.



However, it confirms that the dataset contains a substantial dormant or intermittent customer population.



\---



\## 18. Historical Fuel and Non-Fuel Participation



Among the final eligible customers:



```text

5,405

```



have historical qualifying fuel activity.



Approximately:



```text

4,069

```



have historical qualifying non-fuel activity.



Customer histories can therefore be divided approximately into:



| Historical behaviour | Customers |

|---|---:|

| Fuel + non-fuel | 3,986 |

| Fuel only | 1,419 |

| Non-fuel only | 83 |



This is an important structural characteristic of the problem.



A substantial customer population has historically purchased fuel but has not yet participated in qualifying convenience categories.



These customers may be particularly relevant when later modelling new-category adoption.



No targeting or predictive conclusion is drawn during Phase 01.



\---



\## 19. Category Breadth



Eligible customers vary substantially in the number of categories they have historically purchased from.



Approximate observed distribution:



| Statistic | Historical category breadth |

|---|---:|

| Median | 3 |

| 75th percentile | 6 |

| 90th percentile | 8 |

| Minimum | 1 |



This suggests another potentially useful behavioural concept:



```text

customer category maturity

```



A customer with one historical category has a very different adoption opportunity space from a customer already purchasing across many categories.



Category breadth will therefore be considered during feature engineering.



\---



\## 20. Literal `NA` Category



The dataset contains a literal Level-2 category named:



```text

NA

```



with approximately:



```text

1,699

```



observed transaction rows.



This value is not missing data.



The official rules explicitly distinguish the literal `"NA"` category from an empty or missing value.



`NA` contributes to qualifying activity and CLV calculations according to the official rules.



This creates an important implementation requirement.



The transaction CSV must be loaded without automatically converting `"NA"` into a missing value.



For pandas-based pipelines, default NA conversion must therefore be handled explicitly.



Failure to preserve the literal category would change customer histories, Opportunity labels and non-fuel CLV calculations.



\---



\## 21. Negative Transaction Values



The source data contains genuine signed transaction values.



Observed values included approximately:



```text

322 rows with negative Quantity

352 rows with negative TotalExclAmount

```



These observations must not automatically be removed, converted to absolute values or clipped to zero.



The official label rules state that genuine product returns reduce net totals.



Therefore negative source values retain their economic meaning.



Future feature engineering may distinguish among:



```text

gross purchases

returns

net spend

return frequency

```



if error analysis suggests that this behaviour is predictive.



\---



\## 22. Fuel Source Anomalies



The official label documentation states that three non-standard source Fuel/Sale rows exist and must not contribute to the fuel-litre regression target.



The dataset contains exactly:



```text

3 Fuel rows with LineItemType = Sale

```



in addition to:



```text

342,760 Fuel rows with LineItemType = Sale\_Fuel

```



This agrees with the official documentation.



Only correctly defined `Fuel + Sale\_Fuel` records will contribute to the official fuel target.



\---



\## 23. Adjustment Items



The following named adjustment items occur in the transaction history:



| Item | Observed rows |

|---|---:|

| `TOTAL/ROUNDING DISCO` | 309 |

| `WOW\_PROMO` | 4 |

| `PENSIONER DISCOUNT` | 4 |



These records remain in the supplied historical dataset.



However, the official target rules exclude them from target construction and qualifying purchase activity.



They must therefore be removed when reproducing Sasol's label-generation rules.



They should not simply be deleted globally from the raw dataset because the raw layer is intended to preserve the source data unchanged.



\---



\## 24. Raw Data Preservation Principle



The project will follow a layered data architecture.



\### Raw layer



`data/raw/`



Contains unchanged official competition files.



Raw data will never be manually edited.



\### Processed layer



`data/processed/`



Contains reproducibly generated intermediate datasets.



Any transformation from raw to processed data must be implemented in code.



This ensures:



```text

raw data

&#x20;   ->

reproducible processing

&#x20;   ->

processed data

```



rather than:



```text

raw data

&#x20;   ->

manual spreadsheet edits

&#x20;   ->

unknown state

```



\---



\## 25. Leakage Control Principle



For every forecasting origin \\(t\\), predictive features may use only information satisfying:



\\\[

TransactionTime < t

\\]



Outcome information must never enter feature construction.



Similarly, customer eligibility must be determined only from transactions available before that forecasting origin.



This restriction applies to:



\- recency,

\- frequency,

\- value,

\- category history,

\- tenure,

\- trends,

\- customer segmentation variables,

\- transaction counts,

\- behavioural ratios,

\- product history,

\- location behaviour,

\- and every future engineered feature.



If a quantity depends on information occurring on or after the forecasting cutoff, it cannot be used as a predictor for that snapshot.



\---



\## 26. Data-Cleaning Decision



Phase 01 found no evidence that aggressive generic data cleaning is required.



The project will therefore not perform arbitrary operations such as:



```text

dropping all negative values

removing large transactions

replacing unusual categories

winsorising values without evidence

filling legitimate category values

deduplicating already-unique rows

```



Instead, data transformations will be driven by:



1\. official Sasol rules,

2\. demonstrated data-quality problems,

3\. clearly stated feature hypotheses,

4\. or error analysis.



This avoids silently changing the data-generating process.



\---



\## 27. Phase 01 Modelling Implications



Phase 01 establishes that the core modelling challenge is not data cleaning.



The challenge is representing changing customer behaviour across time.



The future modelling system should therefore capture, among other concepts:



\\\[

Recency

\\]



\\\[

Frequency

\\]



\\\[

Monetary/Volume\\ Value

\\]



\\\[

Customer\\ Tenure

\\]



\\\[

Behavioural\\ Trend

\\]



\\\[

Purchase\\ Regularity

\\]



\\\[

Category\\ History

\\]



\\\[

Category\\ Breadth

\\]



\\\[

Fuel/NonFuel\\ Relationship

\\]



\\\[

Customer\\ Lifecycle

\\]



These concepts have not yet been converted into final model features.



Feature engineering will begin only after the targets have been reconstructed and analysed in Phase 02.



\---



\## 28. Decisions Locked at the End of Phase 01



The following decisions are now fixed unless later evidence requires documented revision:



| Decision | Status |

|---|---|

| Preserve raw competition files unchanged | Locked |

| Use `TransactionId` for basket frequency | Locked |

| Never sum repeated basket totals across line items | Locked |

| Preserve literal `NA` category | Locked |

| Preserve genuine negative source transactions | Locked |

| Recalculate eligibility separately at every cutoff | Locked |

| Use temporal rather than random validation | Locked |

| Model observation = Customer × Forecasting Origin | Locked |

| Use historical forecasting snapshots | Locked |

| Prevent all post-cutoff information from entering features | Locked |

| Delay feature engineering until labels are understood | Locked |

| Delay modelling until target forensics is complete | Locked |



\---



\## 29. Phase 01 Exit Criteria



Phase 01 is considered complete because:



\- the row, basket and customer hierarchy is understood;

\- the complete historical period has been established;

\- the final prediction cutoff has been established;

\- the test population has been reconciled with the official eligibility rule;

\- major source-data integrity checks have passed;

\- fuel and non-fuel transaction structures have been investigated;

\- important Sasol-specific edge cases have been identified;

\- the appropriate temporal modelling unit has been defined;

\- leakage-control principles have been established;

\- the raw-data preservation policy has been established;

\- no modelling or feature-engineering decisions have been made using future information.



\---



\# 30. Phase 01 Conclusion



The Sasol transaction dataset is structurally clean and sufficiently rich for longitudinal customer modelling.



The main analytical challenge is not repairing the raw data but converting line-item transaction histories into leakage-free representations of customer state at historical forecasting origins.



The core modelling architecture is therefore:



\\\[

Raw\\ Line\\ Items

\\]



\\\[

\\downarrow

\\]



\\\[

Transactions/Baskets

\\]



\\\[

\\downarrow

\\]



\\\[

Customer\\ History\\ Before\\ Cutoff

\\]



\\\[

\\downarrow

\\]



\\\[

Customer\\ Snapshot

\\]



\\\[

\\downarrow

\\]



\\\[

Future\\ Three\\ Month\\ Outcomes

\\]



The fundamental supervised-learning unit is:



\\\[

\\boxed{Customer \\times Forecasting\\ Origin}

\\]



rather than a static customer record.



Phase 02 will now reproduce the official Sasol labels at each historical forecasting cutoff and analyse the statistical structure of the three prediction targets before feature engineering begins.



\---



\## 31. Next Phase



\*\*Phase 02 — Label Reconstruction and Target Forensics\*\*



The next phase will:



1\. execute the official Sasol label generator at each valid historical cutoff;

2\. verify that the implementation reproduces the documented rules;

3\. measure the number of eligible customers at each forecasting origin;

4\. analyse the distributions of raw fuel and non-fuel future value;

5\. analyse transformed `CLV\_fuel` and `CLV\_nonfuel`;

6\. measure zero-target prevalence;

7\. analyse all 17 Opportunity classes;

8\. quantify class imbalance;

9\. analyse how target distributions shift across time;

10\. examine customer transitions between historical snapshots;

11\. establish the exact modelling and evaluation dataset for Phase 03.



No predictive features will be engineered until this work is complete.

