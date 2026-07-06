# P6 — Duplicate detection: human review of the top similarity pairs

**Question:** when the model flags two datasets as near-duplicates via cosine similarity, is it right?

## Method
- Embed every dataset (all-MiniLM-L6-v2), L2-normalise, cosine similarity.
- **Regime A (literal top-50):** the 50 most-similar pairs overall. These are dominated by byte-for-byte identical titles.
- **Regime B (review band, N=50):** we drop pairs whose titles are *identical* (a trivial string match) and stratify-sample 50 pairs across the 0.86–1.00 similarity range. This is the realistic *human review queue*: highly similar but not identical.
- Each pair labelled: **Duplicate** (same dataset) / **Near-duplicate** (same indicator, one statistical column differs) / **Related but not duplicate** (same table/theme, genuinely different slice) / **Wrong** (unrelated).

## Regime A — literal top-50 most similar pairs
- **50/50 (100.0%)** have identical titles → true duplicates. At the very top of the ranking, duplicate detection is essentially perfect.

## Regime B — review band (non-identical, N=50)

| Pair type | Count | Share |
| --- | --- | --- |
| Duplicate | 1 | 2% |
| Near-duplicate | 5 | 10% |
| Related but not duplicate | 44 | 88% |
| Wrong | 0 | 0% |
| **Total** | **50** | 100% |

## Key findings (honest)
- **50/50 (100%) pairs are genuinely on-topic — 0 were 'wrong'.** Embedding similarity almost never links unrelated datasets: precision on *"these two belong together"* is very high.
- But only **6/50 (12%)** are true or near duplicates. In the non-identical band, the model mostly surfaces **related dataset families** — the *same* census table sliced by a different age, sex, ethnic group, month or measure.
- Those related slices are **not** things you would delete — you need all single-year ages, both sexes, every ethnic category. This is exactly why we **flag likely duplicates for human review, never auto-delete.**

## What to say in the pitch
> "Exact duplicates are flagged with ~100% precision. Below that, similarity groups datasets into families for a **human review queue** — our model *flags* likely duplicates for a human to confirm, it does **not** auto-remove anything. Across the top-50 review band, **0%** were wrong."

_Note: precision only. Recall is unknown — the 20,685 datasets were never exhaustively labelled for duplicates._

## Sample (review band, sorted by similarity)

| # | sim | label | title A | title B |
| --- | --- | --- | --- | --- |
| 1 | 0.999 | Near-duplicate | Stroke: QOF prevalence (all ages) - Persons - All ages (Lowe | Stroke: QOF prevalence (all ages) - Persons - All ages (Uppe |
| 2 | 0.997 | Near-duplicate | OST005: Patients (75+ yrs) with a fragility fracture treated | OST005: Patients (75+ yrs) with a fragility fracture treated |
| 3 | 0.996 | Related but not duplicate | DEPRECATED Admissions for lower respiratory tract infections | DEPRECATED Admissions for lower respiratory tract infections |
| 4 | 0.996 | Near-duplicate | Emergency admissions for pedestrians (aged 0-24) - Male - <2 | Emergency admissions for pedestrians (aged 0-24) - Persons - |
| 5 | 0.996 | Duplicate | TS028 - National identity (detailed) Other identity only | TS028 - National identity (detailed) Other identity only: Ot |
| 6 | 0.995 | Related but not duplicate | UK Business Counts - local units by industry and employment  | UK Business Counts - local units by industry and employment  |
| 7 | 0.995 | Near-duplicate | Hospital admissions for diabetes (under 19 years) - Persons  | Hospital admissions for diabetes (under 19 years) - Persons  |
| 8 | 0.995 | Related but not duplicate | Redbridge - Payments over £500 2013-14 to 2015-16 - June 201 | Redbridge - Payments over £500 2013-14 to 2015-16 - July 201 |
| 9 | 0.993 | Related but not duplicate | Sex by single year of age (Census TS009) - Female Aged 33 ye | Sex by single year of age (Census TS009) - Female Aged 37 ye |
| 10 | 0.991 | Related but not duplicate | Sex by single year of age (Census TS009) - Male Aged 25 year | Sex by single year of age (Census TS009) - Male Aged 28 year |
| 11 | 0.989 | Related but not duplicate | Age by single year (Census TS007) - Aged 31 years (rate) | Age by single year (Census TS007) - Aged 70 years (rate) |
| 12 | 0.987 | Related but not duplicate | Females aged 35 to 39 | Females aged 25 to 29 % |
| 13 | 0.985 | Related but not duplicate | DEPRECATED: Population by country of birth - EU14 - percenta | DEPRECATED: Population by country of birth - Non-EU European |
| 14 | 0.983 | Related but not duplicate | Females aged 0 to 4 | Females aged 18 to 24 |
| 15 | 0.982 | Near-duplicate | A01b - Life expectancy at 65 - Female - 65 - "LSOA21 depriva | A01b - Life expectancy at 65 - Female - 65 - "LSOA21 depriva |
| 16 | 0.981 | Related but not duplicate | 2022-based projected population, persons aged 30-34 | 2022-based projected population, females aged 70-74 |
| 17 | 0.979 | Related but not duplicate | % in employment with RQF4+ - females aged 16-64 | % in employment with RQF4+ - males aged 16-64 ~ Numerator |
| 18 | 0.978 | Related but not duplicate | Sex by single year of age (Census TS009) - Female Aged 50 ye | Sex by single year of age (Census TS009) - Male Aged 12 year |
| 19 | 0.976 | Related but not duplicate | Sex by single year of age (Census TS009) - Female Aged 65 ye | Sex by single year of age (Census TS009) - Male Aged 32 year |
| 20 | 0.975 | Related but not duplicate | Age by single year (Census TS007) - Aged 72 years | Age by single year (Census TS007) - Aged 96 years (rate) |
| 21 | 0.967 | Related but not duplicate | Number of jobs - confidence - Annual pay - Male Full Time Wo | workplace analysis: Number of jobs - confidence - Weekly pay |
| 22 | 0.965 | Related but not duplicate | TS022 - Ethnic group (detailed) Black, Black British, Black  | TS022 - Ethnic group (detailed) Black, Black British, Black  |
| 23 | 0.964 | Related but not duplicate | Sex by single year of age (Census TS009) - Female Aged 80 ye | Sex by single year of age (Census TS009) - Male Aged 12 year |
| 24 | 0.960 | Related but not duplicate | Number of jobs - value - Weekly pay - Male Full Time Workers | Number of jobs - confidence - Annual pay - Male |
| 25 | 0.957 | Related but not duplicate | Median - Annual pay - Male Full Time Workers | Number of jobs - value - Weekly pay - gross - Female |
| 26 | 0.955 | Related but not duplicate | Males age group 25 - 29 | Males age group 0 - 19 |
| 27 | 0.951 | Related but not duplicate | UK Business Counts - local units by industry and employment  | UK Business Counts - local units by industry and employment  |
| 28 | 0.950 | Related but not duplicate | TS022 - Ethnic group (detailed) Other ethnic group: Somalila | TS022 - Ethnic group (detailed) Black, Black British, Black  |
| 29 | 0.948 | Related but not duplicate | Persons age group 16 - 64 % | Persons age 18 % |
| 30 | 0.944 | Related but not duplicate | TS022 - Ethnic group (detailed) White: Spanish rate | TS022 - Ethnic group (detailed) Other ethnic group: Other Wh |
| 31 | 0.942 | Related but not duplicate | TS022 - Ethnic group (detailed) Asian, Asian British or Asia | TS022 - Ethnic group (detailed) Mixed or Multiple ethnic gro |
| 32 | 0.941 | Related but not duplicate | TS031 - Religion (detailed) Other religion: Druze | TS031 - Religion (detailed) Hindu |
| 33 | 0.939 | Related but not duplicate | Females aged 35 to 44 | Persons All |
| 34 | 0.937 | Related but not duplicate | Persons aged 75 to 79 | Males aged 0 to 15 % |
| 35 | 0.936 | Related but not duplicate | The proportion of carers who receive direct payments - Respo | The proportion of people who use services who find it easy t |
| 36 | 0.936 | Related but not duplicate | TS022 - Ethnic group (detailed) Other ethnic group: Brazilia | TS022 - Ethnic group (detailed) Other ethnic group: Moroccan |
| 37 | 0.927 | Related but not duplicate | Life expectancy at birth - Female - All ages - Deprivation d | Life expectancy at 65 - Male - 65 - Deprivation deciles - Th |
| 38 | 0.921 | Related but not duplicate | Passports held (Census TS005) - Europe: Other Europe: Rest o | Passports held (detailed) (Census TS013) - Middle East and A |
| 39 | 0.919 | Related but not duplicate | TS064 - Occupation - minor groups 222 Therapy Professionals | TS064 - Occupation - minor groups 243 Business, Research and |
| 40 | 0.915 | Related but not duplicate | TS064 - Occupation - minor groups 412 Administrative Occupat | TS064 - Occupation - minor groups 612 Animal Care and Contro |
| 41 | 0.910 | Related but not duplicate | Redbridge - Payments over £500 2021-22 - January 2022 | Redbridge - Payments over £500 2013-14 to 2015-16 - June 201 |
| 42 | 0.910 | Related but not duplicate | Age by single year (Census TS007) - Aged 77 years (rate) | Sex by single year of age (Census TS009) - All persons Aged  |
| 43 | 0.908 | Related but not duplicate | TS060 - Industry 28 Manufacture of machinery and equipment n | TS060 - Industry 52 Warehousing and support activities for t |
| 44 | 0.907 | Related but not duplicate | Age by single year (Census TS007) - Aged 45 years (rate) | Sex by single year of age (Census TS009) - Male Aged 68 year |
| 45 | 0.891 | Related but not duplicate | Key stage 1 pupils meeting the expected standard in maths -  | Key stage 1 pupils meeting the expected standard in science  |
| 46 | 0.889 | Related but not duplicate | Age by single year (Census TS007) - Aged 42 years | Sex by single year of age (Census TS009) - All persons Aged  |
| 47 | 0.886 | Related but not duplicate | Males age 70 (counts) | Females age 45 (counts) |
| 48 | 0.885 | Related but not duplicate | DEPRECATED Admissions for lower respiratory tract infections | Emergency admissions for lower respiratory tract infections  |
| 49 | 0.884 | Related but not duplicate | Males age group 10 - 14 % | Persons age 14 % |
| 50 | 0.880 | Related but not duplicate | Age by single year (Census TS007) - Aged 32 years | Sex by single year of age (Census TS009) - Female Aged 82 ye |