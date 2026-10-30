# Linear validation report

**Production decision: not approved.** This is a structured challenge by the same person who trained the model. It is not the bank’s independent validation function.

---

## 0. How to read this

You will meet the file, the data-quality flags, the split, the development winner, then validation steps V1 through V8 in the order they happened. Each chapter starts where the last one stopped. Terms are defined the first time they appear. A later chapter may repeat a short reminder in place; you do not need another file to understand a metric.

The candidate under review is an L2 logistic regression with penalty `C = 0.1`, original (not log) loan amount, fourteen inputs including `interest_rate`, and `recovery_flag_after_90d` kept out. Frozen file: `models/selected/selected_pipeline.joblib`.

---

## 1. What we were given

The assignment is a pickle of 20,000 rows and 16 columns. The brief calls the 0/1 column `default` “the actual observed default outcome.” Treating 1 as default is a **working convention**, not a legal definition.

**Row** means one line in the table. We score each row as one case. We do not know whether a row is one person, one loan, or a mix.

**Label** (also **target**) is that 0/1 column. An **event** is a row with label 1. In the full file there are 3,146 events, so the **observed rate** (share of ones) is 15.73 percent.

Fourteen columns are allowed into the model:

`client_history_in_bank`, `age`, `income`, `employment_years`, `loan_amount`, `interest_rate`, `num_loans`, `credit_utilization`, `past_due_30`, `past_due_60`, `past_due_90`, `past_default`, `dti`, `region_risk`.

`recovery_flag_after_90d` is in the pickle and is **not** a model input. The target is not an input either.

The brief does not give a scoring date (call it t0), a horizon H such as twelve months, or a confirmed 90-plus days-past-due rule. So the model can only estimate the probability of the **supplied file label**, given the supplied columns.

What was fitted:

$$
\hat{p}(x)=\widehat{P}(Y_{\mathrm{file}}=1\mid X=x)
$$

This is “among rows that look like x in this file, what share had label 1.”

What a production origination probability of default would need:

$$
P\big(Y_{(t_0,\,t_0+H]}=1 \mid X(t_0),\; \text{eligible population}\big)
$$

That second object is not identified from the pickle alone. We do not invent a twelve-month or 90-plus target to fill the gap.

**Working use case (assumption, not confirmed by the employer):** score an existing customer before a new credit decision, using information that would be known at scoring.

Raw file SHA-256: `9d69a6155fec0e9b006a4c09594f676cd4ae57a4af4ca588f2c8bd32868ca11e`. Values were not edited. Rows were not dropped.

The next chapter only **describes** the 20,000 rows. It does not split or train.

---

## 2. Raw-file investigation (before any split)

Question: what is actually in the table, and which quality flags should stay visible?

How: count and compare columns. No imputation, no deletions to make later AUC prettier.

### Two file pieces, not dates

The pandas index is not unique: 7,000 extra repeated index labels. When the original index jumps backward, we mark an **audit block**. Block 1 has 13,000 rows; block 2 has 7,000. These blocks are **not** calendar periods. There is no customer ID.

### Income, age, utilization

- Income missing: 700 rows (all in block 2).
- Income exactly 30,000: 12,505 of the recorded incomes (possible cap; not proven).
- Credit utilization missing: 17 rows.
- Age exactly 99: 120 rows (all in block 2). Sentinel versus real age is unknown.
- Age minus employment years below 14: 970 rows. An illustrative check, not a legal working-age rule.

### DTI versus loan and income

**DTI** is usually “debt relative to income.” If `dti` in this file is loan over income, then for rows with income filled in:

$$
\frac{\mathrm{loan\ amount}}{\mathrm{income}\times\mathrm{dti}}\approx 1
$$

Tiny example: income 2,000 and DTI 0.5 should go with loan 1,000 (ratio 1). Loan 10,000 with the same income and DTI gives ratio 10: units or a keying error.

How we counted: only rows with income present; ratio within 0.05 of 1, or within 0.5 of 10.

Result: **19,227** near 1; **73** near 10; 700 missing-income rows cannot be checked.

### Past-due columns

`past_due_30`, `past_due_60`, and `past_due_90` are small integers (`int64`), not dates and not a continuous “days late.” No missing values. Values that appear:

- `past_due_30`: 0 to 4 (most rows 0).
- `past_due_60`: 0 to 4 (most rows 0).
- `past_due_90`: 0, 1, or 2 (about 95 percent are 0).

`past_default` is 0/1 (2,049 ones). It is a separate column.

Names suggest counts of 30-plus / 60-plus / 90-plus delinquencies. If those buckets were nested, you would expect 90-count ≤ 60-count ≤ 30-count. The file does not always obey that.

How we counted: `(past_due_60 > past_due_30)` and `(past_due_90 > past_due_60)`.

Result: **2,558** rows with 60 greater than 30; **797** with 90 greater than 60. Also **773** rows where the largest of the three past-due fields is greater than `num_loans` (scopes may differ).

These are flags, not automatic deletions.

### Interest rate and recovery

`interest_rate` runs from about 0.03 to 0.24. Units are unverified (could be a decimal annual rate).

`recovery_flag_after_90d` equals 1 on **3,148** rows. The brief: the client entered recovery **after** becoming 90-plus days past due. That is typically after serious arrears, often the same story as default, not a fact known at pre-approval. It stays out of the model so nobody “adds it back” because it predicts well.

The next chapter freezes who is development and who is test, without using these flags to hunt a convenient split.

---

## 3. Split

Question: which 16,000 rows may we use to choose a model, and which 4,000 must wait?

How: 80/20 by **physical row**, stratified by target crossed with audit block, so both pieces keep a similar mix of defaults and of the two file blocks. Holdout seed 42. Five development **folds** (pieces of the 16,000) use seed 43.

A **fold** is one of those five pieces, about 3,200 rows. Later, training will hold one fold out at a time.


| Partition   | Rows   | Events | Event rate | Block 1 | Block 2 | Income missing |
| ----------- | ------ | ------ | ---------- | ------- | ------- | -------------- |
| Full        | 20,000 | 3,146  | 0.1573     | 13,000  | 7,000   | 700            |
| Development | 16,000 | 2,517  | 0.1573     | 10,400  | 5,600   | 563            |
| Test        | 4,000  | 629    | 0.1573     | 2,600   | 1,400   | 137            |


Read the test line: same default rate as development, mixed blocks, physically disjoint rows. It is **not** borrower-disjoint and **not** out-of-time (a later calendar period). Generalization to a new book can be overstated (this becomes finding VAL-05).

Membership lives in `data/splits/partition_assignments.csv`. Membership SHA-256:

`1603d6e29c5de95d299ae6ac0f7b712d62e8cce9a11d1bbff1a4c11e9dbbd8ef`

Until a later chapter named V4, **nobody scores the 4,000 test rows**.

The next chapter uses only development folds to pick a winner.

---

## 4. Development training and OOF (selection, not validation)

Question: which scoring recipe is best on development data?

This chapter does **not** answer holdout quality, leakage, or production PD.

### Shared recipe for every candidate

Inside each fold’s **training** rows only we:

1. take the fourteen columns;
2. fill numeric gaps with the **median** and add **missingness indicators**;
3. scale numeric fields with **StandardScaler** (mean 0, variance 1 on that fold’s train);
4. fill `region_risk` with the most frequent value and **one-hot** encode it.

`interest_rate` is in. Recovery is out. We do not use class weights, resampling, a calibrator, or a lending cutoff search.

Tiny example of scaling: if DTI on the fold-train has mean 0.4 and standard deviation 0.2, a raw DTI of 0.6 becomes (0.6 − 0.4) / 0.2 = 1.0 after scaling.

### Three families

**Logistic regression** turns a weighted sum of the transformed fields into a probability with a sigmoid:

$$
z=\beta_0+\sum_j \beta_j \tilde{x}_j
$$

$$
\hat{p}=\frac{1}{1+e^{-z}}
$$

Larger z means higher predicted probability of label 1. **L2** shrinks coefficients toward 0. **C** is the inverse penalty: small C means stronger shrink. Grid: C in {0.001, 0.01, 0.1, 1, 10, 100}, and loan amount raw versus log1p.

**Random forest:** 250 trees; each tree sees a bootstrap of rows and 80 percent of features at a split; we vary minimum leaf size.

**Histogram gradient boosting:** 200 iterations, no early stopping, we vary maximum leaf nodes.

### What OOF means

**OOF (out-of-fold)** is a predicted probability for a development row from a model that **did not see that row while fitting**.

How: for fold 1, fit on folds 2–5, predict fold 1; repeat for folds 2, 3, 4, 5. Each of the 16,000 rows gets exactly one OOF score.

If we instead fit on all 16,000 and score those same 16,000, that is **in-sample**, not OOF: the model has already seen the row.

### How we chose the winner

**Log loss** is a penalty for a wrong probability, especially a confident one. If the label is 1, the penalty is minus log of p-hat. If the label is 0, it is minus log of (1 − p-hat). Average over the n rows. Lower is better.

$$
\mathrm{log\ loss}=-\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\Big]
$$

This is the average lie penalty when the label is 0 or 1.

Tiny example: true 1 and p-hat 0.8 → about 0.22. True 1 and p-hat 0.1 → about 2.3.

**Mean fold log loss** is the average of this quantity on each of the five folds. That was the **selection** rule.

We also record three extras on **pooled OOF** scores (all 16,000 OOF probabilities at once). They did **not** choose the winner.

**Brier** is mean squared error of p-hat versus 0/1. Lower is better.

$$
\mathrm{Brier}=\frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i-y_i)^2
$$

This is ordinary exam-style error for a probability.

**ROC AUC** (this is “AUC” here) is ranking: with n1 defaults and n0 non-defaults, look at every (default, non-default) pair. Count 1 if the default has the higher p-hat, 0.5 if they tie, 0 if the non-default is higher. Then average.

$$
\mathrm{ROC\ AUC}=\frac{1}{n_1 n_0}\sum_{i:y_i=1}\sum_{j:y_j=0}\Big[\mathbf{1}(\hat{p}_i>\hat{p}_j)+\tfrac{1}{2}\mathbf{1}(\hat{p}_i=\hat{p}_j)\Big]
$$

This is the share of such pairs where the default scores above the non-default. 1 is perfect order; 0.5 is chance.

**Average precision (AP)** walks the list from largest p-hat to smallest. At rank k, P(k) is precision among the top k. Delta R(k) is the jump in recall at k (usually 1/n1 if that row is a default, else 0).

$$
\mathrm{AP}=\sum_k P(k)\,\Delta R(k)
$$

This is sklearn’s `average_precision_score` (with its usual interpolation). A random ranking sits near the event rate (here about 0.157).

The holdout chapter repeats these four ideas with **test** numbers. Chapter 4 only uses them on **development OOF**.

Family winners:


| Family         | Config                        | Mean fold log loss | OOF ROC AUC | OOF AP | OOF Brier |
| -------------- | ----------------------------- | ------------------ | ----------- | ------ | --------- |
| Logistic       | `C=0.1`, original loan amount | 0.38923            | 0.7167      | 0.3741 | 0.1180    |
| Hist. boosting | 7 leaves                      | 0.39274            | 0.7100      | 0.3595 | 0.1192    |
| Random forest  | leaf 50                       | 0.39651            | 0.7017      | 0.3485 | 0.1203    |


Winner: logistic `C=0.1`, original loan amount.

Then we **refit that spec on all 16,000** development rows and save `selected_pipeline.joblib`. Those 16,000 have now been seen by the frozen model. Scores of the frozen model on development are **not** OOF.

### Development score bands (frozen model, before validation)

After the freeze we still stay on development. We sort the 16,000 frozen p-hat values and cut at the 10th, 20th, …, 90th percentiles (plus 0 and 1). Ten bands. Because the edges come from **these same** scores, each band has **exactly 1,600** rows. (On test later, the same edges need not give equal n.)

Limitation: this is **in-sample**. The model was fit on these rows, so bands can look a little too neat. They are a baseline to carry into holdout, not independent confirmation.


| Band (p-hat) | n    | Mean p-hat | Observed rate | Events |
| ------------ | ---- | ---------- | ------------- | ------ |
| up to 0.061  | 1600 | 0.054      | 0.051         | 82     |
| 0.061–0.073  | 1600 | 0.067      | 0.064         | 103    |
| 0.073–0.085  | 1600 | 0.079      | 0.077         | 123    |
| 0.085–0.098  | 1600 | 0.091      | 0.101         | 162    |
| 0.098–0.114  | 1600 | 0.106      | 0.097         | 155    |
| 0.114–0.135  | 1600 | 0.124      | 0.136         | 218    |
| 0.135–0.166  | 1600 | 0.149      | 0.148         | 237    |
| 0.166–0.220  | 1600 | 0.191      | 0.185         | 296    |
| 0.220–0.318  | 1600 | 0.261      | 0.258         | 413    |
| 0.318–1      | 1600 | 0.451      | 0.455         | 728    |


Read the first row: the safest tenth has mean p-hat 0.054 and observed default rate 0.051. Read the last row: the riskiest tenth has mean p-hat 0.451 and observed rate 0.455. Mean p-hat and observed rate travel together on development. 

### Groups we will examine later 

The investigation flags are not closed by this table. In validation we will reuse **saved** scores and look at these **predeclared** slices (named before any test `predict`):

1. Overall reserved test (once the protocol allows it).
2. Audit block 1 versus block 2 (two file pieces, not dates).
3. Income missing versus income observed.
4. `past_due_90 > past_due_60` versus the rest (the nested-count oddity).
5. Age = 99 (likely too small to rank; we will withhold AUC if either class has fewer than 20 rows).

We will not drop those rows here to make development bands look cleaner. We will not hunt extra slices after seeing test AUC.

---

## 5. V1 — Scope and findings

Question: what is verified in the files versus assumed? What must close before anyone calls this an origination PD?

All ten items stay **open**. A good development log loss does not close them.


| ID     | Severity | Finding                                                                          | Status                    |
| ------ | -------- | -------------------------------------------------------------------------------- | ------------------------- |
| VAL-01 | High     | Target event, coding, horizon, and outcome maturity are unverified               | Open                      |
| VAL-02 | High     | As-of availability of the 14 inputs is assumed, not proven                       | Open                      |
| VAL-03 | High     | `interest_rate` is in the model; quote timing and pricing leakage are unverified | Open                      |
| VAL-04 | High     | DTI formula and factor-ten loan/income/DTI rows are unresolved                   | Open                      |
| VAL-05 | High     | Population, row unit, borrower IDs, and dates are missing                        | Open                      |
| VAL-06 | Medium   | Recovery flag excluded; residual timing risk if a similar field were added later | Open (exclusion recorded) |
| VAL-07 | Medium   | Income 30,000 concentration and block-2 missingness                              | Open                      |
| VAL-08 | Medium   | Age 99 and employment versus age flags                                           | Open                      |
| VAL-09 | Medium   | Delinquency count order and scope                                                | Open                      |
| VAL-10 | Medium   | Same person developed and is reviewing — not organizational independence         | Open                      |


**VAL-01.** No confirmed 90-plus definition, twelve-month window, censoring, or t0. Impact: p-hat is P(label in the file = 1 given X), not a booked horizon PD.

**VAL-02.** No snapshot timestamps. Names sound like customer or bureau fields; that does not prove they were known at t0 or outside the outcome window. `past_due_`* and `past_default` are the highest timing concern after recovery. `past_due_90` could overlap a 90-plus default definition **if** that were the target — the target itself is unconfirmed.

**VAL-03.** Rate is in the model (range about 0.03 to 0.24). In live origination the quoted rate is often set using a risk view, so rate can **contain a previous PD decision**. Tiny example: policy gives riskier customers 18 percent instead of 10 percent; a model that sees 18 percent “learns” default; at application time the rate may not exist yet. Stance: this candidate is a score **conditional on observing the supplied rate**, not a verified application-time PD. We will not drop the field on the test set later to “fix” leakage.

**VAL-04.** The DTI identity from the investigation chapter: 19,227 near 1, 73 near 10. Affordability and any story that “credits DTI” are unstable if units are wrong on those 73. No silent divide-by-ten.

**VAL-05.** 20,000 rows, two blocks, no IDs, no dates. The 16,000/4,000 split is physically disjoint rows, mixed blocks, same default rate. It is not borrower-disjoint or out-of-time.

**VAL-06.** Recovery stays out. Tiny example: using “already in recovery” to predict default is telling the model the ending. 3,148 rows have the flag = 1. If a **prior**, separately dated recovery episode exists, a **new** candidate could consider it. This version does not.

**VAL-07, VAL-08, VAL-09.** The investigation flags (income 30,000, missing income, age 99, 60>30, 90>60, delinquency versus `num_loans`) stay in the data. We do not drop those rows to lift AUC. The 90>60 pattern will be named as a holdout **slice** in the next chapter.

**VAL-10.** A reserved test does not create a second team. Interview line: a critical review with a locked protocol is not the bank’s independent validation function.

Completing this register does not approve the model. Holdout numbers, when they exist, still cannot close VAL-01 through VAL-05 without source evidence.

The 4,000 rows are still unused. Next we lock what we will compute on them, and what we refuse.

---

## 6. V2 — Holdout protocol (locked before any test score)

Question: what will we compute **once** on the 4,000 reserved rows, and what will we refuse?

The protocol was locked; the test was then scored once. Findings stay open; we do not wait for them to close.

**Compare** the frozen model to a **constant** probability equal to the development positive rate (about 0.1573), using only test rows.

Promised skill checks (defined in the holdout chapter): ROC AUC, average precision, log loss, Brier, mean p-hat versus observed rate, reliability in development-defined score bands.

Promised later, still using **saved** test scores: segments named below; cutoff illustrations from development OOF tails; no profit search.

**Predeclared segments** (groups named now, before test scores):

1. Overall test set.
2. Audit block 1 versus block 2.
3. Missing income versus observed income.
4. `past_due_90 > past_due_60` versus the rest.
5. Age = 99 (likely too small for stable ranking; withhold ranking if either class is below 20).

**Refuse:** choose another C, family, or feature using test metrics; fit a calibrator on the test; search an “optimal” cutoff on the test (F1, Youden, profit); drop age 99, factor-ten DTI rows, or the 90>60 group to raise AUC; remove `interest_rate` after seeing test results; call the result a twelve-month PD or out-of-time validation; close VAL-01–VAL-05 because AUC looks similar to development.

**Gate:** implementation checks on development must pass before any test `predict`. If identity fails, stop.

The next chapter is that gate. Test still unused.

---

## 7. V3 — Implementation checks

Question: is the joblib the logistic we think it is?

How: load `selected_pipeline.joblib`. Use **development rows only**. Do not score test.


| Check                                 | How                                                                                                | Result                   |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------ |
| `C_equals_0.1`                        | Read `model.C`                                                                                     | 0.1, pass                |
| `estimator_is_logistic`               | Class name                                                                                         | LogisticRegression, pass |
| `fourteen_model_features`             | Columns except target and recovery, compared to the agreed list of 14                              | Pass                     |
| `recovery_not_a_feature`              | Recovery must not be in the feature list (it may still exist in the pickle)                        | Pass                     |
| Finite probabilities                  | `predict_proba` on 16,000 development rows                                                         | Pass                     |
| Probabilities in the unit interval    | Every p-hat in [0, 1]                                                                              | Pass                     |
| Hand logistic matches `predict_proba` | After preprocess, compute sigmoid of X beta plus intercept; max absolute difference versus sklearn | 0.0, pass                |


The reconstruction is:

$$
\hat{p}=\sigma(X_{\mathrm{transformed}}\beta+\beta_0)
$$

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Side observation (not the gate): on development, mean p-hat 0.1573 versus observed rate 0.1573. A fitted intercept often sits on the training average.

**Core identity: pass. Test rows scored: 0.** This is not a skill test. It does not speak to leakage or to VAL-01.

The next chapter scores the 4,000 rows **once**.

---

## 8. V4 — Reserved-test assessment (one scoring)

The joblib matched. We now score the 4,000 test rows once and freeze those probabilities. No retune.

**Constant baseline:** ignore all features; always output the development default rate 0.1573. Question: did using X beat “quote the average”?

**Counts:** 4,000 rows, 629 events, observed rate 0.1573. Mean predicted probability of the frozen model: 0.1596.

When people say **AUC** here, they mean **ROC AUC**.

### Ranking: ROC AUC

**Ranking** means: if you sort people from high p-hat to low p-hat, do the ones tend to sit above the zeros? You can rank well even if every score is too high, as long as the **order** is right.

Tiny example: labels 0, 0, 1, 1 with scores 0.05, 0.10, 0.25, 0.40. Order is perfect. A second model that adds 0.40 to every score has the **same ranking** and worse calibration.

**ROC AUC** in words: pick a random person with label 1 and a random person with label 0. How often does the 1 have the higher score? 1.00 is perfect order; 0.50 is a coin (the constant, everyone tied); 0.00 is reversed.

How we computed it: `sklearn.metrics.roc_auc_score` on test labels and the frozen probabilities.

**Holdout ROC AUC: 0.7183** versus **0.5000** for the constant. Development OOF was 0.7167. About 72 times in 100, a random labelled default scores above a random non-default. Modest ranking, stable versus OOF, not a star PD, not proof of a twelve-month event.

### Ranking that cares about the top: average precision

**Precision** at a cutoff: of the people you flagged, what share were truly 1. **Recall** (capture): of all true 1s, what share you flagged.

**Average precision** walks the ranked list from riskiest to safest and averages precision along that path. A mindless ranking gets AP about equal to the prevalence (here about 0.157). That is the floor.

How: `sklearn.metrics.average_precision_score`.

**Holdout AP: 0.3912** versus **0.1573** for the constant. The top of the list is richer in defaults than a random list. This is still not a single operating policy.

### Probability quality: log loss

Same penalty as in selection, now on test.

$$
\mathrm{log\ loss}=-\frac{1}{n}\sum_i\Big[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\Big]
$$

This is the average lie penalty when the label is 0 or 1. Lower is better. Cocky wrong probabilities are expensive.

How: `sklearn.metrics.log_loss` with labels {0, 1}.

**Holdout log loss: 0.3865** versus **0.4351** for the constant. The fourteen fields beat always saying 15.7 percent. This is a statistical penalty, not a dollar loss.

### Probability quality: Brier score

$$
\mathrm{Brier}=\frac{1}{n}\sum_i(\hat{p}_i-y_i)^2
$$

This is mean squared error of p-hat versus 0/1. Lower is better. Less explosive than log loss when p-hat is near 0 or 1.

Tiny example: truth 1 and p-hat 0.7 → error 0.09. Truth 0 and p-hat 0.7 → error 0.49.

How: `sklearn.metrics.brier_score_loss`.

**Holdout Brier: 0.1168** versus **0.1325** for the constant.

### One-number calibration: mean p-hat versus observed rate

Mean p-hat **0.1596** versus observed **0.1573**. Overall we are almost unbiased on this mixture. The constant matches the observed rate by construction. This does not prove honesty **inside** groups.

### Band calibration: reliability table

A **band** is a bucket of similar scores. We took quantile edges from **development scores of the frozen model** (those people helped fit the model: in-sample edges, a limitation), then applied the same edges to **test** so we did not redraw buckets after seeing test labels.

How: ten quantiles, cut test p-hat, in each bucket report n, mean p-hat, observed rate, events.


| Band (test p-hat) | n   | Mean p-hat | Observed rate | Events |
| ----------------- | --- | ---------- | ------------- | ------ |
| up to 0.061       | 353 | 0.054      | 0.048         | 17     |
| 0.061–0.073       | 398 | 0.067      | 0.070         | 28     |
| 0.073–0.085       | 420 | 0.079      | 0.079         | 33     |
| 0.085–0.098       | 400 | 0.092      | 0.103         | 41     |
| 0.098–0.114       | 430 | 0.106      | 0.102         | 44     |
| 0.114–0.135       | 386 | 0.124      | 0.119         | 46     |
| 0.135–0.166       | 405 | 0.150      | 0.141         | 57     |
| 0.166–0.220       | 381 | 0.190      | 0.192         | 73     |
| 0.220–0.318       | 447 | 0.264      | 0.248         | 111    |
| 0.318–1           | 380 | 0.470      | 0.471         | 179    |


Read the last row: among people the model called quite risky, about 47 in 100 were labelled 1, and the average p-hat was also 0.47.

### What V4 did not do

No bootstrap (protocol allowed it; we skipped). No 0.5 lending cutoff (0.5 appears later only as a diagnostic). No second `predict`. Scores saved for later chapters.

V1 findings stay open. AUC 0.718 does not close VAL-01–VAL-05.

The next chapter cuts these **same** saved scores into the groups named in the protocol.

---

## 9. V5 — Predeclared segments

We already have one p-hat per test row. Question: does overall AUC 0.72 hide a broken pocket?

How: filter the saved scores; recompute n, events, observed rate, mean p-hat, and — if both classes have at least 20 rows — ROC AUC, AP, and log loss. ROC AUC still means: how often a random default in **this slice** scores above a random non-default in **this slice**. No new `predict`. No new groups after seeing numbers.


| Segment                   | n    | Events | Observed | Mean p-hat | AUC    | AP     | Log loss | Note             |
| ------------------------- | ---- | ------ | -------- | ---------- | ------ | ------ | -------- | ---------------- |
| overall test              | 4000 | 629    | 0.1573   | 0.1596     | 0.7183 | 0.3912 | 0.3865   | same as V4       |
| audit block 1             | 2600 | 409    | 0.1573   | 0.1591     | 0.7058 | 0.3871 | 0.3899   |                  |
| audit block 2             | 1400 | 220    | 0.1571   | 0.1607     | 0.7418 | 0.3999 | 0.3801   |                  |
| income missing            | 137  | 30     | 0.2190   | 0.2647     | 0.7312 | 0.5462 | 0.4587   | small            |
| income observed           | 3863 | 599    | 0.1551   | 0.1559     | 0.7177 | 0.3841 | 0.3839   |                  |
| past_due_90 > past_due_60 | 160  | 51     | 0.3187   | 0.2909     | 0.5985 | 0.4773 | 0.6097   |                  |
| complement of that rule   | 3840 | 578    | 0.1505   | 0.1542     | 0.7138 | 0.3817 | 0.3772   |                  |
| age = 99                  | 32   | 2      | 0.0625   | 0.1726     | —      | —      | —        | ranking withheld |
| age not 99                | 3968 | 627    | 0.1580   | 0.1595     | 0.7195 | 0.3938 | 0.3871   |                  |


**Blocks (VAL-05).** Same default rate, different ranking: 0.706 versus 0.742. The headline 0.72 is a blend of two file pieces.

**Missing income (VAL-07).** 137 people, dirtier book (22 percent defaults); the model’s mean p-hat is also higher (0.26). AUC on 30 events is noisy.

**90 greater than 60 (VAL-09).** 160 people, 32 percent defaults, AUC 0.599 (weak ranking), log loss 0.61 (probabilities fit this pocket poorly). We do not delete these 160 and republish 0.72.

**Age 99 (VAL-08).** 32 rows, 2 events: ranking withheld on purpose.

A strong pattern for `interest_rate` in a slice would not close VAL-03.

The next chapter explains the **equation** on development, not a new test score.

---

## 10. V6 — Logistic equation (development)

Question: which transformed fields move log-odds in this fitted surface?

No SHAP. After median impute, missing indicators, scaling, and region one-hot:

$$
\log\frac{\hat{p}}{1-\hat{p}}=\beta_0+\sum_j \beta_j x_j^{\mathrm{transformed}}
$$

This is log-odds equal to intercept plus a weighted sum of transformed inputs.

Intercept beta0 = **−1.7325**. C = 0.1. There are 18 coefficients (scaled numerics, two missingness indicators, three region dummies). All region levels are kept; L2 makes that identifiable. Do not read a dummy as “versus a dropped base region.”

For a scaled numeric field, plus one transformed unit is about plus one development standard deviation after impute. Odds multiplier is exp(beta): holding other **transformed** fields fixed, that move multiplies odds of the **file label**.

Largest absolute coefficients:


| Term                          | Coefficient | Odds per +1 transformed |
| ----------------------------- | ----------- | ----------------------- |
| `numeric__past_default`       | 0.4480      | 1.565                   |
| `numeric__dti`                | 0.4172      | 1.518                   |
| `categorical__region_risk_0`  | −0.3052     | 0.737                   |
| `numeric__past_due_60`        | 0.2974      | 1.346                   |
| `categorical__region_risk_2`  | 0.2506      | 1.285                   |
| `numeric__past_due_30`        | 0.2469      | 1.280                   |
| `numeric__past_due_90`        | 0.2200      | 1.246                   |
| `numeric__credit_utilization` | 0.1208      | 1.128                   |


`numeric__interest_rate` is small: coefficient about **−0.016**. That does not close VAL-03. Leakage is about when the field is known, not only about a large coefficient.

Worked development row `source_row_id = 0`: reconstructed p-hat **0.089565**, matching the pipeline. Associations, not causes. This does not close VAL-01–VAL-05.

The next chapter draws **lines** on the saved test scores using development OOF tails, not a search on test.

---

## 11. V7 — Cutoff illustrations (not a policy)

Question: if we flagged the riskiest tail, what workload and what capture of the **file** label?

**Threshold** means a line such as “flag if p-hat is at least 0.32.” The model does not need a threshold to exist. A threshold is an extra rule.

How the lines were drawn: reconstruct **OOF** for the frozen spec only (logistic C=0.1, original loan amount, same five folds — not a new grid). Take the development OOF scores at the 90th, 80th, and 70th percentiles (riskiest 10, 20, 30 percent). Apply those numbers to **saved** test scores.

**Flag rate:** share of test above the line (workload).  
**Capture rate:** share of the 629 labelled defaults above the line.  
**Non-defaults flagged:** goods in the pile.  
**Event rate among flagged:** default share inside the pile.  
**Event rate below the cutoff:** default share among people **not** flagged.


| Rule                          | Threshold | Flag rate | Defaults captured | Capture rate | Non-defaults flagged | Event rate among flagged | Event rate below |
| ----------------------------- | --------- | --------- | ----------------- | ------------ | -------------------- | ------------------------ | ---------------- |
| Top 10% of development OOF    | 0.317     | 9.55%     | 179               | 28.5%        | 203                  | 0.469                    | 0.124            |
| Top 20%                       | 0.219     | 20.7%     | 290               | 46.1%        | 539                  | 0.350                    | 0.107            |
| Top 30%                       | 0.166     | 30.3%     | 363               | 57.7%        | 847                  | 0.300                    | 0.095            |
| Diagnostic 0.5 (not a policy) | 0.500     | 3.15%     | 84                | 13.4%        | 42                   | 0.667                    | 0.141            |


Read the first row: review about one in ten files, catch about 28 percent of labelled defaults, pile default rate 47 percent versus 16 percent in the book, accepted tail 12.4 percent — cleaner, not zero.

**Why 0.5 is not a credit policy here.** On a 16 percent book almost nobody has p-hat ≥ 0.5 (3 percent of rows). You catch only 13 percent of defaults.

Costs, LGD, and exposure are unknown. There is no profit-maximising cutoff in this pack. Do not pick a line from this table to raise a test metric.

The last chapter turns the same findings into a monitoring **spec** and a production sentence.

---

## 12. V8 — Monitoring spec and conclusion

Question: if a score like this were used later, what would we watch? Do we approve production?

This is a specification, not a live platform. Alert cut-points belong to the bank. Baselines are **development**; do not retune alerts on the holdout.


| Watch                                                | Why                         | Starting baseline                              |
| ---------------------------------------------------- | --------------------------- | ---------------------------------------------- |
| Volume and mix of scored applications                | Population shift (VAL-05)   | Development versus any later book              |
| Missingness of the 14 inputs, especially income      | VAL-07                      | Development missing rates                      |
| Age = 99 rate                                        | VAL-08                      | Development share                              |
| Factor-ten DTI / loan / income breaks                | VAL-04                      | 73 rows in the full file                       |
| `past_due_90 > past_due_60` rate                     | VAL-09                      | Development share                              |
| `interest_rate` distribution and missingness         | VAL-03                      | Development 0.03–0.24 (units unverified)       |
| Mean p-hat versus matured observed rate              | Calibration                 | Development rate about 0.157                   |
| Log loss / Brier / ROC AUC once labels exist         | Skill of the supplied event | Holdout log loss 0.3865, ROC AUC 0.7183        |
| Share flagged at the predeclared OOF top-decile rule | Workload drift              | Test flag rate 9.55 percent at threshold 0.317 |
| Recovery-like fields entering the pipeline           | VAL-06                      | Must stay excluded                             |


Investigate (idea, not bank policy) if missingness or age-99 rate doubles, if mean p-hat and observed rate diverge for a long time, or if a recovery-style column appears.

**What would have to close before production:** dictionary for the event, horizon H, t0, incomplete outcomes (VAL-01); as-of lineage for every input (VAL-02); confirmation that `interest_rate` is known at the intended decision point and is not circular with pricing (VAL-03); DTI formula and the 73 rows (VAL-04); population, row unit, IDs, dates (VAL-05); a reviewer who did not train the model (VAL-10).

**Production: not approved.**

Honest line: useful ranking of the **file label** on a random-row holdout; not a production PD.

Interview close: we selected on development log loss, locked the test, checked the joblib, scored once, and did not retune. Discrimination is real on this mixture and is not a twelve-month PD. SHAP was not required. Cutoffs are illustrations. Monitoring here is a spec, not a platform.

---

## 13. Closing inventory


| Step          | Question                      | What we computed                                            | Result                                        |
| ------------- | ----------------------------- | ----------------------------------------------------------- | --------------------------------------------- |
| Investigation | What is in the file?          | Counts and identities                                       | Flags kept; nothing dropped                   |
| Split         | Who may we train on?          | 16,000 / 4,000, seeds 42/43                                 | Test locked                                   |
| Training      | Which recipe?                 | Mean fold log loss, OOF extras                              | Logistic C=0.1 original loan                  |
| V1            | What is assumed?              | VAL-01–10                                                   | All open                                      |
| V2            | What once on test?            | Protocol                                                    | Approved 2026-10-29                           |
| V3            | Is the joblib that logistic?  | Seven identity checks                                       | Pass; test unused                             |
| V4            | Skill on the file label?      | ROC AUC, AP, log loss, Brier, mean p-hat, bands vs constant | 0.718 / 0.391 / 0.387 / 0.117; bands coherent |
| V5            | Where does 0.72 hide failure? | Same metrics in predeclared slices                          | Weak 90>60 ranking; age 99 withheld           |
| V6            | Why this p-hat?               | Coefficients, one worked row                                | past_default and dti dominate; rate small     |
| V7            | Workload vs capture?          | OOF top 10/20/30 percent on saved scores                    | Illustration only                             |
| V8            | Watch list and go/no-go?      | Spec + decision                                             | Not approved                                  |


