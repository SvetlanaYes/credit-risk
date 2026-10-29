# V2 — Holdout protocol (locked before any test score)

**Answers:** What we will compute **once** on the 4,000 reserved rows, and what we refuse to do.

**Does not answer:** The numeric results. Those exist only after this protocol is approved and implementation checks pass.

**Approval:** unread / **not approved yet**. Do not score the test until this line is changed to approved and V3 has passed its gate.

---

## Candidate and data

- Frozen pipeline: `models/selected/selected_pipeline.joblib` (logistic `C=0.1`, original loan amount, 14 inputs).
- Membership: `data/splits/partition_assignments.csv` (`partition = test`, 4,000 rows, no `cv_fold`).
- Expected events at the same mixture as development: 629 labelled 1 if membership is unchanged.
- Scores will be saved once and **reused** for later segment and cutoff tables. No second scoring to try another model.

---

## Gate before scoring

1. V1 findings remain open; we do not “wait until they close.”
2. V3 implementation (still to be coded): confirm `C=0.1`, 14 column names, development-row probability reconstruction. If core identity fails, **stop**; do not score the test.
3. No change to features, hyperparameters, or preprocessing after looking at test labels (there must be no look).

---

## What we will compute (once)

Compare the frozen model to a **constant** probability equal to the **development** positive rate (~0.1573), using only test rows.

| Metric | Purpose |
| --- | --- |
| ROC AUC | Ranking |
| Average precision | Ranking with ~16% prevalence |
| Log loss | Probability quality (primary analogue of the selection metric) |
| Brier score | Probability MSE |
| Mean $\hat{p}$ vs observed rate | One-number calibration |
| Reliability in **development-defined** score bands | Band calibration; not retuned on test |

Optional uncertainty: row bootstrap **conditional on the frozen scores**, with the VAL-05 independence caveat stated.

Diagnostic confusion counts at 0.5 may be printed. **0.5 is not a lending policy.**

---

## What we refuse

- Choose another `C`, family, or feature using test metrics
- Fit a calibrator on the test
- Search an “optimal” cutoff on the test (F1, Youden, profit)
- Drop age 99, factor-ten DTI rows, or the 90>60 group to raise AUC
- Remove `interest_rate` after seeing test results
- Call the result a 12-month PD or out-of-time validation
- Close VAL-01–VAL-05 because AUC looks similar to development

---

## Cutoff illustrations (only after overall metrics)

Cutoffs taken from **development OOF** score quantiles (e.g. top 10%, 20%, 30%), then applied to **saved** test scores.

Report: flag rate, defaults captured, non-defaults flagged, event rate below the cutoff.

This is a **workload vs capture** picture. It is not a profit-maximising policy (costs, LGD, exposure unknown).

---

## Predeclared segments (V5)

Defined **now**, before test scores:

1. Overall test set  
2. File block 1 vs block 2  
3. Missing income vs observed income  
4. `past_due_90 > past_due_60` vs complement  
5. Age = 99 (likely too small for stable AUC; report counts/events and **withhold ranking** if either class is below 20)

Withhold ranking for other groups below 20 events or 20 non-events. Intervals, if computed, are pointwise, not multiplicity-adjusted. VAL-03: do not treat a strong `interest_rate` effect in a segment as proof of a valid application-time input.

---

## Explanations (V6)

Use **development** rows and the fitted logistic equation (exact log-odds). Do not require SHAP. Interpret any `interest_rate` coefficient under VAL-03.

---

## After the one scoring

Write a short conclusion: useful discrimination of the **supplied** label on this mixture, or not, **with** V1 restrictions. Production suitability remains unestablished until High findings close or are formally accepted by the bank.

---

## How to approve

Replace the approval line at the top with the date and “approved.” Only then add `src` scoring code and run it **once**.
