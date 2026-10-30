# V8 — Monitoring spec and conclusion

**Production decision:** **not approved.**  
This pack is a **specification** of what would be watched if this score were used later. It is not a live monitoring service.

All High findings VAL-01–VAL-05 remain **open**. VAL-10: this is not organizationally independent validation.

---

## What was shown

Frozen logistic `C=0.1`, 14 inputs including `interest_rate`, recovery excluded.

| | Frozen model | Constant development rate |
| --- | ---: | ---: |
| Holdout ROC AUC | 0.7183 | 0.5000 |
| Holdout AP | 0.3912 | 0.1573 |
| Holdout log loss | 0.3865 | 0.4351 |
| Holdout Brier | 0.1168 | 0.1325 |

Development OOF AUC was 0.717. Holdout AUC is 0.718 on 4,000 rows / 629 events. Ranking of the **supplied label** on this mixture is modest and stable versus a constant rate. Mean $\hat{p}$ 0.160 vs observed 0.157. Score bands tracked observed rates.

That does **not** make $\hat{p}$ a verified origination PD:

$$
\hat{p}(x)=\widehat{P}(Y_{\mathrm{supplied}}=1\mid X=x)
$$

not

$$
P\big(Y_{(t_0,\,t_0+H]}=1\mid X(t_0),\;\text{eligible population}\big).
$$

Segments: block 2 ranked higher than block 1; `past_due_90>past_due_60` was weak; age 99 ranking withheld. Coefficients: `past_default` and `dti` dominate the scaled log-odds; `interest_rate` is small and still covered by VAL-03.

Example (not a policy): flagging development-OOF top 10% on test (threshold 0.317) flags 9.6% of rows, captures 28.5% of labelled defaults, and leaves event rate 0.124 below the cutoff.

---

## Monitoring we would specify (if used later)

Baselines are **development**. Do not retune alerts on the holdout.

| Watch | Why | Starting baseline |
| --- | --- | --- |
| Volume and mix of scored applications | Population shift (VAL-05) | Development vs any later book |
| Missingness of the 14 inputs, especially income | VAL-07 | Development missing rates |
| Age = 99 rate | VAL-08 | Development share |
| Factor-ten DTI / loan / income breaks | VAL-04 | 73 rows in the full file |
| `past_due_90 > past_due_60` rate | VAL-09 | Development share |
| `interest_rate` distribution and missingness | VAL-03 | Development 0.03–0.24 range (units unverified) |
| Mean $\hat{p}$ vs matured observed rate | Calibration | Development rate ~0.157 |
| Log loss / Brier / AUC once labels exist | Skill of the supplied event | Holdout log loss 0.3865, AUC 0.7183 |
| Share flagged at the **predeclared** OOF top-decile rule | Workload drift | Test flag rate in the V7 table |
| Recovery-like fields entering the pipeline | VAL-06 | Must stay excluded |

Traffic-light idea (not bank policy): investigate if missingness or age-99 rate doubles, if mean $\hat{p}$ and observed rate diverge by a large persistent gap, or if a recovery-style column appears. Exact thresholds belong to the bank.

---

## What would have to close before production

1. Dictionary for the event, horizon $H$, $t_0$, and incomplete outcomes (VAL-01).  
2. As-of lineage for every input, especially delinquency history (VAL-02).  
3. Confirmation that `interest_rate` is known at the intended decision point and is not circular with pricing (VAL-03).  
4. DTI formula and the 73 factor-ten rows (VAL-04).  
5. Population, row unit, IDs, dates (VAL-05).  
6. A reviewer who did not train the model (VAL-10).

Until then the honest line is: **useful ranking of the file label on a random-row holdout; not a production PD.**

---

## Interview close

We selected on development log loss, locked the test, checked the joblib, scored once, and did not retune. Discrimination is real on this mixture and not a 12-month PD. SHAP was not required. Cutoffs are illustrations. Monitoring here is a spec, not a platform.
