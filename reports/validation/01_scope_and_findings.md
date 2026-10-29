# V1 — Scope and findings

**Answers:** What would have to be true for this candidate to be a pre-approval PD? What is verified in the files vs assumed?

**Does not answer:** Holdout performance. No test labels were used to write this register.

Evidence: assignment brief; `reports/investigation/`; `config/features.json`; selected logistic `C=0.1`.

Working use case (**assumption**): score an existing customer before a new credit decision, using information that would be known at scoring. The employer did not confirm this use.

---

## Findings register

All items below are **open**. Favourable development log loss does not close them.

| ID | Severity | Finding | Status |
| --- | --- | --- | --- |
| VAL-01 | High | Target event, coding, horizon, and outcome maturity are unverified | Open |
| VAL-02 | High | As-of availability of the 14 inputs is assumed, not proven | Open |
| VAL-03 | High | `interest_rate` is in the model; quote timing and pricing leakage are unverified | Open |
| VAL-04 | High | DTI formula and factor-ten loan/income/DTI rows are unresolved | Open |
| VAL-05 | High | Population, row unit, borrower IDs, and dates are missing | Open |
| VAL-06 | Medium | Recovery flag excluded; residual timing risk if a similar field were added later | Open (exclusion recorded) |
| VAL-07 | Medium | Income 30,000 concentration and block-2 missingness | Open |
| VAL-08 | Medium | Age 99 and employment vs age flags | Open |
| VAL-09 | Medium | Delinquency count order and scope | Open |
| VAL-10 | Medium | Same person developed and is reviewing — not organizational independence | Open |

---

## VAL-01 — Target contract

The brief says only “actual observed default outcome.” Values are 0/1. Treating 1 as default is a **working convention**.

There is no confirmed 90+ DPD definition, 12-month window, censoring rule, or scoring date $t_0$.

**Impact:** $\hat{p}$ is $P(\text{label in the file}=1 \mid X)$, not a booked horizon PD.

**Closure:** Source dictionary for event, $H$, observation window, and incomplete outcomes.

---

## VAL-02 — Feature timing (all 14)

No snapshot timestamps. Names sound like customer and bureau-style fields; that does not prove they were extracted at $t_0$ or outside the outcome window.

History fields (`past_due_*`, `past_default`) are the highest timing concern after recovery (which is already out). `past_due_90` could overlap a 90+ default definition **if** that were the target — the target itself is unconfirmed (VAL-01).

**Impact:** CV skill can be leakage, not origination skill.

**Closure:** As-of lineage for every retained column.

---

## VAL-03 — `interest_rate` (full write-up)

**What we did in development.** Included the field. Observed range about **0.03 to 0.24**. Units are not documented (could be a decimal annual rate; not verified).

**Business meaning.** Price of the loan. In live origination, the quoted rate is often **set using** the risk view (risk-based pricing). Then rate **contains the previous PD decision**. Using it to predict default looks strong and can still be unusable **before** terms are fixed (pre-approval).

**Tiny example.** Policy: riskier customers get 18% instead of 10%. A model that sees 18% “learns” default. At application time the rate is not chosen yet, so that input would be missing or circular.

**What CV does not prove.** Lower log loss when rate is included does not prove the field is known at $t_0$ or independent of the decision.

**Validation stance.** Keep the current candidate as a **labelled scenario**: “PD-style score **conditional on observing the supplied rate**.” Do not describe it as a verified application-time PD. After holdout (if approved), interpret any rate coefficient with this caveat. Do not drop the field on the test set to “fix” leakage.

**Closure:** Bank confirms when rate is known, whether pricing uses this (or another) risk score, and units.

---

## VAL-04 — DTI

For recorded income, `loan_amount ≈ income × dti` on **19,227** rows and a factor of about **10** on **73** rows. 700 missing-income rows cannot be checked that way.

**Impact:** Affordability may be wrong for some rows; explanations that credit DTI are unstable if units are wrong.

**Closure:** Numerator, denominator, dates, missing-income rule; disposition of the 73 rows. No silent divide-by-ten.

---

## VAL-05 — Population and dependence

20,000 rows; index labels repeat (7,000 extra index duplicates) and form two file blocks (13,000 + 7,000). Blocks are **not** dates. No customer ID.

The 16,000/4,000 split is physically disjoint rows, mixed blocks, same default rate (~15.73%). It is **not** borrower-disjoint or out-of-time.

**Impact:** Row bootstrap and holdout can overstate generalization.

**Closure:** Entity keys, timestamps, sampling frame (booked vs through-the-door).

---

## VAL-06 — `recovery_flag_after_90d`

**Not a model input.** Brief: indicator that the client entered recovery **after** becoming 90+ DPD.

That sequence is typically **after** serious arrears — often the same story as default, not a fact known at pre-approval.

**Tiny example.** Using “already in recovery” to predict default is telling the model the ending.

**Why document it anyway.** So nobody “adds it back” because it predicts well. **3,148** rows have the flag = 1 in the raw file.

**Closure:** If a **prior**, separately dated recovery episode exists, a **new** candidate could consider it. This version stays without the flag.

---

## VAL-07, VAL-08, VAL-09 — Quality flags, not deletions

| Flag | Count | Stance |
| --- | --- | --- |
| Income = 30,000 | 12,505 of recorded incomes | Preserve; possible cap |
| Income missing (all block 2) | 700 | Median + indicator; pattern may be a source process |
| Utilization missing | 17 | Same |
| Age = 99 (all block 2) | 120 | Preserve; sentinel vs real age unknown |
| Age − employment years < 14 | 970 | Illustrative check, not a legal rule |
| 60+ count > 30+ | 2,558 | Only a contradiction if buckets are nested |
| 90+ count > 60+ | 797 | Predeclare as a **segment** in the holdout protocol |
| Delinquency count > `num_loans` | 773 | Scopes may differ |

Do not drop these rows to lift AUC.

---

## VAL-10 — Independence

This review is a structured challenge by the same person who trained the model. A reserved test does not create a second team.

**Interview line:** “I performed a critical review with a locked holdout protocol. That is not the bank’s independent validation function.”

---

## Production implication of V1

Production use as a horizon PD is **not established**. Completing V1 does not approve the model. Holdout metrics, when they exist, still cannot close VAL-01–VAL-05 without source evidence.
