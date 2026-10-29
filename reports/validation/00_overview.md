# Validation overview

**Status:** documents first. The reserved test has **not** been scored.  
**Holdout lock:** `final_test_scored: false` in `models/selected/selection_record.json`.

Read this file, then [01_scope_and_findings.md](01_scope_and_findings.md), then [02_holdout_protocol.md](02_holdout_protocol.md). Do not implement scoring until the protocol is approved.

## Candidate under review

| Item | Recorded state |
| --- | --- |
| Family | L2 logistic regression |
| Penalty | `C = 0.1` |
| Loan amount | original scale (not `log1p`) |
| Artifact | `models/selected/selected_pipeline.joblib` |
| Inputs | 14 fields including `interest_rate` |
| Excluded | `recovery_flag_after_90d`, `default` |
| Development evidence | mean fold log loss 0.38923; OOF AUC 0.717; OOF AP 0.374; OOF Brier 0.118 |
| Test | 4,000 rows reserved; **unused** |

Development OOF is **selection evidence**. It is not independent confirmation and not a 12-month PD.

## What probability is in scope

What was fitted:

$$
\hat{p}(x) = \widehat{P}(Y_{\mathrm{supplied}} = 1 \mid X = x)
$$

What a production origination PD would require:

$$
P\big(Y_{(t_0,\, t_0+H]} = 1 \mid X(t_0),\; \text{eligible population}\big)
$$

The brief does not give $t_0$, $H$, or a verified event. Validation must not invent a 12-month or 90+ DPD target.

## Layers and the question each step answers

| Step | When | Question it answers | Forbidden until later |
| --- | --- | --- | --- |
| V1 Scope and findings | **Now** | What must be true for a pre-approval PD? What is assumption vs fact? | Closing High findings with CV AUC |
| V2 Holdout protocol | **Now** | What will we compute **once** on test, and what will we refuse? | Any test `predict` |
| V3 Implementation | After you approve V2 | Does the joblib match `C=0.1` and the 14 columns? | Patching the model to “pass” |
| V4 Holdout scoring | After V3 gate | Ranking and probability quality of the **supplied** label on this mixture | Retuning, extra features, peeking to choose a cutoff |
| V5 Segments | After V4, reuse saved scores | Where the average can hide failure | Redefining groups to look better |
| V6 Explanations | Development rows | Why this logistic equation produced $\hat{p}$ | Causal stories; SHAP not required |
| V7 Cutoff illustrations | After V4 | Workload vs capture | Cost-optimal policy without costs |
| V8 Monitoring spec and conclusion | Last | What we would watch if used later; production yes/no | Claiming a live monitoring service |

## Forbidden in this docs-first slice

- Load test rows into `predict` / `predict_proba`
- Change features, `C`, or preprocessing using any test outcome
- Drop age 99 or “fix” DTI to improve scores
- Call this organizationally independent validation (same person as development)
- Call development OOF a holdout result

## Sign-off gate

1. Review V1 findings (especially `interest_rate` and recovery).
2. Review V2 protocol; confirm the metric list and predeclared segments.
3. Only then write implementation checks and score the test **once**.
