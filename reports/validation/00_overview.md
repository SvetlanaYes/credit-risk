# Validation overview

**Status:** V1–V8 complete. Holdout scored once. Production **not** established. No SHAP.  
**Holdout lock:** `final_test_scored: true` in `models/selected/selection_record.json`.

Conclusion: [08_monitoring_and_conclusion.md](08_monitoring_and_conclusion.md).  
Findings register: [01_scope_and_findings.md](01_scope_and_findings.md).  
Locked protocol (what was promised): [02_holdout_protocol.md](02_holdout_protocol.md).  
Step-by-step narrative: [linear_report.md](linear_report.md).

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
| Test | 4,000 rows scored once; AUC 0.718; AP 0.391; log loss 0.387; Brier 0.117 vs constant 0.435 / 0.133 |

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

The brief does not give a scoring date, a horizon, or a verified event. This review does not invent a 12-month or 90+ DPD target.

## Layers and the question each step answers

| Step | Status | Question it answers | Still forbidden |
| --- | --- | --- | --- |
| V1 Scope and findings | Done | What must be true for a pre-approval PD? What is in the file vs assumed? | Closing High findings with CV AUC |
| V2 Holdout protocol | Done (locked, then scored) | What to compute once on test, and what to refuse | A second `predict` to try another model |
| V3 Implementation | Done | Does the joblib match `C=0.1` and the 14 columns? | Patching the model to “pass” |
| V4 Holdout scoring | Done | Ranking and probability quality of the supplied label on this mixture | Retuning, extra features, peeking to choose a cutoff |
| V5 Segments | Done | Where the average can hide failure | Redefining groups to look better |
| V6 Explanations | Done | Why this logistic equation produced p-hat | Causal stories; SHAP not required |
| V7 Cutoff illustrations | Done | Workload vs capture | Cost-optimal policy without costs |
| V8 Monitoring spec and conclusion | Done | What would be watched if used later; production yes/no | Claiming a live monitoring service |

## Standing rules after scoring

- Do not score the reserved test again.
- Do not change features, `C`, or preprocessing using test outcomes.
- Do not drop age 99 or “fix” DTI to raise scores.
- Do not call this organizationally independent validation (same person as development).
- Do not call development OOF a holdout result.
