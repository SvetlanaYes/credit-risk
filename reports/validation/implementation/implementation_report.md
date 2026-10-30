# V3 — Implementation checks

Development rows only. The reserved test was **not** scored.

- Core identity pass: **True**
- Development rows: 16000
- Mean predicted vs observed: 0.1573 vs 0.1573

| Check | Pass |
| --- | --- |
| C_equals_0.1 | True |
| estimator_is_logistic | True |
| fourteen_model_features | True |
| recovery_not_a_feature | True |
| development_probabilities_finite | True |
| development_probabilities_in_unit_interval | True |
| hand_logistic_matches_predict_proba | True |

If core identity fails, do not score the holdout.
