# V4 — Reserved-test assessment

Frozen logistic scored the 4,000 test rows **once**. No retuning.
This is discrimination of the **supplied label** on this mixture, not a verified horizon PD.
V1 findings stay open.

| Metric | Frozen model | Constant development rate |
| --- | ---: | ---: |
| ROC AUC | 0.7183 | 0.5000 |
| Average precision | 0.3912 | 0.1573 |
| Log loss | 0.3865 | 0.4351 |
| Brier | 0.1168 | 0.1325 |
| Mean predicted | 0.1596 | 0.1573 |
| Observed rate | 0.1573 | 0.1573 |

Rows 4000, events 629.
Reliability bands use quantile edges from **development** scores of the frozen model (in-sample edges; limitation).
See `reliability_bands.csv` and `test_scores.csv`.

Production use is **not** established.
