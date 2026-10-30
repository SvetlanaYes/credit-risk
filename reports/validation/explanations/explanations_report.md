# V6 — Logistic equation (development)

No SHAP. After median impute, missing indicators, scaling, and region one-hot:

$$
\log\frac{\hat{p}}{1-\hat{p}} = \beta_0 + \sum_j \beta_j x_j^{\mathrm{transformed}}
$$

Intercept $\beta_0$ = -1.7325. $C=0.1$. Coefficients live in **transformed** space.
For scaled numeric fields, $+1$ transformed unit is $+1$ development SD after impute.
Region dummies are all retained (no dropped reference). L2 makes the set identifiable; do not read them as contrasts vs a base region.
**Associations in this fitted surface, not causes.** `interest_rate` is read under VAL-03.

Largest $|\beta|$ (transformed names):

| Term | $\beta$ | Odds per +1 transformed |
| --- | ---: | ---: |
| `numeric__past_default` | 0.4480 | 1.5652 |
| `numeric__dti` | 0.4172 | 1.5177 |
| `categorical__region_risk_0` | -0.3052 | 0.7369 |
| `numeric__past_due_60` | 0.2974 | 1.3463 |
| `categorical__region_risk_2` | 0.2506 | 1.2848 |
| `numeric__past_due_30` | 0.2469 | 1.2801 |
| `numeric__past_due_90` | 0.2200 | 1.2461 |
| `numeric__credit_utilization` | 0.1208 | 1.1284 |

Worked development row `source_row_id=0`: reconstructed $\hat{p}$ = 0.089565 (pipeline 0.089565). See `coefficients.csv` and `example_row_0.csv`.

This does not close VAL-01–VAL-05.
