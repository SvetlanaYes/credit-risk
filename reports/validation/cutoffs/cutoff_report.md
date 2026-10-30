# V7 — Cutoff illustrations

Thresholds come from **development OOF** of the frozen spec (reconstructed; not a new search).
Rules are applied to **saved** test scores. This is workload vs capture, not a lending policy.
Costs, LGD, and exposure are unknown, so no profit-maximising cutoff.
`0.5` is a diagnostic only.

| Rule | Threshold | Flag rate | Defaults captured | Capture rate | Non-defaults flagged | Event rate below cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_10_percent_of_development_oof | 0.3169 | 0.0955 | 179 | 0.2846 | 203 | 0.1244 |
| top_20_percent_of_development_oof | 0.2189 | 0.2072 | 290 | 0.4610 | 539 | 0.1069 |
| top_30_percent_of_development_oof | 0.1660 | 0.3025 | 363 | 0.5771 | 847 | 0.0953 |
| diagnostic_0.5_not_a_policy | 0.5000 | 0.0315 | 84 | 0.1335 | 42 | 0.1407 |

A lower event rate below the cutoff than the sample rate (~0.157) means the tail that is **not** flagged is cleaner on this label.
Do not pick a cutoff from this table to raise a metric.
