# V5 — Predeclared holdout segments

Uses **saved** test scores. Groups were locked in V2. No retune, no new `predict`.
Ranking is withheld if either class has fewer than 20 rows.
`past_due_90 > past_due_60` is False when either field is missing (those rows sit in the complement).
A strong `interest_rate` pattern in a slice would not close VAL-03.

| Segment | n | Events | Observed | Mean $\hat{p}$ | AUC | AP | Log loss | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| overall_test | 4000 | 629 | 0.1573 | 0.1596 | 0.7183 | 0.3912 | 0.3865 |  |
| audit_block=1 | 2600 | 409 | 0.1573 | 0.1591 | 0.7058 | 0.3871 | 0.3899 |  |
| audit_block=2 | 1400 | 220 | 0.1571 | 0.1607 | 0.7418 | 0.3999 | 0.3801 |  |
| income_missing | 137 | 30 | 0.2190 | 0.2647 | 0.7312 | 0.5462 | 0.4587 |  |
| income_observed | 3863 | 599 | 0.1551 | 0.1559 | 0.7177 | 0.3841 | 0.3839 |  |
| past_due_90>past_due_60 | 160 | 51 | 0.3187 | 0.2909 | 0.5985 | 0.4773 | 0.6097 |  |
| past_due_90<=past_due_60_or_incomparable | 3840 | 578 | 0.1505 | 0.1542 | 0.7138 | 0.3817 | 0.3772 |  |
| age=99 | 32 | 2 | 0.0625 | 0.1726 | — | — | — | ranking withheld (<20 events or non-events) |
| age_not_99 | 3968 | 627 | 0.1580 | 0.1595 | 0.7195 | 0.3938 | 0.3871 |  |

Production use is **not** established.
