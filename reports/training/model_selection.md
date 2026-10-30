# Selected development model

Winner is chosen by **mean five-fold log loss** on development data only.
This is not production approval and not a reserved-test result.

- Family: `logistic`
- Config: `logistic_C0.1_original`
- Mean fold log loss: 0.389228
- Development OOF AUC: 0.7167
- Development OOF AP: 0.3741
- Development OOF Brier: 0.118014

Fourteen inputs including `interest_rate`. Recovery flag excluded.

See `family_comparison.csv` for the three family-best rows.
