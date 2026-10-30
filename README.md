# Credit-risk candidate

This folder is a development model plus a structured review of the assignment pickle.

The fitted object is L2 logistic regression with `C = 0.1` and original (not log) loan amount. It uses fourteen columns, including `interest_rate`, and does not use `recovery_flag_after_90d`. The output is a probability of the **supplied file label** being 1, given those columns. The brief does not define a scoring date, a horizon, or a verified default event, so this is not a confirmed origination PD.

**Production use is not established.** Findings VAL-01 through VAL-05 remain open.

## What to read

1. [reports/validation/08_monitoring_and_conclusion.md](reports/validation/08_monitoring_and_conclusion.md) — decision and monitoring spec  
2. [reports/validation/01_scope_and_findings.md](reports/validation/01_scope_and_findings.md) — what is in the file versus assumed  
3. [reports/validation/holdout/holdout_report.md](reports/validation/holdout/holdout_report.md) and [reports/validation/segments/segment_report.md](reports/validation/segments/segment_report.md) — reserved-test numbers  
4. Optional: [reports/validation/linear_report.md](reports/validation/linear_report.md) — step-by-step narrative of metrics, not the decision memo  

## Data

Symlink or copy the assignment pickle to `data/raw/data.pkl` (gitignored). A local copy may sit at `../data.pkl`.

## How this was run

From this folder:

```sh
python src/data_investigation.py
python src/split_data.py
python src/train_all.py
python src/select_winner.py
python src/validate_implementation.py
python src/score_holdout.py
python src/segment_holdout.py
python src/explain_logistic.py
python src/illustrate_cutoffs.py
python src/write_v8_conclusion.py
```

The reserved 4,000 rows were scored **once**. `src/score_holdout.py` exits if `reports/validation/holdout/test_scores.csv` already exists. Do not retune from those scores.

## Split and selection

80/20 by physical row, stratified by target and file block. Seeds 42 (holdout) and 43 (five development folds). Membership is saved and hashed.

One trainer compares logistic regression, random forest, and histogram gradient boosting on the same folds. The winner is the lowest **mean fold log loss**.
