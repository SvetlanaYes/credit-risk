# Credit-risk development (Ameriabank assignment)

This project trains a probability-of-default candidate on **14 inputs** (including `interest_rate`) and excludes only `recovery_flag_after_90d` and the target `default`.

The reserved test is **not** scored here. Timing and leakage for `interest_rate` belong in a later **validation** step.

## Data

Place or symlink the assignment pickle at `data/raw/data.pkl` (gitignored). The Desktop source file is `../data.pkl`.

## Run stages (from this folder)

```sh
python src/data_investigation.py
python src/split_data.py
python src/train_all.py
python src/select_winner.py
```

Validation documents (holdout still locked): `reports/validation/00_overview.md`. Do not score the reserved test until `02_holdout_protocol.md` is approved.

## Split

80/20 by physical row, stratified by observed file block and target. Seeds 42 (holdout) and 43 (five development folds). Membership is saved and hashed so the split can be reused unchanged.

## Models

One entrypoint, `src/train_all.py`, compares logistic regression, random forest, and histogram gradient boosting on the **same saved folds**. Selection metric: mean validation **log loss**.
