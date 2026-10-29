# Credit-risk development (Ameriabank assignment)

This folder is a **new development repo**. It trains on **14 inputs** (including `interest_rate`) and excludes only `recovery_flag_after_90d` and the target `default`.

It is **not** a copy of the earlier 13-input freeze. The reserved test is **not** scored here. Full discussion of `interest_rate` timing belongs in a later **validation** step.

## Data

Place or symlink the assignment pickle at `data/raw/data.pkl` (gitignored). The Desktop source file is `../data.pkl`.

## Run stages (from this folder)

```sh
python src/data_investigation.py
python src/split_data.py
python src/train_all.py
python src/select_winner.py
```

Do not run a holdout evaluation until validation is planned separately.

## Split

80/20 by physical row, stratified by observed file block and target. Seeds 42 (holdout) and 43 (five development folds). After splitting we hash-check against the original membership file when it is available.

## Models

One entrypoint, `src/train_all.py`, compares logistic regression, random forest, and histogram gradient boosting on the **same saved folds**. Selection metric: mean validation **log loss**.
