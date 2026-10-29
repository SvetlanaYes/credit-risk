"""Train logistic, random forest, and histogram boosting on saved development folds only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from dataset import (
    ASSIGNMENTS_PATH,
    CONFIG_DIR,
    PROJECT_ROOT,
    TARGET,
    feature_columns,
    load_json,
    load_membership,
    load_partition,
    load_raw_data,
    sha256_file,
)

REPORT_DIR = PROJECT_ROOT / "reports" / "training"
MODELS_DIR = PROJECT_ROOT / "models"


def log1p_loan_amount(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["loan_amount"] = np.log1p(out["loan_amount"].astype(float))
    return out


def build_preprocess(feature_names: list[str], categorical: list[str], log_loan: bool) -> Pipeline:
    numeric = [name for name in feature_names if name not in categorical]
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")
    steps = []
    if log_loan:
        steps.append(("log_loan", FunctionTransformer(log1p_loan_amount, validate=False)))
    steps.append(("columns", transformer))
    return Pipeline(steps)


def make_estimator(family: str, spec: dict):
    if family == "logistic":
        return LogisticRegression(
            C=spec["C"],
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=spec["min_samples_leaf"],
            max_features=0.8,
            random_state=42,
            n_jobs=2,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=spec["max_leaf_nodes"],
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
    raise ValueError(f"Unknown family {family}")


def score_probabilities(y: pd.Series, probability: np.ndarray) -> dict:
    predicted_label = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "accuracy_at_0_5": float(accuracy_score(y, predicted_label)),
        "recall_at_0_5": float(recall_score(y, predicted_label, pos_label=1, zero_division=0)),
    }


def iter_logistic_grid(model_config: dict):
    for processing in model_config["logistic"]["loan_amount_processing"]:
        for C in model_config["logistic"]["C_values"]:
            name = f"logistic_C{C}_{processing}"
            yield name, {"C": C, "loan_amount": processing}, processing == "log1p"


def iter_forest_grid(model_config: dict):
    for leaf in model_config["random_forest"]["min_samples_leaf_values"]:
        name = f"random_forest_leaf{leaf}"
        yield name, {"min_samples_leaf": leaf}, False


def iter_boosting_grid(model_config: dict):
    for leaves in model_config["hist_gradient_boosting"]["max_leaf_nodes_values"]:
        name = f"hist_gb_leaves{leaves}"
        yield name, {"max_leaf_nodes": leaves}, False


def run_family(family: str, grid, X, y, fold_ids, feature_names, categorical, family_dir: Path):
    family_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    summaries = []
    for config_name, spec, log_loan in grid:
        oof = np.full(len(X), np.nan)
        fold_losses = []
        last_pipeline = None
        for fold in sorted(fold_ids.dropna().unique()):
            train_mask = fold_ids.ne(fold).to_numpy()
            valid_mask = fold_ids.eq(fold).to_numpy()
            X_train, y_train = X.loc[train_mask], y.loc[train_mask]
            X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
            pipeline = Pipeline([
                ("preprocess", build_preprocess(feature_names, categorical, log_loan)),
                ("model", make_estimator(family, spec)),
            ])
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_valid)[:, 1]
            oof[valid_mask] = probability
            metrics = score_probabilities(y_valid, probability)
            fold_rows.append({"family": family, "config": config_name, "fold": int(fold), **metrics})
            fold_losses.append(metrics["log_loss"])
            last_pipeline = pipeline
        mean_log_loss = float(np.mean(fold_losses))
        pooled = score_probabilities(y, oof)
        summaries.append({
            "family": family,
            "config": config_name,
            "spec": spec,
            "mean_fold_log_loss": mean_log_loss,
            "pooled_oof": pooled,
        })
        # Refit the last spec on all development rows after the fold loop ends.
        # Family winner is chosen later; we still save each config's OOF metrics.
        del last_pipeline
        print(f"  {config_name}: mean fold log loss {mean_log_loss:.6f}")

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(family_dir / "fold_metrics.csv", index=False)
    summary_table = pd.DataFrame([
        {
            "family": row["family"],
            "config": row["config"],
            "mean_fold_log_loss": row["mean_fold_log_loss"],
            **{f"oof_{key}": value for key, value in row["pooled_oof"].items()},
            "spec_json": json.dumps(row["spec"]),
        }
        for row in summaries
    ]).sort_values("mean_fold_log_loss")
    summary_table.to_csv(family_dir / "config_summary.csv", index=False)

    winner_row = min(summaries, key=lambda row: row["mean_fold_log_loss"])
    log_loan = False
    if family == "logistic":
        log_loan = winner_row["spec"].get("loan_amount") == "log1p"
    winner_pipeline = Pipeline([
        ("preprocess", build_preprocess(feature_names, categorical, log_loan)),
        ("model", make_estimator(family, winner_row["spec"])),
    ])
    winner_pipeline.fit(X, y)
    joblib.dump(winner_pipeline, family_dir / "family_best.joblib")
    (family_dir / "family_best.json").write_text(json.dumps(winner_row, indent=2))
    return winner_row


def main() -> None:
    set_config(transform_output="pandas")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_policy = load_json(CONFIG_DIR / "features.json")
    model_config = load_json(CONFIG_DIR / "models.json")
    membership = load_membership()
    development = load_partition(load_raw_data(), membership, "development")
    feature_names = feature_columns(development)
    if "interest_rate" not in feature_names:
        raise ValueError("interest_rate must be included in this development version.")
    if "recovery_flag_after_90d" in feature_names:
        raise ValueError("recovery_flag_after_90d must stay out of the model.")
    X = development[feature_names].copy()
    y = development[TARGET].copy()
    fold_ids = membership.set_index("source_row_id").loc[development.index, "cv_fold"]
    categorical = feature_policy["categorical_features"]

    print("Logistic grid")
    logistic_best = run_family(
        "logistic",
        iter_logistic_grid(model_config),
        X, y, fold_ids, feature_names, categorical,
        MODELS_DIR / "logistic",
    )
    print("Random forest grid")
    forest_best = run_family(
        "random_forest",
        iter_forest_grid(model_config),
        X, y, fold_ids, feature_names, categorical,
        MODELS_DIR / "random_forest",
    )
    print("Histogram gradient boosting grid")
    boosting_best = run_family(
        "hist_gradient_boosting",
        iter_boosting_grid(model_config),
        X, y, fold_ids, feature_names, categorical,
        MODELS_DIR / "hist_gradient_boosting",
    )

    comparison = pd.DataFrame([
        {
            "family": row["family"],
            "config": row["config"],
            "mean_fold_log_loss": row["mean_fold_log_loss"],
            "oof_roc_auc": row["pooled_oof"]["roc_auc"],
            "oof_average_precision": row["pooled_oof"]["average_precision"],
            "oof_brier_score": row["pooled_oof"]["brier_score"],
        }
        for row in [logistic_best, forest_best, boosting_best]
    ]).sort_values("mean_fold_log_loss")
    comparison.to_csv(REPORT_DIR / "family_comparison.csv", index=False)
    payload = {
        "features": feature_names,
        "n_development": int(len(development)),
        "membership_sha256": sha256_file(ASSIGNMENTS_PATH),
        "final_test_scored": False,
        "interest_rate_included": True,
        "family_best": [logistic_best, forest_best, boosting_best],
        "leader_by_mean_fold_log_loss": comparison.iloc[0].to_dict(),
    }
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(payload, indent=2))
    print(comparison.to_string(index=False))
    print("Training used development folds only. Test was not scored.")


if __name__ == "__main__":
    main()
