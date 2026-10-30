"""V3 — Check the frozen joblib on development rows only. Do not score the test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn import set_config

from dataset import (
    PROJECT_ROOT,
    TARGET,
    feature_columns,
    load_membership,
    load_partition,
    load_raw_data,
)

REPORT_DIR = PROJECT_ROOT / "reports" / "validation" / "implementation"
MODEL_PATH = PROJECT_ROOT / "models" / "selected" / "selected_pipeline.joblib"
EXPECTED_FEATURES = [
    "client_history_in_bank",
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "num_loans",
    "credit_utilization",
    "past_due_30",
    "past_due_60",
    "past_due_90",
    "past_default",
    "dti",
    "region_risk",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    set_config(transform_output="pandas")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline.named_steps["model"]
    checks = []

    c_value = float(model.C)
    checks.append({"check": "C_equals_0.1", "pass": abs(c_value - 0.1) < 1e-12, "observed": c_value})
    checks.append({
        "check": "estimator_is_logistic",
        "pass": type(model).__name__ == "LogisticRegression",
        "observed": type(model).__name__,
    })

    development = load_partition(load_raw_data(), load_membership(), "development")
    features = feature_columns(development)
    checks.append({
        "check": "fourteen_model_features",
        "pass": features == EXPECTED_FEATURES,
        "observed": features,
    })
    checks.append({
        "check": "recovery_not_a_feature",
        "pass": "recovery_flag_after_90d" not in features,
        "observed": "recovery_flag_after_90d" in development.columns,
    })

    X = development[features]
    y = development[TARGET]
    probability = pipeline.predict_proba(X)[:, 1]
    finite = bool(np.isfinite(probability).all())
    in_unit = bool(((probability >= 0) & (probability <= 1)).all())
    checks.append({"check": "development_probabilities_finite", "pass": finite, "n": int(len(probability))})
    checks.append({"check": "development_probabilities_in_unit_interval", "pass": in_unit})

    transformed = pipeline.named_steps["preprocess"].transform(X)
    coef = np.asarray(model.coef_).ravel()
    intercept = float(model.intercept_[0])
    reconstructed = sigmoid(transformed.to_numpy() @ coef + intercept)
    max_abs = float(np.max(np.abs(reconstructed - probability)))
    checks.append({
        "check": "hand_logistic_matches_predict_proba",
        "pass": max_abs < 1e-10,
        "max_abs_difference": max_abs,
    })

    core_pass = all(item["pass"] is True for item in checks)
    summary = {
        "development_rows": int(len(development)),
        "test_rows_scored": 0,
        "core_pass": core_pass,
        "development_mean_predicted": float(probability.mean()),
        "development_observed_rate": float(y.mean()),
        "checks": checks,
    }
    (REPORT_DIR / "implementation_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# V3 — Implementation checks",
        "",
        "Development rows only. The reserved test was **not** scored.",
        "",
        f"- Core identity pass: **{core_pass}**",
        f"- Development rows: {len(development)}",
        f"- Mean predicted vs observed: {probability.mean():.4f} vs {y.mean():.4f}",
        "",
        "| Check | Pass |",
        "| --- | --- |",
    ]
    for item in checks:
        lines.append(f"| {item['check']} | {item['pass']} |")
    lines.extend(["", "If core identity fails, do not score the holdout.", ""])
    (REPORT_DIR / "implementation_report.md").write_text("\n".join(lines))
    print(f"core_pass={core_pass} max_abs_difference={max_abs:.3e}")
    if not core_pass:
        raise SystemExit("Implementation gate failed.")


if __name__ == "__main__":
    main()
