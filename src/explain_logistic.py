"""V6 — Exact logistic equation on development. No SHAP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config

from dataset import PROJECT_ROOT, feature_columns, load_membership, load_partition, load_raw_data

MODEL_PATH = PROJECT_ROOT / "models" / "selected" / "selected_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports" / "validation" / "explanations"


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -50, 50))
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    set_config(transform_output="pandas")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(MODEL_PATH)
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    columns = preprocess.named_steps["columns"]
    names = list(columns.get_feature_names_out())
    coef = np.asarray(model.coef_).ravel()
    intercept = float(model.intercept_[0])

    scaler = columns.named_transformers_["numeric"].named_steps["scale"]
    n_numeric_out = scaler.mean_.shape[0]
    numeric_names = names[:n_numeric_out]
    raw_scale = {name: float(scale) for name, scale in zip(numeric_names, scaler.scale_)}

    rows = []
    for name, beta in zip(names, coef):
        item = {
            "transformed_name": name,
            "coefficient": float(beta),
            "odds_per_plus1_transformed": float(np.exp(beta)),
            "odds_per_plus1_raw_numeric": None,
            "kind": "other",
        }
        if name.startswith("numeric__"):
            item["kind"] = "missing_indicator" if "missingindicator" in name else "scaled_numeric"
            scale = raw_scale.get(name)
            if scale not in (None, 0.0) and "missingindicator" not in name:
                item["odds_per_plus1_raw_numeric"] = float(np.exp(beta / scale))
        elif name.startswith("categorical__"):
            item["kind"] = "region_dummy"
        rows.append(item)

    table = pd.DataFrame(rows).sort_values("coefficient", key=np.abs, ascending=False)
    table.to_csv(REPORT_DIR / "coefficients.csv", index=False)

    development = load_partition(load_raw_data(), load_membership(), "development")
    features = feature_columns(development)
    example = development.iloc[[0]]
    example_id = int(example.index[0])
    transformed = preprocess.transform(example[features])
    contributions = transformed.to_numpy().ravel() * coef
    z = intercept + float(contributions.sum())
    probability = sigmoid(z)
    pipeline_p = float(pipeline.predict_proba(example[features])[0, 1])

    example_rows = [{"term": "intercept", "value": intercept, "contribution": intercept}]
    for name, value, contribution in zip(names, transformed.to_numpy().ravel(), contributions):
        example_rows.append({
            "term": name,
            "value": float(value),
            "contribution": float(contribution),
        })
    example_frame = pd.DataFrame(example_rows)
    example_frame["abs_contribution"] = example_frame["contribution"].abs()
    example_frame = example_frame.sort_values("abs_contribution", ascending=False)
    example_frame.to_csv(REPORT_DIR / f"example_row_{example_id}.csv", index=False)

    (REPORT_DIR / "explanation_summary.json").write_text(json.dumps({
        "intercept": intercept,
        "C": float(model.C),
        "n_coefficients": int(len(coef)),
        "example_source_row_id": example_id,
        "example_log_odds": z,
        "example_probability": probability,
        "pipeline_probability": pipeline_p,
        "shap_used": False,
    }, indent=2))

    top = table.head(8)
    lines = [
        "# V6 — Logistic equation (development)",
        "",
        "No SHAP. After median impute, missing indicators, scaling, and region one-hot:",
        "",
        "$$",
        r"\log\frac{\hat{p}}{1-\hat{p}} = \beta_0 + \sum_j \beta_j x_j^{\mathrm{transformed}}",
        "$$",
        "",
        f"Intercept $\\beta_0$ = {intercept:.4f}. $C=0.1$. Coefficients live in **transformed** space.",
        "For scaled numeric fields, $+1$ transformed unit is $+1$ development SD after impute.",
        "Region dummies are all retained (no dropped reference). L2 makes the set identifiable; do not read them as contrasts vs a base region.",
        "**Associations in this fitted surface, not causes.** `interest_rate` is read under VAL-03.",
        "",
        "Largest $|\\beta|$ (transformed names):",
        "",
        "| Term | $\\beta$ | Odds per +1 transformed |",
        "| --- | ---: | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['transformed_name']}` | {row['coefficient']:.4f} | {row['odds_per_plus1_transformed']:.4f} |"
        )
    lines.extend([
        "",
        f"Worked development row `source_row_id={example_id}`: reconstructed $\\hat{{p}}$ = {probability:.6f} "
        f"(pipeline {pipeline_p:.6f}). See `coefficients.csv` and `example_row_{example_id}.csv`.",
        "",
        "This does not close VAL-01–VAL-05.",
        "",
    ])
    (REPORT_DIR / "explanations_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
