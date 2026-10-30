"""V4 — Score the reserved test once. Do not retune."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from dataset import (
    PROJECT_ROOT,
    TARGET,
    feature_columns,
    load_membership,
    load_partition,
    load_raw_data,
)

REPORT_DIR = PROJECT_ROOT / "reports" / "validation" / "holdout"
MODEL_PATH = PROJECT_ROOT / "models" / "selected" / "selected_pipeline.joblib"
SCORES_PATH = REPORT_DIR / "test_scores.csv"
SELECTION_RECORD = PROJECT_ROOT / "models" / "selected" / "selection_record.json"


def metrics(y: pd.Series, probability: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "mean_predicted": float(np.mean(probability)),
        "observed_rate": float(y.mean()),
        "n": int(len(y)),
        "events": int(y.sum()),
    }


def reliability_table(dev_p: np.ndarray, test_p: np.ndarray, test_y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(dev_p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    frame = pd.DataFrame({"probability": test_p, "target": test_y})
    frame["band"] = pd.cut(frame["probability"], bins=edges, include_lowest=True)
    grouped = frame.groupby("band", observed=True)
    return grouped.agg(
        n=("target", "size"),
        mean_predicted=("probability", "mean"),
        observed_rate=("target", "mean"),
        events=("target", "sum"),
    ).reset_index()


def main() -> None:
    set_config(transform_output="pandas")
    if SCORES_PATH.exists():
        raise SystemExit(f"{SCORES_PATH} already exists. Protocol allows one scoring only.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(MODEL_PATH)
    membership = load_membership()
    raw = load_raw_data()
    development = load_partition(raw, membership, "development")
    test = load_partition(raw, membership, "test")
    features = feature_columns(development)
    if list(test.columns) and "recovery_flag_after_90d" in features:
        raise ValueError("Recovery flag must not be scored as a feature.")

    y_dev = development[TARGET]
    y_test = test[TARGET]
    X_dev = development[features]
    X_test = test[features]

    development_probability = pipeline.predict_proba(X_dev)[:, 1]
    test_probability = pipeline.predict_proba(X_test)[:, 1]

    scores = pd.DataFrame({
        "source_row_id": test.index.to_numpy(),
        "target": y_test.to_numpy(),
        "probability": test_probability,
    })
    scores.to_csv(SCORES_PATH, index=False)

    constant = np.full(len(y_test), float(y_dev.mean()))
    candidate = metrics(y_test, test_probability)
    reference = metrics(y_test, constant)
    bands = reliability_table(development_probability, test_probability, y_test.to_numpy())
    bands.to_csv(REPORT_DIR / "reliability_bands.csv", index=False)

    summary = {
        "protocol": "V2 approved; one frozen scoring; no retune",
        "candidate": candidate,
        "constant_development_rate_reference": reference,
        "log_loss_improvement": reference["log_loss"] - candidate["log_loss"],
        "brier_improvement": reference["brier_score"] - candidate["brier_score"],
        "scores_path": str(SCORES_PATH.relative_to(PROJECT_ROOT)),
        "retuned": False,
        "production_approved": False,
    }
    (REPORT_DIR / "holdout_summary.json").write_text(json.dumps(summary, indent=2))

    record = json.loads(SELECTION_RECORD.read_text())
    record["final_test_scored"] = True
    record["holdout_roc_auc"] = candidate["roc_auc"]
    SELECTION_RECORD.write_text(json.dumps(record, indent=2))

    report = [
        "# V4 — Reserved-test assessment",
        "",
        "Frozen logistic scored the 4,000 test rows **once**. No retuning.",
        "This is discrimination of the **supplied label** on this mixture, not a verified horizon PD.",
        "V1 findings stay open.",
        "",
        "| Metric | Frozen model | Constant development rate |",
        "| --- | ---: | ---: |",
        f"| ROC AUC | {candidate['roc_auc']:.4f} | {reference['roc_auc']:.4f} |",
        f"| Average precision | {candidate['average_precision']:.4f} | {reference['average_precision']:.4f} |",
        f"| Log loss | {candidate['log_loss']:.4f} | {reference['log_loss']:.4f} |",
        f"| Brier | {candidate['brier_score']:.4f} | {reference['brier_score']:.4f} |",
        f"| Mean predicted | {candidate['mean_predicted']:.4f} | {reference['mean_predicted']:.4f} |",
        f"| Observed rate | {candidate['observed_rate']:.4f} | {reference['observed_rate']:.4f} |",
        "",
        f"Rows {candidate['n']}, events {candidate['events']}.",
        "Reliability bands use quantile edges from **development** scores of the frozen model (in-sample edges; limitation).",
        "See `reliability_bands.csv` and `test_scores.csv`.",
        "",
        "Production use is **not** established.",
        "",
    ]
    (REPORT_DIR / "holdout_report.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
