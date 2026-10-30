"""V7 — Workload vs capture using development OOF quantiles on saved test scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

from dataset import (
    CONFIG_DIR,
    PROJECT_ROOT,
    TARGET,
    feature_columns,
    load_json,
    load_membership,
    load_partition,
    load_raw_data,
)
from train_all import build_preprocess, make_estimator

SCORES_PATH = PROJECT_ROOT / "reports" / "validation" / "holdout" / "test_scores.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "validation" / "cutoffs"
TOP_SHARES = (0.10, 0.20, 0.30)


def reconstruct_oof() -> tuple[np.ndarray, pd.Series]:
    """Same frozen spec, development folds only. Not a new model search."""
    membership = load_membership()
    development = load_partition(load_raw_data(), membership, "development")
    feature_names = feature_columns(development)
    categorical = load_json(CONFIG_DIR / "features.json")["categorical_features"]
    X = development[feature_names]
    y = development[TARGET]
    fold_ids = membership.set_index("source_row_id").loc[development.index, "cv_fold"]
    oof = np.full(len(X), np.nan)
    spec = {"C": 0.1, "loan_amount": "original"}
    for fold in sorted(fold_ids.dropna().unique()):
        train_mask = fold_ids.ne(fold).to_numpy()
        valid_mask = fold_ids.eq(fold).to_numpy()
        pipeline = Pipeline([
            ("preprocess", build_preprocess(feature_names, categorical, False)),
            ("model", make_estimator("logistic", spec)),
        ])
        pipeline.fit(X.loc[train_mask], y.loc[train_mask])
        oof[valid_mask] = pipeline.predict_proba(X.loc[valid_mask])[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError("OOF reconstruction left missing scores.")
    return oof, y


def evaluate_rule(y: np.ndarray, probability: np.ndarray, flagged: np.ndarray, label: str) -> dict:
    n = int(len(y))
    events = int(y.sum())
    n_flagged = int(flagged.sum())
    captured = int((flagged & (y == 1)).sum())
    false_flags = int((flagged & (y == 0)).sum())
    below = ~flagged
    return {
        "rule": label,
        "threshold": None,
        "n": n,
        "flag_rate": n_flagged / n if n else None,
        "defaults_captured": captured,
        "capture_rate": captured / events if events else None,
        "non_defaults_flagged": false_flags,
        "false_flag_rate_among_non_defaults": false_flags / (n - events) if n > events else None,
        "event_rate_among_flagged": captured / n_flagged if n_flagged else None,
        "event_rate_below_cutoff": float(y[below].mean()) if below.any() else None,
        "n_below": int(below.sum()),
    }


def main() -> None:
    set_config(transform_output="pandas")
    if not SCORES_PATH.exists():
        raise SystemExit("Missing saved test scores. Run V4 first.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    oof, y_dev = reconstruct_oof()
    oof_path = REPORT_DIR / "development_oof_scores.csv"
    pd.DataFrame({
        "source_row_id": y_dev.index.to_numpy(),
        "target": y_dev.to_numpy(),
        "oof_probability": oof,
    }).to_csv(oof_path, index=False)

    test = pd.read_csv(SCORES_PATH)
    y = test["target"].to_numpy()
    p = test["probability"].to_numpy()

    rows = []
    for share in TOP_SHARES:
        threshold = float(np.quantile(oof, 1.0 - share))
        flagged = p >= threshold
        row = evaluate_rule(y, p, flagged, f"top_{int(share * 100)}_percent_of_development_oof")
        row["threshold"] = threshold
        row["development_oof_share"] = share
        rows.append(row)

    diagnostic = evaluate_rule(y, p, p >= 0.5, "diagnostic_0.5_not_a_policy")
    diagnostic["threshold"] = 0.5
    diagnostic["development_oof_share"] = None
    rows.append(diagnostic)

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "cutoff_table.csv", index=False)
    (REPORT_DIR / "cutoff_summary.json").write_text(json.dumps({
        "source": "development OOF quantiles of logistic C=0.1 original; applied to saved test scores",
        "oof_mean": float(oof.mean()),
        "rows": rows,
        "policy": False,
        "costs_unknown": True,
    }, indent=2))

    lines = [
        "# V7 — Cutoff illustrations",
        "",
        "Thresholds come from **development OOF** of the frozen spec (reconstructed; not a new search).",
        "Rules are applied to **saved** test scores. This is workload vs capture, not a lending policy.",
        "Costs, LGD, and exposure are unknown, so no profit-maximising cutoff.",
        "`0.5` is a diagnostic only.",
        "",
        "| Rule | Threshold | Flag rate | Defaults captured | Capture rate | Non-defaults flagged | Event rate below cutoff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rule} | {threshold:.4f} | {flag_rate:.4f} | {defaults_captured} | {capture_rate:.4f} | {non_defaults_flagged} | {event_rate_below_cutoff:.4f} |".format(
                **{k: row[k] for k in row}
            )
        )
    lines.extend([
        "",
        "A lower event rate below the cutoff than the sample rate (~0.157) means the tail that is **not** flagged is cleaner on this label.",
        "Do not pick a cutoff from this table to raise a metric.",
        "",
    ])
    (REPORT_DIR / "cutoff_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
