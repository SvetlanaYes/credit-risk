"""V5 — Predeclared segments on saved holdout scores. No second predict."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from dataset import PROJECT_ROOT, TARGET, load_membership, load_partition, load_raw_data

SCORES_PATH = PROJECT_ROOT / "reports" / "validation" / "holdout" / "test_scores.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "validation" / "segments"
MIN_CLASS = 20


def ranking_ok(y: pd.Series) -> bool:
    events = int(y.sum())
    nonevents = int((1 - y).sum())
    return events >= MIN_CLASS and nonevents >= MIN_CLASS


def segment_metrics(name: str, y: pd.Series, probability: np.ndarray) -> dict:
    n = int(len(y))
    events = int(y.sum())
    row = {
        "segment": name,
        "n": n,
        "events": events,
        "nonevents": n - events,
        "observed_rate": float(y.mean()) if n else None,
        "mean_predicted": float(np.mean(probability)) if n else None,
        "ranking_reported": False,
        "roc_auc": None,
        "average_precision": None,
        "log_loss": None,
        "brier_score": None,
        "note": "",
    }
    if n == 0:
        row["note"] = "empty"
        return row
    if not ranking_ok(y):
        row["note"] = f"ranking withheld (<{MIN_CLASS} events or non-events)"
        return row
    row["ranking_reported"] = True
    row["roc_auc"] = float(roc_auc_score(y, probability))
    row["average_precision"] = float(average_precision_score(y, probability))
    row["log_loss"] = float(log_loss(y, probability, labels=[0, 1]))
    row["brier_score"] = float(brier_score_loss(y, probability))
    return row


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    if not SCORES_PATH.exists():
        raise SystemExit("Missing saved test scores. Run V4 first.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORES_PATH)
    membership = load_membership()
    test = load_partition(load_raw_data(), membership, "test")
    frame = scores.merge(membership[["source_row_id", "audit_block"]], on="source_row_id", how="left")
    frame = frame.merge(
        test.reset_index()[["source_row_id", "income", "past_due_90", "past_due_60", "age", TARGET]],
        on="source_row_id",
        how="left",
        suffixes=("", "_raw"),
    )
    if not frame["target"].equals(frame[TARGET]):
        raise ValueError("Saved scores do not match the test target.")

    y = frame["target"]
    p = frame["probability"].to_numpy()
    block1 = frame["audit_block"].eq(1).to_numpy()
    block2 = frame["audit_block"].eq(2).to_numpy()
    missing_income = frame["income"].isna().to_numpy()
    delinquency_order = (frame["past_due_90"] > frame["past_due_60"]).to_numpy()
    age_99 = frame["age"].eq(99).to_numpy()

    rows = [
        segment_metrics("overall_test", y, p),
        segment_metrics("audit_block=1", y[block1], p[block1]),
        segment_metrics("audit_block=2", y[block2], p[block2]),
        segment_metrics("income_missing", y[missing_income], p[missing_income]),
        segment_metrics("income_observed", y[~missing_income], p[~missing_income]),
        segment_metrics("past_due_90>past_due_60", y[delinquency_order], p[delinquency_order]),
        segment_metrics("past_due_90<=past_due_60_or_incomparable", y[~delinquency_order], p[~delinquency_order]),
        segment_metrics("age=99", y[age_99], p[age_99]),
        segment_metrics("age_not_99", y[~age_99], p[~age_99]),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "segment_metrics.csv", index=False)
    (REPORT_DIR / "segment_summary.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# V5 — Predeclared holdout segments",
        "",
        "Uses **saved** test scores. Groups were locked in V2. No retune, no new `predict`.",
        "Ranking is withheld if either class has fewer than 20 rows.",
        "`past_due_90 > past_due_60` is False when either field is missing (those rows sit in the complement).",
        "A strong `interest_rate` pattern in a slice would not close VAL-03.",
        "",
        "| Segment | n | Events | Observed | Mean $\\hat{p}$ | AUC | AP | Log loss | Note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {segment} | {n} | {events} | {observed_rate} | {mean_predicted} | {roc_auc} | {average_precision} | {log_loss} | {note} |".format(
                segment=row["segment"],
                n=row["n"],
                events=row["events"],
                observed_rate=fmt(row["observed_rate"]),
                mean_predicted=fmt(row["mean_predicted"]),
                roc_auc=fmt(row["roc_auc"]),
                average_precision=fmt(row["average_precision"]),
                log_loss=fmt(row["log_loss"]),
                note=row["note"] or "",
            )
        )
    lines.extend(["", "Production use is **not** established.", ""])
    (REPORT_DIR / "segment_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
