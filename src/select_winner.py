"""Freeze the family with the lowest mean development-fold log loss. Do not score the test."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from dataset import PROJECT_ROOT

REPORT_DIR = PROJECT_ROOT / "reports" / "training"
MODELS_DIR = PROJECT_ROOT / "models"
SELECTED_DIR = MODELS_DIR / "selected"


def main() -> None:
    comparison = pd.read_csv(REPORT_DIR / "family_comparison.csv")
    winner = comparison.sort_values("mean_fold_log_loss").iloc[0]
    family = winner["family"]
    source_dir = MODELS_DIR / family
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_dir / "family_best.joblib", SELECTED_DIR / "selected_pipeline.joblib")
    shutil.copy(source_dir / "family_best.json", SELECTED_DIR / "selected_spec.json")

    note = [
        "# Selected development model",
        "",
        "Winner is chosen by **mean five-fold log loss** on development data only.",
        "This is not production approval and not a reserved-test result.",
        "",
        f"- Family: `{family}`",
        f"- Config: `{winner['config']}`",
        f"- Mean fold log loss: {winner['mean_fold_log_loss']:.6f}",
        f"- Development OOF AUC: {winner['oof_roc_auc']:.4f}",
        f"- Development OOF AP: {winner['oof_average_precision']:.4f}",
        f"- Development OOF Brier: {winner['oof_brier_score']:.6f}",
        "",
        "Fourteen inputs including `interest_rate`. Recovery flag excluded.",
        "",
        "See `family_comparison.csv` for the three family-best rows.",
        "",
    ]
    (REPORT_DIR / "model_selection.md").write_text("\n".join(note))
    record = {
        "family": family,
        "config": winner["config"],
        "mean_fold_log_loss": float(winner["mean_fold_log_loss"]),
        "final_test_scored": False,
    }
    (SELECTED_DIR / "selection_record.json").write_text(json.dumps(record, indent=2))
    print("\n".join(note))


if __name__ == "__main__":
    main()
