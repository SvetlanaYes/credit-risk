"""Create a frozen 80/20 split and five development folds. Do not train or score the test."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from dataset import (
    ASSIGNMENTS_PATH,
    CONFIG_DIR,
    ORIGINAL_MEMBERSHIP_SHA256,
    PROJECT_ROOT,
    RAW_DATA,
    SPLITS_DIR,
    TARGET,
    load_json,
    load_raw_data,
    observed_file_block,
    sha256_file,
)
from plots import save_grouped_bars

REPORT_DIR = PROJECT_ROOT / "reports" / "split"
ORIGINAL_ASSIGNMENTS = Path(
    "/Users/admin/Documents/Codex/2026-10-21/ww/outputs/ameria-credit-risk/data/splits/partition_assignments.csv"
)


def build_assignments(data: pd.DataFrame, plan: dict) -> pd.DataFrame:
    target = data[plan["target_column"]].reset_index(drop=True)
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Evaluation requires a known 0/1 target on every row.")
    original_index = data.index.to_numpy()
    block = observed_file_block(original_index).to_numpy()
    strata = pd.Series(block).astype(str) + "|" + target.astype(str)
    if strata.value_counts().min() < plan["cv_splits"]:
        raise ValueError("A target/block stratum is too small for the planned folds.")
    row_ids = np.arange(len(data))
    development_ids, test_ids = train_test_split(
        row_ids,
        test_size=plan["test_fraction"],
        random_state=plan["holdout_random_state"],
        stratify=strata,
    )
    development_ids = np.sort(development_ids)
    assignment = pd.DataFrame({
        "source_row_id": row_ids,
        "original_index": original_index,
        "audit_block": block,
        "partition": "test",
        "cv_fold": pd.Series(pd.NA, index=np.arange(len(data)), dtype="Int64"),
    })
    assignment.loc[development_ids, "partition"] = "development"
    development_strata = strata.iloc[development_ids].reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=plan["cv_splits"],
        shuffle=plan["cv_shuffle"],
        random_state=plan["cv_random_state"],
    )
    for fold, (_, validation_positions) in enumerate(
        splitter.split(np.zeros(len(development_ids)), development_strata),
        start=1,
    ):
        assignment.loc[development_ids[validation_positions], "cv_fold"] = fold
    return assignment


def partition_stats(data: pd.DataFrame, membership: pd.DataFrame, label: str) -> dict:
    if label == "full":
        subset = data.reset_index(drop=True)
        block = membership["audit_block"]
    else:
        ids = membership.loc[membership["partition"].eq(label), "source_row_id"]
        subset = data.reset_index(drop=True).iloc[ids]
        block = membership.loc[membership["partition"].eq(label), "audit_block"]
    return {
        "partition": label,
        "rows": int(len(subset)),
        "target_ones": int(subset[TARGET].eq(1).sum()),
        "target_one_rate": float(subset[TARGET].eq(1).mean()),
        "block_1": int((block == 1).sum()),
        "block_2": int((block == 2).sum()),
        "income_missing": int(subset["income"].isna().sum()),
        "utilization_missing": int(subset["credit_utilization"].isna().sum()),
    }


def main() -> None:
    plan = load_json(CONFIG_DIR / "evaluation.json")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_raw_data()
    raw_hash = sha256_file(RAW_DATA)
    generated = build_assignments(data, plan)

    used_original_file = False
    generated.to_csv(ASSIGNMENTS_PATH, index=False)
    membership_hash = sha256_file(ASSIGNMENTS_PATH)
    if membership_hash != ORIGINAL_MEMBERSHIP_SHA256:
        if ORIGINAL_ASSIGNMENTS.exists():
            shutil.copy(ORIGINAL_ASSIGNMENTS, ASSIGNMENTS_PATH)
            used_original_file = True
            membership_hash = sha256_file(ASSIGNMENTS_PATH)
        # If the original file is missing, keep the generated split and record the mismatch.

    membership = pd.read_csv(ASSIGNMENTS_PATH, dtype={"cv_fold": "Int64"})
    stats = pd.DataFrame([
        partition_stats(data, membership, "full"),
        partition_stats(data, membership, "development"),
        partition_stats(data, membership, "test"),
    ])
    stats.to_csv(REPORT_DIR / "partition_sizes.csv", index=False)

    rate_table = stats.set_index("partition")[["target_one_rate"]]
    rate_table.index = ["full file", "development", "reserved test"]
    save_grouped_bars(
        REPORT_DIR / "target_rate_by_partition.png",
        rate_table,
        "Target=1 rate: full file vs split",
        "Share labelled 1",
    )
    block_table = stats.set_index("partition")[["block_1", "block_2"]]
    block_table.index = ["full file", "development", "reserved test"]
    save_grouped_bars(
        REPORT_DIR / "block_mix_by_partition.png",
        block_table,
        "Observed file blocks in each partition",
        "Rows",
    )
    missing_table = stats.set_index("partition")[["income_missing", "utilization_missing"]]
    missing_table.index = ["full file", "development", "reserved test"]
    save_grouped_bars(
        REPORT_DIR / "missingness_by_partition.png",
        missing_table,
        "Missing income and utilization by partition",
        "Rows",
    )

    summary = {
        "raw_sha256": raw_hash,
        "membership_sha256": membership_hash,
        "matches_original_membership_hash": membership_hash == ORIGINAL_MEMBERSHIP_SHA256,
        "used_original_assignment_file": used_original_file,
        "final_test_scored": False,
        "partitions": stats.to_dict(orient="records"),
    }
    (REPORT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    (REPORT_DIR / "split_report.md").write_text(
        "\n".join([
            "# Split report",
            "",
            "Development = 16,000 rows used for five-fold comparison. Test = 4,000 rows **reserved**.",
            "This test is a holdout of physical rows from the same file mixture. It is not out-of-time",
            "and not independent validation. Missingness was not used to search for a convenient split.",
            "",
            f"- Membership SHA-256: `{membership_hash}`",
            f"- Matches original recorded hash: {summary['matches_original_membership_hash']}",
            f"- Copied original assignment file after a seed mismatch: {used_original_file}",
            "",
            "See `partition_sizes.csv` for row counts, target rates, block mix, and missingness.",
            "",
        ])
    )
    print(f"Wrote split reports under {REPORT_DIR}")
    print(f"membership hash {membership_hash} original_match={summary['matches_original_membership_hash']}")


if __name__ == "__main__":
    main()
