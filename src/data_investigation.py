"""Describe the raw assignment table. Do not split, impute, or train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PROJECT_ROOT, RAW_DATA, TARGET, feature_columns, load_raw_data, observed_file_block, sha256_file
from plots import save_bar

REPORT_DIR = PROJECT_ROOT / "reports" / "investigation"


def column_profile(data: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame({
        "dtype": data.dtypes.astype(str),
        "missing_count": data.isna().sum(),
        "missing_percent": data.isna().mean().mul(100).round(3),
        "distinct_nonmissing": data.nunique(dropna=True),
    })
    profile.index.name = "column"
    return profile


def quality_flags(data: pd.DataFrame) -> dict:
    flags = {}
    flags["age_equal_99"] = int((data["age"] == 99).sum())
    flags["income_equal_30000"] = int((data["income"] == 30000).sum())
    flags["income_missing"] = int(data["income"].isna().sum())
    flags["utilization_missing"] = int(data["credit_utilization"].isna().sum())
    recorded = data["income"].notna()
    ratio = data.loc[recorded, "loan_amount"] / (data.loc[recorded, "income"] * data.loc[recorded, "dti"])
    flags["dti_ratio_near_one"] = int(((ratio - 1).abs() < 0.05).sum())
    flags["dti_ratio_near_ten"] = int(((ratio - 10).abs() < 0.5).sum())
    flags["past_due_60_gt_30"] = int((data["past_due_60"] > data["past_due_30"]).sum())
    flags["past_due_90_gt_60"] = int((data["past_due_90"] > data["past_due_60"]).sum())
    flags["delinquency_gt_num_loans"] = int(
        (data[["past_due_30", "past_due_60", "past_due_90"]].max(axis=1) > data["num_loans"]).sum()
    )
    young_tenure = (data["age"] - data["employment_years"]) < 14
    flags["age_minus_employment_below_14"] = int(young_tenure.sum())
    flags["interest_rate_min"] = float(data["interest_rate"].min())
    flags["interest_rate_max"] = float(data["interest_rate"].max())
    flags["recovery_flag_positive"] = int((data["recovery_flag_after_90d"] == 1).sum())
    return flags


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_raw_data()
    features = feature_columns(data)
    block = observed_file_block(data.index)

    profile = column_profile(data)
    profile.to_csv(REPORT_DIR / "column_profile.csv")

    numeric = data.select_dtypes(include="number").describe().T
    numeric.to_csv(REPORT_DIR / "numeric_summary.csv")

    flags = quality_flags(data)
    missing = data.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        save_bar(
            REPORT_DIR / "missingness.png",
            missing.index,
            missing.values,
            "Missing value counts in the raw file",
            "Rows missing",
        )

    save_bar(
        REPORT_DIR / "target_counts.png",
        ["target=0", "target=1"],
        data[TARGET].value_counts().sort_index().values,
        "Supplied default label counts",
        "Rows",
    )

    block_missing = pd.DataFrame({
        "block": block.to_numpy(),
        "income_missing": data["income"].isna().to_numpy(),
        "utilization_missing": data["credit_utilization"].isna().to_numpy(),
    }).groupby("block").sum()
    block_missing.to_csv(REPORT_DIR / "missingness_by_block.csv")

    summary = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "target_ones": int(data[TARGET].eq(1).sum()),
        "target_one_rate": float(data[TARGET].eq(1).mean()),
        "index_is_unique": bool(data.index.is_unique),
        "duplicate_index_beyond_first": int(data.index.duplicated().sum()),
        "file_blocks": {str(key): int(value) for key, value in block.value_counts().sort_index().items()},
        "model_features": features,
        "excluded_from_models": ["recovery_flag_after_90d"],
        "quality_flags": flags,
        "raw_sha256": sha256_file(RAW_DATA),
        "values_changed": False,
        "rows_dropped": False,
        "split_created": False,
    }
    (REPORT_DIR / "investigation_summary.json").write_text(json.dumps(summary, indent=2))

    report = REPORT_DIR / "investigation_report.md"
    report.write_text(
        "\n".join([
            "# Data investigation",
            "",
            "This is a description of the **raw** 20,000-row file. No split, no imputation, no training.",
            "Recorded values were **not** changed. Age 99 and other flags are left as-is.",
            "",
            f"- Rows: {summary['rows']}",
            f"- Target = 1: {summary['target_ones']} ({100 * summary['target_one_rate']:.2f}%)",
            f"- Unique pandas index: {summary['index_is_unique']} (repeated labels are not treated as customer IDs)",
            f"- Features that will be modelled later: {', '.join(features)}",
            "- Excluded from models: `recovery_flag_after_90d` (timing). `interest_rate` is included now; validation will challenge it.",
            "",
            "## Quality flags (not automatic deletions)",
            "",
            f"- Income missing: {flags['income_missing']}; utilization missing: {flags['utilization_missing']}",
            f"- Income equal to 30,000: {flags['income_equal_30000']}",
            f"- Age equal to 99: {flags['age_equal_99']}",
            f"- Loan/(income×DTI) near 1 / near 10: {flags['dti_ratio_near_one']} / {flags['dti_ratio_near_ten']}",
            f"- 60+ count > 30+ count: {flags['past_due_60_gt_30']}; 90+ > 60+: {flags['past_due_90_gt_60']}",
            f"- Interest rate range: {flags['interest_rate_min']:.6g} to {flags['interest_rate_max']:.6g} (units unverified)",
            f"- Recovery flag = 1: {flags['recovery_flag_positive']} (field still excluded from models)",
            "",
        ])
    )
    print(f"Wrote investigation reports under {REPORT_DIR}")


if __name__ == "__main__":
    main()
