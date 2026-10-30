"""Load the raw pickle and shared path constants. Values are never edited here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "data.pkl"
CONFIG_DIR = PROJECT_ROOT / "config"
REPORTS = PROJECT_ROOT / "reports"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ASSIGNMENTS_PATH = SPLITS_DIR / "partition_assignments.csv"
TARGET = "default"
EXCLUDED_PREDICTORS = ["recovery_flag_after_90d"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_raw_data(path: Path = RAW_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Symlink or copy the assignment pickle to data/raw/data.pkl."
        )
    data = pd.read_pickle(path)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, received {type(data).__name__}")
    if TARGET not in data.columns:
        raise ValueError(f"Target column {TARGET!r} was not found.")
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Every column except the target and recovery flag."""
    columns = [column for column in data.columns if column != TARGET]
    return [column for column in columns if column not in EXCLUDED_PREDICTORS]


def observed_file_block(index_values) -> pd.Series:
    """Mark concatenated index restarts as audit blocks. These are not dates."""
    original = pd.Index(index_values).to_numpy()
    block = (pd.Series(original).diff().fillna(1) < 0).cumsum() + 1
    return block.astype(int)


def load_membership() -> pd.DataFrame:
    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing split file {ASSIGNMENTS_PATH}")
    return pd.read_csv(ASSIGNMENTS_PATH, dtype={"cv_fold": "Int64"})


def load_partition(data: pd.DataFrame, membership: pd.DataFrame, partition: str) -> pd.DataFrame:
    """Return original row values, indexed by unique physical row id."""
    if partition not in {"development", "test"}:
        raise ValueError("Partition must be development or test.")
    if len(data) != len(membership):
        raise ValueError("Raw data and membership must have the same number of rows.")
    frame = data.reset_index(drop=True)
    ids = membership.loc[membership["partition"].eq(partition), "source_row_id"].to_numpy()
    selected = frame.iloc[ids].copy()
    selected.index = ids
    selected.index.name = "source_row_id"
    return selected
