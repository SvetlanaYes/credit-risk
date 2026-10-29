"""Small matplotlib helpers for investigation and split reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar(path: Path, labels, values, title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar([str(label) for label in labels], values, color="#4c6a92")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_grouped_bars(path: Path, table: pd.DataFrame, title: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    table.plot(kind="bar", ax=axis, rot=0)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend(title=None)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
