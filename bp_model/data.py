"""Load and clean the cardiovascular disease dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cardio_train.csv"

# Plausible ranges for adults. Anything outside is treated as a data-entry error.
LIMITS = {
    "height": (120, 220),     # cm
    "weight": (30, 200),      # kg
    "ap_hi": (80, 240),       # systolic, mmHg
    "ap_lo": (40, 160),       # diastolic, mmHg
}


@dataclass
class CleaningReport:
    """Keeps track of how many rows each cleaning rule removed."""
    start_rows: int = 0
    removed: dict[str, int] = field(default_factory=dict)

    @property
    def end_rows(self) -> int:
        return self.start_rows - sum(self.removed.values())

    def summary(self) -> str:
        lines = [f"Rows before cleaning: {self.start_rows:,}"]
        lines += [f"  - {rule}: {n:,} removed" for rule, n in self.removed.items()]
        lines.append(f"Rows after cleaning:  {self.end_rows:,}")
        return "\n".join(lines)


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the raw Kaggle CSV (semicolon-separated)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    return pd.read_csv(path, sep=";")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove impossible values and standardise columns.

    Returns the cleaned frame and a report of what was dropped.
    """
    report = CleaningReport(start_rows=len(df))
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])
    report.removed["duplicate rows"] = before - len(df)

    for col, (lo, hi) in LIMITS.items():
        before = len(df)
        df = df[df[col].between(lo, hi)]
        report.removed[f"{col} outside {lo}-{hi}"] = before - len(df)

    before = len(df)
    df = df[df["ap_lo"] < df["ap_hi"]]
    report.removed["diastolic >= systolic"] = before - len(df)

    df = df.assign(
        age_years=(df["age"] / 365.25).round(1),
        is_male=(df["gender"] == 2).astype(int),
    ).drop(columns=["age", "gender"])

    return df.reset_index(drop=True), report
