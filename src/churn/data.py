"""Load the telco churn CSV into a frame the schema can check."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.schema import TARGET_COLUMN


def load_telco_csv(
    path: Path,
    *,
    sample_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    if sample_rows is not None and sample_rows < len(frame):
        frame = frame.sample(n=sample_rows, random_state=random_state)
    return frame.reset_index(drop=True)


def target_series(frame: pd.DataFrame) -> pd.Series:
    return (frame[TARGET_COLUMN] == "Yes").astype(int)
