"""Column sets and the preprocessing pipeline shared by train and serve."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn.schema import CAT_COLUMNS, INFER_COLUMNS, NUM_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="constant", fill_value=0.0))])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_COLUMNS),
            ("cat", categorical, CAT_COLUMNS),
        ]
    )


def align_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model inputs in training column order, with blank totals filled."""
    aligned = frame.loc[:, INFER_COLUMNS].copy()
    aligned["TotalCharges"] = pd.to_numeric(aligned["TotalCharges"], errors="coerce").fillna(0.0)
    return aligned
