"""Holdout slice metrics for the committed production model.

The split matches training: validate the IBM sample, then a stratified 80/20
cut with random_state=42. Slice metrics are computed on that test set only.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn.data import load_telco_csv, target_series
from churn.features import align_features
from churn.schema import validate_training_frame

CONTRACT_ORDER = ("Month-to-month", "One year", "Two year")
SENIOR_ORDER = (0, 1)


def _rates(y_true: np.ndarray) -> dict[str, int | float]:
    y_true = np.asarray(y_true)
    return {
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }


def _slice_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, int | float | None]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    labels = (probabilities >= 0.5).astype(int)
    true_negative, false_positive, _false_negative, _true_positive = confusion_matrix(
        y_true, labels, labels=[0, 1]
    ).ravel()
    negatives = int(true_negative + false_positive)
    row: dict[str, int | float | None] = {
        **_rates(y_true),
        "selection_rate": float(labels.mean()) if y_true.size else None,
        "precision_at_0_5": float(precision_score(y_true, labels, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, labels, zero_division=0)),
        "fpr": float(false_positive / negatives) if negatives else None,
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        row["average_precision"] = float(average_precision_score(y_true, probabilities))
    return row


def _group_rows(
    frame: pd.DataFrame,
    target: pd.Series,
    column: str,
    order: tuple[object, ...],
) -> list[dict[str, object]]:
    rows = []
    for value in order:
        mask = frame[column].to_numpy() == value
        rows.append({"column": column, "value": value, **_rates(target.to_numpy()[mask])})
    return rows


def fairness_report(data_path: Path, pipeline_path: Path) -> dict[str, object]:
    """Score the training holdout and slice it by contract and senior citizen."""
    frame = validate_training_frame(load_telco_csv(data_path))
    features = align_features(frame)
    target = target_series(frame)
    x_train, x_test, _y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )
    probabilities = joblib.load(pipeline_path).predict_proba(x_test)[:, 1]
    y_values = y_test.to_numpy()
    slices: list[dict[str, object]] = []
    for value in CONTRACT_ORDER:
        mask = x_test["Contract"].to_numpy() == value
        slices.append(
            {
                "column": "Contract",
                "value": value,
                **_slice_metrics(y_values[mask], probabilities[mask]),
            }
        )
    for value in SENIOR_ORDER:
        mask = x_test["SeniorCitizen"].to_numpy() == value
        slices.append(
            {
                "column": "SeniorCitizen",
                "value": int(value),
                **_slice_metrics(y_values[mask], probabilities[mask]),
            }
        )
    return {
        "n_rows": int(len(frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate": float(target.mean()),
        "roc_auc": float(roc_auc_score(y_values, probabilities)),
        "population": [
            *_group_rows(frame, target, "Contract", CONTRACT_ORDER),
            *_group_rows(frame, target, "SeniorCitizen", SENIOR_ORDER),
        ],
        "slices": slices,
    }
