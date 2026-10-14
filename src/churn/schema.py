"""Pandera schemas for the IBM telco customer churn table."""

from __future__ import annotations

import math
from typing import Any

import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

YES_NO = ["Yes", "No"]
INTERNET_ADDON = ["Yes", "No", "No internet service"]

NUM_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CAT_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
INFER_COLUMNS = NUM_COLUMNS + CAT_COLUMNS
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def _feature_columns() -> dict[str, pa.Column]:
    return {
        "gender": pa.Column(str, pa.Check.isin(["Female", "Male"])),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": pa.Column(str, pa.Check.isin(YES_NO)),
        "Dependents": pa.Column(str, pa.Check.isin(YES_NO)),
        "tenure": pa.Column(int, pa.Check.in_range(0, 100)),
        "PhoneService": pa.Column(str, pa.Check.isin(YES_NO)),
        "MultipleLines": pa.Column(str, pa.Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": pa.Column(str, pa.Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "OnlineBackup": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "DeviceProtection": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "TechSupport": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "StreamingTV": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "StreamingMovies": pa.Column(str, pa.Check.isin(INTERNET_ADDON)),
        "Contract": pa.Column(str, pa.Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": pa.Column(str, pa.Check.isin(YES_NO)),
        "PaymentMethod": pa.Column(
            str,
            pa.Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True),
    }


def _schema(columns: dict[str, pa.Column]) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(columns, coerce=True, strict=False)


INFER_SCHEMA = _schema(_feature_columns())
TRAIN_SCHEMA = _schema(
    {
        ID_COLUMN: pa.Column(str),
        **_feature_columns(),
        TARGET_COLUMN: pa.Column(str, pa.Check.isin(YES_NO)),
    }
)


class InputValidationError(Exception):
    """Raised when a scoring payload fails the Pandera schema."""

    def __init__(self, failures: list[dict[str, Any]]):
        super().__init__("input failed validation")
        self.failures = failures


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def failure_payload(exc: Exception) -> list[dict[str, Any]]:
    if isinstance(exc, SchemaErrors):
        records = exc.failure_cases.head(20).to_dict(orient="records")
        return [{key: _json_safe(val) for key, val in row.items()} for row in records]
    if isinstance(exc, SchemaError):
        return [{"error": str(exc).splitlines()[0]}]
    return [{"error": str(exc)}]


def validate_training_frame(frame):
    return TRAIN_SCHEMA.validate(frame, lazy=True)


def validate_inference_frame(frame):
    missing = [column for column in INFER_COLUMNS if column not in frame.columns]
    if missing:
        raise InputValidationError([{"missing_columns": missing}])
    try:
        return INFER_SCHEMA.validate(frame.loc[:, INFER_COLUMNS], lazy=True)
    except (SchemaError, SchemaErrors) as exc:
        raise InputValidationError(failure_payload(exc)) from exc
