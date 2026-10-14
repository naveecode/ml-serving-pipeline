from pathlib import Path

import pytest
from pandera.errors import SchemaErrors

from churn.data import load_telco_csv
from churn.samples import EXAMPLE_CUSTOMER
from churn.schema import validate_inference_frame, validate_training_frame

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def test_training_schema_accepts_the_telco_extract():
    frame = load_telco_csv(DATA_PATH)
    validated = validate_training_frame(frame)
    assert len(validated) == 7043
    assert validated["TotalCharges"].isna().sum() == 11


def test_training_schema_rejects_unknown_gender():
    frame = load_telco_csv(DATA_PATH).head(8).copy()
    frame.loc[0, "gender"] = "Other"
    with pytest.raises(SchemaErrors):
        validate_training_frame(frame)


def test_inference_schema_accepts_example_customer():
    import pandas as pd

    validated = validate_inference_frame(pd.DataFrame([EXAMPLE_CUSTOMER]))
    assert list(validated["gender"]) == ["Female"]


def test_inference_schema_reports_missing_columns():
    import pandas as pd

    from churn.schema import InputValidationError

    with pytest.raises(InputValidationError) as caught:
        validate_inference_frame(pd.DataFrame([{"gender": "Female"}]))
    assert "tenure" in caught.value.failures[0]["missing_columns"]
