from pathlib import Path

from churn.data import load_telco_csv
from churn.features import align_features, build_preprocessor
from churn.schema import INFER_COLUMNS, validate_training_frame

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def test_preprocessor_expands_categoricals():
    frame = validate_training_frame(load_telco_csv(DATA_PATH).head(40))
    features = align_features(frame)
    transformed = build_preprocessor().fit_transform(features)
    assert transformed.shape[0] == 40
    assert transformed.shape[1] > len(INFER_COLUMNS)


def test_align_fills_blank_total_charges():
    frame = validate_training_frame(load_telco_csv(DATA_PATH))
    blank = frame[frame["TotalCharges"].isna()].head(1)
    aligned = align_features(blank)
    assert aligned["TotalCharges"].iloc[0] == 0.0
    assert list(aligned.columns) == INFER_COLUMNS
