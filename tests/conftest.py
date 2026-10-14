"""Shared fixtures. API tests get a real tiny booster, not a mock score."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import mongomock
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from churn.api import create_app
from churn.config import Settings
from churn.data import load_telco_csv, target_series
from churn.features import align_features
from churn.samples import EXAMPLE_CUSTOMER
from churn.schema import validate_training_frame
from churn.train import classification_metrics, fit_final_pipeline, persist_version

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
FAST_PARAMS = {
    "max_depth": 2,
    "learning_rate": 0.1,
    "n_estimators": 15,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
}


def _register_fitted_version(model_dir: Path, seed: int) -> None:
    frame = validate_training_frame(load_telco_csv(DATA_PATH, sample_rows=400, random_state=seed))
    features = align_features(frame)
    target = target_series(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=seed,
    )
    pipeline = fit_final_pipeline(x_train, y_train, FAST_PARAMS)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = classification_metrics(y_test, probabilities)
    metrics.update(
        {
            "val_roc_auc": metrics["roc_auc"],
            "threshold": 0.5,
            "n_rows": int(len(frame)),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "n_trials": 0,
            "positive_rate": float(target.mean()),
            "best_n_estimators": FAST_PARAMS["n_estimators"],
            "scale_pos_weight": 1.0,
            "mlflow_run_id": None,
            "mlflow_model_version": None,
            "created_at": datetime.now(UTC).isoformat(),
            "quick": True,
            "top_features": [],
        }
    )
    persist_version(
        model_dir=model_dir,
        pipeline=pipeline,
        metrics=metrics,
        params=FAST_PARAMS,
        reference=x_train,
        baseline_current=x_test,
        promote=True,
    )


@pytest.fixture
def api(tmp_path: Path):
    model_dir = tmp_path / "models"
    _register_fitted_version(model_dir, seed=1)
    _register_fitted_version(model_dir, seed=2)
    mongo = mongomock.MongoClient()
    settings = Settings(
        model_dir=model_dir,
        mongo_client=mongo,
        mongodb_db="churn",
        mongodb_collection="predictions",
        drift_min_rows=3,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        predict_threshold=0.5,
    )
    client = TestClient(create_app(settings))
    return client, mongo


@pytest.fixture
def example_customer() -> dict:
    return dict(EXAMPLE_CUSTOMER)
