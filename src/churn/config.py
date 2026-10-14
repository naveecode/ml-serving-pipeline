"""Runtime settings loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    model_dir: Path = Path("models")
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "churn"
    mongodb_collection: str = "predictions"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    predict_threshold: float = 0.5
    drift_min_rows: int = 50
    mongo_client: Any = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            mongodb_db=os.getenv("MONGODB_DB", "churn"),
            mongodb_collection=os.getenv("MONGODB_COLLECTION", "predictions"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            predict_threshold=float(os.getenv("PREDICT_THRESHOLD", "0.5")),
            drift_min_rows=int(os.getenv("DRIFT_MIN_ROWS", "50")),
        )
