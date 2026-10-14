"""Load the production pipeline and swap it on rollback."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn.features import align_features
from churn.registry import (
    apply_production,
    load_registry,
    previous_version,
    save_registry,
    sync_production_alias,
    version_dir,
)
from churn.schema import validate_inference_frame


class ModelNotReady(Exception):
    """Raised when no production model has been registered."""


class ModelService:
    def __init__(self, model_dir: Path, threshold: float = 0.5, tracking_uri: str | None = None):
        self.model_dir = model_dir
        self._threshold = threshold
        self._tracking_uri = tracking_uri
        self._lock = threading.Lock()
        self._pipeline = None
        self._version: str | None = None

    def ensure_loaded(self) -> None:
        if self._pipeline is None:
            self.reload()

    def reload(self) -> str:
        registry = load_registry(self.model_dir)
        version = registry.get("production")
        if not version:
            raise ModelNotReady("no production model is registered")
        pipeline = joblib.load(version_dir(self.model_dir, version) / "pipeline.joblib")
        with self._lock:
            self._pipeline = pipeline
            self._version = version
        return version

    @property
    def production_version(self) -> str:
        self.ensure_loaded()
        assert self._version is not None
        return self._version

    def predict(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        self.ensure_loaded()
        frame = pd.DataFrame.from_records(records)
        validated = validate_inference_frame(frame)
        aligned = align_features(validated)
        with self._lock:
            pipeline = self._pipeline
            version = self._version
            threshold = self._threshold
        probabilities = pipeline.predict_proba(aligned)[:, 1]
        rows = []
        for probability in probabilities:
            score = float(probability)
            rows.append({"churn_probability": score, "churn": bool(score >= threshold)})
        return rows, str(version)

    def describe(self) -> dict[str, Any]:
        registry = load_registry(self.model_dir)
        production = registry.get("production")
        versions = []
        for version in sorted(registry["versions"], key=int):
            meta = dict(registry["versions"][version])
            meta["role"] = "production" if version == production else "available"
            versions.append(meta)
        payload: dict[str, Any] = {
            "model_name": registry["model_name"],
            "production_version": production,
            "versions": versions,
        }
        if production:
            metrics_path = version_dir(self.model_dir, production) / "metrics.json"
            if metrics_path.exists():
                payload["production_metrics"] = json.loads(metrics_path.read_text())
        return payload

    def rollback(self, version: str | None = None) -> dict[str, Any]:
        registry = load_registry(self.model_dir)
        target = previous_version(registry) if version is None else version
        apply_production(registry, target)
        save_registry(self.model_dir, registry)
        self.reload()
        mlflow_version = registry["versions"][target].get("mlflow_model_version")
        synced = False
        if self._tracking_uri:
            synced = sync_production_alias(self._tracking_uri, mlflow_version)
        description = self.describe()
        description["mlflow_alias_synced"] = synced
        return description

    def reference_paths(self) -> tuple[Path, Path]:
        version = self.production_version
        directory = version_dir(self.model_dir, version)
        return directory / "reference.csv", directory / "baseline_current.csv"
