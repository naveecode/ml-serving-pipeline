"""File-backed model registry. MLflow aliases are updated when the tracking store is reachable."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

MODEL_NAME = "telco-churn"
logger = logging.getLogger(__name__)


class NoRollbackTarget(Exception):
    """Raised when rollback is requested and only one version exists."""


def empty_registry() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "production": None,
        "history": [],
        "versions": {},
    }


def registry_path(model_dir: Path) -> Path:
    return model_dir / "registry.json"


def load_registry(model_dir: Path) -> dict[str, Any]:
    path = registry_path(model_dir)
    if not path.exists():
        return empty_registry()
    return json.loads(path.read_text())


def save_registry(model_dir: Path, registry: dict[str, Any]) -> None:
    path = registry_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def next_version(registry: dict[str, Any]) -> str:
    if not registry["versions"]:
        return "1"
    return str(max(int(version) for version in registry["versions"]) + 1)


def apply_production(registry: dict[str, Any], version: str) -> dict[str, Any]:
    if version not in registry["versions"]:
        raise KeyError(version)
    registry["production"] = version
    registry["history"].append(version)
    return registry


def previous_version(registry: dict[str, Any]) -> str:
    current = registry["production"]
    for version in reversed(registry["history"]):
        if version != current and version in registry["versions"]:
            return version
    others = [version for version in registry["versions"] if version != current]
    if not others:
        raise NoRollbackTarget("no prior version to roll back to")
    return sorted(others, key=int)[-1]


def version_dir(model_dir: Path, version: str) -> Path:
    return model_dir / "versions" / version


def sync_production_alias(tracking_uri: str, model_version: str | None) -> bool:
    """Point the MLflow `production` alias at a registry version. Failure is non-fatal."""
    if not model_version:
        return False
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        MlflowClient(tracking_uri).set_registered_model_alias(
            MODEL_NAME, "production", str(model_version)
        )
        return True
    except Exception:
        logger.warning("MLflow alias sync failed", exc_info=True)
        return False
