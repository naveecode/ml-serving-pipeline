"""The scheduled retrain promotes a candidate only when its test ROC AUC is higher."""

from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from churn.registry import empty_registry, load_registry, save_registry
from churn.train import should_promote, train

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"
ROOT = Path(__file__).resolve().parents[1]


def test_should_promote_only_when_auc_is_strictly_higher():
    assert should_promote(0.90, None) is True
    assert should_promote(0.90, 0.85) is True
    assert should_promote(0.85, 0.85) is False
    assert should_promote(0.80, 0.85) is False


def _seed_production(model_dir: Path, roc_auc: float) -> None:
    registry = empty_registry()
    registry["versions"]["1"] = {
        "version": "1",
        "roc_auc": roc_auc,
        "created_at": "2026-01-01T00:00:00+00:00",
        "n_rows": 7043,
        "n_trials": 25,
        "val_roc_auc": roc_auc,
        "mlflow_run_id": None,
        "mlflow_model_version": None,
        "params": {},
    }
    registry["production"] = "1"
    registry["history"] = ["1"]
    save_registry(model_dir, registry)


def _train_against_seed(tmp_path: Path, roc_auc: float) -> tuple[dict, dict]:
    model_dir = tmp_path / "models"
    _seed_production(model_dir, roc_auc)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = train(
        DATA_PATH,
        model_dir,
        n_trials=2,
        sample_rows=500,
        tracking_uri=tracking_uri,
        artifact_dir=tmp_path / "artifacts",
        quick=True,
        promote_if_better=True,
    )
    return result, load_registry(model_dir)


def test_retrain_keeps_the_registered_model_when_auc_does_not_improve(tmp_path):
    result, registry = _train_against_seed(tmp_path, 1.0)
    assert result["registered_roc_auc"] == 1.0
    assert result["promoted"] is False
    assert result["roc_auc"] <= 1.0
    assert registry["production"] == "1"
    assert result["version"] in registry["versions"]
    assert (tmp_path / "models" / "versions" / result["version"] / "pipeline.joblib").exists()

    client = MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("telco-churn", "production")


def test_retrain_promotes_when_auc_beats_the_registered_model(tmp_path):
    result, registry = _train_against_seed(tmp_path, 0.0)
    assert result["registered_roc_auc"] == 0.0
    assert result["roc_auc"] > 0.0
    assert result["promoted"] is True
    assert registry["production"] == result["version"]
    assert registry["production"] != "1"

    client = MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    registered = client.get_model_version_by_alias("telco-churn", "production")
    assert registered.run_id == result["mlflow_run_id"]
    assert str(registered.version) == result["mlflow_model_version"]


def test_retrain_workflow_is_scheduled_and_gates_promotion():
    text = (ROOT / ".github" / "workflows" / "retrain.yml").read_text()
    assert "schedule:" in text
    assert "cron:" in text
    assert "data/Telco-Customer-Churn.csv" in text
    assert "--promote-if-better" in text
    assert "steps.retrain.outputs.promoted == 'true'" in text
