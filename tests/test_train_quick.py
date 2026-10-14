from pathlib import Path

from mlflow.tracking import MlflowClient

from churn.train import train

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def test_quick_training_registers_a_model(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = train(
        DATA_PATH,
        tmp_path / "models",
        n_trials=2,
        sample_rows=500,
        tracking_uri=tracking_uri,
        artifact_dir=tmp_path / "artifacts",
        quick=True,
    )
    assert result["n_rows"] == 500
    assert result["n_trials"] == 2
    assert result["n_test"] == 100
    assert 0.5 < result["roc_auc"] <= 1.0
    assert 0.5 < result["val_roc_auc"] <= 1.0
    version_dir = tmp_path / "models" / "versions" / result["version"]
    assert (version_dir / "pipeline.joblib").exists()
    assert (version_dir / "reference.csv").exists()
    assert (version_dir / "metrics.json").exists()

    client = MlflowClient(tracking_uri)
    registered = client.get_model_version_by_alias("telco-churn", "production")
    assert registered.run_id == result["mlflow_run_id"]
    assert str(registered.version) == result["mlflow_model_version"]
