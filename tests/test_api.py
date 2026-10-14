from churn.schema import InputValidationError


def test_health_and_model_list_two_versions(api):
    client, _mongo = api
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_version"] == "2"

    described = client.get("/model")
    assert described.status_code == 200
    body = described.json()
    assert body["production_version"] == "2"
    assert [item["version"] for item in body["versions"]] == ["1", "2"]
    assert body["production_metrics"]["n_trials"] == 0
    assert 0.0 < body["production_metrics"]["roc_auc"] <= 1.0


def test_predict_logs_to_mongo_and_matches_threshold(api, example_customer):
    client, mongo = api
    response = client.post("/predict", json=example_customer)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "2"
    assert body["logged"] is True
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn"] is (body["churn_probability"] >= 0.5)

    stored = list(mongo["churn"]["predictions"].find())
    assert len(stored) == 1
    assert stored[0]["model_version"] == "2"
    assert stored[0]["route"] == "/predict"
    assert stored[0]["features"]["tenure"] == example_customer["tenure"]
    assert stored[0]["churn_probability"] == body["churn_probability"]


def test_batch_logs_each_row(api, example_customer):
    client, mongo = api
    other = dict(example_customer)
    other["tenure"] = 72
    other["Contract"] = "Two year"
    response = client.post("/predict/batch", json={"records": [example_customer, other]})
    assert response.status_code == 200
    body = response.json()
    assert body["logged"] is True
    assert len(body["predictions"]) == 2
    assert mongo["churn"]["predictions"].count_documents({}) == 2


def test_invalid_payload_is_422(api):
    client, _mongo = api
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422
    empty = client.post("/predict/batch", json={"records": []})
    assert empty.status_code == 422


def test_pandera_runs_on_the_scoring_path(api, example_customer):
    import pytest

    client, _mongo = api
    bad = dict(example_customer)
    bad["gender"] = "Nope"
    with pytest.raises(InputValidationError):
        client.app.state.service.predict([bad])


def test_rollback_switches_production_version(api, example_customer):
    client, mongo = api
    rolled = client.post("/model/rollback", json={})
    assert rolled.status_code == 200
    assert rolled.json()["production_version"] == "1"
    assert rolled.json()["mlflow_alias_synced"] is False

    scored = client.post("/predict", json=example_customer)
    assert scored.status_code == 200
    assert scored.json()["model_version"] == "1"
    assert mongo["churn"]["predictions"].find_one()["model_version"] == "1"

    missing = client.post("/model/rollback", json={"version": "99"})
    assert missing.status_code == 404
    restored = client.post("/model/rollback", json={})
    assert restored.status_code == 200
    assert restored.json()["production_version"] == "2"


def test_drift_uses_holdout_then_prediction_log(api, example_customer):
    client, _mongo = api
    baseline = client.get("/drift", params={"format": "json"})
    assert baseline.status_code == 200
    body = baseline.json()
    assert body["source"] == "baseline_holdout"
    assert body["reference_rows"] > 0
    assert body["current_rows"] > 0
    assert "drifted_columns" in body
    assert isinstance(body["columns"], list)

    html = client.get("/drift")
    assert html.status_code == 200
    assert "text/html" in html.headers["content-type"]
    assert 'id="drift-window"' in html.text
    assert "source=baseline_holdout" in html.text

    for tenure in (1, 12, 40):
        payload = dict(example_customer)
        payload["tenure"] = tenure
        assert client.post("/predict", json=payload).status_code == 200

    live = client.get("/drift", params={"format": "json"})
    assert live.status_code == 200
    assert live.json()["source"] == "prediction_log"
    assert live.json()["current_rows"] == 3


def test_predict_still_scores_when_mongo_is_down(tmp_path, example_customer):
    from fastapi.testclient import TestClient

    from churn.api import create_app
    from churn.config import Settings
    from tests.conftest import _register_fitted_version

    model_dir = tmp_path / "models"
    _register_fitted_version(model_dir, seed=3)
    settings = Settings(
        model_dir=model_dir,
        mongodb_uri="mongodb://127.0.0.1:1",
        mongodb_db="churn",
        mongodb_collection="predictions",
        drift_min_rows=50,
    )
    client = TestClient(create_app(settings))
    response = client.post("/predict", json=example_customer)
    assert response.status_code == 200
    assert response.json()["logged"] is False
    assert 0.0 <= response.json()["churn_probability"] <= 1.0
    denied = client.post("/model/rollback", json={})
    assert denied.status_code == 409
