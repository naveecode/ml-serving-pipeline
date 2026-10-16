"""Keep the README numbers tied to the files produced by training and the latency bench."""

import json
from pathlib import Path

from churn.samples import EXAMPLE_CUSTOMER
from churn.service import ModelService

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_files_from_the_training_run_and_latency_bench():
    metrics = json.loads((ROOT / "reports" / "training_metrics.json").read_text())
    latency = json.loads((ROOT / "reports" / "latency.json").read_text())
    registry = json.loads((ROOT / "models" / "registry.json").read_text())
    version = registry["production"]
    stored = json.loads((ROOT / "models" / "versions" / version / "metrics.json").read_text())
    assert metrics["roc_auc"] == stored["roc_auc"] == registry["versions"][version]["roc_auc"]
    assert metrics["mlflow_run_id"] == registry["versions"][version]["mlflow_run_id"]
    assert latency["sequential_logged"] == latency["sequential"]["requests"]
    assert latency["concurrent_logged"] == latency["concurrent"]["requests"]
    assert latency["model_version"] == version

    readme = (ROOT / "README.md").read_text()
    assert f"{metrics['roc_auc']:.4f}" in readme
    assert f"{metrics['val_roc_auc']:.4f}" in readme
    assert f"{metrics['average_precision']:.4f}" in readme
    assert f"{latency['sequential']['p50_ms']:.2f}" in readme
    assert f"{latency['sequential']['p95_ms']:.2f}" in readme
    assert f"{latency['concurrent']['p95_ms']:.2f}" in readme

    scored, scored_version = ModelService(ROOT / "models").predict([EXAMPLE_CUSTOMER])
    assert scored_version == version
    assert f"{scored[0]['churn_probability']:.4f}" in readme

    hpa = json.loads((ROOT / "reports" / "kind_hpa.json").read_text())
    assert hpa["replicas_ready_max"] > hpa["replicas_ready_before"]
    assert str(hpa["locust_requests"]) in readme
    assert str(hpa["locust_failures"]) in readme
    assert str(hpa["replicas_ready_before"]) in readme
    assert str(hpa["replicas_ready_max"]) in readme
    assert str(hpa["desired_replicas_max"]) in readme
    assert str(hpa["average_utilization_max"]) in readme
    assert f"{hpa['locust_rps']:.2f}" in readme
    assert str(hpa["locust_median_ms"]) in readme
    assert hpa["node_image"] in readme
    assert hpa["metrics_server_image"] in readme
    for endpoint in hpa["locust_endpoints"].values():
        assert str(endpoint["requests"]) in readme
        assert str(endpoint["median_ms"]) in readme
