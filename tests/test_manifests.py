from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "k8s" / "base"
OVERLAY = Path(__file__).resolve().parents[1] / "k8s" / "overlays" / "kind"


def test_base_manifests_include_deployment_service_and_hpa():
    text = "\n".join(path.read_text() for path in sorted(BASE.glob("*.yaml")))
    assert "kind: Deployment" in text
    assert "kind: Service" in text
    assert "kind: HorizontalPodAutoscaler" in text
    assert "kind: Secret" in text
    assert "name: cpu" in text
    assert "/health" in text


def test_kind_overlay_points_at_base():
    text = (OVERLAY / "kustomization.yaml").read_text()
    assert "../../base" in text
    assert "IfNotPresent" in (OVERLAY / "patch-deploy.yaml").read_text()


def test_kind_overlay_runs_mongodb_and_the_script_installs_metrics_server():
    overlay = "\n".join(path.read_text() for path in sorted(OVERLAY.glob("*.yaml")))
    assert "name: mongodb" in overlay
    assert "27017" in overlay
    root = OVERLAY.parents[2]
    metrics = (root / "k8s" / "metrics-server" / "components.yaml").read_text()
    assert "metrics-server:v0.9.0" in metrics
    assert "--kubelet-insecure-tls" in metrics
    script = (root / "scripts" / "kind_smoke.sh").read_text()
    assert "k8s/kind-config.yaml" in script
    assert "kubeProxyMode: nftables" in (root / "k8s" / "kind-config.yaml").read_text()
    assert "k8s/metrics-server/components.yaml" in script
    assert "k8s/overlays/kind/locust-job.yaml" in script
    job = (OVERLAY / "locust-job.yaml").read_text()
    assert "locustio/locust:2.46.7" in job
    assert "http://churn-api" in job
