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
