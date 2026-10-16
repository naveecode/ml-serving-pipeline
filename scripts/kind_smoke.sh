#!/usr/bin/env bash
# Build the API image, load it into a local kind cluster with metrics-server,
# check /health, /predict, and /model, then run the in-cluster Locust job and
# record whether the HPA scaled.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER="${CLUSTER:-churn}"
IMAGE="${IMAGE:-churn-api:local}"
PORT="${PORT:-18000}"
NODE_IMAGE="${NODE_IMAGE:-kindest/node:v1.33.1}"
METRICS_IMAGE="${METRICS_IMAGE:-registry.k8s.io/metrics-server/metrics-server:v0.9.0}"
MONGO_IMAGE="${MONGO_IMAGE:-mongo:7.0}"
LOCUST_IMAGE="${LOCUST_IMAGE:-locustio/locust:2.46.7}"
REPORT="${REPORT:-reports/kind_hpa.json}"
SAMPLES="${SAMPLES:-/tmp/churn-hpa-samples.jsonl}"

docker build -t "$IMAGE" .
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind delete cluster --name "$CLUSTER"
fi
kind create cluster --name "$CLUSTER" --config k8s/kind-config.yaml

# Images pulled into Docker's containerd image store do not survive `kind load`
# (ctr import reports a missing blob). A local `docker build` still loads.
# On that failure, export the filesystem and import a single-layer image that
# keeps the original entrypoint, then load that.
load_into_kind() {
  local image="$1"
  if kind load docker-image "$image" --name "$CLUSTER"; then
    return 0
  fi
  echo "flattening ${image} after kind load failed" >&2
  python3 - "$image" <<'PY'
import json
import subprocess
import sys

image = sys.argv[1]
config = json.loads(subprocess.check_output(["docker", "inspect", image], text=True))[0]["Config"]
container = subprocess.check_output(["docker", "create", image], text=True).strip()
changes = []
if config.get("User"):
    changes.extend(["-c", f"USER {config['User']}"])
if config.get("WorkingDir"):
    changes.extend(["-c", f"WORKDIR {config['WorkingDir']}"])
if config.get("Entrypoint"):
    changes.extend(["-c", "ENTRYPOINT " + json.dumps(config["Entrypoint"])])
if config.get("Cmd"):
    changes.extend(["-c", "CMD " + json.dumps(config["Cmd"])])
env = config.get("Env") or []
if env:
    pairs = []
    for item in env:
        key, value = item.split("=", 1)
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        pairs.append(f'{key}="{escaped}"')
    changes.extend(["-c", "ENV " + " ".join(pairs)])
flat = image + "-flat"
export = subprocess.Popen(["docker", "export", container], stdout=subprocess.PIPE)
try:
    subprocess.check_call(["docker", "import", *changes, "-", flat], stdin=export.stdout)
finally:
    if export.stdout is not None:
        export.stdout.close()
    export.wait()
subprocess.check_call(["docker", "rm", container])
subprocess.check_call(["docker", "tag", flat, image])
PY
  kind load docker-image "$image" --name "$CLUSTER"
}

load_into_kind "$IMAGE"
load_into_kind "$METRICS_IMAGE"
load_into_kind "$MONGO_IMAGE"
load_into_kind "$LOCUST_IMAGE"

kubectl apply -f k8s/metrics-server/components.yaml
kubectl apply -k k8s/overlays/kind
kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s
kubectl rollout status deployment/mongodb --timeout=180s
kubectl rollout status deployment/churn-api --timeout=180s
kubectl get deploy,svc,hpa

metrics_ready=0
for _ in $(seq 1 36); do
  if kubectl top pods -l app=churn-api --no-headers 2>/dev/null | grep -q '[0-9]'; then
    metrics_ready=1
    break
  fi
  sleep 5
done
if [[ "$metrics_ready" -ne 1 ]]; then
  echo "metrics-server did not report churn-api CPU" >&2
  kubectl -n kube-system logs deployment/metrics-server --tail=80 >&2 || true
  exit 1
fi

kubectl port-forward svc/churn-api "$PORT:80" >/tmp/churn-port-forward.log 2>&1 &
FORWARD_PID=$!
cleanup() {
  kill "$FORWARD_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health"; then
    echo
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/model" | head -c 400
echo
curl -fsS -X POST "http://127.0.0.1:${PORT}/predict" \
  -H 'content-type: application/json' \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
echo

sample_once() {
  python3 - "$SAMPLES" <<'PY'
import json
import subprocess
import sys
from datetime import UTC, datetime

destination = sys.argv[1]

def kubectl(*args: str) -> str:
    return subprocess.check_output(["kubectl", *args], text=True)

hpa = json.loads(kubectl("get", "hpa", "churn-api", "-o", "json"))
pods = json.loads(kubectl("get", "pods", "-l", "app=churn-api", "-o", "json"))
ready = 0
for pod in pods["items"]:
    conditions = {item["type"]: item["status"] for item in pod["status"].get("conditions", [])}
    if conditions.get("Ready") == "True":
        ready += 1
try:
    top = kubectl("top", "pods", "-l", "app=churn-api", "--no-headers").strip()
except subprocess.CalledProcessError:
    top = ""
status = hpa.get("status") or {}
metrics = status.get("currentMetrics") or []
utilization = None
if metrics:
    current = (metrics[0].get("resource") or {}).get("current") or {}
    utilization = current.get("averageUtilization")
row = {
    "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "ready": ready,
    "current_replicas": status.get("currentReplicas"),
    "desired_replicas": status.get("desiredReplicas"),
    "average_utilization": utilization,
    "top": top,
}
with open(destination, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row) + "\n")
print(json.dumps(row))
PY
}

: >"$SAMPLES"
sample_once
kubectl delete job locust --ignore-not-found
kubectl apply -f k8s/overlays/kind/locust-job.yaml

for _ in $(seq 1 36); do
  sample_once
  succeeded="$(kubectl get job locust -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$(kubectl get job locust -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  if [[ "$succeeded" == "1" ]]; then
    break
  fi
  if [[ -n "$failed" && "$failed" != "0" ]]; then
    echo "locust job failed" >&2
    kubectl logs job/locust >&2 || true
    exit 1
  fi
  sleep 10
done
sample_once
kubectl logs job/locust > /tmp/churn-locust.log

python3 - "$SAMPLES" /tmp/churn-locust.log "$REPORT" "$NODE_IMAGE" "$METRICS_IMAGE" "$LOCUST_IMAGE" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

samples_path, log_path, report_path, node_image, metrics_image, locust_image = sys.argv[1:]
samples = [json.loads(line) for line in Path(samples_path).read_text().splitlines() if line.strip()]
log = Path(log_path).read_text(errors="replace")
stat_row = (
    r"(\d+)\s+(\d+)\([0-9.]+%\)\s+\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+([0-9.]+)\s+[0-9.]+"
)
aggregated = re.findall(r"Aggregated\s+" + stat_row, log)
if not aggregated:
    raise SystemExit("locust summary had no Aggregated row")
requests, failures, _avg, _min, _max, median, rps = aggregated[-1]
endpoints = {}
for label, pattern in (
    ("GET /health", r"GET\s+/health\s+" + stat_row),
    ("POST /predict", r"POST\s+/predict\s+" + stat_row),
    ("POST /predict/batch", r"POST\s+/predict/batch\s+" + stat_row),
):
    rows = re.findall(pattern, log)
    if not rows:
        continue
    row_requests, row_failures, row_avg, row_min, row_max, row_median, row_rps = rows[-1]
    endpoints[label] = {
        "requests": int(row_requests),
        "failures": int(row_failures),
        "median_ms": int(row_median),
        "rps": float(row_rps),
    }
ready_values = [int(sample["ready"]) for sample in samples]
desired_values = [
    int(sample["desired_replicas"])
    for sample in samples
    if sample.get("desired_replicas") is not None
]
utilizations = [
    int(sample["average_utilization"])
    for sample in samples
    if sample.get("average_utilization") is not None
]
kind_version = subprocess.check_output(["kind", "version"], text=True).strip()
kubelet = subprocess.check_output(
    ["kubectl", "get", "nodes", "-o", "jsonpath={.items[0].status.nodeInfo.kubeletVersion}"],
    text=True,
).strip()
job = json.loads(
    subprocess.check_output(["kubectl", "get", "job", "locust", "-o", "json"], text=True)
)
args = job["spec"]["template"]["spec"]["containers"][0]["args"]
report = {
    "kind_version": kind_version,
    "node_image": node_image,
    "kubelet_version": kubelet,
    "metrics_server_image": metrics_image,
    "locust_image": locust_image,
    "locust_args": args,
    "locust_requests": int(requests),
    "locust_failures": int(failures),
    "locust_rps": float(rps),
    "locust_median_ms": int(median),
    "locust_endpoints": endpoints,
    "replicas_ready_before": ready_values[0],
    "replicas_ready_max": max(ready_values),
    "desired_replicas_max": max(desired_values) if desired_values else None,
    "average_utilization_max": max(utilizations) if utilizations else None,
    "samples": samples,
    "locust_log_tail": "\n".join(log.splitlines()[-25:]),
}
destination = Path(report_path)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + "\n")
print(
    f"ready {report['replicas_ready_before']} -> {report['replicas_ready_max']}"
    f" desired_max={report['desired_replicas_max']}"
    f" util_max={report['average_utilization_max']}"
    f" requests={report['locust_requests']} failures={report['locust_failures']}"
)
if report["replicas_ready_max"] <= report["replicas_ready_before"]:
    raise SystemExit("HPA did not increase ready replicas")
PY

echo "kind smoke passed"
