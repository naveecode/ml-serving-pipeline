#!/usr/bin/env bash
# Build the API image, load it into a local kind cluster, and hit /health, /predict, and /model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLUSTER="${CLUSTER:-churn}"
IMAGE="${IMAGE:-churn-api:local}"
PORT="${PORT:-18000}"

docker build -t "$IMAGE" .
if kind get clusters | grep -qx "$CLUSTER"; then
  kind delete cluster --name "$CLUSTER"
fi
kind create cluster --name "$CLUSTER"
kind load docker-image "$IMAGE" --name "$CLUSTER"
kubectl apply -k k8s/overlays/kind
kubectl rollout status deployment/churn-api --timeout=180s
kubectl get deploy,svc,hpa

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
echo "kind smoke passed"
