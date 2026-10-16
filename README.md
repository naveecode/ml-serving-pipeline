# Telco churn scoring service

**Live demo (static, GitHub Pages):** https://naveecode.github.io/ml-serving-pipeline/

A small production-style path for the IBM telco customer churn table: Pandera checks the frame, Optuna tunes an XGBoost classifier, MLflow tracks the run and registers it, and FastAPI serves single and batch scores. Predictions are written to MongoDB. `/drift` builds an Evidently data-drift report from the training reference and either recent logged scores or the held-out baseline. The same image runs under Docker Compose, a kustomize bundle (Deployment, Service, HPA), and the Render and Fly configs.

## Measured results

Numbers below are copied from `reports/training_metrics.json` and `reports/latency.json`. Classification metrics are shown to 4 decimal places. Latencies are shown to 2 decimal places. Both files keep full precision.

Training run `e6cfd563cfaf4940920c02d4f240a9ee` (MLflow model version 1), 2026-10-14. Stratified 80/20 holdout, `random_state=42`, 7,043 rows (5,634 train / 1,409 test). Optuna TPE, 25 trials, early stopping on an inner 20% of the training split. The test set was scored once. Decision threshold is 0.5 and was not tuned. Positive rate in the full table is 0.2654.

| Metric | Value |
| --- | --- |
| Test ROC AUC | 0.8459 |
| Inner-validation ROC AUC (tuning objective) | 0.8570 |
| Average precision | 0.6587 |
| Precision at 0.5 | 0.5122 |
| Recall at 0.5 | 0.7888 |
| Log loss | 0.4844 |
| Brier score | 0.1631 |

The selected booster has `max_depth` 3 and 45 trees. Full params and gain importances are in the metrics file.

Latency is loopback HTTP to one local uvicorn process on this machine (Intel Xeon, 4 vCPU, Python 3.12.3), model version 1, with each `/predict` also inserting into MongoDB 7 at `mongodb://127.0.0.1:27017`. Every timed request was logged (`sequential_logged` 300, `concurrent_logged` 320). Measured 2026-10-14T13:42:56Z. This is not a production SLO.

| Setup | Requests | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| One client, in flight one at a time | 300 | 28.51 ms | 35.15 ms | 43.36 ms |
| 8 clients, 40 requests each | 320 | 305.60 ms | 401.39 ms | 513.17 ms |

The API process handles requests synchronously, so the 8-client figure is mostly queueing on one worker. Repeat the measurement with `python scripts/bench_latency.py` against a running server and replace both JSON files and this section together. `tests/test_published_metrics.py` fails if the README drifts from the files.

Slice metrics, base rates, and the limits of this fit are in `MODEL_CARD.md`. `tests/test_model_card.py` recomputes the holdout from the production pipeline and fails if the card drifts.

Worked example, first CSV row (customer `7590-VHVEG`), model version 1: churn probability **0.8539**, label `true` at threshold 0.5.

## Architecture

```mermaid
flowchart LR
  csv[Telco CSV] --> schema[Pandera schema]
  schema --> tune[XGBoost + Optuna]
  tune --> mlflow[MLflow tracking and registry]
  tune --> versions[Versioned model files]
  client[Client] --> api[FastAPI]
  versions --> api
  api --> mongo[(MongoDB or Atlas)]
  mongo --> drift[Evidently /drift]
  versions --> drift
```

Serving reads the file registry under `models/`, not the MLflow server. Training registers the pipeline and points the MLflow alias `production` at that version. Rollback updates the file registry and best-effort syncs the alias when `mlflow.db` is reachable. If MongoDB is down, scoring still returns and `logged` is `false`.

## Quickstart

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
uvicorn churn.api:app --host 0.0.0.0 --port 8000
```

Score the example customer:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```

OpenAPI is at `/docs`. With MongoDB from Compose:

```bash
docker compose up --build
```

That starts the API on port 8000, MongoDB on 27017, and an MLflow server on 5000. The image already contains model version 1.

Retrain and register a new production version:

```bash
python -m churn.train --data data/Telco-Customer-Churn.csv --n-trials 25
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlflow.db` and `mlruns/` are local and gitignored. The run id above is the one that produced the committed model.

`.github/workflows/retrain.yml` runs every Monday at 06:17 UTC, and on manual dispatch. It retrains on `data/Telco-Customer-Churn.csv` with the same 25-trial command and passes `--promote-if-better`. The file-registry production pointer and the MLflow `production` alias move only when the new test ROC AUC is strictly higher than the registered model's. An equal score keeps the registered model. When the run does promote, the job commits `models/` and pushes that commit.

## API

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Process is up and a production model is loaded |
| `POST` | `/predict` | One customer. Body is the telco fields except `customerID` and `Churn` |
| `POST` | `/predict/batch` | `{"records": [ ... ]}`, 1 to 1000 rows |
| `GET` | `/model` | Production version, prior versions, and the production metrics file |
| `POST` | `/model/rollback` | `{}` restores the previous production version, or `{"version": "1"}` selects one. Unknown versions return 404. A single-version registry returns 409 |
| `GET` | `/drift` | Evidently HTML report. `?format=json` returns drifted-column count, share, and per-column p-values |

Pydantic rejects a malformed body with 422. Pandera runs again inside the scorer, so the same checks apply to any caller of `ModelService.predict`.

`/drift` uses the last 1,000 logged feature payloads when at least `DRIFT_MIN_ROWS` (default 50) are stored. Otherwise it compares the training reference with the held-out baseline saved next to the model and labels the window `baseline_holdout`.

## Kubernetes

```bash
docker build -t churn-api:local .
kind create cluster --name churn --config k8s/kind-config.yaml
kind load docker-image churn-api:local --name churn
kubectl apply -f k8s/metrics-server/components.yaml
kubectl apply -k k8s/overlays/kind
kubectl rollout status deployment/churn-api
kubectl port-forward svc/churn-api 8000:80
```

`scripts/kind_smoke.sh` does that sequence. `k8s/kind-config.yaml` sets kube-proxy to nftables and the node image to `kindest/node:v1.33.1`. The script installs metrics-server v0.9.0 from `k8s/metrics-server/components.yaml` (`--kubelet-insecure-tls`, which kind needs for the kubelet certificate), loads the image into the node, and waits until `kubectl top` returns CPU for `churn-api`. The kind overlay also starts MongoDB 7.0 as Service `mongodb` on port 27017, which is the URI already in the Secret, so `/predict` can log instead of waiting out a failed connection. The script then calls `/health`, `/model`, and `/predict`, and applies `k8s/overlays/kind/locust-job.yaml`: a Locust 2.46.7 Job, 40 users, spawn rate 10, 3 minutes, host `http://churn-api`.

The base bundle is a Deployment (2 replicas, CPU request so the HPA can scale), a ClusterIP Service, an HPA at 70% CPU (min 2, max 8), and a Secret for `MONGODB_URI`. The kind overlay pins 1 replica and an HPA range of 1–3. Swap the Secret for an Atlas URI using `deploy/atlas.md` when this bundle leaves kind.

The measured kind run is in [Kind HPA run](#kind-hpa-run). `reports/kind_hpa.json` is what that script wrote, and `tests/test_published_metrics.py` checks that this section quotes it.

## Kind HPA run

`scripts/kind_smoke.sh` wrote `reports/kind_hpa.json` on 2026-10-16. kind v0.29.0, kubelet v1.33.1, node image `kindest/node:v1.33.1`, kube-proxy in nftables mode, metrics-server `registry.k8s.io/metrics-server/metrics-server:v0.9.0`. The Locust 2.46.7 job used 40 users, spawn rate 10, and a 3 minute run against `http://churn-api`.

| | |
| --- | ---: |
| Requests | 3834 |
| Failures | 0 |
| Final aggregate rate | 21.32 req/s |
| Aggregate median | 1800 ms |
| `GET /health` | 425 requests, median 59 ms |
| `POST /predict` | 2500 requests, median 1900 ms |
| `POST /predict/batch` | 909 requests, median 1900 ms |

Ready replicas were 1 before the job. At 02:08:48Z the HPA set desired replicas to 3, with average CPU at 292% of the 250m request and the only pod at its 1000m limit. Three pods were Ready at 02:08:58Z. The peak average utilization in the samples was 400% at 02:09:08Z. From 02:09:29Z through the last sample at 02:11:12Z the average stayed at 133% and desired stayed at 3: `kubectl top` showed one pod at 1000m and the other two at 2m. 133% is above the 70% target, so the deployment was still at 3 ready replicas when the job stopped. Every sample is in the JSON file.

## Load test

```bash
locust -f locust/locustfile.py --host http://127.0.0.1:8000
```

The locust file posts the same example customer used above, plus a batch of two rows, and checks `/health`. The kind script runs the same tasks from `k8s/overlays/kind/locust-job.yaml` inside the cluster, with a shorter wait so the pods can cross the HPA CPU target.

## Deploy

Render: `render.yaml` is a Docker web service with health check `/health`. Set `MONGODB_URI` in the Render dashboard. The blueprint marks it `sync: false`, so the file does not contain a database password.

Fly: `fly.toml` listens on 8000 and checks `/health`. The app name `telco-churn-api` is a placeholder. Set the secret before the first deploy:

```bash
fly secrets set MONGODB_URI='mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority'
fly deploy
```

Atlas steps, network access, and the Kubernetes secret live in `deploy/atlas.md`. Local defaults are in `.env.example`.

## Tests and CI

```bash
pytest
ruff check src tests locust scripts
ruff format --check src tests locust scripts
```

GitHub Actions runs those checks and `docker build` on every push and pull request. A separate scheduled workflow, `.github/workflows/retrain.yml`, retrains on the sample and promotes only when test ROC AUC improves. Pytest covers the Pandera schema on the full 7,043-row file, the preprocessor, registry rollback, the API with mongomock (predict, batch, rollback, drift), a Mongo-down path that still scores, a 500-row / 2-trial training run that must beat 0.5 ROC AUC and register an MLflow `production` alias, the promote-if-better gate against a seeded registry, and the model-card slice numbers.

## Layout

```text
src/churn/          schema, training, registry, API, drift, fairness slices
data/               IBM telco extract and source note
models/versions/1/  committed pipeline, metrics, drift reference
reports/            training metrics, latency bench, kind HPA run
k8s/                kustomize base, kind overlay, metrics-server
locust/             load test
MODEL_CARD.md       data, metrics, limitations, contract and senior slices
deploy/atlas.md     Atlas connection notes
```

Dataset source and checksum: `data/SOURCE.md`.
