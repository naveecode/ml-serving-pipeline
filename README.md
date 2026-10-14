# ml-serving-pipeline
Production-style MLOps pipeline for a tabular model (customer churn).

**Goal:** show the full ML lifecycle, not just a notebook: data validation, training + Optuna tuning, MLflow tracking/registry, FastAPI batch+online serving, prediction logging to MongoDB, drift monitoring (Evidently), Docker, Kubernetes (kind) and GitHub Actions CI/CD.

**Scope (v1):** XGBoost churn model · `/predict` and `/predict/batch` · model versioning + rollback · drift report endpoint · K8s manifests + HPA · load test (Locust).

Status: scaffolding. Built in public, roadmap in issues.
