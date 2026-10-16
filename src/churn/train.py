"""Train an XGBoost churn model, tune it with Optuna, and register it in MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.data import load_telco_csv, target_series
from churn.features import align_features, build_preprocessor
from churn.registry import (
    MODEL_NAME,
    apply_production,
    load_registry,
    next_version,
    save_registry,
    version_dir,
)
from churn.schema import validate_training_frame

logger = logging.getLogger(__name__)
THRESHOLD = 0.5
REFERENCE_CAP = 2000
BASELINE_CAP = 1000


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def scale_pos_weight(y: pd.Series) -> float:
    positives = max(int((y == 1).sum()), 1)
    negatives = int((y == 0).sum())
    return negatives / positives


def build_params(trial: optuna.Trial, *, quick: bool) -> dict[str, Any]:
    if quick:
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 20, 40),
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 80, 350),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
    }


def make_xgb(
    params: dict[str, Any],
    *,
    pos_weight: float,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    kwargs: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
        "scale_pos_weight": pos_weight,
        **params,
    }
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**kwargs)


def _fit_early_stopped(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any],
    *,
    early_stopping_rounds: int,
) -> tuple[float, int]:
    preprocessor = build_preprocessor()
    preprocessor.fit(x_fit)
    model = make_xgb(
        params,
        pos_weight=scale_pos_weight(y_fit),
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        preprocessor.transform(x_fit),
        y_fit,
        eval_set=[(preprocessor.transform(x_val), y_val)],
        verbose=False,
    )
    best_n = params["n_estimators"]
    if model.best_iteration is not None:
        best_n = int(model.best_iteration) + 1
    probabilities = model.predict_proba(preprocessor.transform(x_val))[:, 1]
    return float(roc_auc_score(y_val, probabilities)), int(best_n)


def fit_final_pipeline(
    features: pd.DataFrame, target: pd.Series, params: dict[str, Any]
) -> Pipeline:
    """Fit a preprocessor and booster on the full training frame."""
    preprocessor = build_preprocessor()
    preprocessor.fit(features)
    model = make_xgb(params, pos_weight=scale_pos_weight(target))
    model.fit(preprocessor.transform(features), target, verbose=False)
    return Pipeline([("prep", preprocessor), ("model", model)])


def _cap(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed)


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    labels = (probabilities >= THRESHOLD).astype(int)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "precision_at_0_5": float(precision_score(y_true, labels, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, labels, zero_division=0)),
    }


def registered_roc_auc(model_dir: Path) -> float | None:
    """ROC AUC of the file-registry production model, or None when nothing is registered."""
    registry = load_registry(model_dir)
    version = registry.get("production")
    if not version:
        return None
    raw = registry["versions"][version].get("roc_auc")
    if raw is None:
        raise ValueError(f"production version {version} has no roc_auc to compare")
    return float(raw)


def should_promote(candidate_auc: float, registered_auc: float | None) -> bool:
    """Promote when no model is registered, or the candidate test ROC AUC is strictly higher."""
    if registered_auc is None:
        return True
    return float(candidate_auc) > float(registered_auc)


def top_features(pipeline: Pipeline, limit: int = 10) -> list[dict[str, float | str]]:
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())
    scores = model.feature_importances_
    ranked = sorted(zip(names, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "gain": float(score)} for name, score in ranked[:limit]]


def persist_version(
    *,
    model_dir: Path,
    pipeline: Pipeline,
    metrics: dict[str, Any],
    params: dict[str, Any],
    reference: pd.DataFrame,
    baseline_current: pd.DataFrame,
    promote: bool = True,
) -> str:
    registry = load_registry(model_dir)
    version = next_version(registry)
    destination = version_dir(model_dir, version)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination / "pipeline.joblib")
    reference.to_csv(destination / "reference.csv", index=False)
    baseline_current.to_csv(destination / "baseline_current.csv", index=False)
    payload = _jsonable({**metrics, "params": params, "version": version})
    (destination / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry["versions"][version] = {
        "version": version,
        "created_at": metrics["created_at"],
        "roc_auc": payload["roc_auc"],
        "val_roc_auc": payload.get("val_roc_auc"),
        "n_rows": payload["n_rows"],
        "n_trials": payload["n_trials"],
        "mlflow_run_id": payload.get("mlflow_run_id"),
        "mlflow_model_version": payload.get("mlflow_model_version"),
        "params": payload["params"],
    }
    if promote or not registry["production"]:
        apply_production(registry, version)
    save_registry(model_dir, registry)
    return version


def _ensure_experiment(name: str, artifact_dir: Path) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=str(artifact_dir.resolve()))
    mlflow.set_experiment(name)


def train(
    data_path: Path,
    model_dir: Path,
    *,
    n_trials: int = 25,
    sample_rows: int | None = None,
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_dir: Path = Path("mlruns"),
    quick: bool = False,
    promote: bool = True,
    promote_if_better: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """Validate, tune on an inner split, score once on a held-out test set, and register."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    frame = validate_training_frame(
        load_telco_csv(data_path, sample_rows=sample_rows, random_state=random_state)
    )
    features = align_features(frame)
    target = target_series(frame)
    if target.nunique() < 2:
        raise ValueError("training sample contains a single class")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state,
    )
    stopping_rounds = 5 if quick else 20

    def objective(trial: optuna.Trial) -> float:
        params = build_params(trial, quick=quick)
        score, best_n = _fit_early_stopped(
            x_fit,
            y_fit,
            x_val,
            y_val,
            params,
            early_stopping_rounds=stopping_rounds,
        )
        trial.set_user_attr("params", params)
        trial.set_user_attr("best_n_estimators", best_n)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = dict(study.best_trial.user_attrs["params"])
    best_n = int(study.best_trial.user_attrs["best_n_estimators"])
    final_params = {**best_params, "n_estimators": best_n}
    pipeline = fit_final_pipeline(x_train, y_train, final_params)
    test_probabilities = pipeline.predict_proba(x_test)[:, 1]
    scored = classification_metrics(y_test, test_probabilities)
    registered_auc = registered_roc_auc(model_dir)
    if promote_if_better:
        promote = should_promote(scored["roc_auc"], registered_auc)
    has_production = registered_auc is not None
    will_promote = bool(promote or not has_production)

    import mlflow
    from mlflow.models import infer_signature

    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment("telco-churn", artifact_dir)
    with mlflow.start_run(run_name=f"telco-rows-{len(frame)}-trials-{n_trials}") as run:
        mlflow.set_tag("dataset", "telco-customer-churn")
        mlflow.log_params({key: value for key, value in final_params.items()})
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_rows", len(frame))
        mlflow.log_param("quick", quick)
        mlflow.log_metric("val_roc_auc", float(study.best_value))
        for name, value in scored.items():
            mlflow.log_metric(name, value)
        signature = infer_signature(x_train.head(20), test_probabilities[:20])
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=x_train.head(2),
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        model_version = info.registered_model_version
        run_id = run.info.run_id
    if will_promote and model_version is not None:
        from mlflow.tracking import MlflowClient

        MlflowClient(tracking_uri).set_registered_model_alias(
            MODEL_NAME, "production", str(model_version)
        )

    from datetime import datetime

    metrics: dict[str, Any] = {
        **scored,
        "val_roc_auc": float(study.best_value),
        "threshold": THRESHOLD,
        "n_rows": int(len(frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_trials": int(n_trials),
        "positive_rate": float(target.mean()),
        "best_n_estimators": best_n,
        "scale_pos_weight": scale_pos_weight(y_train),
        "mlflow_run_id": run_id,
        "mlflow_model_version": None if model_version is None else str(model_version),
        "mlflow_tracking_uri": tracking_uri,
        "created_at": datetime.now(UTC).isoformat(),
        "quick": quick,
        "top_features": top_features(pipeline),
        "registered_roc_auc": registered_auc,
        "promoted": will_promote,
    }
    version = persist_version(
        model_dir=model_dir,
        pipeline=pipeline,
        metrics=metrics,
        params=final_params,
        reference=_cap(x_train, REFERENCE_CAP, random_state),
        baseline_current=_cap(x_test, BASELINE_CAP, random_state),
        promote=will_promote,
    )
    metrics["version"] = version
    logger.info("registered version %s test ROC AUC %.4f", version, metrics["roc_auc"])
    return _jsonable(metrics)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train and register the telco churn model")
    parser.add_argument("--data", type=Path, default=Path("data/Telco-Customer-Churn.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument(
        "--promote-if-better",
        action="store_true",
        help="Point production at this run only when its test ROC AUC exceeds the registered model",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.promote_if_better and args.no_promote:
        parser.error("--promote-if-better cannot be combined with --no-promote")
    result = train(
        args.data,
        args.model_dir,
        n_trials=args.n_trials,
        sample_rows=args.sample_rows,
        tracking_uri=args.tracking_uri,
        artifact_dir=args.artifact_dir,
        quick=args.quick,
        promote=not args.no_promote,
        promote_if_better=args.promote_if_better,
    )
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
