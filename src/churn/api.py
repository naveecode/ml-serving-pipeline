"""FastAPI scoring service: predict, batch, model rollback, and drift."""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pymongo import MongoClient

from churn import __version__
from churn.config import Settings
from churn.drift import load_feature_csv, prepare_frame, render_report
from churn.registry import NoRollbackTarget
from churn.samples import EXAMPLE_CUSTOMER
from churn.schema import InputValidationError
from churn.service import ModelNotReady, ModelService
from churn.store import PredictionStore


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictBatch(BaseModel):
    records: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class RollbackRequest(BaseModel):
    version: str | None = None


def _mongo_client(settings: Settings):
    if settings.mongo_client is not None:
        return settings.mongo_client
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=1500)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Telco Churn API",
        version=__version__,
        summary="Score customer churn, roll back the production model, and inspect drift.",
    )
    app.state.settings = settings
    app.state.service = ModelService(
        settings.model_dir,
        threshold=settings.predict_threshold,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    app.state.store = PredictionStore(
        _mongo_client(settings),
        settings.mongodb_db,
        settings.mongodb_collection,
    )

    @app.exception_handler(InputValidationError)
    async def _invalid_input(_request: Request, exc: InputValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.failures})

    @app.exception_handler(ModelNotReady)
    async def _not_ready(_request: Request, exc: ModelNotReady):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "telco-churn",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict/batch",
            "model": "/model",
            "drift": "/drift",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": app.state.service.production_version}

    @app.post("/predict")
    def predict(body: CustomerFeatures) -> dict:
        record = body.model_dump()
        results, version = app.state.service.predict([record])
        logged = app.state.store.log("/predict", version, [record], results)
        return {"model_version": version, "logged": logged, **results[0]}

    @app.post("/predict/batch")
    def predict_batch(body: PredictBatch) -> dict:
        records = [item.model_dump() for item in body.records]
        results, version = app.state.service.predict(records)
        logged = app.state.store.log("/predict/batch", version, records, results)
        return {"model_version": version, "logged": logged, "predictions": results}

    @app.get("/model")
    def model_info() -> dict:
        return app.state.service.describe()

    @app.post("/model/rollback")
    def rollback(body: RollbackRequest) -> dict:
        try:
            return app.state.service.rollback(body.version)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown model version '{exc.args[0]}'"
            ) from exc
        except NoRollbackTarget as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/drift")
    def drift(
        report_format: Literal["html", "json"] = Query(default="html", alias="format"),
    ):
        service: ModelService = app.state.service
        settings: Settings = app.state.settings
        reference_path, baseline_path = service.reference_paths()
        if not reference_path.exists():
            raise HTTPException(status_code=409, detail="training reference for drift is missing")
        reference = load_feature_csv(reference_path)
        logged = app.state.store.recent_features(limit=1000)
        if len(logged) >= settings.drift_min_rows:
            current = prepare_frame(pd.DataFrame.from_records(logged))
            source = "prediction_log"
        elif baseline_path.exists():
            current = load_feature_csv(baseline_path)
            source = "baseline_holdout"
        else:
            raise HTTPException(status_code=409, detail="no current window is available for drift")
        html, summary = render_report(current, reference)
        banner = (
            f'<p id="drift-window">source={source}; '
            f"model_version={service.production_version}; "
            f"reference_rows={len(reference)}; current_rows={len(current)}</p>"
        )
        if report_format == "json":
            return {
                "source": source,
                "model_version": service.production_version,
                "reference_rows": len(reference),
                "current_rows": len(current),
                **summary,
            }
        if "<body>" in html:
            html = html.replace("<body>", "<body>" + banner, 1)
        else:
            html = banner + html
        return HTMLResponse(html)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("churn.api:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
