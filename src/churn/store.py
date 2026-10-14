"""Prediction log backed by MongoDB. Writes fail open so scoring still returns."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class PredictionStore:
    def __init__(self, client: Any, database: str, collection: str):
        self._col = client[database][collection]
        self._indexed = False

    def _ensure_index(self) -> None:
        if self._indexed:
            return
        self._indexed = True
        try:
            self._col.create_index("logged_at")
        except Exception:
            logger.debug("prediction log index was not created", exc_info=True)

    def log(self, route: str, version: str, records: list[dict], results: list[dict]) -> bool:
        now = datetime.now(UTC)
        request_id = str(uuid4())
        documents = [
            {
                "request_id": request_id,
                "logged_at": now,
                "model_version": version,
                "route": route,
                "features": record,
                "churn_probability": result["churn_probability"],
                "churn": result["churn"],
            }
            for record, result in zip(records, results, strict=True)
        ]
        if not documents:
            return True
        self._ensure_index()
        try:
            self._col.insert_many(documents, ordered=False)
            return True
        except Exception:
            logger.warning("prediction log write failed", exc_info=True)
            return False

    def recent_features(self, limit: int) -> list[dict]:
        self._ensure_index()
        try:
            cursor = (
                self._col.find({}, {"features": 1, "_id": 0}).sort("logged_at", -1).limit(limit)
            )
            return [
                document["features"]
                for document in cursor
                if isinstance(document.get("features"), dict)
            ]
        except Exception:
            logger.warning("prediction log read failed", exc_info=True)
            return []
