"""Evidently data-drift report comparing live scores with the training reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn.schema import CAT_COLUMNS, INFER_COLUMNS, NUM_COLUMNS


def load_feature_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return prepare_frame(frame)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.loc[:, INFER_COLUMNS].copy()
    for column in NUM_COLUMNS:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    aligned["TotalCharges"] = aligned["TotalCharges"].fillna(0.0)
    aligned["SeniorCitizen"] = aligned["SeniorCitizen"].fillna(0).astype(int)
    aligned["tenure"] = aligned["tenure"].fillna(0).astype(int)
    for column in CAT_COLUMNS:
        aligned[column] = aligned[column].astype(str)
    return aligned


def summarize_snapshot(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
    drifted_columns = None
    drift_share = None
    columns: list[dict[str, Any]] = []
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config") or {}
        metric_type = config.get("type", "")
        if metric_type.endswith("DriftedColumnsCount"):
            value = metric.get("value") or {}
            drifted_columns = value.get("count")
            drift_share = value.get("share")
            continue
        if metric_type.endswith("ValueDrift"):
            threshold = config.get("threshold")
            p_value = metric.get("value")
            drifted = None
            if isinstance(p_value, (int, float)) and isinstance(threshold, (int, float)):
                drifted = bool(p_value < threshold)
            columns.append(
                {
                    "column": config.get("column"),
                    "method": config.get("method"),
                    "threshold": threshold,
                    "p_value": p_value,
                    "drifted": drifted,
                }
            )
    return {
        "drifted_columns": drifted_columns,
        "drift_share": drift_share,
        "columns": columns,
    }


def render_report(current: pd.DataFrame, reference: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    report = Report([DataDriftPreset()])
    snapshot = report.run(current, reference, name="telco-churn-drift")
    html = snapshot.get_html_str(as_iframe=False)
    summary = summarize_snapshot(snapshot.dict())
    return html, summary
