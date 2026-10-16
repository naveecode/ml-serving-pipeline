"""The model card quotes the production model's holdout, including contract and senior slices."""

import json
from pathlib import Path

from churn.fairness import fairness_report

ROOT = Path(__file__).resolve().parents[1]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def test_model_card_matches_the_production_holdout():
    version = json.loads((ROOT / "models" / "registry.json").read_text())["production"]
    metrics = json.loads((ROOT / "models" / "versions" / version / "metrics.json").read_text())
    report = fairness_report(
        ROOT / "data" / "Telco-Customer-Churn.csv",
        ROOT / "models" / "versions" / version / "pipeline.joblib",
    )
    card = (ROOT / "MODEL_CARD.md").read_text()
    source = (ROOT / "data" / "SOURCE.md").read_text()

    assert report["roc_auc"] == metrics["roc_auc"]
    assert report["n_rows"] == metrics["n_rows"]
    assert report["n_test"] == metrics["n_test"]
    assert card.count("Limitations") >= 1
    sha = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
    assert sha in source
    assert sha in card

    for key in (
        "roc_auc",
        "average_precision",
        "precision_at_0_5",
        "recall_at_0_5",
        "log_loss",
        "brier",
    ):
        assert _fmt(metrics[key]) in card
    assert _fmt(report["positive_rate"]) in card
    assert str(report["n_rows"]) in card
    assert str(report["n_test"]) in card

    for row in [*report["population"], *report["slices"]]:
        assert str(row["value"]) in card
        assert str(row["n"]) in card
        assert str(row["n_positive"]) in card
        assert _fmt(row["positive_rate"]) in card
    for row in report["slices"]:
        for key in (
            "selection_rate",
            "precision_at_0_5",
            "recall_at_0_5",
            "fpr",
            "roc_auc",
            "average_precision",
        ):
            assert _fmt(row[key]) in card, (row["column"], row["value"], key)
