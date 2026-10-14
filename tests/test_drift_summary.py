from churn.drift import summarize_snapshot


def test_summarize_snapshot_reads_drift_counts():
    summary = summarize_snapshot(
        {
            "metrics": [
                {
                    "config": {
                        "type": "evidently:metric_v2:DriftedColumnsCount",
                        "drift_share": 0.5,
                    },
                    "value": {"count": 2.0, "share": 0.5},
                },
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "tenure",
                        "method": "K-S p_value",
                        "threshold": 0.05,
                    },
                    "value": 0.01,
                },
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": "gender",
                        "method": "Z-test p_value",
                        "threshold": 0.05,
                    },
                    "value": 0.4,
                },
            ]
        }
    )
    assert summary["drifted_columns"] == 2.0
    assert summary["drift_share"] == 0.5
    assert summary["columns"][0]["drifted"] is True
    assert summary["columns"][1]["drifted"] is False
