"""Build the static GitHub Pages demo.

Uses the committed training metrics, latency numbers and an Evidently drift report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.drift import load_feature_csv, render_report  # noqa: E402

OUT = ROOT / "site"


def row(key: object, value: object) -> str:
    return f"<tr><td>{key}</td><td>{value}</td></tr>"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m = json.loads((ROOT / "reports/training_metrics.json").read_text())
    lat = json.loads((ROOT / "reports/latency.json").read_text())
    version_dir = ROOT / "models/versions" / m["version"]
    current = load_feature_csv(version_dir / "baseline_current.csv")
    reference = load_feature_csv(version_dir / "reference.csv")
    html, summary = render_report(current, reference)
    (OUT / "drift.html").write_text(html)
    (OUT / ".nojekyll").write_text("")

    seq, conc = lat["sequential"], lat["concurrent"]
    metrics = [
        ("ROC AUC", f"{m['roc_auc']:.3f}"),
        ("Average precision", f"{m['average_precision']:.3f}"),
        ("Precision @0.5", f"{m['precision_at_0_5']:.3f}"),
        ("Recall @0.5", f"{m['recall_at_0_5']:.3f}"),
        ("Brier", f"{m['brier']:.3f}"),
        ("Optuna trials", m["n_trials"]),
    ]
    latency = [
        ("Sequential p50 / p95 ms", f"{seq['p50_ms']:.1f} / {seq['p95_ms']:.1f}"),
        (
            f"{conc['workers']} concurrent clients p50 / p95 ms",
            f"{conc['p50_ms']:.1f} / {conc['p95_ms']:.1f}",
        ),
    ]
    drift_rows = [
        ("Drifted columns", int(summary["drifted_columns"])),
        ("Drift share", f"{summary['drift_share']:.3f}"),
    ]
    page = Template((ROOT / "scripts/site_template.html").read_text()).substitute(
        version=m["version"],
        n_test=m["n_test"],
        metrics="".join(row(k, v) for k, v in metrics),
        features="".join(row(f["feature"], f"{f['gain']:.3f}") for f in m["top_features"]),
        latency="".join(row(k, v) for k, v in latency),
        latency_note=lat["note"],
        drift="".join(row(k, v) for k, v in drift_rows),
    )
    (OUT / "index.html").write_text(page)
    print("site built:", dict(drift_rows))


if __name__ == "__main__":
    main()
