"""Build the static GitHub Pages demo: real training metrics, latency and an Evidently drift report."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn.drift import load_feature_csv, render_report  # noqa: E402

OUT = ROOT / "site"
OUT.mkdir(exist_ok=True)
m = json.loads((ROOT / "reports/training_metrics.json").read_text())
lat = json.loads((ROOT / "reports/latency.json").read_text())
v = ROOT / "models/versions" / m["version"]
html, summary = render_report(load_feature_csv(v / "baseline_current.csv"), load_feature_csv(v / "reference.csv"))
(OUT / "drift.html").write_text(html)
(OUT / ".nojekyll").write_text("")
row = lambda k, val: f"<tr><td>{k}</td><td>{val}</td></tr>"
feats = "".join(row(f["feature"], f"{f['gain']:.3f}") for f in m["top_features"])
drift = "".join(row(k, val) for k, val in summary.items() if not isinstance(val, (dict, list)))
s, c = lat["sequential"], lat["concurrent"]
(OUT / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ml-serving-pipeline demo</title><style>body{{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;color:#111}}table{{border-collapse:collapse;margin:.5rem 0 1.5rem}}td{{border:1px solid #ddd;padding:.3rem .6rem}}code{{background:#f3f3f3;padding:0 .3rem}}</style>
<script type="module">import m from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";m.initialize({{startOnLoad:true}});</script></head><body>
<h1>ml-serving-pipeline</h1><p>Telco churn: Pandera validation, XGBoost tuned with Optuna, MLflow registry, FastAPI serving, MongoDB prediction log, Evidently drift, Docker/K8s.
<a href="https://github.com/naveecode/ml-serving-pipeline">Source on GitHub</a>. Static demo built from committed artifacts by CI; the live API is not hosted.</p>
<h2>Model v{m['version']} (held-out test, n={m['n_test']})</h2><table>{row('ROC AUC', f"{m['roc_auc']:.3f}")}{row('Average precision', f"{m['average_precision']:.3f}")}{row('Precision @0.5', f"{m['precision_at_0_5']:.3f}")}{row('Recall @0.5', f"{m['recall_at_0_5']:.3f}")}{row('Brier', f"{m['brier']:.3f}")}{row('Optuna trials', m['n_trials'])}</table>
<h3>Top features (gain)</h3><table>{feats}</table>
<h2>Latency (/predict, local loopback, one uvicorn worker)</h2><table>{row('Sequential p50 / p95 ms', f"{s['p50_ms']:.1f} / {s['p95_ms']:.1f}")}{row(f"{c['workers']} concurrent clients p50 / p95 ms", f"{c['p50_ms']:.1f} / {c['p95_ms']:.1f}")}</table><p><small>{lat['note']}</small></p>
<h2>Drift (held-out baseline vs training reference)</h2><table>{drift}</table><p><a href="drift.html">Full Evidently report</a></p>
<h2>Architecture</h2><pre class="mermaid">flowchart LR
  csv[Telco CSV] --> val[Pandera] --> train[XGBoost + Optuna] --> mlflow[MLflow registry]
  mlflow --> api[FastAPI /predict /predict/batch /model]
  api --> mongo[(MongoDB prediction log)]
  mongo --> drift[Evidently /drift]
  api --> k8s[Docker / K8s HPA]</pre></body></html>""")
print("site built", summary)
