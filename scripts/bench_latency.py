"""Measure loopback HTTP latency against a running churn API.

Writes reports/latency.json. This is a single-machine measurement, not a production SLO.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import httpx
import numpy as np

from churn.samples import EXAMPLE_CUSTOMER


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), pct, method="linear"))


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "requests": len(samples),
        "min_ms": _percentile(samples, 0),
        "p50_ms": _percentile(samples, 50),
        "p95_ms": _percentile(samples, 95),
        "p99_ms": _percentile(samples, 99),
        "max_ms": _percentile(samples, 100),
        "mean_ms": sum(samples) / len(samples),
    }


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _sequential(client: httpx.Client, url: str, count: int) -> tuple[list[float], int]:
    samples: list[float] = []
    logged = 0
    for _ in range(count):
        started = time.perf_counter()
        response = client.post(url, json=EXAMPLE_CUSTOMER)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        if response.json().get("logged") is True:
            logged += 1
        samples.append(elapsed_ms)
    return samples, logged


def _concurrent(url: str, workers: int, each: int) -> tuple[list[float], int]:
    def run() -> tuple[list[float], int]:
        with httpx.Client(timeout=30) as client:
            return _sequential(client, url, each)

    samples: list[float] = []
    logged = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch, batch_logged in pool.map(lambda _: run(), range(workers)):
            samples.extend(batch)
            logged += batch_logged
    return samples, logged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--each", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("reports/latency.json"))
    parser.add_argument(
        "--mongodb-uri",
        default="mongodb://127.0.0.1:27017",
        help="Recorded for provenance. The script does not open MongoDB itself.",
    )
    args = parser.parse_args()

    health = urlopen(f"{args.base_url}/health", timeout=10)
    model_version = json.loads(health.read())["model_version"]
    url = f"{args.base_url}/predict"
    with httpx.Client(timeout=30) as client:
        _sequential(client, url, args.warmup)
        sequential, sequential_logged = _sequential(client, url, args.requests)
    loaded, loaded_logged = _concurrent(url, args.workers, args.each)
    payload = {
        "measured_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "endpoint": "/predict",
        "model_version": model_version,
        "mongodb_uri": args.mongodb_uri,
        "sequential_logged": sequential_logged,
        "concurrent_logged": loaded_logged,
        "warmup_requests": args.warmup,
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "note": "Loopback HTTP against one local uvicorn process. Not a production SLO.",
        "sequential": _summary(sequential),
        "concurrent": {
            "workers": args.workers,
            "requests_per_worker": args.each,
            **_summary(loaded),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
