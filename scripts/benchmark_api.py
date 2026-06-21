from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class EndpointCase:
    name: str
    method: str
    path: str
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class Measurement:
    endpoint: str
    status_code: int
    duration_ms: float
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkRun:
    measurements: list[Measurement]
    elapsed_seconds: float


def benchmark_cases() -> list[EndpointCase]:
    return [
        EndpointCase(name="ready", method="GET", path="/ready"),
        EndpointCase(
            name="predict",
            method="POST",
            path="/predict",
            payload={
                "station_no": "500101002",
                "bikes_available": 8,
                "temperature": 33,
                "rain": 3,
            },
        ),
        EndpointCase(
            name="risk",
            method="POST",
            path="/stations/risk",
            payload={
                "temperature": 25,
                "rain": 0,
                "stations": [
                    {
                        "station_no": "500101002",
                        "bikes_available": 5,
                        "spaces_available": 15,
                    },
                    {
                        "station_no": "500101003",
                        "bikes_available": 18,
                        "spaces_available": 2,
                    },
                    {
                        "station_no": "500101004",
                        "bikes_available": 10,
                        "spaces_available": 10,
                    },
                ],
            },
        ),
    ]


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def request_once(
    base_url: str,
    case: EndpointCase,
    request_id: str,
    timeout: float,
) -> Measurement:
    url = f"{base_url.rstrip('/')}{case.path}"
    body = None
    headers = {"X-Request-ID": request_id}
    if case.payload is not None:
        body = json.dumps(case.payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=case.method,
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            status_code = response.status
            ok = 200 <= status_code < 400
            error = None
    except urllib.error.HTTPError as exc:
        exc.read()
        status_code = exc.code
        ok = False
        error = f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        status_code = 0
        ok = False
        error = str(exc.reason)

    duration_ms = (time.perf_counter() - start) * 1000
    return Measurement(
        endpoint=case.name,
        status_code=status_code,
        duration_ms=duration_ms,
        ok=ok,
        error=error,
    )


def summarize(
    measurements: list[Measurement],
    elapsed_seconds: float | None = None,
) -> list[dict[str, Any]]:
    summaries = []
    endpoints = sorted({measurement.endpoint for measurement in measurements})
    for endpoint in endpoints:
        endpoint_measurements = [
            measurement for measurement in measurements if measurement.endpoint == endpoint
        ]
        durations = [measurement.duration_ms for measurement in endpoint_measurements]
        error_count = sum(1 for measurement in endpoint_measurements if not measurement.ok)
        endpoint_elapsed_seconds = elapsed_seconds
        if endpoint_elapsed_seconds is None:
            endpoint_elapsed_seconds = sum(durations) / 1000
        throughput_rps = (
            len(endpoint_measurements) / endpoint_elapsed_seconds
            if endpoint_elapsed_seconds and endpoint_elapsed_seconds > 0
            else 0.0
        )
        summaries.append(
            {
                "endpoint": endpoint,
                "requests": len(endpoint_measurements),
                "errors": error_count,
                "error_rate_percent": (
                    error_count / len(endpoint_measurements) * 100
                    if endpoint_measurements
                    else 0.0
                ),
                "throughput_rps": throughput_rps,
                "min_ms": min(durations) if durations else 0.0,
                "avg_ms": statistics.fmean(durations) if durations else 0.0,
                "p50_ms": percentile(durations, 0.50),
                "p95_ms": percentile(durations, 0.95),
                "p99_ms": percentile(durations, 0.99),
                "max_ms": max(durations) if durations else 0.0,
            }
        )
    return summaries


def format_markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| endpoint | requests | errors | error % | rps | min ms | avg ms | p50 ms | p95 ms | p99 ms | max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| {endpoint} | {requests} | {errors} | {error_rate_percent:.2f} | "
            "{throughput_rps:.2f} | {min_ms:.2f} | {avg_ms:.2f} | {p50_ms:.2f} | "
            "{p95_ms:.2f} | {p99_ms:.2f} | {max_ms:.2f} |".format(**summary)
        )
    return "\n".join(lines)


def run_benchmark(
    base_url: str,
    requests_per_endpoint: int,
    warmup_requests: int,
    timeout: float,
    concurrency: int,
) -> BenchmarkRun:
    cases = benchmark_cases()

    for case in cases:
        for index in range(warmup_requests):
            request_once(
                base_url=base_url,
                case=case,
                request_id=f"benchmark-warmup-{case.name}-{index}",
                timeout=timeout,
            )

    measurements = []
    recorded_requests = [
        (case, index)
        for index in range(requests_per_endpoint)
        for case in cases
    ]
    start = time.perf_counter()
    if concurrency == 1:
        for case, index in recorded_requests:
            measurements.append(
                request_once(
                    base_url=base_url,
                    case=case,
                    request_id=f"benchmark-{case.name}-{index}",
                    timeout=timeout,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    request_once,
                    base_url=base_url,
                    case=case,
                    request_id=f"benchmark-{case.name}-{index}",
                    timeout=timeout,
                )
                for case, index in recorded_requests
            ]
            for future in concurrent.futures.as_completed(futures):
                measurements.append(future.result())

    elapsed_seconds = time.perf_counter() - start
    return BenchmarkRun(measurements=measurements, elapsed_seconds=elapsed_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small local latency/concurrency benchmark against the YouBike FastAPI service."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--requests", type=int, default=20, help="Recorded requests per endpoint.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests per endpoint.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent workers for recorded requests.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-request timeout in seconds.")
    parser.add_argument("--json-output", type=Path, help="Optional path for raw benchmark results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests <= 0:
        raise SystemExit("--requests must be greater than 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be greater than or equal to 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be greater than 0")

    result = run_benchmark(
        base_url=args.base_url,
        requests_per_endpoint=args.requests,
        warmup_requests=args.warmup,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    summaries = summarize(result.measurements, elapsed_seconds=result.elapsed_seconds)
    print(format_markdown_table(summaries))

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "requests_per_endpoint": args.requests,
                    "warmup_requests": args.warmup,
                    "concurrency": args.concurrency,
                    "elapsed_seconds": result.elapsed_seconds,
                    "summary": summaries,
                    "measurements": [asdict(measurement) for measurement in result.measurements],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return 1 if any(not measurement.ok for measurement in result.measurements) else 0


if __name__ == "__main__":
    raise SystemExit(main())
