import pytest

from scripts import benchmark_api


def test_percentile_interpolates_values():
    assert benchmark_api.percentile([10, 20, 30, 40], 0.50) == 25
    assert benchmark_api.percentile([10, 20, 30, 40], 0.95) == 38.5


def test_summarize_groups_measurements_by_endpoint():
    measurements = [
        benchmark_api.Measurement("ready", 200, 10.0, True),
        benchmark_api.Measurement("ready", 200, 20.0, True),
        benchmark_api.Measurement("predict", 500, 50.0, False, "HTTP 500"),
    ]

    summary = benchmark_api.summarize(measurements, elapsed_seconds=2.0)

    assert summary[1]["p99_ms"] == pytest.approx(19.9)
    summary[1]["p99_ms"] = 19.9

    assert summary == [
        {
            "endpoint": "predict",
            "requests": 1,
            "errors": 1,
            "error_rate_percent": 100.0,
            "throughput_rps": 0.5,
            "min_ms": 50.0,
            "avg_ms": 50.0,
            "p50_ms": 50.0,
            "p95_ms": 50.0,
            "p99_ms": 50.0,
            "max_ms": 50.0,
        },
        {
            "endpoint": "ready",
            "requests": 2,
            "errors": 0,
            "error_rate_percent": 0.0,
            "throughput_rps": 1.0,
            "min_ms": 10.0,
            "avg_ms": 15.0,
            "p50_ms": 15.0,
            "p95_ms": 19.5,
            "p99_ms": 19.9,
            "max_ms": 20.0,
        },
    ]


def test_format_markdown_table_includes_latency_columns():
    table = benchmark_api.format_markdown_table(
        [
            {
                "endpoint": "ready",
                "requests": 2,
                "errors": 0,
                "error_rate_percent": 0.0,
                "throughput_rps": 10.0,
                "min_ms": 1.0,
                "avg_ms": 2.0,
                "p50_ms": 2.0,
                "p95_ms": 3.0,
                "p99_ms": 3.8,
                "max_ms": 4.0,
            }
        ]
    )

    assert (
        "| endpoint | requests | errors | error % | rps | min ms | avg ms | "
        "p50 ms | p95 ms | p99 ms | max ms |"
    ) in table
    assert "| ready | 2 | 0 | 0.00 | 10.00 | 1.00 | 2.00 | 2.00 | 3.00 | 3.80 | 4.00 |" in table


def test_assess_summaries_marks_pass_watch_and_fail():
    profile = benchmark_api.BenchmarkProfile(
        name="test",
        description="test profile",
        requests_per_endpoint=1,
        warmup_requests=0,
        concurrency=1,
        timeout=1.0,
        p95_warn_ms=100.0,
        p95_fail_ms=200.0,
        max_error_rate_percent=0.0,
    )

    assessments = benchmark_api.assess_summaries(
        [
            {"endpoint": "ready", "error_rate_percent": 0.0, "p95_ms": 50.0},
            {"endpoint": "predict", "error_rate_percent": 0.0, "p95_ms": 150.0},
            {"endpoint": "risk", "error_rate_percent": 2.0, "p95_ms": 50.0},
        ],
        profile=profile,
    )

    assert assessments[0]["status"] == "pass"
    assert assessments[1]["status"] == "watch"
    assert assessments[2]["status"] == "fail"


def test_format_assessment_table_includes_threshold_status():
    table = benchmark_api.format_assessment_table(
        [
            {
                "endpoint": "predict",
                "status": "watch",
                "reason": "p95 exceeded watch threshold",
            }
        ]
    )

    assert "| endpoint | status | reason |" in table
    assert "| predict | watch | p95 exceeded watch threshold |" in table


def test_resolve_config_uses_profile_defaults_and_overrides():
    args = type(
        "Args",
        (),
        {
            "profile": "capacity",
            "requests": 7,
            "warmup": None,
            "concurrency": 3,
            "timeout": None,
        },
    )()

    config = benchmark_api.resolve_config(args)

    assert config.profile.name == "capacity"
    assert config.requests_per_endpoint == 7
    assert (
        config.warmup_requests
        == benchmark_api.BENCHMARK_PROFILES["capacity"].warmup_requests
    )
    assert config.concurrency == 3
    assert config.timeout == benchmark_api.BENCHMARK_PROFILES["capacity"].timeout


def test_run_benchmark_supports_concurrency(monkeypatch):
    request_ids = []

    def fake_request_once(base_url, case, request_id, timeout):
        request_ids.append(request_id)
        return benchmark_api.Measurement(
            endpoint=case.name,
            status_code=200,
            duration_ms=1.0,
            ok=True,
        )

    monkeypatch.setattr(benchmark_api, "request_once", fake_request_once)

    result = benchmark_api.run_benchmark(
        base_url="http://example.test",
        requests_per_endpoint=3,
        warmup_requests=1,
        timeout=1.0,
        concurrency=3,
    )

    assert len(result.measurements) == 9
    assert result.elapsed_seconds >= 0
    assert "benchmark-warmup-ready-0" in request_ids
    assert "benchmark-predict-2" in request_ids
    assert "benchmark-risk-2" in request_ids
