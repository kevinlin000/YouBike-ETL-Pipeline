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

    summary = benchmark_api.summarize(measurements)

    assert summary == [
        {
            "endpoint": "predict",
            "requests": 1,
            "errors": 1,
            "min_ms": 50.0,
            "avg_ms": 50.0,
            "p50_ms": 50.0,
            "p95_ms": 50.0,
            "max_ms": 50.0,
        },
        {
            "endpoint": "ready",
            "requests": 2,
            "errors": 0,
            "min_ms": 10.0,
            "avg_ms": 15.0,
            "p50_ms": 15.0,
            "p95_ms": 19.5,
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
                "min_ms": 1.0,
                "avg_ms": 2.0,
                "p50_ms": 2.0,
                "p95_ms": 3.0,
                "max_ms": 4.0,
            }
        ]
    )

    assert "| endpoint | requests | errors | min ms | avg ms | p50 ms | p95 ms | max ms |" in table
    assert "| ready | 2 | 0 | 1.00 | 2.00 | 2.00 | 3.00 | 4.00 |" in table
