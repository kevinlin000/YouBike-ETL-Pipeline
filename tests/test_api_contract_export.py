import json

from scripts import export_api_contract


def test_build_openapi_schema_includes_served_endpoints():
    schema = export_api_contract.build_openapi_schema()

    assert schema["info"]["title"] == "YouBike LSTM Prediction API"
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "/predict" in schema["paths"]
    assert "/stations/risk" in schema["paths"]
    assert "/metrics" in schema["paths"]
    assert (
        "text/plain"
        in schema["paths"]["/metrics"]["get"]["responses"]["200"]["content"]
    )


def test_write_contract_files_creates_openapi_and_http_examples(tmp_path):
    written_files = export_api_contract.write_contract_files(tmp_path)

    assert {path.name for path in written_files} == {
        "openapi.json",
        "api_examples.http",
    }

    openapi = json.loads((tmp_path / "openapi.json").read_text(encoding="utf-8"))
    examples = (tmp_path / "api_examples.http").read_text(encoding="utf-8")

    assert "/predict" in openapi["paths"]
    assert "POST {{base_url}}/predict" in examples
    assert "POST {{base_url}}/stations/risk" in examples
    assert "X-Request-ID: {{request_id}}" in examples
    assert "Validation 範例：負數車輛應回 422" in examples
