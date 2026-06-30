import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import api_client  # noqa: E402
from dashboard import runtime_config  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_get_station_data_returns_station_map(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    def fake_get(url, headers, timeout):
        assert url == "http://api:8000/stations"
        assert headers == {}
        assert timeout == 5
        return FakeResponse(body={"stations": {"500101001": "測試站 (中正區)"}})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    assert api_client.get_station_data("http://api:8000") == {
        "500101001": "測試站 (中正區)"
    }


def test_get_demo_station_data_returns_copy():
    station_map = api_client.get_demo_station_data()
    station_map["500101001"] = "changed"

    assert api_client.get_demo_station_data()["500101001"] == "捷運公館站 (大安區)"


def test_get_station_data_raises_on_api_error(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)

    def fake_get(_url, headers, timeout):
        assert headers == {}
        assert timeout == 5
        return FakeResponse(status_code=503, text="model not ready")

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    with pytest.raises(api_client.DashboardApiError, match="503"):
        api_client.get_station_data("http://api:8000")


def test_predict_station_posts_expected_payload(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse(body={"station_no": "500101001", "predicted_bikes_next_hour": 9})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.predict_station(
        "http://api:8000/",
        "500101001",
        12,
        27.5,
        0,
    )

    assert captured == {
        "url": "http://api:8000/predict",
        "json": {
            "station_no": "500101001",
            "bikes_available": 12,
            "temperature": 27.5,
            "rain": 0.0,
        },
        "headers": {},
        "timeout": 10,
    }
    assert result["predicted_bikes_next_hour"] == 9


def test_live_requests_include_api_key_header(monkeypatch):
    monkeypatch.setenv("API_KEY", "portfolio-demo-key")
    captured = {}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse(body={"stations": {"500101001": "測試站 (中正區)"}})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    result = api_client.get_station_data("http://api:8000")

    assert captured == {
        "url": "http://api:8000/stations",
        "headers": {api_client.API_KEY_HEADER: "portfolio-demo-key"},
        "timeout": 5,
    }
    assert result == {"500101001": "測試站 (中正區)"}


def test_rank_station_risks_posts_expected_payload(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse(
            body={
                "risks": [
                    {
                        "station_no": "500101001",
                        "risk_level": "stock_out",
                        "risk_score": 101,
                    }
                ]
            }
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.rank_station_risks(
        "http://api:8000",
        [{"station_no": "500101001", "bikes_available": 2, "spaces_available": 18}],
        27.5,
        0,
    )

    assert captured == {
        "url": "http://api:8000/stations/risk",
        "json": {
            "temperature": 27.5,
            "rain": 0.0,
            "stations": [
                {
                    "station_no": "500101001",
                    "bikes_available": 2,
                    "spaces_available": 18,
                }
            ],
        },
        "headers": {},
        "timeout": 10,
    }
    assert result[0]["risk_level"] == "stock_out"


def test_demo_predict_station_is_deterministic():
    result = api_client.demo_predict_station(
        "500101002",
        bikes_available=8,
        temperature=27.5,
        rain=0,
    )

    assert result == {
        "station_no": "500101002",
        "predicted_bikes_next_hour": 4,
        "forecast_horizon": api_client.FORECAST_HORIZON,
        "forecast_horizon_description": api_client.FORECAST_HORIZON_DESCRIPTION,
        **api_client.DEMO_MODEL_LINEAGE,
    }


def test_demo_rank_station_risks_returns_sorted_results():
    result = api_client.demo_rank_station_risks(
        [
            {"station_no": "500101003", "bikes_available": 18, "spaces_available": 2},
            {"station_no": "500101002", "bikes_available": 4, "spaces_available": 16},
        ],
        temperature=34,
        rain=6,
    )

    assert [row["station_no"] for row in result] == ["500101002", "500101003"]
    assert result[0]["risk_level"] == "stock_out"
    assert result[0]["suggested_action"] == "rebalance_in"
    assert result[0]["forecast_horizon"] == api_client.FORECAST_HORIZON
    assert result[0]["model_version"] == api_client.DEMO_MODEL_LINEAGE["model_version"]


def test_station_display_options_are_sorted_and_parseable():
    options = api_client.station_display_options(
        {
            "500101002": "B 站 (大安區)",
            "500101001": "A 站 (中正區)",
        }
    )

    assert options == ["A 站 (中正區) [500101001]", "B 站 (大安區) [500101002]"]
    assert api_client.parse_station_option(options[0]) == ("500101001", "A 站 (中正區)")


def test_risk_and_action_labels():
    assert api_client.risk_level_label("stock_out") == "嚴重缺車"
    assert api_client.risk_level_label("unknown") == "unknown"
    assert api_client.suggested_action_label("rebalance_in") == "建議補車"
    assert api_client.suggested_action_label("unknown") == "unknown"


def test_dashboard_runtime_config_prefers_environment(monkeypatch):
    monkeypatch.setenv("DASHBOARD_DEMO_MODE", "true")
    monkeypatch.setenv("API_BASE_URL", "http://localhost:8000")

    secrets = {
        "DASHBOARD_DEMO_MODE": False,
        "API_BASE_URL": "https://example.invalid",
    }

    assert runtime_config.dashboard_demo_mode_default(secrets) is True
    assert runtime_config.dashboard_api_base_url(secrets) == "http://localhost:8000"


def test_dashboard_runtime_config_reads_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("DASHBOARD_DEMO_MODE", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    secrets = {
        "DASHBOARD_DEMO_MODE": True,
        "API_BASE_URL": "https://dashboard-api.example.com",
    }

    assert runtime_config.dashboard_demo_mode_default(secrets) is True
    assert runtime_config.dashboard_api_base_url(secrets) == "https://dashboard-api.example.com"


def test_dashboard_runtime_config_defaults_to_demo_without_api(monkeypatch):
    monkeypatch.delenv("DASHBOARD_DEMO_MODE", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    assert runtime_config.dashboard_demo_mode_default({}) is True


def test_dashboard_runtime_config_defaults_to_live_when_api_is_configured(monkeypatch):
    monkeypatch.delenv("DASHBOARD_DEMO_MODE", raising=False)
    monkeypatch.setenv("API_BASE_URL", "http://api:8000")

    assert runtime_config.dashboard_demo_mode_default({}) is False
