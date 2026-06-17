import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import api_client  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_get_station_data_returns_station_map(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://api:8000/stations"
        assert timeout == 5
        return FakeResponse(body={"stations": {"500101001": "測試站 (中正區)"}})

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    assert api_client.get_station_data("http://api:8000") == {
        "500101001": "測試站 (中正區)"
    }


def test_get_station_data_raises_on_api_error(monkeypatch):
    def fake_get(_url, timeout):
        assert timeout == 5
        return FakeResponse(status_code=503, text="model not ready")

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    with pytest.raises(api_client.DashboardApiError, match="503"):
        api_client.get_station_data("http://api:8000")


def test_predict_station_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
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
        "timeout": 10,
    }
    assert result["predicted_bikes_next_hour"] == 9


def test_rank_station_risks_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
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
        "timeout": 10,
    }
    assert result[0]["risk_level"] == "stock_out"


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
