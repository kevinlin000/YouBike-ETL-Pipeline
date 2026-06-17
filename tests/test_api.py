import os
import sys

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import main as api_main  # noqa: E402


@pytest.fixture(autouse=True)
def reset_model_resources(monkeypatch):
    """Keep API tests independent from real model artifacts."""
    monkeypatch.setattr(api_main, "model", None)
    monkeypatch.setattr(api_main, "scaler", None)
    monkeypatch.setattr(api_main, "station_mapping", None)
    monkeypatch.setattr(api_main, "station_info_map", None)


@pytest.fixture
def client():
    return TestClient(api_main.app)


class FakeScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float)


class FakeModel:
    def __call__(self, _input_tensor):
        return torch.tensor([[7.2]])


def test_home_returns_service_metadata(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "online"
    assert body["model"] == "LSTM Multi-Station"
    assert "Bikes" in body["features"]


def test_get_stations_requires_model_information(client):
    response = client.get("/stations")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model information not initialized"


def test_get_stations_returns_loaded_station_map(client, monkeypatch):
    monkeypatch.setattr(api_main, "station_info_map", {"500101001": "測試站 (中正區)"})

    response = client.get("/stations")

    assert response.status_code == 200
    assert response.json() == {"stations": {"500101001": "測試站 (中正區)"}}


def test_predict_requires_ready_model(client):
    response = client.post(
        "/predict",
        json={
            "station_no": "500101001",
            "bikes_available": 12,
            "temperature": 27.5,
            "rain": 0,
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not ready"


@pytest.mark.parametrize(
    "payload",
    [
        {"station_no": "500101001", "bikes_available": -1, "temperature": 27.5, "rain": 0},
        {"station_no": "500101001", "bikes_available": 12, "temperature": 99, "rain": 0},
        {"station_no": "500101001", "bikes_available": 12, "temperature": 27.5, "rain": -0.1},
    ],
)
def test_predict_validates_request_payload(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_unknown_station(client, monkeypatch):
    monkeypatch.setattr(api_main, "model", FakeModel())
    monkeypatch.setattr(api_main, "scaler", FakeScaler())
    monkeypatch.setattr(api_main, "station_mapping", {"500101001": 0})

    response = client.post(
        "/predict",
        json={
            "station_no": "unknown",
            "bikes_available": 12,
            "temperature": 27.5,
            "rain": 0,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Station ID not supported by model"


def test_predict_returns_prediction_with_mocked_model(client, monkeypatch):
    monkeypatch.setattr(api_main, "model", FakeModel())
    monkeypatch.setattr(api_main, "scaler", FakeScaler())
    monkeypatch.setattr(api_main, "station_mapping", {"500101001": 0})

    response = client.post(
        "/predict",
        json={
            "station_no": "500101001",
            "bikes_available": 12,
            "temperature": 27.5,
            "rain": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "station_no": "500101001",
        "predicted_bikes_next_hour": 7,
    }
