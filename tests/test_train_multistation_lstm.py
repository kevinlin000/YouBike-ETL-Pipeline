import json
from argparse import Namespace
from datetime import datetime, timedelta

import joblib
import pandas as pd
import torch

from scripts import train_multistation_lstm as trainer


def synthetic_training_frame() -> pd.DataFrame:
    rows = []
    start = datetime(2025, 1, 1, 8, 0)
    stations = [
        ("A-main", "Alpha Main", "Alpha", 14, 10),
        ("A-small", "Alpha Small", "Alpha", 5, 5),
        ("B-main", "Beta Main", "Beta", 14, 20),
    ]
    for station_no, name, district, count, base_bikes in stations:
        for offset in range(count):
            rows.append(
                {
                    "station_no": station_no,
                    "name_tw": name,
                    "district": district,
                    "record_time": start + timedelta(minutes=10 * offset),
                    "bikes_available": base_bikes + (offset % 4),
                    "spaces_available": 30 - base_bikes,
                    "temperature": 25.0 + (offset % 3),
                    "rain": [0.0, 0.5, 3.0, 12.0][offset % 4],
                }
            )
    return pd.DataFrame(rows)


def test_get_rain_category_matches_api_thresholds():
    assert trainer.get_rain_category(0) == 0
    assert trainer.get_rain_category(0.5) == 1
    assert trainer.get_rain_category(2) == 1
    assert trainer.get_rain_category(3) == 2
    assert trainer.get_rain_category(10) == 2
    assert trainer.get_rain_category(10.1) == 3


def test_prepare_datasets_selects_one_representative_station_per_district():
    bundle = trainer.prepare_datasets(
        synthetic_training_frame(),
        station_selection="district-representative",
        time_steps=3,
        horizon_steps=1,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    assert bundle.selected_station_ids == ["A-main", "B-main"]
    assert bundle.station_mapping == {"A-main": 0, "B-main": 1}
    assert bundle.station_info_map == {
        "A-main": "Alpha Main (Alpha)",
        "B-main": "Beta Main (Beta)",
    }
    assert bundle.x_train.shape[1:] == (3, 5)
    assert bundle.split_counts == {"train": 10, "validation": 6, "test": 6}


def test_current_value_baseline_uses_last_observed_bike_count():
    bundle = trainer.prepare_datasets(
        synthetic_training_frame(),
        station_selection="district-representative",
        time_steps=3,
        horizon_steps=1,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    metrics = trainer.evaluate_current_value_baseline(
        bundle.scaler,
        bundle.x_train,
        bundle.y_train,
    )

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]


def test_baseline_suite_includes_rolling_and_previous_day_metrics():
    bundle = trainer.prepare_datasets(
        synthetic_training_frame(),
        station_selection="district-representative",
        time_steps=3,
        horizon_steps=1,
        train_ratio=0.6,
        val_ratio=0.2,
    )

    metrics = trainer.evaluate_all_baselines(bundle)

    assert set(metrics) == {"current_value", "rolling_mean", "same_time_previous_day"}
    assert metrics["rolling_mean"]["train"]["n"] == bundle.split_counts["train"]
    assert metrics["rolling_mean"]["train"]["mae"] >= 0
    assert metrics["same_time_previous_day"]["train"] == {}


def test_train_and_save_writes_api_compatible_artifacts(tmp_path):
    data_path = tmp_path / "training.csv"
    output_dir = tmp_path / "artifacts"
    synthetic_training_frame().to_csv(data_path, index=False)

    args = Namespace(
        data_path=data_path,
        output_dir=output_dir,
        station_selection="district-representative",
        max_stations=None,
        time_steps=3,
        horizon_steps=1,
        train_ratio=0.6,
        val_ratio=0.2,
        hidden_size=8,
        embedding_dim=3,
        epochs=1,
        learning_rate=0.001,
        batch_size=4,
        seed=7,
    )

    metadata = trainer.train_and_save(args)

    assert (output_dir / "youbike_lstm_multistation.pth").exists()
    assert (output_dir / "scaler.pkl").exists()
    assert (output_dir / "station_mapping.pkl").exists()
    assert (output_dir / "station_info_map.pkl").exists()
    assert (output_dir / "model_metadata.json").exists()

    station_mapping = joblib.load(output_dir / "station_mapping.pkl")
    station_info_map = joblib.load(output_dir / "station_info_map.pkl")
    model_state = torch.load(
        output_dir / "youbike_lstm_multistation.pth",
        map_location="cpu",
        weights_only=True,
    )
    saved_metadata = json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))

    assert station_mapping == {"A-main": 0, "B-main": 1}
    assert station_info_map["A-main"] == "Alpha Main (Alpha)"
    assert "lstm.weight_ih_l0" in model_state
    assert saved_metadata["feature_columns"] == trainer.FEATURE_COLUMNS
    assert saved_metadata["split"]["sequence_counts"] == {"train": 10, "validation": 6, "test": 6}
    assert saved_metadata["metrics"].keys() == metadata["metrics"].keys()
    assert set(saved_metadata["metrics"]) == {
        "lstm",
        "baselines",
        "lstm_vs_baselines",
        "baseline_current_value",
        "lstm_vs_baseline",
    }
    assert set(saved_metadata["metrics"]["lstm"]) == {"train", "validation", "test"}
    assert set(saved_metadata["metrics"]["baselines"]) == {
        "current_value",
        "rolling_mean",
        "same_time_previous_day",
    }
    assert set(saved_metadata["metrics"]["baseline_current_value"]) == {"train", "validation", "test"}
    assert "mae" in saved_metadata["metrics"]["baseline_current_value"]["test"]
    assert "mae" in saved_metadata["metrics"]["baselines"]["rolling_mean"]["test"]
    assert saved_metadata["metrics"]["baselines"]["same_time_previous_day"]["test"] == {}
    assert "mae_delta" in saved_metadata["metrics"]["lstm_vs_baseline"]["test"]
    assert saved_metadata["model"]["type"] == "MultiStationLSTM"
