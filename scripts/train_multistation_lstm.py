"""Train the Multi-Station LSTM artifacts used by the FastAPI service.

This script turns the historical notebook flow into a reproducible CLI. It does
not run in CI against the full dataset because the processed training CSV is not
committed to Git.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLUMNS = ["bikes_available", "temperature", "rain", "Rain_Cat"]
TARGET_COLUMN = "bikes_available"
REQUIRED_COLUMNS = {
    "station_no",
    "bikes_available",
    "record_time",
    "temperature",
    "rain",
}


class MultiStationLSTM(nn.Module):
    """Model architecture kept in sync with `api/app/main.py`."""

    def __init__(
        self,
        num_stations: int,
        input_size: int = 4,
        hidden_size: int = 64,
        output_size: int = 1,
        embedding_dim: int = 5,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)
        self.lstm_input_size = input_size + embedding_dim
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numeric_features = x[:, :, :-1]
        station_ids = x[:, :, -1].long()
        station_embeds = self.station_embedding(station_ids)
        combined = torch.cat((numeric_features, station_embeds), dim=2)
        lstm_out, _ = self.lstm(combined)
        final_hidden_state = lstm_out[:, -1, :]
        out = self.dropout(final_hidden_state)
        return self.fc(out)


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    station_mapping: dict[str, int]
    station_info_map: dict[str, str]
    split_counts: dict[str, int]
    selected_station_ids: list[str]
    baseline_predictions: dict[str, dict[str, np.ndarray]]


def get_rain_category(rain: float) -> int:
    if rain == 0:
        return 0
    if rain <= 2:
        return 1
    if rain <= 10:
        return 2
    return 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_columns(df: pd.DataFrame, station_selection: str) -> None:
    required = set(REQUIRED_COLUMNS)
    if station_selection == "district-representative":
        required.add("district")

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")


def station_label(row: pd.Series) -> str:
    name = row.get("name_tw")
    district = row.get("district")
    if pd.notna(name) and pd.notna(district):
        return f"{name} ({district})"
    if pd.notna(name):
        return str(name)
    if pd.notna(district):
        return f"{row['station_no']} ({district})"
    return str(row["station_no"])


def select_stations(
    df: pd.DataFrame,
    station_selection: str,
    max_stations: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    if station_selection not in {"district-representative", "all"}:
        raise ValueError("station_selection must be 'district-representative' or 'all'")

    if station_selection == "all":
        station_ids = sorted(df["station_no"].astype(str).unique().tolist())
    else:
        station_ids = []
        for _, district_df in df.sort_values(["district", "station_no"]).groupby("district"):
            counts = district_df["station_no"].astype(str).value_counts()
            station_ids.append(str(counts.index[0]))

    if max_stations is not None:
        station_ids = station_ids[:max_stations]

    first_rows = (
        df[df["station_no"].astype(str).isin(station_ids)]
        .sort_values(["station_no", "record_time"])
        .drop_duplicates("station_no")
        .set_index("station_no")
    )
    station_info_map = {
        station_id: station_label(first_rows.loc[station_id])
        for station_id in station_ids
        if station_id in first_rows.index
    }
    return station_ids, station_info_map


def prepare_dataframe(
    df: pd.DataFrame,
    station_selection: str,
    max_stations: int | None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str], list[str]]:
    validate_columns(df, station_selection)

    prepared = df.copy()
    prepared["station_no"] = prepared["station_no"].astype(str)
    prepared["record_time"] = pd.to_datetime(prepared["record_time"], errors="coerce")
    prepared = prepared.dropna(subset=["record_time"])
    prepared["rain"] = prepared["rain"].fillna(0)
    prepared["Rain_Cat"] = prepared["rain"].apply(get_rain_category)

    selected_station_ids, station_info_map = select_stations(
        prepared,
        station_selection=station_selection,
        max_stations=max_stations,
    )
    if not selected_station_ids:
        raise ValueError("No stations selected for training")

    prepared = prepared[prepared["station_no"].isin(selected_station_ids)].copy()
    prepared = prepared.sort_values(["station_no", "record_time"])
    prepared[FEATURE_COLUMNS] = prepared.groupby("station_no")[FEATURE_COLUMNS].transform(
        lambda series: series.ffill().bfill()
    )
    prepared = prepared.dropna(subset=FEATURE_COLUMNS)

    station_mapping = {station_id: idx for idx, station_id in enumerate(selected_station_ids)}
    prepared["station_idx"] = prepared["station_no"].map(station_mapping)
    return prepared, station_mapping, station_info_map, selected_station_ids


def fit_scaler_on_training_rows(df: pd.DataFrame, train_ratio: float) -> MinMaxScaler:
    train_rows = []
    for _, station_df in df.groupby("station_no", sort=False):
        cutoff = max(1, int(len(station_df) * train_ratio))
        train_rows.append(station_df.iloc[:cutoff])

    scaler = MinMaxScaler()
    scaler.fit(pd.concat(train_rows, ignore_index=True)[FEATURE_COLUMNS])
    return scaler


def stack_or_empty(items: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not items:
        return np.empty(shape, dtype=np.float32)
    return np.asarray(items, dtype=np.float32)


def empty_baseline_prediction_lists() -> dict[str, dict[str, list[float]]]:
    return {
        "current_value": {"train": [], "validation": [], "test": []},
        "rolling_mean": {"train": [], "validation": [], "test": []},
        "same_time_previous_day": {"train": [], "validation": [], "test": []},
    }


def find_same_time_previous_day_value(
    times: np.ndarray,
    raw_bikes: np.ndarray,
    target_idx: int,
    tolerance: np.timedelta64 = np.timedelta64(30, "m"),
) -> float:
    """Return the prior-day same-time bike count, or NaN when no close row exists."""
    target_time = times[target_idx]
    desired_time = target_time - np.timedelta64(1, "D")
    insert_pos = int(np.searchsorted(times, desired_time))
    candidate_indices = [idx for idx in (insert_pos - 1, insert_pos) if 0 <= idx < len(times)]
    if not candidate_indices:
        return float("nan")

    best_idx = min(candidate_indices, key=lambda idx: abs(times[idx] - desired_time))
    if abs(times[best_idx] - desired_time) > tolerance:
        return float("nan")
    return float(raw_bikes[best_idx])


def finalize_baseline_predictions(
    predictions: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        baseline_name: {
            split_name: np.asarray(values, dtype=np.float32)
            for split_name, values in split_values.items()
        }
        for baseline_name, split_values in predictions.items()
    }


def build_datasets(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    time_steps: int,
    horizon_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
    dict[str, dict[str, np.ndarray]],
]:
    if time_steps < 1:
        raise ValueError("time_steps must be >= 1")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    train_x: list[np.ndarray] = []
    train_y: list[float] = []
    val_x: list[np.ndarray] = []
    val_y: list[float] = []
    test_x: list[np.ndarray] = []
    test_y: list[float] = []
    baseline_prediction_lists = empty_baseline_prediction_lists()

    scaled = df.copy()
    scaled["_raw_bikes_available"] = scaled["bikes_available"].astype(float)
    scaled[FEATURE_COLUMNS] = scaler.transform(scaled[FEATURE_COLUMNS])

    for _, station_df in scaled.groupby("station_no", sort=False):
        station_df = station_df.sort_values("record_time")
        values = station_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        raw_bikes = station_df["_raw_bikes_available"].to_numpy(dtype=np.float32)
        times = station_df["record_time"].to_numpy(dtype="datetime64[ns]")
        station_ids = station_df["station_idx"].to_numpy(dtype=np.float32).reshape(-1, 1)
        n_rows = len(station_df)
        train_cutoff = int(n_rows * train_ratio)
        val_cutoff = int(n_rows * (train_ratio + val_ratio))
        max_start = n_rows - time_steps - horizon_steps + 1

        for start_idx in range(max(0, max_start)):
            target_idx = start_idx + time_steps + horizon_steps - 1
            sequence = np.hstack(
                (
                    values[start_idx : start_idx + time_steps],
                    station_ids[start_idx : start_idx + time_steps],
                )
            )
            target = float(values[target_idx, 0])
            raw_window = raw_bikes[start_idx : start_idx + time_steps]
            baseline_values = {
                "current_value": float(raw_window[-1]),
                "rolling_mean": float(np.mean(raw_window)),
                "same_time_previous_day": find_same_time_previous_day_value(
                    times,
                    raw_bikes,
                    target_idx,
                ),
            }

            if target_idx < train_cutoff:
                train_x.append(sequence)
                train_y.append(target)
                split_name = "train"
            elif target_idx < val_cutoff:
                val_x.append(sequence)
                val_y.append(target)
                split_name = "validation"
            else:
                test_x.append(sequence)
                test_y.append(target)
                split_name = "test"

            for baseline_name, baseline_value in baseline_values.items():
                baseline_prediction_lists[baseline_name][split_name].append(baseline_value)

    feature_shape = (0, time_steps, len(FEATURE_COLUMNS) + 1)
    x_train = stack_or_empty(train_x, feature_shape)
    x_val = stack_or_empty(val_x, feature_shape)
    x_test = stack_or_empty(test_x, feature_shape)
    y_train = np.asarray(train_y, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(val_y, dtype=np.float32).reshape(-1, 1)
    y_test = np.asarray(test_y, dtype=np.float32).reshape(-1, 1)

    split_counts = {
        "train": int(len(x_train)),
        "validation": int(len(x_val)),
        "test": int(len(x_test)),
    }
    if split_counts["train"] == 0:
        raise ValueError("No training sequences were created; lower time_steps or provide more rows")

    baseline_predictions = finalize_baseline_predictions(baseline_prediction_lists)

    return x_train, y_train, x_val, y_val, x_test, y_test, split_counts, baseline_predictions


def prepare_datasets(
    df: pd.DataFrame,
    station_selection: str = "district-representative",
    max_stations: int | None = None,
    time_steps: int = 3,
    horizon_steps: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DatasetBundle:
    prepared, station_mapping, station_info_map, selected_station_ids = prepare_dataframe(
        df,
        station_selection=station_selection,
        max_stations=max_stations,
    )
    scaler = fit_scaler_on_training_rows(prepared, train_ratio=train_ratio)
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        split_counts,
        baseline_predictions,
    ) = build_datasets(
        prepared,
        scaler=scaler,
        time_steps=time_steps,
        horizon_steps=horizon_steps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        station_mapping=station_mapping,
        station_info_map=station_info_map,
        split_counts=split_counts,
        selected_station_ids=selected_station_ids,
        baseline_predictions=baseline_predictions,
    )


def train_model(
    bundle: DatasetBundle,
    hidden_size: int,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[MultiStationLSTM, list[dict[str, float]]]:
    set_seed(seed)
    model = MultiStationLSTM(
        num_stations=len(bundle.station_mapping),
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(
        torch.tensor(bundle.x_train, dtype=torch.float32),
        torch.tensor(bundle.y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        history.append({"epoch": epoch, "train_loss": total_loss / len(dataset)})

    return model, history


def inverse_bike_values(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    dummy = np.zeros((len(values), len(FEATURE_COLUMNS)), dtype=np.float32)
    dummy[:, 0] = values.reshape(-1)
    return scaler.inverse_transform(dummy)[:, 0]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    finite_mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[finite_mask]
    predicted = predicted[finite_mask]
    if len(actual) == 0:
        return {}
    errors = predicted - actual
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def evaluate_split(
    model: MultiStationLSTM,
    scaler: MinMaxScaler,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, float]:
    if len(x_values) == 0:
        return {}
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(x_values, dtype=torch.float32)).numpy()

    actual_raw = inverse_bike_values(scaler, y_values)
    predicted_raw = inverse_bike_values(scaler, predictions)
    return regression_metrics(actual_raw, predicted_raw)


def evaluate_current_value_baseline(
    scaler: MinMaxScaler,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, float]:
    """Predict the target as the last observed bike count in the input window."""
    if len(x_values) == 0:
        return {}

    baseline_scaled = x_values[:, -1, 0].reshape(-1, 1)
    actual_raw = inverse_bike_values(scaler, y_values)
    baseline_raw = inverse_bike_values(scaler, baseline_scaled)
    return regression_metrics(actual_raw, baseline_raw)


def evaluate_rolling_mean_baseline(
    scaler: MinMaxScaler,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict[str, float]:
    """Predict the target as the average bike count in the input window."""
    if len(x_values) == 0:
        return {}

    baseline_scaled = np.mean(x_values[:, :, 0], axis=1).reshape(-1, 1)
    actual_raw = inverse_bike_values(scaler, y_values)
    baseline_raw = inverse_bike_values(scaler, baseline_scaled)
    return regression_metrics(actual_raw, baseline_raw)


def evaluate_raw_baseline_predictions(
    scaler: MinMaxScaler,
    y_values: np.ndarray,
    predictions_raw: np.ndarray,
) -> dict[str, float]:
    if len(y_values) == 0:
        return {}
    actual_raw = inverse_bike_values(scaler, y_values)
    return regression_metrics(actual_raw, predictions_raw)


def build_tabular_lag_features(x_values: np.ndarray, num_stations: int) -> np.ndarray:
    if len(x_values) == 0:
        feature_count = x_values.shape[1] * (x_values.shape[2] - 1) + num_stations
        return np.empty((0, feature_count), dtype=np.float32)

    numeric_features = x_values[:, :, :-1].reshape(len(x_values), -1)
    station_ids = x_values[:, -1, -1].astype(int)
    station_one_hot = np.zeros((len(x_values), num_stations), dtype=np.float32)
    station_one_hot[np.arange(len(x_values)), station_ids] = 1.0
    return np.hstack((numeric_features, station_one_hot)).astype(np.float32)


def evaluate_ridge_lag_regression_baseline(
    bundle: DatasetBundle,
    alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Train a simple tabular baseline on the same lag window used by the LSTM."""
    train_features = build_tabular_lag_features(bundle.x_train, len(bundle.station_mapping))
    if len(train_features) == 0:
        return {"train": {}, "validation": {}, "test": {}}

    model = Ridge(alpha=alpha)
    model.fit(train_features, bundle.y_train.reshape(-1))

    split_values = {
        "train": (bundle.x_train, bundle.y_train),
        "validation": (bundle.x_val, bundle.y_val),
        "test": (bundle.x_test, bundle.y_test),
    }
    metrics = {}
    for split_name, (x_values, y_values) in split_values.items():
        if len(x_values) == 0:
            metrics[split_name] = {}
            continue

        features = build_tabular_lag_features(x_values, len(bundle.station_mapping))
        predicted_scaled = model.predict(features).reshape(-1, 1)
        actual_raw = inverse_bike_values(bundle.scaler, y_values)
        predicted_raw = inverse_bike_values(bundle.scaler, predicted_scaled)
        metrics[split_name] = regression_metrics(actual_raw, predicted_raw)

    return metrics


def compare_against_baseline(
    model_metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for split_name, split_model_metrics in model_metrics.items():
        split_baseline_metrics = baseline_metrics.get(split_name, {})
        if not split_model_metrics or not split_baseline_metrics:
            comparison[split_name] = {}
            continue

        comparison[split_name] = {
            "mae_delta": split_baseline_metrics["mae"] - split_model_metrics["mae"],
            "rmse_delta": split_baseline_metrics["rmse"] - split_model_metrics["rmse"],
        }
    return comparison


def evaluate_model(model: MultiStationLSTM, bundle: DatasetBundle) -> dict[str, dict[str, float]]:
    return {
        "train": evaluate_split(model, bundle.scaler, bundle.x_train, bundle.y_train),
        "validation": evaluate_split(model, bundle.scaler, bundle.x_val, bundle.y_val),
        "test": evaluate_split(model, bundle.scaler, bundle.x_test, bundle.y_test),
    }


def evaluate_current_value_baselines(bundle: DatasetBundle) -> dict[str, dict[str, float]]:
    return evaluate_all_baselines(bundle)["current_value"]


def evaluate_all_baselines(bundle: DatasetBundle) -> dict[str, dict[str, dict[str, float]]]:
    split_arrays = {
        "train": bundle.y_train,
        "validation": bundle.y_val,
        "test": bundle.y_test,
    }
    naive_baselines = {
        baseline_name: {
            split_name: evaluate_raw_baseline_predictions(
                bundle.scaler,
                split_arrays[split_name],
                predictions,
            )
            for split_name, predictions in split_predictions.items()
        }
        for baseline_name, split_predictions in bundle.baseline_predictions.items()
    }
    naive_baselines["ridge_lag_regression"] = evaluate_ridge_lag_regression_baseline(bundle)
    return naive_baselines


def find_best_baseline_for_metric(
    baseline_metrics: dict[str, dict[str, dict[str, float]]],
    split_name: str,
    metric_name: str,
) -> dict[str, float | str] | None:
    candidates = []
    for baseline_name, split_metrics in baseline_metrics.items():
        metrics = split_metrics.get(split_name, {})
        if metric_name in metrics:
            candidates.append((baseline_name, metrics))

    if not candidates:
        return None

    best_name, best_metrics = min(candidates, key=lambda item: item[1][metric_name])
    return {
        "name": best_name,
        "mae": best_metrics["mae"],
        "rmse": best_metrics["rmse"],
        "n": best_metrics["n"],
    }


def build_model_selection_summary(
    model_metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, dict[str, float]]],
    primary_split: str = "test",
) -> dict:
    split_summary = {}
    for split_name, split_model_metrics in model_metrics.items():
        best_by_mae = find_best_baseline_for_metric(baseline_metrics, split_name, "mae")
        best_by_rmse = find_best_baseline_for_metric(baseline_metrics, split_name, "rmse")
        if not split_model_metrics or best_by_mae is None or best_by_rmse is None:
            split_summary[split_name] = {}
            continue

        split_summary[split_name] = {
            "lstm": {
                "mae": split_model_metrics["mae"],
                "rmse": split_model_metrics["rmse"],
                "n": split_model_metrics["n"],
            },
            "best_baseline_by_mae": best_by_mae,
            "best_baseline_by_rmse": best_by_rmse,
            "lstm_beats_best_baseline_by_mae": split_model_metrics["mae"] < best_by_mae["mae"],
            "lstm_beats_best_baseline_by_rmse": split_model_metrics["rmse"] < best_by_rmse["rmse"],
        }

    primary_summary = split_summary.get(primary_split, {})
    candidate_beats_best_baseline = bool(
        primary_summary.get("lstm_beats_best_baseline_by_mae")
        and primary_summary.get("lstm_beats_best_baseline_by_rmse")
    )
    recommendation = (
        "Candidate LSTM beats the best baseline on the test split for both MAE and RMSE; "
        "inspect lineage before replacing served artifacts."
        if candidate_beats_best_baseline
        else "Do not replace served artifacts; candidate LSTM does not beat the best test baseline."
    )

    return {
        "primary_split": primary_split,
        "promotion_rule": "LSTM must beat the best available baseline on the test split for both MAE and RMSE.",
        "candidate_beats_best_baseline": candidate_beats_best_baseline,
        "recommendation": recommendation,
        "splits": split_summary,
    }


def build_metadata(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    history: list[dict[str, float]],
    model_metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, dict[str, float]]],
) -> dict:
    metric_comparison = {
        baseline_name: compare_against_baseline(model_metrics, split_metrics)
        for baseline_name, split_metrics in baseline_metrics.items()
    }
    current_value_metrics = baseline_metrics["current_value"]
    current_value_comparison = metric_comparison["current_value"]
    model_selection = build_model_selection_summary(model_metrics, baseline_metrics)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_path": str(args.data_path),
        "notebook_lineage": "notebooks/05_multistation_lstm.ipynb",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "station_selection": args.station_selection,
        "selected_station_count": len(bundle.station_mapping),
        "selected_station_ids": bundle.selected_station_ids,
        "time_steps": args.time_steps,
        "horizon_steps": args.horizon_steps,
        "split": {
            "train_ratio": args.train_ratio,
            "validation_ratio": args.val_ratio,
            "test_ratio": 1 - args.train_ratio - args.val_ratio,
            "sequence_counts": bundle.split_counts,
        },
        "model": {
            "type": "MultiStationLSTM",
            "hidden_size": args.hidden_size,
            "embedding_dim": args.embedding_dim,
            "dropout": 0.2,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "final_train_loss": history[-1]["train_loss"] if history else None,
        },
        "metrics": {
            "lstm": model_metrics,
            "baselines": baseline_metrics,
            "lstm_vs_baselines": metric_comparison,
            "baseline_current_value": current_value_metrics,
            "lstm_vs_baseline": current_value_comparison,
        },
        "model_selection": model_selection,
        "limitations": [
            "Metrics are only meaningful when the full processed dataset is available.",
            "The warehouse-backed inference path can fetch recent bike counts but still needs aligned weather history.",
        ],
    }


def save_artifacts(
    output_dir: Path,
    model: MultiStationLSTM,
    bundle: DatasetBundle,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "youbike_lstm_multistation.pth")
    joblib.dump(bundle.scaler, output_dir / "scaler.pkl")
    joblib.dump(bundle.station_mapping, output_dir / "station_mapping.pkl")
    joblib.dump(bundle.station_info_map, output_dir / "station_info_map.pkl")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_and_save(args: argparse.Namespace) -> dict:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. Generate it from notebooks/03_data_merge.ipynb "
            "or pass --data-path to a processed YouBike/weather CSV."
        )

    df = pd.read_csv(data_path)
    bundle = prepare_datasets(
        df,
        station_selection=args.station_selection,
        max_stations=args.max_stations,
        time_steps=args.time_steps,
        horizon_steps=args.horizon_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    model, history = train_model(
        bundle,
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model_metrics = evaluate_model(model, bundle)
    baseline_metrics = evaluate_all_baselines(bundle)
    metadata = build_metadata(args, bundle, history, model_metrics, baseline_metrics)
    save_artifacts(Path(args.output_dir), model, bundle, metadata)
    return metadata


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def bounded_ratio(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("ratio must be between 0 and 1")
    return parsed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/youbike_weather_merged.csv"),
        help="Processed YouBike/weather CSV generated by notebooks/03_data_merge.ipynb.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".scratch/model_training"),
        help="Directory for model artifacts. Use api/model_files only when intentionally replacing served artifacts.",
    )
    parser.add_argument(
        "--station-selection",
        choices=["district-representative", "all"],
        default="district-representative",
        help="Select one frequent station per district, matching the notebook, or train on all stations.",
    )
    parser.add_argument("--max-stations", type=positive_int, default=None)
    parser.add_argument("--time-steps", type=positive_int, default=3)
    parser.add_argument("--horizon-steps", type=positive_int, default=1)
    parser.add_argument("--train-ratio", type=bounded_ratio, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-size", type=positive_int, default=64)
    parser.add_argument("--embedding-dim", type=positive_int, default=5)
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("--train-ratio + --val-ratio must be less than 1")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = train_and_save(args)
    print(json.dumps(metadata["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(metadata["model_selection"], ensure_ascii=False, indent=2))
    print(f"Saved model artifacts to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
