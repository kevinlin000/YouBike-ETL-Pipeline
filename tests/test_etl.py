"""ETL 單元測試：僅測試 transform 邏輯，不依賴 DB 或 GCP。"""
import os
import sys

import pandas as pd
import pytest

# 測試前注入環境變數，避免 import etl_job 時連到 GCP
os.environ.setdefault("DB_PASSWORD", "test_no_gcp")
os.environ.setdefault("DB_HOST", "127.0.0.1")

# 專案根目錄加入 path 以便 import etl_job
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import etl_job  # noqa: E402


def test_transform_data_empty_raises():
    """Extract 回傳空資料時應拋出 ValueError。"""
    with pytest.raises(ValueError, match="空資料"):
        etl_job.transform_data([])


def test_transform_data_missing_columns_raises():
    """缺少必要欄位時應拋出 KeyError。"""
    raw = [{"sno": "1", "sna": "A"}]
    with pytest.raises(KeyError, match="station_info 缺少欄位"):
        etl_job.transform_data(raw)


def test_transform_data_success():
    """正常資料應產出 df_info 與 df_status，欄位與型別正確。"""
    raw = [
        {
            "sno": "500101001",
            "sna": "測試站",
            "sarea": "中正區",
            "latitude": 25.04,
            "longitude": 121.52,
            "Quantity": 30,
            "available_rent_bikes": 5,
            "available_return_bikes": 25,
            "srcUpdateTime": "2025-12-10 15:00:00",
        }
    ]
    df_info, df_status = etl_job.transform_data(raw)
    assert df_info is not None and df_status is not None
    assert list(df_info.columns) == [
        "station_no", "name_tw", "district", "lat", "lng", "total_spaces"
    ]
    assert list(df_status.columns) == [
        "station_no", "bikes_available", "spaces_available", "record_time"
    ]
    assert len(df_info) == 1 and len(df_status) == 1
    assert df_info["station_no"].iloc[0] == "500101001"
    assert df_status["bikes_available"].iloc[0] == 5
    assert pd.api.types.is_datetime64_any_dtype(df_status["record_time"])
