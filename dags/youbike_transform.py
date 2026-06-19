import pandas as pd

STATION_INFO_COLUMNS = ["sno", "sna", "sarea", "latitude", "longitude", "Quantity"]
STATION_STATUS_COLUMNS = ["sno", "available_rent_bikes", "available_return_bikes", "srcUpdateTime"]
STATION_INFO_OUTPUT_COLUMNS = ["station_no", "name_tw", "district", "lat", "lng", "total_spaces"]
STATION_STATUS_OUTPUT_COLUMNS = ["station_no", "bikes_available", "spaces_available", "record_time"]


def transform_youbike_data(raw_data):
    """Transform YouBike API rows into station dimension and status fact dataframes."""
    if not raw_data:
        raise ValueError("Extract 回傳空資料，無法 Transform")

    df = pd.DataFrame(raw_data)

    for col in STATION_INFO_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"station_info 缺少欄位: {col}")
    for col in STATION_STATUS_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"station_status 缺少欄位: {col}")

    df_info = df[STATION_INFO_COLUMNS].copy()
    df_info.columns = STATION_INFO_OUTPUT_COLUMNS
    df_info = df_info.drop_duplicates(subset=["station_no"])

    df_status = df[STATION_STATUS_COLUMNS].copy()
    df_status.columns = STATION_STATUS_OUTPUT_COLUMNS

    # API time is Taipei local time. Store UTC as naive datetime for MySQL compatibility.
    ts = pd.to_datetime(df_status["record_time"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Taipei", ambiguous="infer")
    df_status["record_time"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)

    return df_info, df_status


def validate_transformed_data(df_info, df_status):
    """Validate transformed dataframes before loading or downstream testing."""
    missing_info = [col for col in STATION_INFO_OUTPUT_COLUMNS if col not in df_info.columns]
    if missing_info:
        raise ValueError(f"station_info transform 缺少欄位: {missing_info}")

    missing_status = [col for col in STATION_STATUS_OUTPUT_COLUMNS if col not in df_status.columns]
    if missing_status:
        raise ValueError(f"station_status transform 缺少欄位: {missing_status}")

    if df_info["station_no"].duplicated().any():
        raise ValueError("station_info 含重複 station_no")

    duplicated_status = df_status.duplicated(subset=["station_no", "record_time"])
    if duplicated_status.any():
        raise ValueError("station_status 含重複 station_no + record_time")

    total_spaces = pd.to_numeric(df_info["total_spaces"], errors="coerce")
    bikes_available = pd.to_numeric(df_status["bikes_available"], errors="coerce")
    spaces_available = pd.to_numeric(df_status["spaces_available"], errors="coerce")

    if total_spaces.isna().any() or bikes_available.isna().any() or spaces_available.isna().any():
        raise ValueError("station availability 欄位含非數值")

    if (total_spaces < 0).any() or (bikes_available < 0).any() or (spaces_available < 0).any():
        raise ValueError("station availability 欄位含負值")

    if not pd.api.types.is_datetime64_any_dtype(df_status["record_time"]):
        raise ValueError("station_status record_time 必須是 datetime")


def validate_transformed_data_for_load(df_info, df_status, mode="strict"):
    """Run load-time validation.

    Returns a warning message in warn mode and raises in strict mode.
    """
    validation_mode = mode.lower()
    if validation_mode not in {"strict", "warn"}:
        raise ValueError("ETL_VALIDATION_MODE 必須是 strict 或 warn")

    try:
        validate_transformed_data(df_info, df_status)
    except ValueError as exc:
        if validation_mode == "warn":
            return str(exc)
        raise

    return None
