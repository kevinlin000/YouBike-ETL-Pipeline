from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
import time
import logging
from google.cloud import secretmanager
import os

try:
    from youbike_transform import transform_youbike_data, validate_transformed_data_for_load
except ModuleNotFoundError:
    from dags.youbike_transform import transform_youbike_data, validate_transformed_data_for_load

# --- Logging ---
logger = logging.getLogger(__name__)

# --- 設定區 (Configuration) ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "youbike-airflow-server")
SECRET_ID = os.getenv("GCP_SECRET_ID", "mysql_password")
VERSION_ID = "latest"

API_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
API_TIMEOUT_SEC = int(os.getenv("API_TIMEOUT", "30"))
API_RETRIES = int(os.getenv("API_RETRIES", "3"))
API_RETRY_BACKOFF_SEC = float(os.getenv("API_RETRY_BACKOFF", "2.0"))


def get_gcp_secret(secret_id: str) -> str:
    """從 GCP Secret Manager 獲取密碼"""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{VERSION_ID}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.warning("無法從 GCP 獲取密鑰: %s", e)
        env_pw = os.getenv("DB_PASSWORD")
        if env_pw:
            return env_pw
        raise

# --- 使用密碼與連線設定（優先環境變數，其次 GCP Secret） ---
DB_USER = os.getenv("DB_USER", "admin")
_db_password_env = os.getenv("DB_PASSWORD")
if _db_password_env:
    DB_PASSWORD = _db_password_env
else:
    DB_PASSWORD = get_gcp_secret(SECRET_ID)

DB_HOST = os.getenv("DB_HOST", "mysql-db")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "youbike_db")

# 建立資料庫連線引擎（模組級別，共用連線池）
engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    pool_pre_ping=True,
)


def validate_before_load(df_info, df_status):
    """Validate transformed data before loading; strict mode fails, warn mode logs."""
    validation_mode = os.getenv("ETL_VALIDATION_MODE", "strict").lower()
    warning = validate_transformed_data_for_load(df_info, df_status, mode=validation_mode)
    if warning:
        logger.warning("ETL validation failed but continuing because mode=warn: %s", warning)


def etl_process():
    """Extract -> Transform -> Load，任一步失敗即拋出例外讓 Airflow 標記失敗。"""
    # 1. Extract（含 timeout 與 retry）
    last_error = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            response = requests.get(API_URL, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            raw_data = response.json()
            break
        except Exception as e:
            last_error = e
            logger.warning("Extract 失敗 (嘗試 %s/%s): %s", attempt, API_RETRIES, e)
            if attempt < API_RETRIES:
                time.sleep(API_RETRY_BACKOFF_SEC * attempt)
    else:
        logger.error("Extract 最終失敗: %s", last_error)
        raise last_error

    # 2. Transform（失敗即拋出）
    df_info, df_status = transform_youbike_data(raw_data)
    validate_before_load(df_info, df_status)

    # 3. Load（失敗即拋出，唯一鍵衝突僅記錄）
    with engine.connect() as conn:
        existing_ids = pd.read_sql("SELECT station_no FROM station_info", conn)
    new_stations = df_info[~df_info["station_no"].isin(existing_ids["station_no"])]

    if not new_stations.empty:
        new_stations.to_sql("station_info", con=engine, if_exists="append", index=False)
        logger.info("New stations added: %s", len(new_stations))

    try:
        df_status.to_sql("station_status", con=engine, if_exists="append", index=False)
        logger.info("Status data loaded: %s rows", len(df_status))
    except IntegrityError as ie:
        logger.info("Status rows already exist, skipping duplicates: %s", ie)


# --- Airflow DAG 設定 ---
default_args = {
    "owner": "Kevin",
    "depends_on_past": False,
    "start_date": datetime(2025, 12, 7),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "youbike_etl_v1",
    default_args=default_args,
    description="YouBike ETL Pipeline",
    schedule_interval="*/10 * * * *",
    catchup=False,
    tags=["SDG11"],
) as dag:

    run_etl_task = PythonOperator(
        task_id="run_youbike_etl",
        python_callable=etl_process,
    )
