from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from sqlalchemy import create_engine
import time
from google.cloud import secretmanager
import os

# --- 設定區 (Configuration) ---
PROJECT_ID = 'youbike-airflow-server' 
SECRET_ID = 'mysql_password' 
VERSION_ID = 'latest'

def get_gcp_secret(secret_id):
    """從 GCP Secret Manager 獲取密碼"""
    try:
        # 建立 Client
        client = secretmanager.SecretManagerServiceClient()
        # 組合完整的路徑
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/{VERSION_ID}"
        # 呼叫 API
        response = client.access_secret_version(request={"name": name})
        # 解碼回傳的 payload
        password = response.payload.data.decode("UTF-8")
        return password
    except Exception as e:
        print(f"無法獲取密鑰: {e}")
        # 這裡可以做一個 fallback，如果在本地測試失敗，就用環境變數或預設值
        return "123456" 

# --- 使用密碼 ---
DB_USER = 'admin'
DB_PASSWORD = get_gcp_secret(SECRET_ID) 
DB_HOST = 'mysql-db'
DB_PORT = '3306'
DB_NAME = 'youbike_db'

API_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"

def etl_process():
    print("Start Extracting...")
    # 1. Extract
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        raw_data = response.json()
    except Exception as e:
        print(f"Extract Error: {e}")
        return

    print("Start Transforming...")
    # 2. Transform
    df = pd.DataFrame(raw_data)
    
    # 處理 Info
    try:
        df_info = df[['sno', 'sna', 'sarea', 'latitude', 'longitude', 'Quantity']].copy()
        df_info.columns = ['station_no', 'name_tw', 'district', 'lat', 'lng', 'total_spaces']
        df_info = df_info.drop_duplicates(subset=['station_no'])
        
        # 處理 Status
        df_status = df[['sno', 'available_rent_bikes', 'available_return_bikes', 'srcUpdateTime']].copy() 
        df_status.columns = ['station_no', 'bikes_available', 'spaces_available', 'record_time']
        df_status['record_time'] = pd.to_datetime(df_status['record_time'])
    except Exception as e:
        print(f"Transform Error: {e}")
        return

    print("Start Loading...")
    # 3. Load
    try:
        # 建立連線引擎
        engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 寫入 Info (使用簡單的 append，忽略重複錯誤需進階處理，這邊先求有)
        # 為了避免 Info 重複寫入報錯，我們先只查不存在的寫入 
        with engine.connect() as conn:
            existing_ids = pd.read_sql("SELECT station_no FROM station_info", conn)
        
        new_stations = df_info[~df_info['station_no'].isin(existing_ids['station_no'])]
        
        if not new_stations.empty:
            new_stations.to_sql('station_info', con=engine, if_exists='append', index=False)
            print(f"New stations added: {len(new_stations)}")
            
        # 寫入 Status (Log 檔直接 append)
        df_status.to_sql('station_status', con=engine, if_exists='append', index=False)
        print(f"Status data loaded: {len(df_status)} rows")
        
    except Exception as e:
        print(f"Load Error: {e}")
        raise e # 拋出錯誤讓 Airflow 知道這一次失敗了

# --- Airflow DAG 設定 ---
default_args = {
    'owner': 'Kevin',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 7),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'youbike_etl_v1',              # DAG ID
    default_args=default_args,
    description='YouBike ETL Pipeline',
    schedule_interval='*/10 * * * *', # 每 10 分鐘執行一次
    catchup=False,
    tags=['SDG11'],
) as dag:

    run_etl_task = PythonOperator(
        task_id='run_youbike_etl',
        python_callable=etl_process
    )