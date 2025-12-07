import requests
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import time

# --- 設定區 (Configuration) ---
DB_USER = 'admin'        # 改用新建立的帳號
DB_PASSWORD = '123456'   # 剛剛設定的密碼
DB_HOST = '127.0.0.1'    
DB_PORT = '3306'
DB_NAME = 'youbike_db'

# YouBike 2.0 API URL (台北市)
API_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"

# 建立資料庫連線引擎
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def extract_data():
    """Extract: 從 API 抓取資料"""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
        print(f"[{datetime.now()}] 成功抓取 {len(data)} 筆資料")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def transform_data(raw_data):
    """Transform: 資料清洗與整理 (修正版)"""
    if not raw_data:
        return None, None
    
    df = pd.DataFrame(raw_data)
    
    # --- 1. 整理 [station_info] 站點基本資料 ---
    # 使用正確的 API 欄位名稱: sno, sna, sarea, latitude, longitude, Quantity
    # 注意：這裡選取的欄位名稱必須跟 API 給的一模一樣
    try:
        df_info = df[['sno', 'sna', 'sarea', 'latitude', 'longitude', 'Quantity']].copy()
        
        # 將欄位重新命名，對應到我們 MySQL 資料庫的欄位名稱
        df_info.columns = ['station_no', 'name_tw', 'district', 'lat', 'lng', 'total_spaces']
        
        # 去除重複 (確保同一個站點只有一筆資料)
        df_info = df_info.drop_duplicates(subset=['station_no'])
    except KeyError as e:
        print(f"資料轉換錯誤 (Info): 找不到欄位 {e}")
        return None, None

    # --- 2. 整理 [station_status] 狀態紀錄資料 ---
    # 使用正確的 API 欄位名稱: sno, available_rent_bikes, available_return_bikes, srcUpdateTime
    try:
        df_status = df[['sno', 'available_rent_bikes', 'available_return_bikes', 'srcUpdateTime']].copy() 
        
        # 將欄位重新命名，對應到我們 MySQL 資料庫的欄位名稱
        df_status.columns = ['station_no', 'bikes_available', 'spaces_available', 'record_time']
        
        # 轉換時間格式 (以防萬一 API 給的時間格式 SQL 不吃)
        df_status['record_time'] = pd.to_datetime(df_status['record_time'])
        
    except KeyError as e:
        print(f"資料轉換錯誤 (Status): 找不到欄位 {e}")
        return None, None
    
    return df_info, df_status

def load_data(df_info, df_status):
    """Load: 寫入 MySQL 資料庫"""
    if df_info is None or df_status is None:
        return

    # 1. 更新站點資訊 (使用 'replace' 會 drop table，這裡我們用 'append' 搭配 SQL 語法處理重複比較好)
    # 但為了簡化示範 Data Engineer 常用技巧：Upsert (Update or Insert)
    # 這裡我們先簡單做：先寫入 info (如果是第一次)，實務上通常檢查是否有新站點
    try:
        # 使用 SQLAlchemy 的一招：先查詢現有站點，只 Insert 新的
        existing_ids = pd.read_sql("SELECT station_no FROM station_info", engine)
        new_stations = df_info[~df_info['station_no'].isin(existing_ids['station_no'])]
        
        if not new_stations.empty:
            new_stations.to_sql('station_info', con=engine, if_exists='append', index=False)
            print(f"新增 {len(new_stations)} 個新站點")
        
        # 2. 寫入狀態紀錄 (這是 Log，一直 append 即可)
        df_status.to_sql('station_status', con=engine, if_exists='append', index=False)
        print("狀態資料寫入完成")
        
    except Exception as e:
        print(f"Database Error: {e}")

if __name__ == "__main__":
    # 這裡模擬排程，每 10 分鐘跑一次
    # 在真正的 DE 專案中，這裡會用 Airflow 或 Crontab 來觸發，而不是用 while 迴圈
    # 但為了讓你現在就能跑，我們先用 loop
    print("ETL Service Started...")
    while True:
        data = extract_data()
        df_info, df_status = transform_data(data)
        load_data(df_info, df_status)
        
        print("等待 10 分鐘...\n")
        time.sleep(600) # 600秒 = 10分鐘