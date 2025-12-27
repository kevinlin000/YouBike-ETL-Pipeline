import streamlit as st
import requests
import pandas as pd
import os

# --- 頁面設定 ---
st.set_page_config(page_title="YouBike 預測系統", layout="centered")
st.title("🚲 台北市 YouBike 2.0 流量預測系統")
st.markdown("---")

# 讀取環境變數 (Docker 會自動傳入 http://api:8000)
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

# --- 站點名稱對照表 (這裡可以擴充) ---
STATION_NAME_MAP = {
    '500101001': '捷運科技大樓站 (大安區)', 
    '500103001': '延平國宅 (大同區)', 
    '500104001': '劍潭抽水站 (士林區)', 
    '500105001': '台北花木批發市場 (文山區)', 
    '500106001': '臺北自來水事業處 (中正區)', 
    '500107001': '通北街65巷口 (中山區)', 
    '500108001': '文湖街21巷118弄口 (內湖區)', 
    '500109001': '承德路七段304巷口 (北投區)', 
    '500110002': '捷運松山站(4號出口) (松山區)', 
    '500111001': '南港公園(東新街) (南港區)', 
    '500112001': '黎忠區民活動中心 (信義區)', 
    '500113001': '德昌寶興街口(西北角) (萬華區)', 
    '500119005': '臺大水源舍區A棟 (臺大公館校區)'
}

# --- 1. 取得站點列表 ---
# 使用 ttl=60 (秒) 讓它每分鐘會嘗試重新抓一次，避免永遠卡在錯誤
@st.cache_data(ttl=60)
def get_supported_stations():
    try:
        url = f"{API_BASE_URL}/stations"
        response = requests.get(url, timeout=5) # 設定超時避免卡死
        if response.status_code == 200:
            return response.json().get("supported_stations", [])
        else:
            return []
    except Exception:
        return []

# 執行取得站點
raw_station_list = get_supported_stations()
station_list = [str(s) for s in raw_station_list]

# --- 側邊欄：輸入參數 ---
st.sidebar.header("🔧 環境參數設定")

# 站點選擇器邏輯
if station_list:
    # 如果 API 活著，顯示漂亮的下拉選單
    display_options = []
    for s_id in station_list:
        name = STATION_NAME_MAP.get(s_id, s_id) # 查不到名字就顯示 ID
        display_options.append(f"{name} [{s_id}]")
    
    selected_option = st.sidebar.selectbox("選擇預測站點", display_options)
    selected_station = selected_option.split("[")[-1].replace("]", "")
    selected_station_name = selected_option.split(" [")[0]
else:
    # 如果 API 連不上，顯示紅字但允許手動輸入 (Fallback)
    st.sidebar.error("⚠️ 無法連線至 API Server")
    selected_station = st.sidebar.text_input("手動輸入站點編號", "500101001")
    selected_station_name = "自訂站點"

st.sidebar.markdown("---")
bikes_now = st.sidebar.slider("目前車輛數", 0, 100, 15)
temp_now = st.sidebar.slider("氣溫 (°C)", 10.0, 40.0, 25.0)
rain_now = st.sidebar.slider("降雨量 (mm)", 0.0, 50.0, 0.0)

# --- 主畫面顯示 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📍 {selected_station_name}")
    st.caption(f"站點編號：{selected_station}")

with col2:
    # 顯示即時狀態卡片
    st.metric("目前車輛", bikes_now)

# --- 預測按鈕與邏輯 ---
if st.button("🚀 開始預測流量", type="primary", use_container_width=True):
    
    # 準備進度條
    progress_text = "正在呼叫 AI 模型進行運算..."
    my_bar = st.progress(0, text=progress_text)

    api_url = f"{API_BASE_URL}/predict"
    payload = {
        "station_no": str(selected_station), 
        "bikes_available": bikes_now,
        "temperature": temp_now,
        "rain": rain_now
    }

    try:
        my_bar.progress(50, text="連線至 API...")
        response = requests.post(api_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            my_bar.progress(100, text="運算完成！")
            result = response.json()
            prediction = result['predicted_bikes_next_hour']
            
            # --- 顯示漂亮結果 ---
            st.success("✅ 預測成功")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("1小時後預測", f"{prediction} 台", delta=f"{prediction - bikes_now}")
            c2.metric("氣溫條件", f"{temp_now}°C")
            c3.metric("降雨條件", f"{rain_now}mm")
            
            # 智慧建議
            st.markdown("### 💡 調度建議")
            if prediction < 5:
                st.error(f"**缺車警示 (High Demand)**\n\n預計 1 小時後車輛極少 ({prediction}台)，建議即刻調度補車。")
            elif prediction > 25:
                st.warning(f"**滿站警示 (High Supply)**\n\n預計 1 小時後車輛過多 ({prediction}台)，建議暫停補車以免無位可還。")
            else:
                st.info(f"**供需平衡 (Balanced)**\n\n預計車輛數為 {prediction} 台，維持現狀即可。")
                
        else:
            my_bar.empty()
            st.error(f"API 請求失敗: {response.text}")
            
    except Exception as e:
        my_bar.empty()
        st.error(f"連線錯誤: {e}")
        st.caption("請檢查 API 容器是否已啟動")

# 頁尾
st.markdown("---")
st.caption("Created by YouBike Data Engineering Team | Powered by PyTorch & FastAPI")