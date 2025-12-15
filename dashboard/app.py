import streamlit as st
import requests
import pandas as pd

# --- 頁面設定 ---
st.set_page_config(page_title="YouBike 預測系統", layout="centered")
st.title("台北市 YouBike 2.0 流量預測系統")
st.caption("技術：PyTorch Multi-Station LSTM & FastAPI")

API_BASE_URL = "http://127.0.0.1:8000"

# --- 站點名稱對照表 ---
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

# --- 輔助函數 ---
@st.cache_data
def get_supported_stations():
    try:
        url = f"{API_BASE_URL}/stations"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()["supported_stations"]
        else:
            st.error("[錯誤] 無法取得站點列表")
            return []
    except Exception as e:
        st.error(f"[錯誤] API 連線失敗: {e}")
        return []

# 取得列表並確保型別為字串 (Type Safety)
raw_station_list = get_supported_stations()
station_list = [str(s) for s in raw_station_list] if raw_station_list else []

# --- 側邊欄設定 ---
st.sidebar.header("環境參數設定")

# [修改] 站點選擇：顯示中文名稱
if station_list:
    # 1. 製作顯示用的清單： "中文站名 [ID]"
    display_options = []
    for s_id in station_list:
        # 查字典，查不到就用 ID 代替
        name = STATION_NAME_MAP.get(s_id, s_id)
        display_options.append(f"{name} [{s_id}]")
    
    # 2. 顯示下拉選單
    selected_option = st.sidebar.selectbox("選擇預測站點", display_options)
    
    # 3. 從選單字串中把 ID 拆出來傳給 API
    # 例如 "捷運科技大樓站 (大安區) [500101001]" -> 取出 "500101001"
    selected_station = selected_option.split("[")[-1].replace("]", "")
    
    # 為了畫面好看，我們把中文名字也存下來顯示在主標題
    selected_station_name = selected_option.split(" [")[0]
else:
    selected_station = st.sidebar.text_input("輸入站點編號", "500101001")
    selected_station_name = "未知站點"

st.sidebar.markdown("---")

# 特徵輸入
bikes_now = st.sidebar.slider("目前車輛數", 0, 100, 15)
temp_now = st.sidebar.slider("氣溫 (°C)", 10.0, 40.0, 25.0)
rain_now = st.sidebar.slider("降雨量 (mm)", 0.0, 50.0, 0.0)

# --- 主畫面顯示 ---
st.write(f"### 站點：`{selected_station_name}`")
st.caption(f"站點編號：{selected_station}")

col1, col2, col3 = st.columns(3)
col1.metric("車輛數", bikes_now)
col2.metric("氣溫", f"{temp_now} °C")
col3.metric("降雨量", f"{rain_now} mm")

# --- 預測邏輯 ---
if st.button("開始預測流量", type="primary"):
    api_url = f"{API_BASE_URL}/predict"
    
    # 建構 Payload (確保型別正確)
    payload = {
        "station_no": str(selected_station), 
        "bikes_available": bikes_now,
        "temperature": temp_now,
        "rain": rain_now
    }

    try:
        with st.spinner(f'正在分析 {selected_station_name} 的數據...'):
            response = requests.post(api_url, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            prediction = result['predicted_bikes_next_hour']
            
            st.success("預測成功")
            st.markdown(f"### 預測一小時後車輛數： **{prediction}** 台")
            
            # 商業邏輯建議
            if prediction < 3:
                st.error("[警示] 預期缺車 (Low Supply)，建議調度補車。")
            elif prediction > 30:
                st.warning("[警示] 預期滿站 (High Supply)，建議暫停補車。")
            else:
                st.info("供需平衡狀態。")
                
            with st.expander("查看 API 原始回傳資料"):
                st.json(result)
                
        else:
            st.error(f"API 請求失敗: {response.text}")
            
    except Exception as e:
        st.error(f"連線錯誤: {e}")

st.divider()
st.markdown("Created by **[Kevin Lin]** | End-to-End Data Engineering Project")