import streamlit as st
import requests
import os

# --- 頁面設定 ---
st.set_page_config(page_title="YouBike 預測系統", layout="centered", page_icon="🚲")
st.title("🚲 台北市 YouBike 2.0 流量預測系統")
st.caption("Powered by LSTM Deep Learning Model (Part H)")
st.markdown("---")

# 讀取環境變數
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

# --- 1. 取得站點列表 (動態版) ---
# 讓 API 告訴我們有哪些站點，以及它們的中文名字
@st.cache_data(ttl=60)
def get_station_data():
    try:
        url = f"{API_BASE_URL}/stations"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # 預期 API 回傳格式: {"stations": {"500101": "台大 (大安區)", ...}}
            return response.json().get("stations", {})
        else:
            return {}
    except Exception:
        return {}

# 執行取得站點
station_map = get_station_data()

# --- 側邊欄：輸入參數 ---
st.sidebar.header("🔧 環境參數設定")

# 站點選擇器邏輯
if station_map:
    # 製作下拉選單選項： "台大 (大安區) [500101]"
    station_options = [f"{name} [{sid}]" for sid, name in station_map.items()]
    selected_option = st.sidebar.selectbox("選擇預測站點", station_options)
    
    # 解析出 ID 與 名稱
    selected_station = selected_option.split("[")[-1].replace("]", "")
    selected_station_name = selected_option.split(" [")[0]
else:
    st.sidebar.error("⚠️ 無法連線至 API 或模型未載入")
    selected_station = st.sidebar.text_input("手動輸入站點編號", "500119005") # 預設給個 ID
    selected_station_name = "未知站點"

st.sidebar.markdown("---")
bikes_now = st.sidebar.slider("目前車輛數", 0, 100, 15)
temp_now = st.sidebar.slider("氣溫 (°C)", 10.0, 40.0, 25.0)
rain_now = st.sidebar.slider("降雨量 (mm)", 0.0, 50.0, 0.0)

# --- 視覺化輔助：顯示降雨等級 ---
def get_rain_label(val):
    if val == 0: return "☀️ 晴朗/陰天 (Dry)"
    elif val <= 2: return "🌦️ 毛毛雨 (Drizzle)"
    elif val <= 10: return "🌧️ 下雨 (Rain)"
    else: return "⛈️ 豪大雨 (Heavy)"

st.sidebar.info(f"天氣狀態：{get_rain_label(rain_now)}")

# --- 主畫面顯示 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📍 {selected_station_name}")
    st.caption(f"站點編號：{selected_station}")

with col2:
    st.metric("目前車輛", bikes_now)

# --- 預測按鈕與邏輯 ---
if st.button("🚀 開始預測流量", type="primary", use_container_width=True):
    
    progress_text = "正在呼叫 AI 模型進行運算..."
    my_bar = st.progress(0, text=progress_text)

    api_url = f"{API_BASE_URL}/predict"
    
    # Payload 只需要傳原始數據，API 會自己算 Rain_Cat
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
            c2.metric("氣溫", f"{temp_now}°C")
            c3.metric("降雨", f"{rain_now}mm")
            
            # 智慧建議
            st.markdown("### 💡 AI 調度建議")
            if prediction < 5:
                st.error(f"**🔴 嚴重缺車警示**\n\n預計 1 小時後車輛將耗盡 ({prediction}台)，建議立即調度補車！")
            elif prediction > 30:
                st.warning(f"**🟠 滿站警示**\n\n預計 1 小時後車輛過多 ({prediction}台)，請注意無位可還風險。")
            else:
                st.success(f"**🟢 供需平衡**\n\n預計車輛數為 {prediction} 台，營運狀況良好。")
                
        else:
            my_bar.empty()
            st.error(f"API 錯誤: {response.text}")
            
    except Exception as e:
        my_bar.empty()
        st.error(f"連線錯誤: {e}")