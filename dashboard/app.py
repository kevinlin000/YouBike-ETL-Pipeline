import os

import pandas as pd
import requests
import streamlit as st

from api_client import (
    DashboardApiError,
    get_station_data,
    parse_station_option,
    predict_station,
    rank_station_risks,
    risk_level_label,
    station_display_options,
    suggested_action_label,
)


st.set_page_config(page_title="YouBike Prediction Dashboard", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")


@st.cache_data(ttl=60)
def load_station_map(api_base_url: str) -> dict:
    try:
        return get_station_data(api_base_url)
    except (DashboardApiError, requests.RequestException):
        return {}


def rain_label(value: float) -> str:
    if value == 0:
        return "Dry"
    if value <= 2:
        return "Drizzle"
    if value <= 10:
        return "Rain"
    return "Heavy rain"


def station_name(station_map: dict, station_no: str) -> str:
    return station_map.get(station_no, station_no)


def render_single_prediction(
    selected_station: str,
    selected_station_name: str,
    temperature: float,
    rain: float,
) -> None:
    st.subheader("單站一小時預測")

    bikes_now = st.slider("目前可借車輛數", 0, 100, 15)
    st.markdown(f"**站點：** {selected_station_name}")
    st.caption(f"站點編號：{selected_station}")

    col1, col2 = st.columns(2)
    col1.metric("目前車輛", bikes_now)
    col2.metric("天氣", rain_label(rain))

    if st.button("執行單站預測", type="primary", use_container_width=True):
        try:
            result = predict_station(
                API_BASE_URL,
                selected_station,
                bikes_now,
                temperature,
                rain,
            )
        except (DashboardApiError, requests.RequestException) as exc:
            st.error(f"API 呼叫失敗：{exc}")
            return

        prediction = result["predicted_bikes_next_hour"]
        delta = prediction - bikes_now
        st.success("預測完成")

        p1, p2, p3 = st.columns(3)
        p1.metric("一小時後預測車輛", f"{prediction} 台", delta=delta)
        p2.metric("氣溫", f"{temperature:.1f}°C")
        p3.metric("降雨", f"{rain:.1f} mm")

        if prediction <= 2:
            st.error(f"嚴重缺車風險：預測剩餘 {prediction} 台，建議優先補車。")
        elif prediction <= 5:
            st.warning(f"車輛偏低：預測剩餘 {prediction} 台，建議持續監控。")
        else:
            st.info(f"供需相對穩定：預測剩餘 {prediction} 台。")


def default_risk_rows(station_options: list[str]) -> list[dict]:
    rows = []
    for option in station_options[:5]:
        station_no, station_label = parse_station_option(option)
        rows.append(
            {
                "station_no": station_no,
                "station_name": station_label,
                "bikes_available": 12,
                "spaces_available": 8,
            }
        )
    return rows


def render_risk_ranking(
    station_map: dict,
    station_options: list[str],
    temperature: float,
    rain: float,
) -> None:
    st.subheader("多站點風險排序")

    if not station_options:
        st.warning("無法從 API 取得模型支援站點，請確認 FastAPI 服務與模型檔是否已載入。")
        return

    selected_options = st.multiselect(
        "選擇要評估的站點",
        station_options,
        default=station_options[: min(5, len(station_options))],
    )
    if not selected_options:
        st.warning("請至少選擇一個站點。")
        return

    editable_rows = default_risk_rows(selected_options)
    edited_df = st.data_editor(
        pd.DataFrame(editable_rows),
        hide_index=True,
        use_container_width=True,
        disabled=["station_no", "station_name"],
        column_config={
            "station_no": "站點編號",
            "station_name": "站點名稱",
            "bikes_available": st.column_config.NumberColumn("目前可借車輛", min_value=0, max_value=200),
            "spaces_available": st.column_config.NumberColumn("目前可還空位", min_value=0, max_value=200),
        },
    )

    if st.button("評估多站點風險", type="primary", use_container_width=True):
        stations = edited_df[["station_no", "bikes_available", "spaces_available"]].to_dict("records")
        if not stations:
            st.warning("請至少選擇一個站點。")
            return

        try:
            risks = rank_station_risks(API_BASE_URL, stations, temperature, rain)
        except (DashboardApiError, requests.RequestException) as exc:
            st.error(f"API 呼叫失敗：{exc}")
            return

        if not risks:
            st.warning("API 未回傳風險排序結果。")
            return

        result_df = pd.DataFrame(risks)
        result_df["station_name"] = result_df["station_no"].map(lambda sid: station_name(station_map, sid))
        result_df["risk_level"] = result_df["risk_level"].map(risk_level_label)
        result_df["suggested_action"] = result_df["suggested_action"].map(suggested_action_label)
        result_df = result_df[
            [
                "station_no",
                "station_name",
                "current_bikes_available",
                "current_spaces_available",
                "predicted_bikes_next_hour",
                "predicted_spaces_next_hour",
                "risk_level",
                "risk_score",
                "suggested_action",
            ]
        ]

        st.dataframe(
            result_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "station_no": "站點編號",
                "station_name": "站點名稱",
                "current_bikes_available": "目前車輛",
                "current_spaces_available": "目前空位",
                "predicted_bikes_next_hour": "預測車輛",
                "predicted_spaces_next_hour": "預測空位",
                "risk_level": "風險等級",
                "risk_score": "風險分數",
                "suggested_action": "建議動作",
            },
        )


st.title("台北市 YouBike 2.0 預測與調度輔助")
st.caption("FastAPI + PyTorch LSTM + Streamlit")

station_map = load_station_map(API_BASE_URL)
station_options = station_display_options(station_map)

with st.sidebar:
    st.header("輸入參數")
    st.caption(f"API: {API_BASE_URL}")

    if station_options:
        selected_option = st.selectbox("單站預測站點", station_options)
        selected_station, selected_station_name = parse_station_option(selected_option)
    else:
        st.error("無法連線至 API 或模型尚未載入")
        selected_station = st.text_input("手動輸入站點編號", "500119005")
        selected_station_name = "未知站點"

    temperature = st.slider("氣溫 (°C)", 10.0, 40.0, 25.0)
    rain = st.slider("降雨量 (mm)", 0.0, 50.0, 0.0)
    st.info(f"天氣狀態：{rain_label(rain)}")

single_tab, risk_tab = st.tabs(["單站預測", "多站風險排序"])

with single_tab:
    render_single_prediction(
        selected_station,
        selected_station_name,
        temperature,
        rain,
    )

with risk_tab:
    render_risk_ranking(
        station_map,
        station_options,
        temperature,
        rain,
    )
