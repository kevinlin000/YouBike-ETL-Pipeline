import os
from html import escape

import pandas as pd
import requests
import streamlit as st

from api_client import (
    DashboardApiError,
    demo_predict_station,
    demo_rank_station_risks,
    get_station_data,
    get_demo_station_data,
    parse_station_option,
    predict_station,
    rank_station_risks,
    risk_level_label,
    station_display_options,
    suggested_action_label,
)


st.set_page_config(page_title="YouBike 調度風險工作台", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
DEMO_MODE_DEFAULT = os.getenv("DASHBOARD_DEMO_MODE", "").lower() in {"1", "true", "yes", "on"}

RISK_CLASS_MAP = {
    "嚴重缺車": "risk-critical",
    "滿站風險": "risk-full",
    "車輛偏低": "risk-warning",
    "空位偏低": "risk-dock",
    "供需穩定": "risk-normal",
}

DEMO_RISK_DEFAULTS = {
    "500101001": (4, 16),
    "500101002": (5, 15),
    "500101003": (18, 2),
    "500101004": (10, 10),
    "500101005": (16, 4),
    "500101006": (13, 7),
}


def apply_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #64748b;
            --line: #dbe3ed;
            --panel: #ffffff;
            --soft: #f8fafc;
            --soft-blue: #eef6ff;
            --teal: #0f766e;
            --blue: #2563eb;
            --amber: #b45309;
            --red: #b91c1c;
            --green: #15803d;
        }

        .stApp {
            background: #ffffff;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.2rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background: #f6f8fb;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h2 {
            color: var(--ink);
            font-size: 1.12rem;
            margin-bottom: 0.6rem;
        }

        #MainMenu, footer, [data-testid="stDecoration"], .stAppDeployButton {
            visibility: hidden;
        }

        .dashboard-hero {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0.8rem 0 1rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.9rem;
        }

        .dashboard-hero h1 {
            margin: 0.15rem 0 0.35rem;
            color: var(--ink);
            font-size: clamp(1.8rem, 2.4vw, 2.35rem);
            line-height: 1.12;
            letter-spacing: 0;
        }

        .dashboard-hero p {
            color: var(--muted);
            margin: 0;
            max-width: 780px;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .eyebrow {
            color: var(--teal) !important;
            font-size: 0.82rem !important;
            font-weight: 700;
            letter-spacing: 0;
        }

        .run-state {
            min-width: 190px;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
            background: #ffffff;
        }

        .run-state span {
            display: block;
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.2rem;
        }

        .run-state strong {
            color: var(--ink);
            font-size: 0.98rem;
        }

        .section-note {
            color: var(--muted);
            margin-top: -0.25rem;
            margin-bottom: 0.9rem;
            line-height: 1.5;
        }

        .sidebar-note {
            border: 1px solid #bfdbfe;
            background: var(--soft-blue);
            color: #1e3a8a;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0.35rem 0 1rem;
        }

        .sidebar-note strong {
            display: block;
            color: #172554;
            margin-bottom: 0.18rem;
        }

        .sidebar-status {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin-top: 0.9rem;
        }

        .sidebar-status span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .sidebar-status strong {
            color: var(--ink);
            font-size: 1rem;
        }

        .priority-list {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.5rem 0 1rem;
        }

        .priority-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 0.85rem;
            min-height: 136px;
        }

        .priority-card.risk-critical {
            border-left: 5px solid var(--red);
        }

        .priority-card.risk-full {
            border-left: 5px solid var(--amber);
        }

        .priority-card.risk-warning,
        .priority-card.risk-dock {
            border-left: 5px solid var(--blue);
        }

        .priority-card.risk-normal {
            border-left: 5px solid var(--green);
        }

        .priority-card .rank {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.32rem;
        }

        .priority-card .station {
            color: var(--ink);
            font-weight: 700;
            line-height: 1.35;
            min-height: 2.7rem;
        }

        .priority-card .risk {
            display: inline-block;
            margin: 0.5rem 0 0.45rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .priority-card .stats {
            display: flex;
            gap: 0.75rem;
            color: var(--muted);
            font-size: 0.84rem;
        }

        .priority-card .action {
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 0.45rem;
        }

        div.stButton > button[kind="primary"] {
            background: var(--teal);
            border-color: var(--teal);
            color: white;
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.55rem;
        }

        div.stButton > button[kind="primary"]:hover {
            background: #115e59;
            border-color: #115e59;
            color: white;
        }

        [data-baseweb="tag"] {
            background: #eaf5ff !important;
            color: #0f172a !important;
            border-radius: 7px !important;
        }

        [data-baseweb="tag"] svg {
            fill: #475569 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: var(--teal);
        }

        @media (max-width: 900px) {
            .dashboard-hero {
                align-items: stretch;
                flex-direction: column;
                gap: 0.9rem;
            }

            .priority-list {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def load_station_map(api_base_url: str, demo_mode: bool) -> dict:
    if demo_mode:
        return get_demo_station_data()
    try:
        return get_station_data(api_base_url)
    except (DashboardApiError, requests.RequestException):
        return {}


def rain_label(value: float) -> str:
    if value == 0:
        return "無雨"
    if value <= 2:
        return "小雨"
    if value <= 10:
        return "降雨"
    return "大雨"


def station_name(station_map: dict, station_no: str) -> str:
    return station_map.get(station_no, station_no)


def render_dashboard_header(demo_mode: bool) -> None:
    mode_label = "資料來源"
    source_label = "固定範例資料" if demo_mode else f"FastAPI：{API_BASE_URL}"
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div>
                <p class="eyebrow">站點供需監控</p>
                <h1>YouBike 調度風險工作台</h1>
                <p>整合模型時窗預測與規則化風險分數，檢視單站水位、多站缺車風險與滿站風險，協助排序調度處理順序。</p>
            </div>
            <div class="run-state">
                <span>{escape(mode_label)}</span>
                <strong>{escape(source_label)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_priority_cards(result_df: pd.DataFrame) -> None:
    top_rows = result_df.head(3).to_dict("records")
    cards = []
    for index, row in enumerate(top_rows, start=1):
        risk_class = RISK_CLASS_MAP.get(row["risk_label"], "risk-normal")
        cards.append(
            "<div class=\"priority-card "
            f"{risk_class}\">"
            f"<div class=\"rank\">調度順位 {index}</div>"
            f"<div class=\"station\">{escape(row['station_name'])}</div>"
            f"<div class=\"risk\">{escape(row['risk_label'])} · {int(row['risk_score'])} 分</div>"
            "<div class=\"stats\">"
            f"<span>預測車輛 {int(row['predicted_bikes_next_hour'])} 台</span>"
            f"<span>預測空位 {int(row['predicted_spaces_next_hour'])} 格</span>"
            "</div>"
            f"<div class=\"action\">{escape(row['action_label'])}</div>"
            "</div>"
        )

    st.markdown(
        f"""<div class="priority-list">{''.join(cards)}</div>""",
        unsafe_allow_html=True,
    )


def render_single_prediction(
    selected_station: str,
    selected_station_name: str,
    temperature: float,
    rain: float,
    demo_mode: bool,
) -> None:
    st.subheader("單站水位預測")
    st.markdown(
        '<p class="section-note">依目前車輛數與天氣條件，檢查指定站點在模型時窗內的供需狀態。</p>',
        unsafe_allow_html=True,
    )

    bikes_now = st.slider("目前可借車輛", 0, 100, 15)
    st.markdown(f"**站點：** {selected_station_name}")
    st.caption(f"站點編號：{selected_station}")

    col1, col2 = st.columns(2)
    col1.metric("目前車輛", bikes_now)
    col2.metric("天氣", rain_label(rain))

    if st.button("更新單站預測", type="primary", width="stretch"):
        if demo_mode:
            result = demo_predict_station(selected_station, bikes_now, temperature, rain)
        else:
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
        p1.metric("模型時窗預測車輛", f"{prediction} 台", delta=delta)
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
        bikes_available, spaces_available = DEMO_RISK_DEFAULTS.get(station_no, (12, 8))
        rows.append(
            {
                "station_no": station_no,
                "station_name": station_label,
                "bikes_available": bikes_available,
                "spaces_available": spaces_available,
            }
        )
    return rows


def render_risk_ranking(
    station_map: dict,
    station_options: list[str],
    temperature: float,
    rain: float,
    demo_mode: bool,
) -> None:
    st.subheader("多站風險排序")
    st.markdown(
        '<p class="section-note">比較多個站點的預測水位，將缺車、滿站與觀察名單依風險分數排序。</p>',
        unsafe_allow_html=True,
    )

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
        width="stretch",
        disabled=["station_no", "station_name"],
        column_config={
            "station_no": "站點編號",
            "station_name": "站點名稱",
            "bikes_available": st.column_config.NumberColumn("目前可借車輛", min_value=0, max_value=200),
            "spaces_available": st.column_config.NumberColumn("目前可還空位", min_value=0, max_value=200),
        },
    )

    if st.button("更新風險排序", type="primary", width="stretch"):
        stations = edited_df[["station_no", "bikes_available", "spaces_available"]].to_dict("records")
        if not stations:
            st.warning("請至少選擇一個站點。")
            return

        if demo_mode:
            risks = demo_rank_station_risks(stations, temperature, rain)
        else:
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
        result_df["risk_label"] = result_df["risk_level"].map(risk_level_label)
        result_df["action_label"] = result_df["suggested_action"].map(suggested_action_label)

        render_priority_cards(result_df)

        display_df = result_df[
            [
                "station_no",
                "station_name",
                "current_bikes_available",
                "current_spaces_available",
                "predicted_bikes_next_hour",
                "predicted_spaces_next_hour",
                "risk_label",
                "risk_score",
                "action_label",
            ]
        ]

        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            column_config={
                "station_no": "站點編號",
                "station_name": "站點名稱",
                "current_bikes_available": "目前車輛",
                "current_spaces_available": "目前空位",
                "predicted_bikes_next_hour": "預測車輛",
                "predicted_spaces_next_hour": "預測空位",
                "risk_label": "風險等級",
                "risk_score": "風險分數",
                "action_label": "建議動作",
            },
        )


apply_dashboard_styles()

with st.sidebar:
    st.header("條件設定")
    demo_mode = st.checkbox("使用固定範例資料", value=DEMO_MODE_DEFAULT)
    if demo_mode:
        st.markdown(
            """
            <div class="sidebar-note">
                <strong>本機檢視模式</strong>
                使用固定站點與模擬推論結果，不需要啟動 FastAPI 或載入模型檔。
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"FastAPI：{API_BASE_URL}")

station_map = load_station_map(API_BASE_URL, demo_mode)
station_options = station_display_options(station_map)

render_dashboard_header(demo_mode)

with st.sidebar:

    if station_options:
        selected_option = st.selectbox("單站檢視站點", station_options)
        selected_station, selected_station_name = parse_station_option(selected_option)
    else:
        st.error("無法連線至 API 或模型尚未載入")
        selected_station = st.text_input("手動輸入站點編號", "500119005")
        selected_station_name = "未知站點"

    temperature = st.slider("氣溫 (°C)", 10.0, 40.0, 25.0)
    rain = st.slider("降雨量 (mm)", 0.0, 50.0, 0.0)
    st.markdown(
        f"""
        <div class="sidebar-status">
            <span>天氣狀態</span>
            <strong>{escape(rain_label(rain))}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

single_tab, risk_tab = st.tabs(["單站預測", "多站風險排序"])

with single_tab:
    render_single_prediction(
        selected_station,
        selected_station_name,
        temperature,
        rain,
        demo_mode,
    )

with risk_tab:
    render_risk_ranking(
        station_map,
        station_options,
        temperature,
        rain,
        demo_mode,
    )
