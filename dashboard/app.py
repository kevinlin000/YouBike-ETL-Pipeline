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

PREDICTION_RISK_LABELS = {
    "risk-critical": "嚴重缺車",
    "risk-warning": "車輛偏低",
    "risk-normal": "供需穩定",
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
            --ink: #162033;
            --muted: #65748b;
            --faint: #8a97aa;
            --line: #d7e0ea;
            --panel: #ffffff;
            --soft: #f7f9fc;
            --teal: #0f7b72;
            --teal-dark: #0b615a;
            --critical: #b42318;
            --critical-soft: #fff4f2;
            --critical-line: #fecdca;
            --full: #b54708;
            --full-soft: #fffaeb;
            --full-line: #fedf89;
            --watch: #175cd3;
            --watch-soft: #eff8ff;
            --watch-line: #b2ddff;
            --stable: #067647;
            --stable-soft: #ecfdf3;
            --stable-line: #abefc6;
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
            background: #f5f7fa;
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

        [data-testid="stMarkdownContainer"] a[href^="#"] {
            display: none;
        }

        .dashboard-hero {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0.55rem 0 0.9rem;
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

        .overview-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.25rem 0 1.2rem;
        }

        .overview-item {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 0.7rem 0.8rem;
            min-height: 76px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }

        .overview-item span {
            display: block;
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.25rem;
        }

        .overview-item strong {
            display: block;
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.3;
        }

        .overview-item small {
            display: block;
            color: var(--muted);
            margin-top: 0.15rem;
            line-height: 1.35;
        }

        .sidebar-note {
            border: 1px solid var(--line);
            background: #ffffff;
            color: var(--muted);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0.35rem 0 1rem;
        }

        .sidebar-note strong {
            display: block;
            color: var(--ink);
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
            padding: 0.9rem;
            min-height: 154px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }

        .priority-card-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .priority-card .rank {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .priority-card .rank::before {
            content: "";
            display: inline-block;
            width: 0.45rem;
            height: 0.45rem;
            margin-right: 0.38rem;
            border-radius: 999px;
            background: var(--faint);
            vertical-align: 0.08rem;
        }

        .priority-card.risk-critical .rank::before {
            background: var(--critical);
        }

        .priority-card.risk-full .rank::before {
            background: var(--full);
        }

        .priority-card.risk-warning .rank::before,
        .priority-card.risk-dock .rank::before {
            background: var(--watch);
        }

        .priority-card.risk-normal .rank::before {
            background: var(--stable);
        }

        .priority-card .station {
            color: var(--ink);
            font-weight: 700;
            line-height: 1.35;
            min-height: 2.45rem;
            margin-bottom: 0.55rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            border: 1px solid #e5eaf0;
            background: #f6f8fb;
            color: var(--ink);
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.35;
            white-space: nowrap;
        }

        .status-pill.risk-critical {
            border-color: var(--critical-line);
            background: var(--critical-soft);
            color: var(--critical);
        }

        .status-pill.risk-full {
            border-color: var(--full-line);
            background: var(--full-soft);
            color: var(--full);
        }

        .status-pill.risk-warning,
        .status-pill.risk-dock {
            border-color: var(--watch-line);
            background: var(--watch-soft);
            color: var(--watch);
        }

        .status-pill.risk-normal {
            border-color: var(--stable-line);
            background: var(--stable-soft);
            color: var(--stable);
        }

        .priority-card .status-pill {
            background: #ffffff;
            border-color: #e1e7ef;
        }

        .priority-card .stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            color: var(--muted);
            font-size: 0.84rem;
        }

        .priority-card .stats strong {
            color: var(--ink);
            font-weight: 700;
        }

        .priority-card .action-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            border-top: 1px solid var(--line);
            margin-top: 0.72rem;
            padding-top: 0.68rem;
            color: var(--muted);
            font-size: 0.82rem;
        }

        .priority-card .action-row strong {
            color: var(--ink);
            font-weight: 700;
            text-align: right;
        }

        .input-panel,
        .forecast-panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 1rem;
            min-height: 268px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }

        .panel-title {
            margin: 0 0 0.45rem;
            color: var(--ink);
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.35;
        }

        .input-panel p,
        .forecast-panel p {
            margin: 0;
            color: var(--muted);
            line-height: 1.5;
        }

        .station-readout {
            border-top: 1px solid var(--line);
            margin-top: 0.9rem;
            padding-top: 0.8rem;
        }

        .station-readout span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.18rem;
        }

        .station-readout strong {
            display: block;
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.35;
        }

        .forecast-panel {
            display: flex;
            flex-direction: column;
        }

        .forecast-heading {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .panel-kicker {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .forecast-number-row {
            display: flex;
            align-items: baseline;
            gap: 0.4rem;
            margin: 0.65rem 0 0.3rem;
        }

        .forecast-number {
            color: var(--ink);
            font-size: 2.55rem;
            font-weight: 750;
            line-height: 1;
            margin: 0;
        }

        .forecast-unit {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 700;
        }

        .forecast-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            border-top: 1px solid var(--line);
            margin-top: 0.85rem;
            padding-top: 0.75rem;
        }

        .forecast-grid span {
            display: block;
            color: var(--muted);
            font-size: 0.74rem;
            margin-bottom: 0.18rem;
        }

        .forecast-grid strong {
            color: var(--ink);
            font-size: 0.96rem;
        }

        .forecast-action {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            border-top: 1px solid var(--line);
            margin-top: 0.85rem;
            padding-top: 0.75rem;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .forecast-action strong {
            color: var(--ink);
            font-weight: 700;
            text-align: right;
        }

        .forecast-stale {
            color: var(--amber);
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.45rem;
        }

        .risk-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.75rem 0 0.9rem;
        }

        .risk-summary-item {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 0.7rem 0.8rem;
        }

        .risk-summary-item span {
            display: block;
            color: var(--muted);
            font-size: 0.76rem;
            margin-bottom: 0.22rem;
        }

        .risk-summary-item strong {
            color: var(--ink);
            font-size: 1.05rem;
        }

        .table-title {
            color: var(--ink);
            font-weight: 700;
            margin: 0.5rem 0 0.35rem;
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
            background: var(--teal-dark);
            border-color: var(--teal-dark);
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

            .overview-grid,
            .risk-summary-grid,
            .forecast-grid {
                grid-template-columns: 1fr;
            }

            .priority-list {
                grid-template-columns: 1fr;
            }

            .forecast-heading,
            .priority-card-head,
            .forecast-action,
            .priority-card .action-row {
                align-items: flex-start;
                flex-direction: column;
            }

            .priority-card .action-row strong,
            .forecast-action strong {
                text-align: left;
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


def display_mode_label(demo_mode: bool) -> str:
    return "固定範例資料" if demo_mode else "FastAPI 即時服務"


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


def render_overview_grid(
    demo_mode: bool,
    station_count: int,
    selected_station_name: str,
    temperature: float,
    rain: float,
) -> None:
    st.markdown(
        f"""
        <div class="overview-grid">
            <div class="overview-item">
                <span>服務狀態</span>
                <strong>{escape(display_mode_label(demo_mode))}</strong>
                <small>{escape("不依賴後端服務" if demo_mode else "連線模型 API")}</small>
            </div>
            <div class="overview-item">
                <span>支援站點</span>
                <strong>{station_count} 站</strong>
                <small>目前可供工作台檢視</small>
            </div>
            <div class="overview-item">
                <span>單站焦點</span>
                <strong>{escape(selected_station_name)}</strong>
                <small>側欄可切換站點</small>
            </div>
            <div class="overview-item">
                <span>天氣條件</span>
                <strong>{temperature:.1f}°C / {escape(rain_label(rain))}</strong>
                <small>降雨量 {rain:.1f} mm</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_risk(prediction: int) -> tuple[str, str, str]:
    if prediction <= 2:
        return "risk-critical", "建議優先補車", f"預測剩餘 {prediction} 台，短時間內可能影響借車。"
    if prediction <= 5:
        return "risk-warning", "列入觀察名單", f"預測剩餘 {prediction} 台，需留意下一輪調度。"
    return "risk-normal", "維持監控", f"預測剩餘 {prediction} 台，供需暫時穩定。"


def render_single_input_panel(selected_station: str, selected_station_name: str, bikes_now: int) -> None:
    st.markdown(
        f"""
        <div class="input-panel">
            <div class="panel-title">目前站點狀態</div>
            <p>調整目前可借車輛與天氣條件後，工作台會重新評估模型時窗內的站點水位。</p>
            <div class="station-readout">
                <span>站點</span>
                <strong>{escape(selected_station_name)}</strong>
            </div>
            <div class="station-readout">
                <span>站點編號</span>
                <strong>{escape(selected_station)}</strong>
            </div>
            <div class="station-readout">
                <span>目前可借車輛</span>
                <strong>{bikes_now} 台</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_panel(
    result: dict | None,
    bikes_now: int,
    temperature: float,
    rain: float,
    stale: bool = False,
) -> None:
    if not result:
        st.markdown(
            """
            <div class="forecast-panel">
                <div class="forecast-heading">
                    <div>
                        <div class="panel-kicker">預測結果</div>
                        <div class="panel-title">模型時窗供需</div>
                    </div>
                    <span class="status-pill">等待輸入</span>
                </div>
                <p>尚未取得結果。設定站點與目前車輛後，按下更新即可檢視模型時窗內的供需風險。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction = int(result["predicted_bikes_next_hour"])
    delta = prediction - bikes_now
    tone_class, action, description = prediction_risk(prediction)
    delta_label = f"+{delta} 台" if delta > 0 else f"{delta} 台"
    stale_message = (
        '<div class="forecast-stale">輸入條件已變更，請重新更新結果。</div>' if stale else ""
    )
    panel_html = (
        f'<div class="forecast-panel {tone_class}">'
        '<div class="forecast-heading">'
        "<div>"
        '<div class="panel-kicker">預測結果</div>'
        '<div class="panel-title">模型時窗供需</div>'
        "</div>"
        f'<span class="status-pill {tone_class}">{escape(PREDICTION_RISK_LABELS[tone_class])}</span>'
        "</div>"
        '<div class="forecast-number-row">'
        f'<div class="forecast-number">{prediction}</div>'
        '<div class="forecast-unit">台</div>'
        "</div>"
        f"<p>{escape(description)}</p>"
        f"{stale_message}"
        '<div class="forecast-grid">'
        "<div><span>相對目前</span>"
        f"<strong>{escape(delta_label)}</strong></div>"
        "<div><span>氣溫</span>"
        f"<strong>{temperature:.1f}°C</strong></div>"
        "<div><span>降雨</span>"
        f"<strong>{rain:.1f} mm</strong></div>"
        "</div>"
        '<div class="forecast-action">'
        "<span>建議動作</span>"
        f"<strong>{escape(action)}</strong>"
        "</div>"
        "</div>"
    )

    st.markdown(panel_html, unsafe_allow_html=True)


def normalize_risk_rows(rows_df: pd.DataFrame) -> list[dict]:
    rows = rows_df[["station_no", "bikes_available", "spaces_available"]].to_dict("records")
    return [
        {
            "station_no": str(row["station_no"]),
            "bikes_available": int(row["bikes_available"]),
            "spaces_available": int(row["spaces_available"]),
        }
        for row in rows
    ]


def risk_request_signature(stations: list[dict], temperature: float, rain: float, demo_mode: bool) -> tuple:
    return (
        demo_mode,
        round(float(temperature), 2),
        round(float(rain), 2),
        tuple(
            (
                station["station_no"],
                int(station["bikes_available"]),
                int(station["spaces_available"]),
            )
            for station in stations
        ),
    )


def render_risk_summary(result_df: pd.DataFrame) -> None:
    severe_count = int(result_df["risk_label"].isin(["嚴重缺車", "滿站風險"]).sum())
    monitor_count = int(result_df["risk_label"].isin(["車輛偏低", "空位偏低"]).sum())
    stable_count = int((result_df["risk_label"] == "供需穩定").sum())
    top_action = str(result_df.iloc[0]["action_label"])
    avg_score = result_df["risk_score"].mean()

    st.markdown(
        f"""
        <div class="risk-summary-grid">
            <div class="risk-summary-item">
                <span>高優先處理</span>
                <strong>{severe_count} 站</strong>
            </div>
            <div class="risk-summary-item">
                <span>觀察名單</span>
                <strong>{monitor_count} 站</strong>
            </div>
            <div class="risk-summary-item">
                <span>供需穩定</span>
                <strong>{stable_count} 站</strong>
            </div>
            <div class="risk-summary-item">
                <span>首要動作</span>
                <strong>{escape(top_action)} / 均分 {avg_score:.0f}</strong>
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
        bike_flow = (
            f"{int(row['current_bikes_available'])} → "
            f"{int(row['predicted_bikes_next_hour'])} 台"
        )
        space_flow = (
            f"{int(row['current_spaces_available'])} → "
            f"{int(row['predicted_spaces_next_hour'])} 格"
        )
        cards.append(
            "<div class=\"priority-card "
            f"{risk_class}\">"
            "<div class=\"priority-card-head\">"
            f"<div class=\"rank\">處理順位 {index}</div>"
            f"<span class=\"status-pill {risk_class}\">{escape(row['risk_label'])} · {int(row['risk_score'])} 分</span>"
            "</div>"
            f"<div class=\"station\">{escape(row['station_name'])}</div>"
            "<div class=\"stats\">"
            f"<span>車輛 <strong>{escape(bike_flow)}</strong></span>"
            f"<span>空位 <strong>{escape(space_flow)}</strong></span>"
            "</div>"
            "<div class=\"action-row\">"
            "<span>建議動作</span>"
            f"<strong>{escape(row['action_label'])}</strong>"
            "</div>"
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

    input_col, result_col = st.columns([0.52, 0.48], gap="medium")

    with input_col:
        bikes_now = st.slider("目前可借車輛", 0, 100, 15)
        single_inputs = (
            demo_mode,
            selected_station,
            int(bikes_now),
            round(float(temperature), 2),
            round(float(rain), 2),
        )

        if demo_mode:
            st.session_state["single_prediction_result"] = demo_predict_station(
                selected_station,
                bikes_now,
                temperature,
                rain,
            )
            st.session_state["single_prediction_inputs"] = single_inputs

        render_single_input_panel(selected_station, selected_station_name, bikes_now)
        update_clicked = st.button("更新單站預測", type="primary", width="stretch")

    if update_clicked:
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

        st.session_state["single_prediction_result"] = result
        st.session_state["single_prediction_inputs"] = single_inputs

    stored_inputs = st.session_state.get("single_prediction_inputs")
    result = st.session_state.get("single_prediction_result")
    if stored_inputs and stored_inputs[0] != demo_mode:
        result = None

    stale = bool(result and stored_inputs != single_inputs and not demo_mode)
    with result_col:
        render_prediction_panel(result, bikes_now, temperature, rain, stale=stale)


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

    stations = normalize_risk_rows(edited_df)
    current_signature = risk_request_signature(stations, temperature, rain, demo_mode)

    if demo_mode and st.session_state.get("risk_ranking_signature") != current_signature:
        st.session_state["risk_ranking_result"] = demo_rank_station_risks(
            stations,
            temperature,
            rain,
        )
        st.session_state["risk_ranking_signature"] = current_signature

    if st.button("更新風險排序", type="primary", width="stretch"):
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

        st.session_state["risk_ranking_result"] = risks
        st.session_state["risk_ranking_signature"] = current_signature

    risks = st.session_state.get("risk_ranking_result")
    stored_signature = st.session_state.get("risk_ranking_signature")
    if stored_signature and stored_signature[0] != demo_mode:
        risks = None

    if risks:
        result_df = pd.DataFrame(risks)
        result_df["station_name"] = result_df["station_no"].map(lambda sid: station_name(station_map, sid))
        result_df["risk_label"] = result_df["risk_level"].map(risk_level_label)
        result_df["action_label"] = result_df["suggested_action"].map(suggested_action_label)

        if stored_signature != current_signature and not demo_mode:
            st.warning("輸入條件已變更，請重新更新風險排序。")

        render_risk_summary(result_df)
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

        st.markdown('<div class="table-title">完整排序表</div>', unsafe_allow_html=True)
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
    else:
        st.info("完成站點與水位設定後，按下更新即可產生調度排序。")


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

render_dashboard_header(demo_mode)
render_overview_grid(
    demo_mode,
    len(station_options),
    selected_station_name,
    temperature,
    rain,
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
