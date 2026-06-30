# Streamlit Community Cloud 部署筆記

本文件記錄 dashboard demo 的雲端部署方式。目標是讓面試官可以直接打開展示工作台，不需要在本機啟動 FastAPI、MySQL、模型檔或 Docker Compose。

這個部署應定位為作品集展示，不是正式 production service。雲端 demo 使用固定範例站點與可重現的模擬推論結果，只展示操作流程與 API response shape，不代表模型評估結果。

## 部署設定

在 Streamlit Community Cloud 建立 app 時使用以下設定：

| 欄位 | 值 |
| --- | --- |
| Repository | `kevinlin000/YouBike-ETL-Pipeline` |
| Branch | `main` |
| Main file path | `dashboard/app.py` |

App secrets 設定：

```toml
DASHBOARD_DEMO_MODE = true
```

公開部署如果沒有設定 `API_BASE_URL`，dashboard 會自動使用固定範例資料模式。仍建議明確設定 `DASHBOARD_DEMO_MODE = true`，避免後續調整部署參數時誤切到 live mode。

如果之後有公開可用的 FastAPI 服務，才需要改成 live mode，並新增：

```toml
DASHBOARD_DEMO_MODE = false
API_BASE_URL = "https://your-api.example.com"
```

目前建議維持 `DASHBOARD_DEMO_MODE = true`，因為它最適合求職作品展示：啟動快、沒有外部服務依賴，也不會暴露資料庫或模型部署細節。

## 依賴套件檔案

Streamlit demo 使用 `dashboard/requirements.txt`，只保留 dashboard 必需套件：

- `streamlit`
- `pandas`
- `requests`

不要讓雲端 demo 依賴根目錄的歷史 analysis environment。根目錄 `requirements.txt` 保留課程與 notebook 時期的完整環境脈絡，不適合作為輕量 dashboard demo 的部署 manifest。

## 部署後檢查

部署完成後，檢查以下項目：

1. App 首頁可以開啟，左側預設勾選「使用固定範例資料」。
2. 單站水位預測可以更新結果，不會連到 `http://api:8000`。
3. 多站風險排序可以重新產生排序結果。
4. README 的 live demo 連結只在確認網址可用後再加入，不要先放 placeholder。

## README 更新規則

目前線上展示網址：

```text
https://youbike-etl-pipeline-8dyh8p6fb3m5kxkpwlhezb.streamlit.app
```

若之後重新建立 app、網址改變，再更新 README：

- 在 badges 下方更新線上 dashboard 連結。
- 在「快速展示」補上雲端展示網址。
- 保留文字說明：雲端 demo 使用固定範例資料，不代表模型評估結果。
