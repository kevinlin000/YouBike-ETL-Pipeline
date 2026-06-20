# ADR 0002: Dashboard 固定範例資料模式

## 狀態

Accepted

## 背景

Streamlit dashboard 原本依賴 FastAPI、模型檔、Docker Compose 與本機服務啟動順序。若只是要檢視 dashboard 的操作流程，這些依賴會讓本地展示與截圖重現變得不穩定。

因此 dashboard 需要一個不依賴 live service 的固定範例資料模式，用固定站點資料與模擬推論結果呈現單站預測與多站風險排序流程。

## 決策

保留可重現的固定範例資料模式，可透過以下方式啟用：

```bash
DASHBOARD_DEMO_MODE=true streamlit run dashboard/app.py
```

或：

```bash
make dashboard-demo
```

固定範例資料模式會使用固定 station fixtures 與可重現的模擬推論結果，呈現：

- 單站預測流程。
- 多站缺車與滿站風險排序。
- `stock_out`、`full_load` 等風險標籤。
- `rebalance_in`、`rebalance_out` 等建議動作。

## 後果

- Dashboard 可以在沒有 FastAPI、模型檔、Docker Compose 或 cloud resources 的情況下檢視。
- README 中的 GIF 與截圖可以用穩定資料重現。
- 固定範例資料模式必須明確標示為流程展示，不可作為模型表現證據。
- 模型表現仍以 training/evaluation artifact 和 [`../lstm_evaluation_report.md`](../lstm_evaluation_report.md) 為準。

## English Summary

The dashboard fixed-sample-data mode uses fixed station fixtures and reproducible simulated predictions so the UI flow can be inspected without live FastAPI, model files, Docker Compose, or cloud resources. It is a workflow demonstration, not model-performance evidence.
