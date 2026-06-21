# API Observability 設計

這份文件整理 FastAPI 推論服務目前具備的可觀測性設計：Prometheus-style metrics、JSON request logs、request id tracing，以及若要接到 Grafana / alert rules 時可以怎麼判讀。

本專案目前是作品展示，不是正在營運的 production service。以下內容代表服務已具備可接監控系統的基本訊號，但不代表已部署完整 Prometheus、Grafana、Log Aggregation 或 On-call 流程。

## 目前訊號

| 類型 | 位置 | 用途 |
| --- | --- | --- |
| Liveness | `GET /health` | 確認 API process 活著，並回報模型與資料資源載入狀態 |
| Readiness | `GET /ready` | 確認服務是否具備推論能力；缺模型或 station metadata 時回 `503` |
| Metrics | `GET /metrics` | 輸出 Prometheus-style request count、5xx count 與 latency histogram |
| JSON logs | stdout | 每次 request completion 記錄 request id、path、status code、latency、model version |
| Request tracing | `X-Request-ID` | 呼叫端可帶入 request id，API 會回傳同一個 id 並寫入 log |

## Metrics

`/metrics` 目前輸出以下指標：

| Metric | Type | 說明 |
| --- | --- | --- |
| `youbike_api_requests_total` | counter | API 已處理 request 數 |
| `youbike_api_request_errors_total` | counter | HTTP 5xx response 數 |
| `youbike_api_request_duration_ms_sum` | counter | request duration 毫秒總和，保留給本機快速閱讀 |
| `youbike_api_request_duration_ms_count` | counter | request duration 筆數 |
| `youbike_api_request_duration_ms_max` | gauge | process 啟動後觀察到的最大 request duration |
| `youbike_api_request_duration_seconds_bucket` | histogram bucket | Prometheus histogram bucket，用於 p95 / p99 查詢 |
| `youbike_api_request_duration_seconds_sum` | histogram sum | request duration 秒數總和 |
| `youbike_api_request_duration_seconds_count` | histogram count | request duration 筆數 |

所有 request 指標都使用 `method` 與 `path` 作為 label。`/metrics` 本身不會被納入統計，避免 Prometheus scrape 讓 request count 自我膨脹。

Latency histogram bucket：

```text
0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, +Inf seconds
```

這組 bucket 適合本機 demo 與低延遲 API path。若服務接上真實模型 artifact、外部 DB 或部署到雲端，應依實測 latency 重新調整 bucket。

## Prometheus scrape 範例

```yaml
scrape_configs:
  - job_name: youbike-api
    metrics_path: /metrics
    static_configs:
      - targets:
          - localhost:8000
```

若透過 Docker Compose 或雲端 VM 部署，`targets` 應改成容器網路名稱、VM private IP 或 load balancer endpoint。

## PromQL 範例

每秒 request 數：

```promql
sum by (path) (
  rate(youbike_api_requests_total[5m])
)
```

5xx error rate：

```promql
sum by (path) (
  rate(youbike_api_request_errors_total[5m])
)
/
sum by (path) (
  rate(youbike_api_requests_total[5m])
)
```

p95 latency：

```promql
histogram_quantile(
  0.95,
  sum by (le, path) (
    rate(youbike_api_request_duration_seconds_bucket[5m])
  )
)
```

p99 latency：

```promql
histogram_quantile(
  0.99,
  sum by (le, path) (
    rate(youbike_api_request_duration_seconds_bucket[5m])
  )
)
```

Readiness failure rate：

```promql
sum(
  rate(youbike_api_request_errors_total{path="/ready"}[5m])
)
```

## 建議 Dashboard

如果接到 Grafana，建議保留這幾個 panel：

| Panel | Query | 用途 |
| --- | --- | --- |
| Request rate by path | `sum by (path) (rate(youbike_api_requests_total[5m]))` | 看 API 是否有流量、哪個 endpoint 被打最多 |
| 5xx error rate by path | `sum by (path) (rate(youbike_api_request_errors_total[5m]))` | 快速發現 `/predict` 或 `/stations/risk` 失敗 |
| p95 latency by path | `histogram_quantile(0.95, sum by (le, path) (rate(youbike_api_request_duration_seconds_bucket[5m])))` | 檢查推論路徑延遲是否變慢 |
| p99 latency by path | `histogram_quantile(0.99, sum by (le, path) (rate(youbike_api_request_duration_seconds_bucket[5m])))` | 看尾端延遲 |
| Max latency by path | `max by (path) (youbike_api_request_duration_ms_max)` | 本機展示時快速看 process 內最大值 |

## Alert rule 草案

以下門檻是本機 / demo profile 的初版 guardrail，不是正式 production SLO。正式上線前要先用固定硬體、固定流量模型與長時間測試校準門檻。

```yaml
groups:
  - name: youbike-api-demo
    rules:
      - alert: YouBikeApiHighErrorRate
        expr: |
          (
            sum(rate(youbike_api_request_errors_total[5m]))
            /
            sum(rate(youbike_api_requests_total[5m]))
          ) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: YouBike API 5xx error rate is above 1%.

      - alert: YouBikeApiHighP95Latency
        expr: |
          histogram_quantile(
            0.95,
            sum by (le, path) (
              rate(youbike_api_request_duration_seconds_bucket[5m])
            )
          ) > 0.75
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: YouBike API p95 latency is above 750ms.

      - alert: YouBikeApiReadinessFailing
        expr: |
          sum(rate(youbike_api_request_errors_total{path="/ready"}[5m])) > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: YouBike API readiness endpoint is returning 5xx responses.
```

## Log 查詢方向

API log 是 JSON event 格式。每次 request completion 至少包含：

- `timestamp`
- `service`
- `event`
- `request_id`
- `method`
- `path`
- `status_code`
- `duration_ms`
- `model_version`

排查問題時可以用 `request_id` 從 dashboard 或 API client 回到後端 log。例如使用者回報某次 `/stations/risk` 回應異常時，先看 response header 的 `X-Request-ID`，再查同一個 request id 的 log。

## 面試時的說法

可以這樣講：

> 這個服務不是只有提供 `/predict`，我也補了基本可觀測性。`/health` 和 `/ready` 分開處理 liveness 與 inference readiness；`/metrics` 提供 request count、5xx count 和 Prometheus histogram，所以可以用 `histogram_quantile` 看 p95 / p99 latency；每個 request 都有 `X-Request-ID`，並用 JSON log 記錄 status code、latency 和 model version。這還不是完整 production monitoring，但已經具備接 Prometheus、Grafana 和 alert rules 的基礎。

## English Summary

The FastAPI service exposes basic observability primitives: liveness/readiness endpoints, Prometheus-style request metrics with latency histograms, JSON request logs, and `X-Request-ID` tracing. This document provides Prometheus scrape examples, PromQL queries, Grafana panel suggestions, and draft alert rules. It is a monitoring design note for a portfolio project, not a claim that a full production observability stack is already deployed.
