# API 效能與容量測試說明

這份文件說明本專案如何用可重跑的本機 benchmark 檢查 FastAPI 推論服務的基本延遲、錯誤率與併發行為。它的定位是後端 / AI 應用作品的工程證據，不是正式 production SLO 或雲端壓測報告。

## 測試範圍

benchmark 會固定呼叫三個 endpoint：

| Endpoint | 目的 |
| --- | --- |
| `GET /ready` | 檢查推論服務是否具備可服務狀態 |
| `POST /predict` | 檢查單站模型推論路徑 |
| `POST /stations/risk` | 檢查多站點風險排序路徑 |

測試 payload 使用固定站點與固定天氣條件，確保每次測試都在同一組 API contract 下執行。若 API 啟用 `API_DEMO_MODE=true`，測試不依賴 MySQL、模型檔或 Docker Compose。

## 啟動方式

先啟動 API demo mode：

```bash
make install-app
make api-demo
```

另開一個 terminal 執行短測：

```bash
make api-benchmark
```

若要執行較長的容量探測：

```bash
make api-load-test
```

`make api-load-test` 會把原始測試結果寫到 `.scratch/benchmarks/api-capacity.json`。`.scratch/` 是本機工作區，不會提交到 Git。

## Benchmark profiles

`scripts/benchmark_api.py` 提供三種 profile：

| Profile | Requests / endpoint | Warmup / endpoint | Concurrency | Timeout | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `smoke` | 5 | 1 | 1 | 5s | 快速確認 API contract 與連線正常 |
| `demo` | 30 | 2 | 5 | 5s | 面試展示前的短時間併發檢查 |
| `capacity` | 120 | 5 | 12 | 10s | 較長的本機容量探測，用於討論延遲與錯誤率 |

也可以手動覆蓋 profile 參數：

```bash
python scripts/benchmark_api.py \
  --base-url http://127.0.0.1:8000 \
  --profile capacity \
  --requests 200 \
  --concurrency 16 \
  --json-output .scratch/benchmarks/api-capacity-custom.json
```

## 輸出欄位

benchmark 會輸出每個 endpoint 的摘要：

- `requests`：實際記錄的 request 數。
- `errors`：非 2xx / 3xx 或連線失敗數。
- `error %`：錯誤比例。
- `rps`：整輪測試期間估算的每秒 request 數。
- `min / avg / p50 / p95 / p99 / max ms`：延遲分佈。

同時也會輸出 assessment table：

| Status | 意義 |
| --- | --- |
| `pass` | 錯誤率與 p95 latency 都在本機 profile 門檻內 |
| `watch` | 無錯誤，但 p95 latency 高於觀察門檻，適合進一步確認硬體或服務狀態 |
| `fail` | 錯誤率或 p95 latency 超過失敗門檻，應檢查 API、模型推論路徑或本機資源 |

目前 profile 的門檻設計是本機展示用 guardrail：

| Profile | p95 watch | p95 fail | Max error rate |
| --- | ---: | ---: | ---: |
| `smoke` | 500 ms | 1000 ms | 0% |
| `demo` | 500 ms | 1000 ms | 0% |
| `capacity` | 750 ms | 1500 ms | 1% |

這些門檻不是正式 SLO。它們的用途是讓本機 demo 前可以快速發現 API 沒啟動、endpoint 壞掉、推論路徑異常變慢或錯誤率上升。

## 本機實測摘要

以下是 2026-06-21 在本機 API demo mode 執行 `capacity` profile 的摘要。這筆結果用來確認 benchmark 工具與 demo API path 可正常運作，不代表 production capacity。

| endpoint | requests | errors | error % | rps | p50 ms | p95 ms | p99 ms | max ms | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| predict | 120 | 0 | 0.00 | 120.05 | 29.29 | 65.12 | 128.10 | 130.38 | pass |
| ready | 120 | 0 | 0.00 | 120.05 | 24.03 | 48.90 | 117.87 | 122.78 | pass |
| risk | 120 | 0 | 0.00 | 120.05 | 29.54 | 60.92 | 129.23 | 129.39 | pass |

## 面試時的說法

可以這樣說明：

> 我沒有把這個 benchmark 包裝成 production load test。這裡做的是本機可重跑的容量探測，固定打 `/ready`、`/predict` 和 `/stations/risk`，觀察錯誤率、throughput 和 p95 / p99 latency。短測用來確認 demo 前 API contract 沒壞，capacity profile 則用來討論模型推論服務在本機併發下的延遲分佈。若要上線，下一步會改成 k6 或 Locust，固定硬體規格、ramp-up pattern、長時間測試、Prometheus dashboard 和 alert rule。

## 目前限制

- 測試在本機執行，結果受 CPU、背景程序、Python runtime、uvicorn reload mode 與網路 loopback 影響。
- API demo mode 使用固定資料與 deterministic mock prediction，不代表真實模型 artifact 的 production latency。
- 測試沒有模擬真實流量分佈、尖峰突增、長時間 soak test 或多 instance 部署。
- 測試沒有取代 profiling；若 latency 異常，仍需要用 logs、metrics 與程式 profiling 找瓶頸。

## 後續可擴充方向

1. 用 k6 或 Locust 補 ramp-up、constant arrival rate 與 soak test。
2. 將 benchmark 結果與 `/metrics` 串接，建立 latency dashboard。
3. 對正式模型 artifact 測試 CPU inference latency，區分 demo mode 與 real inference mode。
4. 補 cache 或 precompute 後，比較 `/stations/risk` 的 p95 latency 改善幅度。

## English Summary

This document describes the local FastAPI benchmark profiles for the YouBike prediction service. The benchmark calls `/ready`, `/predict`, and `/stations/risk`, reports error rate, throughput, p50/p95/p99 latency, and emits a simple pass/watch/fail assessment. It is a reproducible local engineering check for interviews and demos, not a production load test or SLO claim.
